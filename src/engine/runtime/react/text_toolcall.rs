//! Recover tool-call arguments from tool-call template markup leaked through
//! the text channel.
//!
//! Background (session 20260923_263149bf, mimo-v2.6-flash): the provider
//! occasionally writes the same tool call on two channels — the native
//! tool_call stream's `arguments` is truncated mid-way (e.g. a 15-byte
//! `{"fork_turns":"` with the finish falsely reporting `Stop`), while the
//! content text channel leaks the chat template's call markup:
//!
//! ```text
//! ...<tool_call><function=spawn_agent><parameter=fork_turns>none<parameter=task>只读调研…
//! ```
//!
//! The leaked markup is often more complete than the truncated native
//! arguments (it carries the model's original intent), so the truncation
//! guard can salvage the call from here before discarding it. The parser
//! recognizes only the two observed shapes: the parameter-markup style
//! (closing tag optional) and the Hermes JSON style
//! (`<tool_call>{"name":…,"arguments":…}</tool_call>`).
//!
//! Known limitation: a `<parameter=` literal appearing again inside a
//! parameter value is cut off early; an incomplete JSON-style block
//! (JSON parse failure) cannot be recovered.

use serde_json::Value;

const OPEN: &str = "<tool_call>";
const CLOSE: &str = "</tool_call>";
const PARAM: &str = "<parameter=";

/// A tool call recovered from the leaked markup: function name + raw string
/// parameters (declaration order preserved; for duplicate names the first one
/// wins). Type coercion is done by [`build_arguments`] based on the schema;
/// this struct only performs plain-text parsing.
pub struct HarvestedCall {
    pub name: String,
    pub params: Vec<(String, String)>,
}

/// Scan text for all `<tool_call>…` blocks and parse each one; blocks that
/// cannot be parsed are silently skipped (the phrase occasionally appearing in
/// normal text must not raise an error, let alone interrupt the dialog).
pub fn harvest(text: &str) -> Vec<HarvestedCall> {
    let mut out = Vec::new();
    let mut rest = text;
    while let Some(start) = rest.find(OPEN) {
        let after_open = &rest[start + OPEN.len()..];
        // Block body: up to the closing tag, the next block's start, or the end
        // of the text (leaked markup often has no closing tag).
        let body_len = after_open
            .find(CLOSE)
            .map(|i| i + CLOSE.len())
            .or_else(|| after_open.find(OPEN))
            .unwrap_or(after_open.len());
        if body_len == 0 {
            break; // Defensive: impossible path, avoid an infinite loop
        }
        if let Some(call) = parse_block(&after_open[..body_len]) {
            out.push(call);
        }
        rest = &after_open[body_len..];
    }
    out
}

fn parse_block(body: &str) -> Option<HarvestedCall> {
    let body = body.strip_suffix(CLOSE).unwrap_or(body);
    let body = body.trim();

    // Hermes JSON style: the block body itself is `{","arguments":{…}}`.
    if body.starts_with('{') {
        let v: Value = serde_json::from_str(body).ok()?;
        let name = v.get("name").and_then(Value::as_str)?.trim().to_string();
        if name.is_empty() {
            return None;
        }
        let args = v
            .get("arguments")
            .cloned()
            .unwrap_or_else(|| Value::Object(serde_json::Map::new()));
        let obj = args.as_object()?;
        // Flatten uniformly into raw string parameters; objects/arrays are
        // serialized to JSON text, and build_arguments restores the types
        // according to the schema.
        let params = obj
            .iter()
            .map(|(k, val)| {
                let raw = match val {
                    Value::String(s) => s.clone(),
                    other => other.to_string(),
                };
                (k.clone(), raw)
            })
            .collect();
        return Some(HarvestedCall { name, params });
    }

    // parameter-markup style: `<function=NAME><parameter=KEY>VALUE…`
    let rest = body.strip_prefix("<function=")?;
    let gt = rest.find('>')?;
    let name = rest[..gt].trim().to_string();
    if name.is_empty() {
        return None;
    }
    let mut params = Vec::new();
    let mut seen = std::collections::HashSet::new();
    let mut cursor = &rest[gt + 1..];
    while let Some(p) = cursor.find(PARAM) {
        let after = &cursor[p + PARAM.len()..];
        // When a parameter header is truncated (no '>'), keep the parameters
        // already collected — partial recovery beats discarding everything.
        let Some(key_gt) = after.find('>') else {
            break;
        };
        let key = after[..key_gt].trim().to_string();
        let value_part = &after[key_gt + 1..];
        // The value ends at the nearest next parameter header / closing tag /
        // end of text.
        let value_len = value_part
            .find(PARAM)
            .into_iter()
            .chain(value_part.find(CLOSE))
            .min()
            .unwrap_or(value_part.len());
        let value = value_part[..value_len].trim().to_string();
        if !key.is_empty() && seen.insert(key.clone()) {
            params.push((key, value));
        }
        cursor = &value_part[value_len..];
    }
    Some(HarvestedCall { name, params })
}

/// Assemble the raw string parameters into an arguments JSON string according
/// to the tool schema.
///
/// - A `required` field declared by the schema is missing → `None`: a salvaged
///   call missing a required field is bound to fail typed parsing at execution
///   time, which merely swaps the death-spiral entry from "truncation" to
///   "missing argument" — better not to salvage it.
/// - Scalars are coerced by schema type (integer/number/boolean); object/array
///   first try JSON parsing to restore them; when the schema is missing or the
///   type is unknown, the value stays a string (in the observed leaks all
///   parameters were long text fields, so string is a safe default).
pub fn build_arguments(call: &HarvestedCall, schema: Option<&Value>) -> Option<String> {
    let mut obj = serde_json::Map::new();
    for (key, raw) in &call.params {
        let ty = schema
            .and_then(|s| s.get("properties"))
            .and_then(|p| p.get(key))
            .and_then(|p| p.get("type"))
            .and_then(Value::as_str);
        obj.insert(key.clone(), coerce(raw, ty));
    }
    if let Some(required) = schema
        .and_then(|s| s.get("required"))
        .and_then(Value::as_array)
    {
        for field in required.iter().filter_map(Value::as_str) {
            if !obj.contains_key(field) {
                return None;
            }
        }
    }
    Some(Value::Object(obj).to_string())
}

fn coerce(raw: &str, ty: Option<&str>) -> Value {
    match ty {
        Some("integer") => raw
            .parse::<i64>()
            .map(Value::from)
            .unwrap_or_else(|_| Value::String(raw.to_string())),
        Some("number") => raw
            .parse::<f64>()
            .map(Value::from)
            .unwrap_or_else(|_| Value::String(raw.to_string())),
        Some("boolean") => match raw {
            "true" => Value::Bool(true),
            "false" => Value::Bool(false),
            _ => Value::String(raw.to_string()),
        },
        Some("object") | Some("array") => {
            serde_json::from_str(raw).unwrap_or_else(|_| Value::String(raw.to_string()))
        }
        _ => Value::String(raw.to_string()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn harvests_observed_leak_with_unterminated_value() {
        // Verbatim leak from session 20260923_263149bf turn 2: no closing tag,
        // the task value was cut off mid-sentence by the server — it must still
        // be salvaged (the content is already 95% usable).
        let text = "参数流被截断了，我改为逐个发起、缩短任务描述。<tool_call>\
<function=spawn_agent><parameter=fork_turns>none<parameter=task>只读调研，不改文件。\
分析 /Users/kangzengchen/source/buka/demo/codex。产出中文结构化报告，";
        let calls = harvest(text);
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].name, "spawn_agent");
        assert_eq!(calls[0].params[0], ("fork_turns".into(), "none".into()));
        assert_eq!(
            calls[0].params[1].1,
            "只读调研，不改文件。分析 /Users/kangzengchen/source/buka/demo/codex。产出中文结构化报告，"
        );
        let args = build_arguments(&calls[0], None).expect("arguments build");
        let v: Value = serde_json::from_str(&args).expect("valid JSON");
        assert_eq!(v["fork_turns"], "none");
        assert!(v["task"].as_str().unwrap().ends_with('，'));
    }

    #[test]
    fn value_stops_at_closing_tag_and_prose_after_is_ignored() {
        let text = "前言<tool_call><function=spawn_agent><parameter=task_name>r1\
</tool_call>后记文字不属于参数。";
        let calls = harvest(text);
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].params.len(), 1);
        assert_eq!(calls[0].params[0], ("task_name".into(), "r1".into()));
    }

    #[test]
    fn hermes_json_style_block_recovered() {
        let text = "before <tool_call>{\"name\":\"read_file\",\"arguments\":\
{\"path\":\"/a/b\",\"limit\":5}}</tool_call> after";
        let calls = harvest(text);
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].name, "read_file");
        // JSON objects are unordered: compare by key, don't rely on the Map's
        // internal ordering.
        let params: std::collections::HashMap<_, _> = calls[0].params.iter().cloned().collect();
        assert_eq!(params.get("path").map(String::as_str), Some("/a/b"));
        assert_eq!(params.get("limit").map(String::as_str), Some("5"));
        assert_eq!(calls[0].params.len(), 2);
    }

    #[test]
    fn coerces_scalars_by_schema_type() {
        let schema = serde_json::json!({
            "type": "object",
            "properties": {
                "limit": { "type": "integer" },
                "all": { "type": "boolean" },
                "task": { "type": "string" },
            },
            "required": ["limit", "task"]
        });
        let call = HarvestedCall {
            name: "t".into(),
            params: vec![
                ("limit".into(), "42".into()),
                ("all".into(), "true".into()),
                ("task".into(), "只读调研".into()),
            ],
        };
        let args = build_arguments(&call, Some(&schema)).expect("build");
        let v: Value = serde_json::from_str(&args).unwrap();
        assert_eq!(v["limit"], 42);
        assert_eq!(v["all"], true);
        assert_eq!(v["task"], "只读调研");
    }

    #[test]
    fn missing_required_field_rejected() {
        // A leak carrying only task_name is missing the required message →
        // reject the salvage, leave it on the re-issue path.
        let schema = serde_json::json!({
            "type": "object",
            "properties": {
                "task_name": { "type": "string" },
                "message": { "type": "string" },
            },
            "required": ["task_name", "message"]
        });
        let call = HarvestedCall {
            name: "spawn_agent".into(),
            params: vec![("task_name".into(), "research".into())],
        };
        assert!(build_arguments(&call, Some(&schema)).is_none());
    }

    #[test]
    fn truncated_parameter_header_keeps_earlier_params() {
        // The text is cut off at `<parameter=tas`: the earlier parameters must
        // be preserved.
        let text = "<tool_call><function=spawn_agent><parameter=fork_turns>none<parameter=tas";
        let calls = harvest(text);
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].params.len(), 1);
        assert_eq!(calls[0].params[0], ("fork_turns".into(), "none".into()));
    }

    #[test]
    fn multiple_blocks_returned_in_order() {
        let text = "<tool_call><function=a_tool><parameter=x>1</tool_call>\
mid<tool_call><function=a_tool><parameter=x>2</tool_call>";
        let calls = harvest(text);
        assert_eq!(calls.len(), 2);
        assert_eq!(calls[0].params[0].1, "1");
        assert_eq!(calls[1].params[0].1, "2");
    }

    #[test]
    fn no_markup_or_plain_mention_yields_empty() {
        assert!(harvest("普通正文，没有调用。").is_empty());
        // Mentions the phrase but is not a real block: no <function= header →
        // skip
        assert!(harvest("我们讨论 <tool_call> 标签的历史").is_empty());
        // Parameter header truncated with no parameters at all → the
        // empty-parameter block is still returned; the caller validates it
        // against the schema
        let calls = harvest("<tool_call><function=spawn_agent>");
        assert_eq!(calls.len(), 1);
        assert!(calls[0].params.is_empty());
    }
}
