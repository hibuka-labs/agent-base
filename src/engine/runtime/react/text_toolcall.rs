//! 从文本通道泄漏的 tool-call 模板 markup 中恢复工具调用参数。
//!
//! 背景（session 20260923_263149bf，mimo-v2.6-flash）：provider 偶发把同一个
//! 工具调用写在两个通道 —— native tool_call 流的 `arguments` 中途截断（如
//! 15 字节的 `{"fork_turns":"`，finish 谎报 `Stop`），而 content 文本通道里
//! 泄漏出聊天模板的调用 markup：
//!
//! ```text
//! ...<tool_call><function=spawn_agent><parameter=fork_turns>none<parameter=task>只读调研…
//! ```
//!
//! 泄漏的 markup 往往比被截断的 native 参数更完整（承载模型的原始意图），
//! 截断守卫在丢弃调用前可从这里救回。解析器只认观察到的两种形状：
//! parameter-markup 风格（可无闭合标签）与 Hermes JSON 风格
//! （`<tool_call>{"name":…,"arguments":…}</tool_call>`）。
//!
//! 已知局限：参数值内部再出现 `<parameter=` 字面量会被提前截断；JSON 风格
//! 块不完整（JSON 解析失败）时无法恢复。

use serde_json::Value;

const OPEN: &str = "<tool_call>";
const CLOSE: &str = "</tool_call>";
const PARAM: &str = "<parameter=";

/// 从泄漏 markup 中恢复出的一个工具调用：函数名 + 原始字符串参数
/// （保持声明顺序，重名取首个）。类型转换由 [`build_arguments`] 依据
/// schema 完成，本结构只做纯文本解析。
pub struct HarvestedCall {
    pub name: String,
    pub params: Vec<(String, String)>,
}

/// 扫描 text 中所有 `<tool_call>…` 块并逐块解析；无法解析的块静默跳过
/// （正常文本偶尔出现该字样不应报错，更不应中断对话）。
pub fn harvest(text: &str) -> Vec<HarvestedCall> {
    let mut out = Vec::new();
    let mut rest = text;
    while let Some(start) = rest.find(OPEN) {
        let after_open = &rest[start + OPEN.len()..];
        // 块体：到闭合标签、下一个块起点或文本末尾为止（泄漏常无闭合标签）。
        let body_len = after_open
            .find(CLOSE)
            .map(|i| i + CLOSE.len())
            .or_else(|| after_open.find(OPEN))
            .unwrap_or(after_open.len());
        if body_len == 0 {
            break; // 防御：不可能路径，避免死循环
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

    // Hermes JSON 风格：块体本身是 `{","arguments":{…}}`。
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
        // 统一展平成原始字符串参数；对象/数组序列化为 JSON 文本，交由
        // build_arguments 按 schema 还原类型。
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

    // parameter-markup 风格：`<function=NAME><parameter=KEY>VALUE…`
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
        // 参数头被截断（无 '>'）时保留已收集的参数——部分恢复优于全弃。
        let Some(key_gt) = after.find('>') else {
            break;
        };
        let key = after[..key_gt].trim().to_string();
        let value_part = &after[key_gt + 1..];
        // 值终止于最近的下一个参数头 / 闭合标签 / 文本末尾。
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

/// 依据工具 schema 把原始字符串参数组装成 arguments JSON 串。
///
/// - schema 声明的 required 字段缺失 → `None`：救回的调用若缺必填项，执行期
///   必然 typed parse 失败，等于把死螺旋入口从"截断"换成"缺参"，不如不救。
/// - 标量按 schema 类型转换（integer/number/boolean）；object/array 先尝试
///   JSON 解析还原；schema 缺失或类型不明时保持字符串（泄漏观测中参数均为
///   长文本字段，字符串是安全默认）。
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
        // session 20260923_263149bf turn 2 的原样泄漏：无闭合标签，task 值
        // 被服务端中断在句中——仍须救回（内容已 95% 可用）。
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
        // JSON 对象无序：按 key 比对，不依赖 Map 的内部排序。
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
        // 只带 task_name 的泄漏缺 required 的 message → 拒绝救回，留在 re-issue 路径。
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
        // 文本在 `<parameter=tas` 处被砍：前面的参数必须保住。
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
        // 提到字样但不是真块：无 <function= 头 → 跳过
        assert!(harvest("我们讨论 <tool_call> 标签的历史").is_empty());
        // 参数头被截断且无任何参数 → 空参数块仍返回，由调用方按 schema 校验
        let calls = harvest("<tool_call><function=spawn_agent>");
        assert_eq!(calls.len(), 1);
        assert!(calls[0].params.is_empty());
    }
}
