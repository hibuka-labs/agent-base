//! Schema budget: keep one tool's arguments from eating the request.
//!
//! A tool definition is charged against the request on **every** turn, whether
//! or not the model uses it. A schema that grows without a ceiling is a tax on
//! every step of every session, so it gets a budget and an automatic way back
//! under it — cutting tools by hand does not scale and does not stop the next
//! one from bloating.
//!
//! The ladder is *best-effort, not a hard cap*, and it never drops a tool: a
//! definition that cannot fit is compressed as far as it goes and logged. A
//! bloated schema is one the model misreads; a missing tool is one it routes
//! around with `execute_command`, which is worse.
//!
//! Budget taken from Codex, which ships this defence in production at the same
//! numbers (`codex-rs/tools/src/json_schema/compaction.rs`, 5,000 bytes and
//! depth 3). See `notes/tool-face-design.md` §3 for the full provenance table.

use serde_json::{Map, Value, json};

/// Compact-JSON byte count standing in for the ~1k tokens a tool schema really
/// costs. A proxy, not the real thing — counting tokens here would mean a
/// tokenizer in the tool layer.
pub(crate) const MAX_TOOL_SCHEMA_BYTES: usize = 5_000;

/// How deep a schema may nest before it is flattened to its bare type. Beyond
/// this the model reads structure it cannot act on.
pub(crate) const MAX_TOOL_SCHEMA_DEPTH: usize = 3;

/// Keys that describe a schema without constraining it. Dropping them can
/// never change what arguments a tool accepts.
const ANNOTATION_KEYS: &[&str] = &[
    "$schema",
    "$id",
    "$comment",
    "title",
    "examples",
    "example",
    "default",
    "deprecated",
    "readOnly",
    "writeOnly",
    "externalDocs",
    "contentEncoding",
    "contentMediaType",
];

/// Keys whose value is a map of name → schema. The map is a pass-through: it
/// is not a schema of its own and must not consume a depth level, or a
/// four-deep argument gets flattened one level early.
const SCHEMA_MAPS: &[&str] = &["properties", "patternProperties", "$defs", "definitions"];

/// Shrink a schema until it fits [`MAX_TOOL_SCHEMA_BYTES`], keeping the
/// argument surface intact.
///
/// Returns how far the ladder had to go, for the log line a caller wants.
pub(crate) fn fit_schema_to_budget(schema: &mut Value) -> FitOutcome {
    if compact_bytes(schema) <= MAX_TOOL_SCHEMA_BYTES {
        return FitOutcome::Untouched;
    }

    // Each pass is more lossy than the one before. The order is the claim:
    // annotations cost nothing to lose, nested prose costs the least to read,
    // depth is structure the model cannot act on, and the tool's own words are
    // the last thing to go. The argument surface — property names, types,
    // `required` — never goes at all.
    for (pass, step) in [
        (
            "annotations",
            (|v: &mut Value| strip_annotations(v)) as fn(&mut Value),
        ),
        ("nested prose", drop_nested_prose),
        ("depth", clamp_depth),
        ("property prose", drop_property_prose),
        ("the tool's own prose", drop_root_prose),
    ] {
        step(schema);
        if compact_bytes(schema) <= MAX_TOOL_SCHEMA_BYTES {
            return FitOutcome::Compressed(pass);
        }
    }

    FitOutcome::OverBudget
}

/// What the ladder managed to do.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum FitOutcome {
    /// Already under budget; sent as authored.
    Untouched,
    /// Under budget after dropping whatever the pass names.
    Compressed(&'static str),
    /// Still over after every pass. The tool is kept, not dropped.
    OverBudget,
}

/// Byte count of the compact JSON — the proxy Codex uses too.
fn compact_bytes(schema: &Value) -> usize {
    serde_json::to_string(schema).map(|s| s.len()).unwrap_or(0)
}

/// Every schema node in the tree, at its schema depth, with container maps
/// skipped so they cost no depth.
fn walk_schemas(
    node: &mut Value,
    depth: usize,
    f: &mut impl FnMut(&mut Map<String, Value>, usize),
) {
    match node {
        Value::Object(map) => {
            let is_container = map.keys().any(|k| SCHEMA_MAPS.contains(&k.as_str()))
                && !map.contains_key("type")
                && !map.contains_key("enum");
            if !is_container {
                f(map, depth);
            }
            for (key, child) in map.iter_mut() {
                let next = if SCHEMA_MAPS.contains(&key.as_str()) {
                    depth
                } else {
                    depth + 1
                };
                walk_schemas(child, next, f);
            }
        }
        Value::Array(items) => {
            for child in items.iter_mut() {
                walk_schemas(child, depth, f);
            }
        }
        _ => {}
    }
}

/// Drop [`ANNOTATION_KEYS`] at every depth. Nothing here constrains arguments.
fn strip_annotations(schema: &mut Value) {
    walk_schemas(schema, 0, &mut |map, _| {
        for key in ANNOTATION_KEYS {
            map.remove(*key);
        }
    });
}

/// Drop `description` on subschemas past the first level of arguments. A
/// nested property's prose is what costs the most to read and tells the model
/// the least.
fn drop_nested_prose(schema: &mut Value) {
    walk_schemas(schema, 0, &mut |map, depth| {
        if depth >= 2 {
            map.remove("description");
        }
    });
}

/// Drop `description` on the top-level arguments themselves.
fn drop_property_prose(schema: &mut Value) {
    walk_schemas(schema, 0, &mut |map, depth| {
        if depth == 1 {
            map.remove("description");
        }
    });
}

/// Drop the tool's own `description` in the schema. The wire copy in the
/// envelope is rendered separately and is not touched here.
fn drop_root_prose(schema: &mut Value) {
    if let Value::Object(map) = schema {
        map.remove("description");
    }
}

/// Flatten anything deeper than [`MAX_TOOL_SCHEMA_DEPTH`] to its bare type.
fn clamp_depth(schema: &mut Value) {
    let mut over: Vec<(Vec<String>, Value)> = Vec::new();
    clamp_depth_at(schema, 0, &mut Vec::new(), &mut over);
    for (path, bare) in over {
        let mut slot = &mut *schema;
        for key in &path {
            slot = match slot {
                Value::Object(m) => m.entry(key.as_str()).or_insert(json!({})),
                Value::Array(a) => &mut a[key.parse::<usize>().unwrap_or(0)],
                other => other,
            };
        }
        *slot = bare;
    }
}

fn clamp_depth_at(
    node: &mut Value,
    depth: usize,
    path: &mut Vec<String>,
    over: &mut Vec<(Vec<String>, Value)>,
) {
    // Only schema objects are ever flattened. A `"type": "object"` string is an
    // attribute of its parent, not a nested schema — flattening it replaces a
    // type name with a bogus schema.
    if !matches!(node, Value::Object(_) | Value::Array(_)) {
        return;
    }
    if depth > MAX_TOOL_SCHEMA_DEPTH
        && let Value::Object(_) = node
    {
        let bare = node.get("type").cloned().unwrap_or_else(|| json!("object"));
        over.push((path.clone(), json!({ "type": bare })));
        return;
    }
    match node {
        Value::Object(map) => {
            let keys: Vec<String> = map.keys().cloned().collect();
            for key in keys {
                let next = if SCHEMA_MAPS.contains(&key.as_str()) {
                    depth
                } else {
                    depth + 1
                };
                path.push(key.clone());
                if let Some(child) = map.get_mut(&key) {
                    clamp_depth_at(child, next, path, over);
                }
                path.pop();
            }
        }
        Value::Array(items) => {
            for (i, child) in items.iter_mut().enumerate() {
                path.push(i.to_string());
                clamp_depth_at(child, depth, path, over);
                path.pop();
            }
        }
        _ => {}
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::Map;

    /// A schema over budget: `props` arguments, each described at length, with
    /// a nested object under it carrying its own prose.
    fn oversized_schema(props: usize, prose: usize) -> Value {
        let mut properties = Map::new();
        for i in 0..props {
            properties.insert(
                format!("arg_{i}"),
                json!({
                    "type": "object",
                    "title": format!("Argument {i}"),
                    "description": "a".repeat(prose),
                    "properties": {
                        "inner": {
                            "type": "string",
                            "description": "b".repeat(prose),
                        }
                    }
                }),
            );
        }
        json!({
            "$schema": "http://json-schema.org/draft-07/schema#",
            "title": "An oversized tool",
            "type": "object",
            "description": "C".repeat(prose),
            "properties": properties,
            "required": ["arg_0"],
        })
    }

    fn bytes(schema: &Value) -> usize {
        compact_bytes(schema)
    }

    /// Value: protects=the 90% of tools whose schema is already small — they
    /// must reach the wire exactly as authored, or the budget silently rewrites
    /// definitions nobody asked it to touch. fails_when=a small schema comes
    /// back altered. why_new=the ladder is new. seam=schema -> wire; no_claim=
    /// that unaltered is optimal, only that it is unchanged.
    #[test]
    fn a_schema_under_budget_is_left_alone() {
        let mut schema = json!({"type": "object", "properties": {"p": {"type": "string"}}});
        let before = schema.clone();
        assert_eq!(fit_schema_to_budget(&mut schema), FitOutcome::Untouched);
        assert_eq!(schema, before, "under budget means untouched");
    }

    /// Value: protects=the argument surface the model calls with — property
    /// names, types and `required` are what a call is made of, and every pass
    /// must spare them; fails_when=any pass eats a property name or a type.
    /// why_new=each pass is new. seam=schema internals; no_claim=that the
    /// surviving schema is pleasant to read, only that it is still callable.
    #[test]
    fn the_argument_surface_survives_every_pass() {
        let mut schema = oversized_schema(40, 400);
        fit_schema_to_budget(&mut schema);
        let props = schema["properties"].as_object().expect("properties kept");
        assert_eq!(props.len(), 40, "no argument may be dropped");
        assert!(props.contains_key("arg_0"), "names survive verbatim");
        assert_eq!(props["arg_0"]["type"], "object", "types survive");
        assert_eq!(schema["required"], json!(["arg_0"]), "`required` survives");
        assert_eq!(schema["type"], "object", "the root type survives");
    }

    /// Value: protects=the ordering of loss — annotations are free to lose and
    /// prose is not, so a pass that claims to drop annotations must not drop a
    /// single description; fails_when=descriptions disappear in pass 1.
    /// why_new=the ladder is new. seam=pass boundaries; no_claim=that this is
    /// the only ordering, only that each pass does what its name says.
    #[test]
    fn annotations_go_and_prose_stays() {
        let mut schema = oversized_schema(3, 1_200);
        assert!(
            bytes(&schema) > MAX_TOOL_SCHEMA_BYTES,
            "fixture starts over"
        );
        strip_annotations(&mut schema);
        assert!(schema["$schema"].is_null(), "`$schema` is an annotation");
        assert!(schema["title"].is_null(), "`title` is an annotation");
        assert!(
            schema["description"].is_string()
                && schema["properties"]["arg_0"]["description"].is_string(),
            "a pass named 'annotations' must not touch prose of any level"
        );
    }

    /// Value: protects=the ordering of loss between levels — nested prose costs
    /// the most to read and the tool's own words carry the most meaning, so the
    /// nested must go first; fails_when=the root description is dropped before
    /// a nested one. why_new=the ladder is new. seam=pass order; no_claim=that
    /// this order is the only good one, only that it is the less lossy one.
    #[test]
    fn nested_prose_goes_before_the_tools_own() {
        // Sized so dropping nested prose alone is enough to fit.
        let mut schema = oversized_schema(3, 900);
        assert!(
            bytes(&schema) > MAX_TOOL_SCHEMA_BYTES,
            "fixture starts over"
        );
        drop_nested_prose(&mut schema);
        assert!(
            bytes(&schema) <= MAX_TOOL_SCHEMA_BYTES,
            "fixture must fit after nested prose alone, or this asserts nothing"
        );
        assert!(
            schema["description"].is_string(),
            "the tool's own words outlive the nested ones"
        );
        assert!(
            schema["properties"]["arg_0"]["description"].is_string(),
            "an argument's own words outlive the nested ones too"
        );
        assert!(
            schema["properties"]["arg_0"]["properties"]["inner"]["description"].is_null(),
            "the nested prose is what went"
        );
    }

    /// Value: protects=readability of what is left — a schema that fits only by
    /// flattening everything is one the model misreads, so the ladder must stop
    /// the moment it fits; fails_when=passes run past the point of fit.
    /// why_new=the ladder is new. seam=pass loop; no_claim=that the stop point
    /// is optimal, only that it is early.
    #[test]
    fn the_ladder_stops_as_soon_as_it_fits() {
        let mut schema = oversized_schema(3, 900);
        let outcome = fit_schema_to_budget(&mut schema);
        assert_eq!(outcome, FitOutcome::Compressed("nested prose"));
        assert!(
            schema["properties"]["arg_0"]["description"].is_string(),
            "a pass the ladder never reached must not have fired"
        );
    }

    /// Value: protects=the model's access to a tool — a definition too big to
    /// fit is one it must still be able to call, so the ladder gives up but
    /// never deletes; fails_when=an oversized tool vanishes from the request.
    /// why_new=the ladder is new. seam=schema -> wire; no_claim=that an
    /// over-budget tool is cheap to send, only that it is sent.
    #[test]
    fn a_tool_that_cannot_fit_is_kept_not_dropped() {
        let mut schema = oversized_schema(200, 400);
        assert_eq!(fit_schema_to_budget(&mut schema), FitOutcome::OverBudget);
        assert_eq!(
            schema["properties"].as_object().map(|p| p.len()),
            Some(200),
            "even a hopeless schema keeps its arguments"
        );
    }

    /// Value: protects=the depth ceiling's meaning — nesting past it is
    /// structure the model cannot act on, and clamping must flatten rather than
    /// truncate (a truncated schema is an invalid one); fails_when=deep
    /// subschemas survive, or are cut mid-object. why_new=the clamp is new.
    /// seam=schema internals; no_claim=that depth 3 is the right ceiling.
    #[test]
    fn deep_nesting_is_flattened_not_truncated() {
        // Four levels of argument nesting: root > a > b > c > d.
        let mut schema = json!({
            "type": "object",
            "properties": {
                "a": {
                    "type": "object",
                    "properties": {
                        "b": {
                            "type": "object",
                            "properties": {
                                "c": {
                                    "type": "object",
                                    "properties": {
                                        "d": { "type": "string", "description": "deep" }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        });
        clamp_depth(&mut schema);
        let c = &schema["properties"]["a"]["properties"]["b"]["properties"]["c"];
        let d = &c["properties"]["d"];
        assert_eq!(
            d.as_object().map(|m| m.len()),
            Some(1),
            "past the ceiling a subschema is its bare type and nothing else"
        );
        assert_eq!(d["type"], "string", "the bare type is kept");
        assert!(
            c["properties"].is_object() && c["type"] == "object",
            "levels at or under the ceiling keep their structure"
        );
        assert!(
            schema["properties"]["a"]["properties"]["b"]["type"] == "object",
            "and are not flattened by a pass aimed deeper down"
        );
    }

    /// Value: protects=the type table itself — `"type": "object"` is an
    /// attribute of its parent, not a nested schema, and treating it as one
    /// turns a type name into a bogus schema (found live: the clamp replaced
    /// `c.type` with `{"type":"object"}` and every reader of that tool got an
    /// invalid definition); fails_when=a scalar is ever a clamp candidate.
    /// why_new=this was a real bug in the first cut. seam=clamp recursion;
    /// no_claim=that every JSON-Schema keyword is modelled correctly.
    #[test]
    fn a_type_name_is_never_treated_as_a_schema() {
        let mut schema = json!({
            "type": "object",
            "properties": {
                "a": {
                    "type": "object",
                    "properties": {
                        "b": {
                            "type": "object",
                            "properties": {
                                "c": {
                                    "type": "object",
                                    "properties": {
                                        "d": { "type": "string", "description": "deep" }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        });
        clamp_depth(&mut schema);
        assert_eq!(
            schema["properties"]["a"]["properties"]["b"]["properties"]["c"]["type"],
            json!("object"),
            "a type name stays a name"
        );
    }
}
