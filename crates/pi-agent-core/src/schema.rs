//! JSON Schema validation for tool-call arguments.

use serde_json::Value;

/// Validate tool-call arguments against a JSON Schema document.
///
/// Non-objects fail before schema validation (matches the previous default
/// `AgentTool::validate`). Schema compile/validation errors become a String
/// suitable for an error tool result. Callers must not invoke this on schemas
/// with external `$ref` from inside a tokio runtime (`jsonschema::validator_for`
/// documents that restriction); our tool schemas have no `$ref`.
pub fn validate_tool_arguments(schema: &Value, args: &Value) -> Result<Value, String> {
    if !args.is_object() {
        return Err("arguments must be an object".to_string());
    }
    let validator = jsonschema::validator_for(schema).map_err(|e| e.to_string())?;
    validator.validate(args).map_err(|e| e.to_string())?;
    Ok(args.clone())
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, clippy::expect_used)]
    use super::*;
    use serde_json::json;

    #[test]
    fn non_object_is_rejected() {
        let err = validate_tool_arguments(&json!({"type": "object"}), &json!([])).unwrap_err();
        assert!(err.contains("object"));
    }

    #[test]
    fn default_object_schema_accepts_any_keys() {
        let v = validate_tool_arguments(&json!({"type": "object"}), &json!({"a": 1})).unwrap();
        assert_eq!(v["a"], 1);
    }

    #[test]
    fn missing_required_is_err() {
        let schema = json!({
            "type": "object",
            "properties": { "path": { "type": "string" } },
            "required": ["path"]
        });
        assert!(validate_tool_arguments(&schema, &json!({})).is_err());
    }
}
