//! Evaluator contract: outputs only, one JSONL record per case. No network by default.

#[path = "compact_tools_support/mod.rs"]
mod support;

use nasiko_llm_router::{compact_tools, ir::ChatRequest};
use nasiko_tool_compact::{StreamDecoder, ToolCall, decode_calls, render_calls};
use serde_json::{Value, json};
use std::{
    fs::File,
    io::{BufWriter, Write},
    path::PathBuf,
};
use support::{Case, EvalSet, LiveConfig, native_request, select_tools};

fn prepare_case(
    native: &Value,
    case: &Case,
) -> Result<(Value, Vec<nasiko_tool_compact::ToolDef>), Box<dyn std::error::Error>> {
    let mut request: ChatRequest = serde_json::from_value(native.clone())?;
    let decision = if compact_tools::supports_wire_tools(native) {
        compact_tools::prepare_request(&mut request).map(|_| ())
    } else {
        Err(compact_tools::Bypass::UnsupportedSchema)
    };
    let compacted = decision.is_ok();
    // Unsupported wire fields bypass this case; they must not abort the dataset.
    let tools = if compacted {
        compact_tools::to_tool_defs(&serde_json::from_value(native.clone())?)?
    } else {
        Vec::new()
    };
    let compact_request = if compacted {
        serde_json::to_value(&request)?
    } else {
        native.clone()
    };
    let (rendered_calls, roundtrip_calls) = if compacted {
        let rendered = render_calls(&case.expected, &tools)?;
        let decoded = decode_calls(&rendered, &tools)?;
        (rendered, decoded)
    } else {
        // Bypass exercises native JSON serialization; makes no compact-codec claim.
        let rendered = serde_json::to_string(&case.expected)?;
        let decoded: Vec<ToolCall> = serde_json::from_str(&rendered)?;
        (rendered, decoded)
    };
    let mut record = json!({"id":case.id,"compact_request":compact_request,"compacted":compacted,"rendered_calls":rendered_calls,"roundtrip_calls":roundtrip_calls});
    if let Err(reason) = decision {
        record["bypass_reason"] = json!(reason.as_str());
    }
    Ok((record, tools))
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let input = std::env::var_os("EVAL_SET")
        .map(PathBuf::from)
        .unwrap_or_else(|| {
            PathBuf::from(env!("CARGO_MANIFEST_DIR"))
                .join("tests/fixtures/compact-tools-eval-v1.json")
        });
    let output = std::env::var_os("OUT")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("compact-tools-out.jsonl"));
    if std::fs::canonicalize(&input).ok() == std::fs::canonicalize(&output).ok() && output.exists()
    {
        return Err("EVAL_SET and OUT must be different files".into());
    }
    let dataset: EvalSet = serde_json::from_reader(File::open(input)?)?;
    let live = LiveConfig::from_env()?;
    let tokenizer = if std::env::var("MEASURE_TOKENS").as_deref() == Ok("1") {
        Some(tiktoken_rs::o200k_base()?)
    } else {
        None
    };
    let mut baseline_tokens = 0usize;
    let mut compact_tokens = 0usize;
    let mut writer = BufWriter::new(File::create(output)?);
    for case in &dataset.cases {
        let mut native = native_request(&dataset, case)?;
        if let Some(live) = &live {
            live.configure_request(&mut native);
        }
        let (mut record, tools) = prepare_case(&native, case)?;
        if let Some(live) = &live {
            // Neither expected calls nor labels are passed to this function or the model.
            live.add_outputs(&mut record, &native, &tools).await?;
        }
        if let Some(tokenizer) = &tokenizer {
            baseline_tokens += tokenizer.encode_ordinary(&native.to_string()).len();
            compact_tokens += tokenizer
                .encode_ordinary(&record["compact_request"].to_string())
                .len();
        }
        serde_json::to_writer(&mut writer, &record)?;
        writeln!(writer)?;
    }
    for case in &dataset.decoder_cases {
        let native = json!({"messages":[],"tools":select_tools(&dataset.tools, &case.tools)?});
        let request: ChatRequest = serde_json::from_value(native)?;
        let tools = compact_tools::to_tool_defs(&request)?;
        let decoded = (|| {
            let mut decoder = StreamDecoder::new(&tools)?;
            for chunk in &case.chunks {
                decoder.push(chunk)?;
            }
            Ok::<_, nasiko_tool_compact::CompactError>(decoder.finish()?.calls)
        })();
        let decoded = match decoded {
            Ok(calls) => json!({"calls":calls}),
            Err(error) => json!({"error":error.code()}),
        };
        let record: Value = json!({"id":case.id,"decoded":decoded});
        serde_json::to_writer(&mut writer, &record)?;
        writeln!(writer)?;
    }
    writer.flush()?;
    if tokenizer.is_some() {
        eprintln!(
            "o200k_base full-request tokens: baseline={baseline_tokens}, compact={compact_tokens}"
        );
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn unsupported_tool_extensions_preserve_native_request_and_continue() {
        let dataset: EvalSet =
            serde_json::from_str(include_str!("../tests/fixtures/compact-tools-eval-v1.json"))
                .unwrap();
        let case = &dataset.cases[0];
        let mut native = native_request(&dataset, case).unwrap();
        native["tools"][0]["custom_contract"] = json!({"must_preserve":true});
        let (record, _) = prepare_case(&native, case).unwrap();
        assert_eq!(record["compacted"], false);
        assert_eq!(record["compact_request"], native);
        assert_eq!(
            record["roundtrip_calls"],
            serde_json::to_value(&case.expected).unwrap()
        );
        let native = native_request(&dataset, &dataset.cases[1]).unwrap();
        assert_eq!(
            prepare_case(&native, &dataset.cases[1]).unwrap().0["compacted"],
            true
        );
    }

    #[test]
    fn changing_expected_answers_cannot_change_the_model_request() {
        let dataset: EvalSet =
            serde_json::from_str(include_str!("../tests/fixtures/compact-tools-eval-v1.json"))
                .unwrap();
        let mut case = dataset.cases.into_iter().next().unwrap();
        let native = json!({"messages":case.messages,"tools":dataset.tools});
        let first = prepare_case(&native, &case).unwrap().0;
        case.expected.clear();
        let second = prepare_case(&native, &case).unwrap().0;
        assert_eq!(first["compact_request"], second["compact_request"]);
        assert_ne!(first["roundtrip_calls"], second["roundtrip_calls"]);
    }
}
