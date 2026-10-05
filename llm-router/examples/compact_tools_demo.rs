//! Local, offline inspection of the production request/response seam.
//! cargo run --release -p nasiko-llm-router --example compact_tools_demo
use std::sync::Arc;

use axum::{
    Json, Router,
    extract::{DefaultBodyLimit, State},
    response::Html,
    routing::{get, post},
};
use nasiko_llm_router::{
    compact_tools,
    ir::{ChatRequest, ChatResponse},
};
use nasiko_tool_compact::{decode_tools, encode_tools};
use serde::Deserialize;
use serde_json::{Value, json};

#[derive(Deserialize)]
struct Input {
    request: Value,
    output: String,
}

fn sample() -> Value {
    let fixture: Value =
        serde_json::from_str(include_str!("../tests/fixtures/compact-tools-eval-v1.json")).unwrap();
    json!({
        "request":{"messages":fixture["cases"][0]["messages"],"tools":fixture["tools"]},
        "output":"<<call create_calendar_event {\"title\":\"Design review\",\"start\":\"2026-10-05T15:00:00+05:30\",\"attendees\":[\"riya@example.com\"]}>>"
    })
}

fn inspect(input: Input, tokenizer: &tiktoken_rs::CoreBPE) -> Result<Value, String> {
    let baseline_tokens = tokenizer.encode_ordinary(&input.request.to_string()).len();
    let mut request: ChatRequest = serde_json::from_value(input.request.clone())
        .map_err(|_| "invalid chat request".to_owned())?;
    let decision = if compact_tools::supports_wire_tools(&input.request) {
        compact_tools::prepare_request(&mut request)
    } else {
        Err(compact_tools::Bypass::UnsupportedSchema)
    };
    let prepared = match decision {
        Ok(prepared) => prepared,
        Err(reason) => {
            return Ok(json!({
                "compacted":false,"reason":reason.as_str(),"baseline_tokens":baseline_tokens,
                "compact_tokens":baseline_tokens,"request":input.request,
                "note":"Native request retained without modification."
            }));
        }
    };
    let compact_request = serde_json::to_value(&request).map_err(|_| "serialization failed")?;
    let compact_tokens = tokenizer
        .encode_ordinary(&compact_request.to_string())
        .len();
    let original: ChatRequest =
        serde_json::from_value(input.request).map_err(|_| "invalid chat request")?;
    let tools = compact_tools::to_tool_defs(&original).map_err(|_| "invalid tool catalog")?;
    let compact = encode_tools(&tools).map_err(|e| e.code().to_owned())?;
    let schema_preserved = decode_tools(&compact).map_err(|e| e.code().to_owned())? == tools;
    let response: ChatResponse = serde_json::from_value(json!({
        "id":"local-inspection","model":"offline-demo",
        "choices":[{"index":0,"message":{"role":"assistant","content":input.output},"finish_reason":"stop"}]
    })).map_err(|_| "invalid response")?;
    let restored = match prepared.restore(response) {
        Ok(response) => json!({"response":response}),
        Err(error) => json!({"error":error.code()}),
    };
    Ok(json!({
        "compacted":true,"baseline_tokens":baseline_tokens,"compact_tokens":compact_tokens,
        "schema_preserved":schema_preserved,"request":compact_request,"restored":restored
    }))
}

async fn inspect_handler(
    State(tokenizer): State<Arc<tiktoken_rs::CoreBPE>>,
    Json(input): Json<Input>,
) -> Json<Value> {
    Json(inspect(input, &tokenizer).unwrap_or_else(|error| json!({"error":error})))
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let tokenizer = Arc::new(tiktoken_rs::o200k_base()?);
    let app = Router::new()
        .route(
            "/",
            get(|| async { Html(include_str!("compact_tools_support/demo.html")) }),
        )
        .route("/api/sample", get(|| async { Json(sample()) }))
        .route("/api/inspect", post(inspect_handler))
        .layer(DefaultBodyLimit::max(65_536))
        .with_state(tokenizer);
    let bind = std::env::var("DEMO_BIND").unwrap_or_else(|_| "127.0.0.1:8765".into());
    let listener = tokio::net::TcpListener::bind(&bind).await?;
    println!("Compact tools offline demo: http://{bind}");
    println!("Runs the production codec and router seam; no hosted inference or tool execution.");
    axum::serve(listener, app).await?;
    Ok(())
}
