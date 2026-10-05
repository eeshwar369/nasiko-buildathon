# Compact tools: measured evidence

Measured on **2026-10-05**, using the code and instructions in this PR. These are
local development measurements, not organizer scores or held-out accuracy.
All live comparisons used OpenAI Chat Completions, temperature 0,
`LIVE_BASELINE=1`, and 1024 maximum completion tokens. GPT-6 Luna used
`LIVE_REASONING_EFFORT=none`, as required for native Chat Completions function
calling by the [OpenAI documentation](https://developers.openai.com/api/docs/models/gpt-6-luna).
The two GPT-4.1 models used the dated IDs in the filenames; GPT-6 Luna is an alias.
Every live request used the organizer-corrected reference date October 3, 2026,
Asia/Kolkata. No answer labels are sent to the model.

## What the measurements show

The seven-tool catalog workload is the clearest observed benefit: **3/3 correct
compact calls and 3/3 correct native calls**, with **23.1% fewer serialized-request
tokens**. Provider-reported input tokens decreased from **1,452 to 1,344 (7.4%)**;
completion tokens were 73 on both paths. This is a small, author-created workload,
and does not establish an average production saving.

Small catalogs do not consistently benefit. The longer call instructions consume
most of their savings, and some models ask for optional fields, narrate actions,
or invent arguments. The feature remains experimental and off by default.

| Workload / configured model | Correct calls, compact path | Correct calls, native path | Requests actually compacted | Full-request tokens, native → compact | Provider input tokens, native → compact |
|---|---:|---:|---:|---:|---:|
| Public / `gpt-4.1-mini-2025-04-14` | 1/3 | 3/3 | 2/3 | 818 → 752 (8.1%) | 484 → 564 (increase) |
| Public / `gpt-4.1-2025-04-14` | 1/3 | 1/3 | 2/3 | 815 → 749 (8.1%) | 484 → 564 (increase) |
| Public / `gpt-6-luna` | 2/3 | 1/3 | 2/3 | 815 → 749 (8.1%) | 736 → 646 (12.2%) |
| Development / `gpt-6-luna` | 12/13 | 11/13 | 4/13 | 3,241 → 3,130 (3.4%) | 3,085 → 2,981 (3.4%) |
| Seven-tool catalog / `gpt-6-luna` | 3/3 | 3/3 | 3/3 | 2,013 → 1,548 (23.1%) | 1,452 → 1,344 (7.4%) |

"Compact path" includes native bypasses: in particular **9/13 development
requests bypassed compaction**. Among the four actually compacted development
requests, three matched the expected calls. The catalog set reuses the first
three development requests with all seven supported tools available; it isolates
catalog size and is **not three additional independent accuracy observations**.
These are individual runs, not confidence intervals or proof of superiority.
All successful live tests here use one provider; no cross-provider claim is made.

The strict local matcher compares call order and argument keys/values, except
documented free-text fields, which are checked for presence and string type.
Valid decoding is not enough to pass: an empty call list for a requested action
or an invented optional argument fails the semantic comparison. The organizer's
private scorer is authoritative and may use different matching details.

## Offline correctness and determinism

| Dataset | Full-request tokens | Round trips | Decoder fixtures | Byte-identical repeated output |
|---|---:|---:|---:|---|
| Unmodified public sample | 656 → 590 (10.1%) | 3/3 | 5/5 | Yes |
| Author-created development set | 2,552 → 2,441 (4.3%) | 13/13 | 8/8 | Yes |

The checked-in public fixture was compared as JSON with a fresh download from
the organizer registry. Offline runs intentionally render the supplied expected
calls and decode them back: these are codec checks, **not inference accuracy**.
The earlier 656 → 459 (30.0%) measurement used shorter instructions and must not
be presented as the result of the current protocol. Token counts include all
instructions, message structure, and bypasses.

## Reproduce

From the repository root, fetch dependencies while online, then:

```sh
EVAL_SET=llm-router/tests/fixtures/compact-tools-eval-v1.json OUT=/tmp/public.jsonl \
  cargo run --release -p nasiko-llm-router --example compact_tools_eval --offline
EVAL_SET=llm-router/tests/fixtures/compact-tools-eval-v1.json OUT=/tmp/public.jsonl \
  cargo run --release -p nasiko-llm-router --example compact_tools_report --offline
```

For hosted inference, export your key locally and set:

```sh
PROVIDER_BASE_URL=https://api.openai.com/v1 MODEL=gpt-6-luna \
LIVE_REASONING_EFFORT=none LIVE_BASELINE=1 \
EVAL_SET=llm-router/tests/fixtures/compact-tools-catalog.json OUT=/tmp/catalog-live.jsonl \
  cargo run --release -p nasiko-llm-router --example compact_tools_eval
```

Change `EVAL_SET` to `compact-tools-generalization.json` or
`compact-tools-eval-v1.json` for the other workloads. Use the same `EVAL_SET` and
`OUT` with `compact_tools_report`. Raw outputs and local reports are committed
beside this document. Provider counters are distinct from the mandated
`tiktoken-rs=0.7.0/o200k_base` serialized-body measurement; no dollar savings or
latency improvement is claimed.

## Running demonstration

[Watch/download the captioned runtime recording](../assets/compact-tools-demo.mp4).
The video records the running Rust example through a real browser: transform,
schema reconstruction, invalid enum rejection, atomic rejection, and native
bypass. Model-output text in this UI is explicitly a sample; the video does not
claim to show hosted inference or the full authenticated Nasiko deployment.

```sh
cargo run --release -p nasiko-llm-router --example compact_tools_demo
```

Open `http://127.0.0.1:8765`. The production codec and request/response seam handle
every action. No API key is needed for this offline demonstration.
