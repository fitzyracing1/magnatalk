# magnatalk

**Use AI chips on purpose.** magnatalk lets an AI agent or a developer choose, deliberately, which GPU/accelerator each job runs on — instead of taking whatever the default happens to be — and keep choosing well as new chips roll out.

Most stacks pick hardware by habit: "just grab an A100" or "whatever the provider defaults to". That habit quietly wastes money (a 70B-class chip serving an 8B model) or wastes time (a bandwidth-starved card running a bandwidth-bound job). magnatalk makes the choice explicit, deterministic, and explainable:

- a **chip catalog** (`chips.yaml`) — plain data, easy to extend as new chips ship
- a **job spec** — a few lines of YAML/JSON declaring what the job actually needs
- a **router** — scores the catalog against the job and returns the pick, a plain-English reason, and the runners-up
- a **decision log** — JSONL audit trail of every choice, with room to record what actually happened
- a **CLI + library + MCP server** — usable by humans, scripts, and AI agents alike

**v0.1 never spends money.** Provider adapters are dry-run only: they print what they *would* launch and stop there. No GPU is rented, no paid API is called. API keys are read only from environment variables, only to report whether they are set, and are never required.

## Quickstart

```bash
pip install .          # or: pip install .[mcp] for the MCP server too

magnatalk list                              # see the catalog
magnatalk validate                          # check the catalog is well-formed
magnatalk pick examples/8b-inference.yaml   # route a job
magnatalk log                               # review past decisions
```

A job spec looks like this:

```yaml
name: llama-8b-chat-serving
task: inference            # inference | fine_tune | training | signal_processing
model_params_b: 8          # or set vram_gb explicitly
precision: fp16            # fp32 | fp16 | bf16 | fp8 | int8 | int4 ...
budget_per_hour_usd: 1.50  # optional hard cap
# budget_per_job_usd / max_runtime_hours / preferred_providers / excluded_providers
```

And a pick looks like this:

```text
PICK: NVIDIA GeForce RTX 4090 on runpod  ($0.89/hr)

WHY : Picked NVIDIA GeForce RTX 4090 on runpod at $0.89/hr (price dated 2026-11-14).
      The job needs ~19.2 GB at fp16; this chip has 24 GB (1.2x headroom). Among 4
      feasible offers it has the best balance of memory bandwidth, which dominates
      inference serving speed, hourly price, and right-sizing (score 0.99 vs 0.65
      for the runner-up). It fits the $1.50/hr budget cap.
```

From Python:

```python
from magnatalk import load_catalog, load_job, route

decision = route(load_job("job.yaml"), load_catalog())
print(decision.chip.name, decision.offer.provider, decision.reason)
```

## How the router decides

Deterministic and explainable, in that order of operations:

1. **Hard filters** — not enough memory, unsupported precision, excluded provider, or over budget ⇒ out, with the rejection reason recorded.
2. **Preferred providers** — if any survive filtering, the choice is restricted to them (otherwise it falls back to all, and says so).
3. **Scoring** — each remaining (chip, provider) offer gets a 0–1 score from three normalized components, weighted by task type:
   - *performance*: memory bandwidth blended with dense tensor throughput at the job's precision (inference and signal processing lean on bandwidth; training and fine-tuning lean on throughput)
   - *cost*: cheaper is better
   - *fit*: prefer the smallest chip that fits comfortably instead of renting a monster for a tiny job
4. **Ties** break lexicographically by chip id, then provider — the same inputs always give the same answer.

VRAM is estimated from `model_params_b × bytes-per-param × task factor` (inference 1.2×, PEFT fine-tune 2.5×, training 4×) unless the job sets `vram_gb` explicitly. These are deliberately simple, documented heuristics — override them when you know better.

## Adding a new chip

Add one entry to [`src/magnatalk/data/chips.yaml`](src/magnatalk/data/chips.yaml) and run `magnatalk validate`. That's the whole process — no code changes:

```yaml
  - id: vendor-new-chip
    name: Vendor New Chip
    vendor: vendor
    memory_gb: 128
    memory_bandwidth_gbps: 6000
    fp16_tflops: 1500        # dense tensor throughput; null if unpublished
    bf16_tflops: 1500
    fp8_tflops: 3000
    supported_precisions: [fp32, fp16, bf16, fp8, int8, int4]
    spec_source: "https://vendor.example/new-chip-datasheet"
    offers:
      - provider: runpod
        price_per_hour_usd: 3.50        # REAL published price only — or null
        price_date: "2026-11-14"        # required when price is non-null
        source: "https://www.runpod.io/pricing"
```

## Cost awareness and pricing honesty

- Every non-null price in the catalog must carry a `price_date` and the `source` URL of the public pricing page it came from — `magnatalk validate` enforces this. **Invented numbers are never presented as real**; when no public price exists, the field stays null (e.g. the AMD MI300X and Google TPU v5e are cataloged with specs but no offers, because RunPod/Modal/Lambda don't list them publicly).
- GPU prices drift. Treat catalog prices as point-in-time estimates for *routing*, not billing truth; re-check the source URL before committing real spend, and refresh `price_date` when you update a number.
- Each pick reports estimated hourly cost, and estimated job cost when the job declares `max_runtime_hours`. Use `budget_per_hour_usd` / `budget_per_job_usd` as hard caps.
- The decision log keeps an `actual` field so you can record what a job really cost and compare it to the estimate: `magnatalk log --annotate <id> --actual '{"cost_usd": 3.10}'`.

## MCP server (for AI agents)

Install the extra and point your agent at the stdio server:

```bash
pip install .[mcp]
```

```json
{ "mcpServers": { "magnatalk": { "command": "magnatalk-mcp" } } }
```

It exposes `pick_chip` (route a job, get the pick + reason + runners-up) and `list_chips`. The server routes and explains only — it never launches hardware or spends money.

## Development

```bash
pip install -e .[dev,mcp]
pytest
ruff check .
magnatalk validate
```

## License

MIT — see [LICENSE](LICENSE). Copyright (c) 2026 Joshua Almeida.
