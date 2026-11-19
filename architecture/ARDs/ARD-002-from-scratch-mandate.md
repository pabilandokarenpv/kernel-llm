# ARD-002: Teacher model (gpt-oss-120b) used only for training-data generation, never for serving end users

**Status:** Accepted — this is the single most load-bearing constraint on
the whole project.

## Context

Early on, Kernel AI's chat endpoint called a hosted Cloudflare Workers AI
model (`@cf/meta/llama-3.1-8b-instruct`, later `@cf/meta/llama-3.2-3b-instruct`
after the first was deprecated) directly to answer users. This is the
standard, fast way to ship a chat product. It also directly contradicts the
project's actual goal: a genuinely from-scratch LLM, "not using anything
existing," the same way ChatGPT/Claude/Llama are each someone's own trained
model, not a wrapper around another lab's API.

## Decision

A strong, free model (`@cf/openai/gpt-oss-120b` via Cloudflare Workers AI)
is used **exclusively** to generate synthetic training examples — a
`datagen-worker` Cloudflare Worker calls it with prompts constrained to a
strict Kernel grammar subset (for `kernel_task`/`kernel_explain`
categories) or general instruction-following topics (for `general`/
`coding`), and the outputs are saved as JSONL training pairs. **This model
never appears in the request path that answers a real user message.** The
chat endpoint serves exclusively from our own trained checkpoint.

## Alternatives considered

- **Hosted model serves users directly (the original Slate setup)** —
  rejected once the "from scratch" requirement was made explicit: this is
  functionally "using an existing AI model," which defeats the stated
  purpose of the entire project.
- **No teacher model at all — only hand-written/generated-by-rule training
  data** — rejected: our own Kernel-execution-trace corpus (4,829 programs)
  and template-based data alone can't teach general conversational
  ability or broad knowledge; a model this small needs every available
  real signal, and synthetic distillation from a free strong model is a
  legitimate, well-established technique (this is literally how the
  original GPT-4 distillation datasets like Alpaca/Vicuna were built) as
  long as the distinction between "data source" and "serving model" is
  kept absolute.

## Consequences

- Every chat response a user sees is provably attributable to our own
  checkpoint's weights, not a hosted model's weights — this is the
  difference between "we built an LLM" and "we built a wrapper."
- Data quality is bounded by the teacher model's output reliability —
  real failure modes hit during this project: `content: null` responses
  (gpt-oss spending its whole token budget on internal reasoning before
  answering, fixed with `max_tokens` + `reasoning: {effort: "low"}`), and
  JSON-parse failures on code-heavy generations (mitigated with
  markdown-fence-tolerant parsing and a smaller batch size).
- The architecture (ARD-001) explicitly supports this split: the served
  model is small enough to actually finish training on free compute, at
  the cost of being honestly less capable than the teacher used to
  generate its training data — documented throughout, not hidden.
