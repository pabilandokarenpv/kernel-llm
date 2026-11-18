export interface Env {
  AI: Ai;
}

// Internal-only data-generation helper: calls a free, strong open-weight
// model to produce synthetic training examples for OUR from-scratch model.
// This endpoint is never used to answer end users — it only ever produces
// training data that we then train our own model on.

export default {
  async fetch(request: Request, env: Env): Promise<Response> {
    if (request.method !== "POST") {
      return new Response("POST only", { status: 405 });
    }
    const body = await request.json<{ prompt: string; system?: string }>();
    const messages = [
      { role: "system", content: body.system ?? "You are a helpful assistant generating training data." },
      { role: "user", content: body.prompt },
    ];
    try {
      const result = await env.AI.run("@cf/openai/gpt-oss-120b", {
        messages,
        stream: false,
        max_tokens: 4096,
        reasoning: { effort: "low" },
      });
      return new Response(JSON.stringify(result), { headers: { "content-type": "application/json" } });
    } catch (err: any) {
      return new Response(JSON.stringify({ error: err.message }), { status: 500 });
    }
  },
};
