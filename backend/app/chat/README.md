# Chat

This module is the answering side of the application: one public endpoint that takes a question, retrieves the law that bears on it, and streams back an answer whose every claim carries a citation — or refuses a question the corpus does not cover.

```bash
curl -N localhost:8000/chat -H 'content-type: application/json' \
  -d '{"question": "how is energy at berth reported"}'
```

## The graph

```
START → retrieve ─┬→ refuse ──────────────→ END   nothing cleared the gate
                  │
                  ├→ assess ⇄ tools              while assess asks and the budget remains
                  │      │
                  └──────┴→ synthesize ────→ END   the context is settled
```

`retrieve` searches the corpus and checks the hits against the refusal gate ([`../retrieval/README.md`](../retrieval/README.md)). If nothing clears the gate the context is empty, and the edge takes us to `refuse`, which returns fixed wording and calls no model.

Otherwise the assess loop runs. `assess` makes one blocking model call over the context so far and answers with the tool calls that would fill what is missing — a fresh `search`, or a `follow_reference` fetching a division the context cites. `tools` runs them and merges what they find into the context, keeping the earlier blocks in place so the `[n]` markers a client already holds keep meaning what they meant. The loop ends when assess asks for nothing or `ASSESS_MAX_ROUNDS` is spent, and `ASSESS_ENABLED=false` skips it entirely, leaving the two-node graph this started as. Whichever way it ends, `synthesize` makes one streamed call answering from the numbered context blocks.

The loop is best-effort: a failed assess call or a failed tool call costs the answer that round's context, never the request. The diagram is hand-drawn, and a test holds the compiled graph to the edge list it was drawn from.

## The stream

`POST /chat` responds with SSE. Four frame types, in this order:

| Event | When | Data |
| ----- | ---- | ---- |
| `sources` | once, as soon as the context settles — after retrieval, or after the loop's last round | the context blocks, each bound to the `[n]` marker the answer will cite it by |
| `text` | repeatedly as the model writes, or once for a refusal | a fragment of the answer |
| `done` | last, on a completed stream | empty |
| `error` | last, in place of everything after it | the app's one error shape, with the request id |

Markers run `1..n` in context order and match the numbering the prompt gave the model, so a client can resolve `[2]` to an act and article on its own.

## The ledger

Every request is recorded however it ended — answered, refused, errored, or abandoned by the client — as a `chat_requests` row with a `chat_request_steps` row per step it ran through, holding the question, the outcome, and the timings and tokens each step spent. A step is a graph node, or one tool call an assess round ran, named `tool_search` / `tool_follow_reference` so one column holds both. It is what a spend cap sums over and what a slow path is diagnosed from.

This is deliberately the tracing, in place of a tracing library: the request row has to exist for the spend cap anyway, and the per-step timings come with it. It is a flat span list ordered by `position`, not a tree — a tool step's parent is the assess step before it — which is enough while the graph nests only one level deep.

The answer text itself is not stored. Keeping it is a separate decision with its own retention question, and nothing needs it yet.
