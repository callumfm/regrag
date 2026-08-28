# Chat

This module is the answering side of the application: one public endpoint that takes a question, retrieves the law that bears on it, and streams back an answer whose every claim carries a citation — or refuses a question the corpus does not cover.

```bash
curl -N localhost:8000/chat -H 'content-type: application/json' \
  -d '{"question": "how is energy at berth reported"}'
```

## The graph

```
START → retrieve ─┬→ synthesize → END
                  └→ refuse     → END
```

`retrieve` searches the corpus and checks the hits against the refusal gate ([`../retrieval/README.md`](../retrieval/README.md)). If nothing clears the gate the context is empty, and the edge takes us to `refuse`, which returns fixed wording and calls no model. Otherwise `synthesize` makes one streamed call answering from the numbered context blocks.

Two nodes do not need a graph framework today. LangGraph is here because multi-turn conversation and tool use are on the roadmap, and both are edges added to this graph rather than a rewrite.

## The stream

`POST /chat` responds with SSE. Four frame types, in this order:

| Event | When | Data |
| ----- | ---- | ---- |
| `sources` | once, as soon as retrieval returns | the context blocks, each bound to the `[n]` marker the answer will cite it by |
| `text` | repeatedly as the model writes, or once for a refusal | a fragment of the answer |
| `done` | last, on a completed stream | empty |
| `error` | last, in place of everything after it | the app's one error shape, with the request id |

Markers run `1..n` in context order and match the numbering the prompt gave the model, so a client can resolve `[2]` to an act and article on its own.

## The ledger

Every request is recorded however it ended — answered, refused, errored, or abandoned by the client — as a `chat_requests` row with a `chat_request_nodes` row per step, holding the question, the outcome, and the timings and tokens each node spent. It is what a spend cap sums over and what a slow path is diagnosed from.

The answer text itself is not stored. Keeping it is a separate decision with its own retention question, and nothing needs it yet.
