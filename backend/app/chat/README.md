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

Otherwise the assess loop runs. `assess` makes one blocking model call and answers with the tool calls that would fill what the context is missing — a fresh `search`, or a `follow_reference` fetching a division the context cites. `tools` runs them and merges what they find in, keeping the earlier blocks in place so the `[n]` markers a client already holds keep meaning what they meant. It ends when assess asks for nothing or `ASSESS_MAX_ROUNDS` is spent; `ASSESS_ENABLED=false` skips it entirely, leaving the two-node graph this started as. Either way, `synthesize` makes one streamed call answering from the numbered blocks.

Three bounds keep the loop honest. A search's hits face the same score bar `retrieve` holds its own to, so what the gate would refuse to answer from cannot arrive by the back door. The merge stops once the loop has added `ASSESS_EXTRA_CHUNKS` on top of what retrieval left. And each call runs on its own session, so one that fails on the database costs its own result and no other's.

Both model calls run at `CHAT_TEMPERATURE`, which is 0 — an answer quoting law back gains nothing from sampling variety, and assess sampling differently changes the context the answer is built from.

The loop is best-effort: a failed assess call or a failed tool call costs the answer that round's context, never the request. The diagram is hand-drawn, and a test holds the compiled graph to the edge list it was drawn from.

### How far the loop can reach

Some questions are only answerable a step or two away from where search lands. Ask what a *voyage* is under FuelEU and search finds FuelEU Article 3 — which does not say. It says the answer is in Article 3 of another act. The text that answers the question is never in the article the question is about.

The loop handles that because of how the context is written for it. Every block assess reads is printed with the addresses it cites, like a footnote list under the paragraph:

```
[2] (32023R1805, Article 3)
    'voyage' means voyage as defined in Article 3, point (c), of Regulation (EU) 2015/757 ...
    cites: 32015R0757 Article 3
```

So assess never has to fetch a block in order to discover where it points — the destination is already on the page. It asks for `32015R0757 Article 3` directly, in one go.

That is why the loop is bounded by **how many things it can fetch, not how far away they are**. `ASSESS_MAX_CALLS` sets the width — four addresses in one round — and a chain three acts long costs the same single round as a chain one act long, as long as each address is visible before it is needed. A second round would only earn its cost if reading a fetched block revealed an address that nothing had shown before. `ASSESS_MAX_ROUNDS` therefore defaults to 1: measured against the multi-hop cases in the golden dataset, a second round changed no score and cost roughly a third of the tokens and two seconds a question.

The reach has a real edge, though. A question needing more than `ASSESS_MAX_CALLS` separate fetches cannot be answered in full however good assess is, and neither can one whose next address only appears in text nobody has fetched yet.

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
