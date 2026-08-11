# RegRag

A retrieval-augmented question answering system over EU maritime emissions law,
where every answer cites the article it came from.

Built end to end: automated corpus discovery from the EU's legislative database,
an HTML parsing and chunking pipeline that preserves article-level structure,
hybrid vector and keyword search, and a React front end.

## Layout

| Directory                | Contents                                                     |
| ------------------------ | ------------------------------------------------------------ |
| [`backend/`](backend/)   | FastAPI service: the ingestion pipeline, retrieval and evals  |
| [`frontend/`](frontend/) | React UI                                                      |

How the corpus is built — and why each stage works the way it does — is in
[`backend/app/ingestion/README.md`](backend/app/ingestion/README.md).

## Local dev

```bash
docker compose up -d db
```

Then follow [`backend/README.md`](backend/README.md) and
[`frontend/README.md`](frontend/README.md).
