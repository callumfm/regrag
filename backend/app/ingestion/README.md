# Ingestion
This module contains the code for the fetching, parsing, chunking and embedding of EU legislative documents. The ingest pipeline takes raw documents and ingests them into the database in a vectorised form that can later be queried by the retrieval stage of the RAG application.

The following sections describe each stage of the ingest pipeline and the design decisions that were taken. Ingest is not a one-off — acts are amended and the pipeline itself is improved — so each stage has to re-run without redoing work that has not changed, embedding especially, since an external model charges per call.

## 1 Fetch
Fetching answers two questions: which documents belong in the corpus, and which version of each to download.

### 1.1 CELEX
Every document in EUR-Lex is identified by a CELEX number and this can be used to find related law documents and versions.

```
3 2023 R 1805
│ │    │ └── act number within that year
│ │    └──── kind: R regulation, L directive, D decision
│ └───────── year of adoption
└─────────── sector: 3 is legislation
```

**Which documents belong in the corpus?**
Sector 3 with a kind letter of R, L or D which means binding legislation. Anything else is discarded — sector 5, for instance, covers the notices, opinions and Commission papers that reference a law without imposing obligations.

**Which version of each?**
An act and its amended versions have matching ids apart from the sector digit and date suffix. The original sits in sector 3 with each consolidation under sector 0 with that date appended.

```
32023R1805              FuelEU as published
02023R1805-20230922     consolidated to 22 September 2023
```

When an act is absorbed into another, its consolidations are filed under the absorbing act's id instead. An act whose consolidations are all filed under another id has been superseded, and the act that absorbed it is the one to fetch.

### 1.2 Discovery
Discovery produces the list of CELEX ids to download. The corpus is built around two regulations: FuelEU (`32023R1805`) and MRV (`32015R0757`).

The EU Publications Office runs a document metadata database, CELLAR, which is queried for every act naming one of those regulations as its legal basis (the law it was made under) along with the regulations themselves.

The results are wider than the corpus, so each act must pass three filters:

- it is legislation: sector 3, with a kind of R, L or D
- it is still in force
- it has not been superseded by an act that absorbed it

CELLAR also reports every consolidated version it holds, and the latest one filed under the act's own id is carried forward as the version to try downloading first.

Discovery refuses a result set that has lost more than a fifth of the acts the previous run held, and at least three of them: a truncated SPARQL response and a mass repeal look identical from here, and only one of them should empty the corpus.

### 1.3 Download
Download takes the list from discovery and pulls each document off EUR-Lex.

EUR-Lex serves documents at a fixed URL, so a CELEX id is enough to locate one.

```
https://eur-lex.europa.eu/legal-content/EN/TXT/HTML/?uri=CELEX:32023R1805
```

HTML is preferred over PDF because it carries the document's structure as markup — articles, paragraphs and tables are each marked up as such. A naive approach to chunking PDF text is to split every N characters with overlap, which cuts through those boundaries.

Each act is downloaded at the latest consolidated version discovery carried forward. That version is not always available: an act that has never been amended has a consolidated id but no consolidated text to serve. Where the consolidated version cannot be downloaded, download falls back to the original act.

Refusal has to be recognised rather than trusted to the status code. EUR-Lex denies a version either with a 404 or with a 200 carrying its "does not exist" page, and answers 202 with an empty body while it is still generating one on demand. Only the first two are refusals and fall back to the next candidate. A 202 is not: falling back there would quietly ingest an older version of an act whose newest one was merely a moment away, so the document is recorded as failed and left for the next run.

Documents are stored as objects — files on disk in dev, a Cloudflare R2 bucket in prod — keyed by act, version and a hash of the bytes:

```
{act}/{version downloaded}/{sha256 of the bytes}.html
32023R1805/32023R1805/9f86d081….html      FuelEU, served as published
```

A new consolidation is therefore a new object rather than a replacement for the old one. This is deliberate: the database keeps one row per document per run, each recording the hash of the bytes that run read, and those bytes are verified against that hash on every later read. Overwriting would leave every earlier run pointing at bytes that no longer match.

### 1.4 Re-fetching
Fetch compares each act's resolved version against the previous run's, so an unchanged document is not downloaded or written again. The comparison is on the version id, not the text — a new consolidation is a new id.

Reusing a version means reading its stored bytes, which also proves they are still there. The row and the object are backed up separately, so a restore can leave them disagreeing; bytes that are missing, or that no longer hash to what the row recorded, are treated as bytes we do not have and the version is downloaded again.

## 2 Parse
Parse turns each downloaded page into a structure: the articles, paragraphs and annexes the document is made of.

### 2.1 Source documents
The HTML comes in two flavours, depending on which version was downloaded. An act as published in the Official Journal uses one set of CSS class names; a consolidated act uses another. The same article heading in each:

```html
<p class="oj-ti-art">Article 6</p>           <!-- as published -->
<p class="title-article-norm">Article 6</p>  <!-- consolidated -->
```

Underneath, the two agree on more than they differ on. Both wrap every article in the same container:

```html
<div class="eli-subdivision" id="art_6">
```

### 2.2 Section tree
A regulation is already a nested document. Articles contain numbered paragraphs, and annexes contain headings, prose and tables:

```
Regulation
├── Article 1
│   ├── paragraph 1
│   └── paragraph 2
└── Annex I
    ├── heading
    └── table
```

That nesting is what parse keeps, because it is what later lets a piece of text know it is Article 6(2) rather than an anonymous extract.

The HTML nests the same way, so parse reads down the page and records each article, paragraph and annex as it recognises it:

```html
<div class="eli-subdivision" id="art_6">
  <p class="oj-ti-art">Article 6</p>
  <div id="006.001"><p>1. Ships shall ...</p></div>
  <div id="006.002"><p>2. The Commission shall ...</p></div>
</div>
```

becomes:

```
Article 6
├── paragraph 1 — "Ships shall ..."
└── paragraph 2 — "The Commission shall ..."
```

The dialects disagree about more than class names. An as-published paragraph carries its number as a leading `1.` in its own text, where a consolidated one keeps it in a marker beside the text; consolidated annexes nest under levelled sub-headings where as-published ones are flat. Each dialect declares those few differences and hands them to one shared procedure, rather than bringing a parser of its own.

The result is the same shape whatever the source was, which is why a PDF parser could be added later without changing anything downstream.

### 2.3 Output
At the end of parsing, only the operative text (articles and annexes) remains. Everything else on the page is dropped:

- **Recitals**, because consolidation strips them — FuelEU has 72 and consolidated MRV none, so keeping them would cover some acts deeply and others not at all.
- **Footnotes and amendment markers** (`▼M2`), which are citation apparatus and change tracking rather than law. The text the markers wrap is kept.
- **Table scaffolding**. Most tables exist only to indent a list (260 in FuelEU against 13 holding real data), so their structure is discarded and the text flattened into the surrounding paragraph. Genuine data tables are kept as a grid.
- **Formulas**, which are images with no readable alternative. Each becomes a `[formula]` placeholder for now; capturing them properly is tracked as a separate piece of work.

## 3 Chunk
Chunking splits the parsed document into the sections to be retrieved by the agent. The goal is to make a chunk short enough to embed, complete enough to read on its own and specific enough to cite.

### 3.1 Boundaries
The naive approach is to slide a fixed window over the text, e.g. 1000 characters with 200 of overlap. After parsing, however, we have a clean section tree for which each leaf can become one chunk: a numbered paragraph, a data table, a block of annex prose.

Although most chunks sit well underneath the embedding limit and pass through whole, there are a few which extend over the limit. For these sections, they are split at the best boundary available: a line break first, a sentence inside an over-long line, and a blunt character cut only where neither exists. Each piece records that it is part 2 of 3, so a fragment can be recognised as one.

Tables split on row boundaries and repeat the header row on every piece, so no row is left without its column names.

### 3.2 Locators
On its own a chunk is an anonymous paragraph of text. Along with the actual text body, we also store location metadata for the chunk, such as *Article 6(2) of FuelEU*, which we collect from the section tree. Each chunk keeps whatever it inherited from everything above it.

```
Article 6  "Additional zero-emission requirements for energy used at berth"
└── paragraph 2   →   article 6, paragraph 2   →   cited as "Article 6(2)"
```

That address is what lets an answer be checked. *Article 6(2) of FuelEU* can be looked up by whoever reads it, whereas an unattributed extract has to be taken on trust.

It is also what makes cross-references usable. A citation in the text names a division rather than a piece of prose, so finding what it points to is a lookup against the numbers other chunks carry (3.3).

### 3.3 Cross-references
Legislation often makes references to other documents or other sections of the current document. These citations are stored as structured fields alongside the text:

```
"... in accordance with Article 25(2) of Directive (EU) 2018/2001"
   →  instrument 32018L2001, article 25, paragraph 2
```

An article named on its own belongs to the document it sits in; one qualified by an instrument takes that instrument's CELEX id, so it can be matched against the rest of the corpus.

This transforms the corpus into a graph for which references can be followed to the source in a deterministic fashion rather than relying on similarity to surface it.

References to the same document always land. References to other instruments mostly do not: a citation only resolves where the act it names is in the corpus, and most of the acts FuelEU cites are not, because discovery collects what is made *under* the seeds rather than what they cite. A few land anyway — FuelEU cites MRV, which is a seed in its own right. The most-cited act of all, Directive 2018/2001, is outside, and it is where fuel certification is defined. Following the citation graph one hop further is tracked as a separate piece of work.

### 3.4 Chunk identity
A chunk is addressed by a hash of its text and locator rather than its position in the document. Reconciling is then a set difference: new hashes inserted, missing ones deleted, the rest left alone with the vectors they already have, so a re-run over an unchanged corpus costs nothing to embed.

Deletion is the irreversible half, so chunks are pruned only when every stage succeeded — a failed run cannot tell absence from failure.

## 4 Embed
Embedding turns each chunk into a vector, so that retrieval can find a provision (a piece of law) by what it means rather than by the words it happens to use.

### 4.1 Hybrid search
In the most basic RAG approach, we search for semantically similar chunks to the user query using a distance metric to compare vectors. If, however, the user explicitly references an article, year or term, we want to be able to deterministically retrieve that chunk too.

Each chunk is therefore indexed twice: once as a vector, and once as keywords. The keyword index is built by the database from the chunk's own columns, weighting citation and title above body text, so a query mentioning Article 6 favours the chunk that *is* Article 6 over the several that merely mention it.

Both indexes are built at ingest. Neither is sufficient alone: keywords cannot match *plug in at berth* to *on-shore power supply*, and vectors cannot be trusted to tell Article 6 from Article 16.

## 5 Runs
Every ingest is recorded as a run. Each stage reports what it added, changed, left alone and failed on, and that report is stored on the run row rather than only printed, so a run can be accounted for after the fact.

A document that fails one stage does not abort the run: the failure is recorded against its CELEX id, the remaining documents carry on, and the run itself closes as failed. A completed run is stamped with the date the corpus last changed plus a fingerprint of it, so an unchanged re-run keeps its version and a failed one gets none.
