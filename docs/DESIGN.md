# Design notes

Decisions that are not obvious from the code, and the measurements behind them.
Numbers come from a 195-document, 1528-chunk personal corpus of mixed Chinese,
Japanese and English technical writing, on an M2 Pro.

Several of these overturned the original design. They are recorded as they
landed rather than tidied up, because the wrong version is usually the more
instructive half.

---

## The index is disposable, the files are not

Everything in SQLite is a projection of files that are still on disk. That single
constraint removes a category of work that a database-of-record needs: no version
table, no soft delete, no backup format, no migration when a column changes, no
export path. Changing the schema means deleting the index and re-running.

It also decides the failure mode. When something is wrong, the fix is almost
always "delete the index and re-run", so the layout draws that boundary as a
directory boundary rather than a convention:

```
~/.zkb/data/     ← irreplaceable
~/.zkb/index/    ← rebuild: zkb index
~/.zkb/models/   ← rebuild: zkb model pull
~/.zkb/run/      ← socket, pid, log
```

A convention would be violated the first time anybody typed `rm -rf ~/.zkb`.

The claim was false for one row until 2026-08-28. A collection's registration —
its root and filters, the thing that decides which files get projected at all —
lived only in `collections`, and nothing on disk recorded it. Measured on a
scratch home: register `notes` and `proj`, run the reset above, and afterwards
only `docs`, `memory` and `numbers` exist. Both user collections are gone,
nothing says so, and the exit code is 0. On the machine this was found on that
would have been five of eight collections, after which `recall --scope` and
`search --collection agent-memory` answer emptily and plausibly.

Registrations now live in `data/collections.csv` and the table is replayed from
it. The replay is folded into `roots.ensureOwn` rather than given a function of
its own: all three callers already invoke that before listing roots, so there is
no fourth place that could forget, and a registration replayed on only some
paths would be worse than one never replayed — it would come back or not
depending on which command ran first.

## Retrieval fuses ranks, not scores

Vector distance and BM25 are different units. Combining them with a weight
requires a calibration constant, and a constant without an experiment behind it
is debt that compounds every time the corpus changes. Reciprocal Rank Fusion
consumes ranks only, so it needs neither.

Two parameters remain, and both are earned:

- `k = 60`, the value from the original RRF paper.
- A first-place bonus expressed as a **fraction of one rank-1 contribution**
  rather than an absolute score. An earlier design used an absolute `+0.05`;
  measured against `k=60` that is three times a whole rank-1 contribution
  (1/61 ≈ 0.0164), which makes "ranked first in either path" beat "found by both
  paths" — inverting the intent, since cross-path agreement is the strongest
  signal available. A unit test caught it.

## CJK needed its own tokenizer

SQLite's `unicode61` does not segment CJK at all: a paragraph of Chinese becomes
one token. `trigram` does segment, but measured **0.000 recall on 7 of 10 Chinese
queries** — Chinese has no spaces, so a whole question becomes a single
multi-character phrase term requiring exact adjacency in the source.

`src/db/fts5_cjk.c` replaces both: overlapping bigrams for Han, kana and Hangul,
whole words with ASCII case folding for Latin. Keyword recall@10 went **0.525 →
0.793** on the same queries and the same chunks.

Bigrams rather than a dictionary segmenter, for three reasons:

1. **Segmentation is context-dependent and fails silently.** If a term segments
   one way in the document and another in the query, the match is simply absent.
   Bigrams cannot fail this way — a given string always yields the same pairs.
2. **One rule for three scripts.** A Chinese dictionary does not help Japanese,
   and neither helps `sqlite-vec`, `frontmatter` or any other identifier, which
   is most of the interesting vocabulary in technical writing.
3. **The inverted index would inherit a dictionary version.** Today changing the
   tokenizer costs seconds, because FTS is rebuilt from `chunks.text` without
   re-embedding.

Measured later: on this corpus the keyword path contributes **no unique recall** —
every relevant document it surfaces is also found by the vector path. Its value
is agreement, which is what drives ranking. A better segmenter therefore has
almost nothing to win here; that would change if a measurable set of documents
were keyword-only.

## Cross-document links

`zkb://docs/projects/x/REQ.md` names a collection and a path inside it, not a
path relative to the document that mentions it. Resolving such a URI as a
relative path was the single largest source of false "broken link" findings —
288 of 348 on one corpus.

**The first segment is the collection**, since 0.0.32. Before that everything
after `://` was one path, looked up by `rel_path` across every collection at
once, and three things followed: `zkb://index.md` was ambiguous (three
collections had one), a link could not name a document outside its own
collection at all, and a reference that resolved today could become ambiguous
tomorrow when an unrelated collection gained a file with the same path. No
fallback to the old reading was kept — "try collection, then path" would resolve
everything today and flip the meaning of `zkb://projects/…` the day somebody
registers a collection called `projects`.

**The scheme is not matched.** Anything that is not a network or filesystem URL
(`http`, `https`, `file`, `ftp`, `data`) is read as collection-rooted, so a
corpus written against another tool's scheme keeps resolving with no migration.
That generosity is for *reading only*. A migration script that rewrote on the
same rule turned `otrans://auth` into `otrans://docs/auth` and `wss://host` into
`wss://docs/host` — 38 edits across 19 files, in prose and code samples that
were never links. An unknown scheme is somebody else's namespace until it is in
the index; only the reading side may assume otherwise.

Wikilinks (`[[name]]`) are global by stem for the same reason: resolving one
relative to the linking document finds the wrong file, or nothing.

**A resolver change must invalidate what the old one produced.** `resolveLinks`
only looks at links with no target yet, so after 0.0.32 shipped, 928 of 932
scheme links still carried a target computed by the old reading and `maintain`
reported the index healthy. Each would have broken silently and separately
whenever its document was next re-indexed. `zkb maintain --relink` exists for
exactly this, and rewrites the graph rather than the index — re-indexing would
also fix it, at the cost of re-embedding every chunk over a problem no embedding
was involved in.

## Numbers are not retrievable

450000 and 480000 are neighbours in a 1024-dimensional space, and "what is my
salary" is a comparison rather than a similarity. So numeric and date columns are
never embedded: they become materialized SQL columns with B-tree indexes, and
only free-text columns enter the vector index.

The same reasoning drives the memory system's fact snapshot. Current values are
**injected** into `recall`, never retrieved — a narrative mentioning last year's
salary would rank just as well as the fact itself, with nothing to mark which is
current.

Combining semantic and exact filtering has one correct order: **filter first,
then KNN over what survives** (`rowid IN (...)`, which sqlite-vec supports for a
single constraint). Taking a large k and then dropping non-matching rows loses
results, and loses them unpredictably.

## Materialized columns, not EAV

An entity-attribute-value store earns its complexity when the truth lives in the
database and adding a field must not force a migration. Here the truth is a csv
file and the index is disposable: change the header, re-index, done. All of EAV's
complexity buys the avoidance of migrations, and there are none to avoid. What it
buys instead is `amount > 1000` going through a B-tree rather than three
self-joins.

Column types are inferred from the header plus the first 200 rows. Two rules were
wrong in the original design and were fixed against real data:

- **Enum detection cannot be a pure ratio.** `distinct/total < 0.1` rejected a
  column with 13 categories across 90 rows, which is unmistakably a category. A
  ratio scales wrong in both directions — under it, a 20-row file can have no
  enum at all, while a 100k-row file would accept 9,000 distinct values. Two
  independent bounds (absolute cardinality, repetition rate) say what was meant.
- **A leading zero means an identifier.** `0120345` parses as a float and loses
  the zero irrecoverably. Leading zeros are meaningful in postcodes and account
  numbers and meaningless in quantities.

Erring is asymmetric: classifying free text as a category removes it from the
embedding and so from semantic search, while classifying a category as text only
costs an index. The thresholds fall towards text.

## Write-time deduplication is required, not an optimisation

The characteristic failure of a memory system is recording the same thing twenty
times, because each session's agent has no idea what the last one wrote. By the
time a maintenance pass notices, the duplicates are already polluting recall.

So `remember` embeds first, runs a KNN inside the memory collection, and refuses
to write above a similarity threshold — printing the candidates and exiting
non-zero. The judgement stays with the caller: update that memory, or assert this
one is genuinely new. Measured: the same preference reworded scored 0.944 and was
refused; an unrelated memory was written normally.

Deduplication compares document-side embeddings on both sides. Using the query
prefix for the new text would compare across two distributions, and the cosines
would not mean what they say.

## Maintenance: suppression matters more than thresholds

Vector-based checks (near-duplicate, island, stale) were calibrated the same way
retrieval was: pool candidates far below any plausible threshold, judge them, then
derive the threshold from the judgements.

The result was decisive and not what the design predicted. Of 170 candidate
pairs, 149 were structurally explainable — tables of contents resembling each
other, the same filename in two projects, two sections of one project's
documentation. **The highest-scoring pair in the entire pool was two link
lists**, while genuine duplicates sat interleaved among the false ones. No
threshold separates them; only classification does.

Three structural rules, needing no maintained list:

1. Chunks that are ≥30% markdown link syntax are navigation, not prose.
2. The same filename in different directories is a template.
3. Two *different* sections of one project overlap by design — but the *same*
   section title restated is a real finding.

The **island** check was measured as not viable and ships disabled. At any
threshold that reports a readable number of chunks, what it reports is ordinary
content on a topic only one project covers. In a corpus of unrelated projects,
"isolated" and "unique subject" are the same thing.

## Two experiments that ended in "no"

**A cross-encoder reranker made retrieval worse**: recall@10 0.928 → 0.814, MRR
0.974 → 0.575, at +6.2 s per query. The integration was verified against
unambiguous pairs before the result was believed. The cause is that the baseline
has no headroom — 14 of 19 judged queries already return every relevant document,
and 18 of 19 put a relevant document first. There is at most +0.026 MRR to win
and −0.398 to lose. Independently, 206 ms per candidate against a 61 ms search is
disqualifying regardless of quality.

**All-pairs comparison replaced KNN** for near-duplicate detection. Per-chunk
`k=6` took 30.7 s and was also wrong for the question: chunks overlap their
neighbours by 80 tokens and share a heading prefix, so a chunk's nearest six are
routinely its own document's, pushing the cross-document pair off the list
entirely. Vectors are normalized, so cosine is a dot product; all pairs over 1528
chunks takes 2.2 s.

## The daemon

One `llama_context` cannot be used concurrently. A plain mutex would let a
background index starve interactive queries for tens of seconds, so embedding
goes through a priority queue with interactive strictly ahead of ingest.

Strict priority turned out to be insufficient, and the measurement said why:
ingest submits one chunk and waits, so its queue is never deeper than one and
there is nothing to jump ahead of. What a query actually waits for is the chunk
already inside `llama_decode`, which cannot be interrupted — about 300 ms. The
fix is a backoff: after an interactive request, ingest holds off briefly, checked
**per chunk** rather than per document.

That backoff then needed a ceiling. Every interactive request refreshes the
timer, so under sustained querying the window never expires and ingestion stops
permanently — measured at zero documents indexed during 20 s of continuous
search, indexed within 3 s once it stopped. An agent searching in a loop is the
design target, and the failure is silent. Deferral is now capped, and the
override count is exported in `stats`.

FSEvents accelerates the scan rather than replacing it. It can fail silently —
a stream that will not start, a network mount that does not report — and as an
optimisation that degrades to "up to 30 s late", while as the only mechanism it
would degrade to "this edit is never indexed", with no way to tell the two apart.

## Cutting UTF-8

Every excerpt, label and preview is bounded in bytes, because its destination is:
a column, a line, a report row. A cut landing inside a multi-byte sequence
produces bytes that are not text, and SQLite stores them without complaint.

This rule was written by hand in four places and wrong in two — one of which fed
`chunks.heading_path`, an FTS-indexed column. It now lives in `src/util/utf8.zig`
and the tests assert the property directly: for *every* limit, what comes out
must be valid UTF-8.
