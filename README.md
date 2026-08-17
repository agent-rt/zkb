# zkb

Local knowledge base and agent memory over your own markdown and csv.
SQLite + sqlite-vec + FTS5 + Qwen3-Embedding, one binary, no services.

zkb does not answer questions. It finds the evidence and hands it over — the
caller is already a language model, and generating the answer is the part it
would only duplicate.

```
zkb daemon start --preload
zkb search "how did I design the retrieval layer"
zkb query  "how did I design the retrieval layer"   # context pack, budgeted
zkb recall                                          # what you already decided
```

## Two file formats, and that is all

**Markdown** for prose, **csv** for numbers. The filesystem is the truth; the
index is derived and can be deleted at any time:

```
rm -rf ~/.zkb/index && zkb index
```

Nothing else in the system is authoritative. There is no proprietary store to
export from, no migration to run when a column changes, and no version table —
your files are already under whatever version control you use.

## Retrieval

Vector KNN and BM25 in parallel, fused with Reciprocal Rank Fusion. The two
answer different questions and neither subsumes the other: vectors find the same
idea in different words, including across languages; BM25 finds the exact
identifier, error code or proper noun that a 1024-dimensional embedding smears
away.

**CJK works.** SQLite's built-in `trigram` tokenizer scored 0.000 recall on 7 of
10 Chinese queries in testing — Chinese has no spaces, so a whole question
becomes one phrase term requiring exact adjacency. zkb ships its own FTS5
tokenizer (`src/db/fts5_cjk.c`): overlapping bigrams for Han, kana and Hangul,
whole words for Latin. Measured keyword recall@10 went from 0.525 to 0.793 on
the same queries.

## Memory

```
zkb remember "prefer RRF over weighted sums for fusion" --type decision
zkb recall
```

`remember` embeds the text, runs a KNN against existing memories, and **refuses
to write** if something above the similarity threshold already exists — printing
the candidates so you can decide whether to update that one or force a new
record. Recording the same thing twenty times is how memory systems die, and
each session's agent has no idea what the last one wrote.

Memories are markdown files with frontmatter, one per memory. `forget` moves a
file to `archive/` and drops it from the index; nothing is ever deleted.

## Numbers

Three kinds of number exist and are not stored the same way:

| | example | where |
|---|---|---|
| derived | age, BMI, balance | **not stored** — store the birth date, compute the age |
| archive fact | birth date, height, salary | `facts.csv` |
| growing series | weight log, expenses | `records/<type>/*.csv` |

Storing a derived quantity creates two truths that drift the moment a birthday
passes. Facts are append-only with two time axes — `at` is when the fact took
effect, `recorded_at` is when it was written down — so correcting a value is a
new row, not an edit.

For a growing series, **the csv header is the schema**. Column types are inferred
from the header plus the first 200 rows; there is no configuration file unless
the inference gets a column wrong.

```
zkb records expenses --where "amount > 1000 AND category = food"
zkb records expenses --agg   "sum(amount) by category"
zkb records weight   --window "avg(kg) over 7 by date"
zkb sql "select ..."          # read-only escape hatch
```

**Numbers never enter the vector index.** 450000 and 480000 are neighbours in
embedding space, and "what is my salary" is a comparison, not a similarity. Only
free-text columns are embedded; the rest are materialized SQL columns with
B-tree indexes. Combining the two filters *first* and runs KNN over what
survives — the other order silently loses results.

## Install

```
brew install agent-rt/tap/zkb
zkb model pull          # 609 MiB, reuses an existing Hugging Face cache
zkb doctor
```

Or build it:

```
zig build llama-cpp     # cmake-builds llama.cpp once
zig build -Doptimize=ReleaseFast
```

Requires Zig 0.16.0. Apple Silicon only for now — Metal is the only accelerator
wired up.

## For agents

`zkb mcp` is an MCP stdio server exposing `zkb_search`, `zkb_query`,
`zkb_recall` and `zkb_records`.

`zkb skill` prints a SKILL.md describing how to use zkb, including which record
types and fact keys exist *on this machine* — the part an agent cannot guess.
Pipe it wherever your agent reads skills from.

## Links between documents

`zkb://projects/x/REQ.md` in a markdown link is rooted at the collection rather
than at the linking document's directory. Any scheme that is not `http`, `https`,
`file`, `ftp` or `data` is treated the same way, so a corpus written against
another tool's URI scheme keeps working. `zkb maintain` reports links that do not
resolve, distinguishing "the target is not an indexed document" from "the target
does not exist" — the two need different fixes.

## Layout

```
~/.zkb/
├── data/      memory/ facts.csv records/   ← the only irreplaceable directory
├── index/     zkb.db                       ← rebuild: zkb index
├── models/    *.gguf                       ← rebuild: zkb model pull
└── run/       socket, pid, log, trace
```

Everything outside `data/` can be deleted at any time. That boundary is a
directory boundary rather than a convention, because "delete the index and
re-run" is the fix for half of what can go wrong and it must never be a sentence
that can destroy a memory.

## Design notes

`docs/DESIGN.md` records the decisions that are not obvious from the code, and
`docs/recipes.md` collects worked SQL for the things the restricted query syntax
deliberately cannot express.

## License

Apache-2.0
