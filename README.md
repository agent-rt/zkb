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

## Measuring retrieval

Both numbers above are claims about recall, and a claim that cannot be re-run
cannot catch a regression. `zkb bench` takes a fixture of queries with known
answers and reports what each retrieval path returns for them:

```
zkb bench fixture.csv --collection docs
```

```
mode        R@1    R@3    R@5   R@10    MRR   ms/q
keyword   0.750  0.917  0.917  0.917  0.833     25
vector    0.917  1.000  1.000  1.000  0.958     57
hybrid    0.917  1.000  1.000  1.000  0.944     76
```

Every mode runs over every case, because the value is the comparison: whether
the vector path earns its 609 MiB, whether a change to the chunker moved
keyword recall, whether fusion beat both. Below the table are a per-kind
breakdown and, for each case, the expected documents that never came back — a
score that dropped says nothing about what got lost.

The fixture is csv, like everything else zkb reads by hand:

```csv
id,kind,query,expected
mqtt-idle,exact,curl 订阅 MQTT 空闲 121 秒被断开,research/scriptc-mqtt-subscribe.md
clip-softmax,semantic,分类模型的 softmax 输出不能当置信度用,research/fashion-clip-classification.md
```

`query` and `expected` are required; `id` and `kind` are optional. `expected`
holds one or more paths separated by `;`, matched against the returned path at
a component boundary — so `DESIGN.md` matches `docs/DESIGN.md`, and never
`docs/xDESIGN.md`. `kind` is a free-form label used only to group the report,
so a drop can be attributed to a kind of query rather than to "the score went
down".

No fixture ships with zkb. A fixture is ground truth about a particular corpus,
and zkb's own repository holds too few documents to produce a number worth
reading — under a handful of documents every term appears in most of them, IDF
collapses, and BM25 stops ranking. Write one against a collection you already
have.

If your corpus keeps index pages — a page per directory listing what is under
it, one line each — you already have several hundred (query, answer) pairs that
nobody wrote while looking at a score:

```
python3 scripts/fixture-from-index-pages.py ~/docs docs > fixture.csv
zkb bench fixture.csv --collection docs
```

494 cases out of this corpus, and the shape it measures is real:

```
mode        R@1    R@3    R@5   R@10    MRR  docs/q   ms/q
keyword   0.117  0.822  0.877  0.893  0.471    6.3     60
vector    0.741  0.883  0.911  0.915  0.812    6.2     57
hybrid    0.652  0.901  0.933  0.955  0.778    6.1    111
```

Fusion beats either path alone at depth, by more than the two paths differ from
each other. Two columns keep that from being read as more than it is. `docs/q`
says `-k 10` returned a median of six *documents* — `-k` counts chunks, and one
document may supply three — so `R@10` means "within about six". And `R@1` is not
usable from a fixture built this way at all: the query is copied verbatim off an
index page, so on the keyword path that page is a perfect lexical match and
outranks the document it points at in 23 of 40 sampled cases.

`bench` runs in-process and never through the daemon: elsewhere the CLI prefers
the daemon because its model is already resident, but a measurement whose
subject depends on which build happens to be running in the background is not a
measurement. The model is loaded once for the whole run.

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

## Collections

A collection is a root plus the filters that select files under it. Registering
one writes it to `~/.zkb/data/collections.csv` and to the index, so the daemon
keeps every collection fresh — registering one is not a one-off import — and the
registration survives the index being thrown away:

```
rm -rf ~/.zkb/index && zkb index    # every collection comes back
```

That file is the record; the `collections` table is the projection of it, rebuilt
on the next scan. Editing it by hand works, and is the way to move a root or
rename filters without re-running the command — `zkb doctor` checks what you
wrote, because a mistyped root indexes nothing and says nothing about why.

```
zkb index --collection notes --root ~/notes --ext md
zkb index --collection agent-memory --root ~/.claude/projects \
          --include '*/memory/*.md' --ext md
zkb search "how did I decide that" --collection agent-memory
```

`--include` globs the path relative to the root: `*` and `?` stop at a `/`, `**`
spans whole directories. It also prunes the walk, which is the difference between
scanning a directory of projects and scanning only the parts you asked for — on a
6020-file tree, 20 files seen in 2.3s instead of 6020 in 10.8s.

`--root` may be repeated, which suits a shell glob
(`--root ~/.claude/projects/*/memory`); the paths are folded into their shared
parent. Prefer `--include` when directories will be added later, since several
roots name only what exists now.

A collection is also reversible, which it was not at first:

```
zkb collection rm agent-memory      # forgets the index; the files stay
```

**Do not make a subdirectory of a collection into its own collection.** A
collection is the identity `zkb://` links resolve against, so carving one out of a
linked tree severs the graph in both directions. To narrow a search, narrow the
search:

```
zkb search "where did I leave off" --path 'agents/handoffs/**'
```

The filter is exact rather than a post-filter on a global top-k: the document set
is resolved by glob, BM25 restricts inside its own query, and the vector side
scores that subset directly.

## Context — what a subtree is

A result is a path and a paragraph. The caller reading it has never seen this
corpus, so `agent-memory/…/feedback_verify_the_instrument.md` could be a note, a
decision, a draft, or somebody else's document quoted in passing. One sentence
from the person who made the tree settles it, and it is knowledge only they have:

```
zkb context add zkb://docs "我自己的知识库：技术笔记、项目文档、会议记录"
zkb context add zkb://docs/research "技术调研与外部资料摘录，多为一次性的实测报告"
zkb context list
```

```
1. docs/research/qmd-teardown.md  [score 0.041 vec#2 fts#1 chunk 7]
   (我自己的知识库：技术笔记、项目文档、会议记录 · 技术调研与外部资料摘录…)
   qmd 拆解 > A 级：直接可抄 > 四后端消融基准
```

**Every matching prefix applies, widest first** — not only the deepest. "These
are my notes" and "this subtree is one-off measurements" are both true, and
neither implies the other. Prefixes match on component boundaries, so
`research` never describes `research-notes/`.

**It never reaches ranking.** A description belongs to a whole subtree, so
scoring with it would lift every document in a described tree above every
document in an undescribed one — a collection-level bias wearing the clothes of
relevance. It is attached after retrieval, for the reader.

Descriptions live in `~/.zkb/data/contexts.csv` beside the collection
registrations, for the same reason: they are authored, not derived. `zkb skill`
prints them too, so an agent reading it starts out knowing what each collection
holds.

## Ignoring files

`.zkbignore` uses gitignore syntax — the real one, checked against
`git check-ignore` across 102 cases, including `!` negation, anchoring, `**`,
character classes, and per-directory nesting.

The repo's own `.gitignore` is respected too, including files above the collection
root, since a file not worth committing is rarely knowledge worth indexing. Rules
live in the corpus rather than the database, so they are versioned with the
documents and unset by deleting a line. `.zkbignore` loads after `.gitignore`, so
it can override it — `!drafts/` indexes a directory git ignores.

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
│           collections.csv                 ← which roots are registered
│           contexts.csv                    ← what each subtree is
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
