#!/usr/bin/env python3
"""Build a `zkb bench` fixture from a corpus that keeps index pages.

Many knowledge bases carry a page per directory listing what is under it, one
line each:

    | [qmd-teardown.md](zkb://research/qmd-teardown.md) | tobi/qmd 拆解，对照 zkb… |

Those lines are (query, answer) pairs somebody already wrote: the description is
how the corpus owner would ask for the document, and the link says which one it
is. That makes them the cheapest honest fixture available — nobody wrote them
while looking at a retrieval score, which is the failure mode of a fixture the
person measuring also authored.

    python3 scripts/fixture-from-index-pages.py ~/docs docs > fixture.csv
    zkb bench fixture.csv --collection docs

**Known artifact, and it is not small.** The query is copied verbatim off an
index page, so on the keyword path that page is a perfect lexical match and
often outranks the document it points at — measured at 23 of 40 sampled cases.
R@1 is therefore not usable from a fixture built this way; R@5 and R@10 are,
since one taken slot is all it costs. Read those, or write queries by hand if
you need R@1.
"""

import csv, os, re, subprocess, sys

ROW = re.compile(r'^\|\s*\[([^\]]+)\]\((?:zkb://)([^)]+)\)\s*\|\s*(.+?)\s*\|\s*$')
MIN_DESC = 12


def indexed_paths(collection: str) -> set[str]:
    out = subprocess.run(
        ["zkb", "sql",
         f"select rel_path from docs d join collections c on c.id=d.collection_id"
         f" where c.name='{collection}'"],
        capture_output=True, text=True, check=True).stdout.splitlines()
    return {line.strip() for line in out[1:] if line.strip()}


def main() -> int:
    if len(sys.argv) != 3:
        print(__doc__.strip(), file=sys.stderr)
        return 2
    root, collection = os.path.expanduser(sys.argv[1]), sys.argv[2]
    have = indexed_paths(collection)

    seen, cases = set(), []
    skipped = {"not indexed": 0, "description is the filename": 0, "too short": 0}
    for dirpath, _, files in os.walk(root):
        if "index.md" not in files:
            continue
        area = os.path.relpath(dirpath, root)
        area = "root" if area == "." else area.split("/")[0]
        with open(os.path.join(dirpath, "index.md"), encoding="utf-8") as fh:
            for line in fh:
                m = ROW.match(line.rstrip("\n"))
                if not m:
                    continue
                name, rel, desc = m.group(1), m.group(2).lstrip("/"), m.group(3).strip()
                if rel not in have:
                    skipped["not indexed"] += 1
                    continue
                stem = os.path.splitext(os.path.basename(rel))[0]
                flat = desc.lower().replace(" ", "")
                if flat in (stem.lower().replace("-", "").replace("_", ""), name.lower()):
                    skipped["description is the filename"] += 1
                    continue
                if len(desc) < MIN_DESC:
                    skipped["too short"] += 1
                    continue
                cid = rel.replace("/", "_")[:60]
                if cid in seen:
                    continue
                seen.add(cid)
                cases.append({"id": cid, "kind": area, "query": desc, "expected": rel})

    # A query that appears on several index pages cannot be scored. `项目历程：
    # 做过什么、推翻过什么、什么时候停的` is the description of twelve different
    # HISTORY.md files here: the case names one of them, so eleven runs are
    # counted wrong no matter what retrieval does. Measured before this filter:
    # 18 such cases out of 512, and they supplied 9 of the 31 misses — a third
    # of the apparent failures were the fixture's own.
    from collections import Counter
    seen_query = Counter(c["query"] for c in cases)
    ambiguous = sum(1 for c in cases if seen_query[c["query"]] > 1)
    cases = [c for c in cases if seen_query[c["query"]] == 1]
    if ambiguous:
        skipped["query names more than one document"] = ambiguous

    w = csv.DictWriter(sys.stdout, fieldnames=["id", "kind", "query", "expected"])
    w.writeheader()
    w.writerows(cases)
    print(f"{len(cases)} cases; skipped " +
          ", ".join(f"{n} {why}" for why, n in skipped.items() if n), file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
