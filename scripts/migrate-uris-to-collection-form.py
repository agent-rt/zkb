#!/usr/bin/env python3
"""Rewrite `zkb://path` links into `zkb://collection/path`.

The scheme's first segment is now the collection name. A link written under the
old reading — everything after `://` being one relative path — names a
collection that does not exist, and `zkb path` says so:

    $ zkb path zkb://research/qmd-teardown.md
    no such collection: research

**The rewrite is the old semantics, spelled out.** `resolveLinks` used to match
inside the *linking document's own* collection, so `zkb://x` in a document of
collection `c` meant `c`'s `x`. This prefixes exactly that. Nothing changes
where a link points; the link now says where it points.

    python3 scripts/migrate-uris-to-collection-form.py            # dry run
    python3 scripts/migrate-uris-to-collection-form.py --write

Idempotent, on one condition that is checked rather than assumed: a link is
skipped when its first segment is already a collection name. That is only safe
while no *old-form* link begins with one — a `zkb://memory/x.md` meaning the
relative path `memory/x.md` would be skipped and silently left meaning the
`memory` collection. The script refuses to run if it finds any, rather than
guessing which reading was meant.
"""

import argparse
import re
import subprocess
import sys
from collections import Counter
from pathlib import Path

# Any non-network scheme is collection-rooted, the same rule `maintain` applies.
LINK = re.compile(r'(?<![A-Za-z0-9])([a-z][a-z0-9+.-]*)://([^\s)\]>"\'`]+)')
EXTERNAL = {"http", "https", "file", "ftp", "data", "mailto"}


def sql(statement: str) -> list[list[str]]:
    out = subprocess.run(["zkb", "sql", statement],
                         capture_output=True, text=True, check=True).stdout.splitlines()
    return [line.split("\t") for line in out[1:] if line.strip()]


def collections() -> dict[str, str]:
    return {row[0]: row[1] for row in sql("select name, root from collections")}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--write", action="store_true", help="apply; otherwise report only")
    ap.add_argument("--collection", action="append", default=None,
                    help="limit to these collections; repeatable. Default: all of them.")
    args = ap.parse_args()

    roots = collections()
    names = set(roots)

    # Which collection each document belongs to decides its links' prefix.
    docs = sql("select c.name, d.rel_path from docs d"
               " join collections c on c.id = d.collection_id")

    # The safety check the docstring promises, made before anything is written —
    # and scoped to what is being migrated. Checking the whole corpus made the
    # script single-use: once one collection was done, its own migrated links
    # tripped the guard for every collection after it.
    scope = ""
    if args.collection:
        names_sql = ",".join("'" + c.replace("'", "''") + "'" for c in args.collection)
        scope = (" and l.doc_id in (select d.id from docs d"
                 " join collections c on c.id = d.collection_id"
                 f" where c.name in ({names_sql}))")
    already = sql(
        "select count(*) from links l where l.raw like '%://%'"
        " and substr(replace(replace(l.raw,'zkb://',''),'lore://',''),1,"
        "   instr(replace(replace(l.raw,'zkb://',''),'lore://','')||'/','/')-1)"
        "   in (select name from collections)" + scope)
    if already and already[0][0] not in ("0", ""):
        where = " in " + ", ".join(args.collection) if args.collection else ""
        print(f"{already[0][0]} link(s){where} already begin with a collection name.\n"
              "Cannot tell an already-migrated link from an old-form one that happens\n"
              "to start with a collection's name. Migrate those by hand first.",
              file=sys.stderr)
        return 1

    changed_files, changed_links = 0, 0
    per_collection: Counter[str] = Counter()

    for coll, rel in docs:
        if args.collection and coll not in args.collection:
            continue
        root = roots.get(coll)
        if root is None:
            continue
        path = Path(root) / rel
        if not path.is_file() or path.suffix.lower() != ".md":
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            continue

        hits = 0

        def repl(m: re.Match[str]) -> str:
            nonlocal hits
            scheme, rest = m.group(1), m.group(2)
            if scheme in EXTERNAL:
                return m.group(0)
            first = rest.lstrip("/").split("/", 1)[0]
            if first in names:  # already in the new form
                return m.group(0)
            hits += 1
            return f"{scheme}://{coll}/{rest.lstrip('/')}"

        new = LINK.sub(repl, text)
        if hits:
            changed_files += 1
            changed_links += hits
            per_collection[coll] += hits
            if args.write:
                path.write_text(new, encoding="utf-8")
            else:
                print(f"  {coll}/{rel}  ({hits} link(s))")

    verb = "rewrote" if args.write else "would rewrite"
    print(f"\n{verb} {changed_links} link(s) in {changed_files} file(s)")
    for coll, n in per_collection.most_common():
        print(f"  {coll}: {n}")
    if not args.write:
        print("\nnothing was written; re-run with --write")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
