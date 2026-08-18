#!/usr/bin/env bash
# 用 git check-ignore 当 oracle，生成 .zkbignore 的对照用例。
#
# 自创的语义只能靠实现者解释；能对账的规范可以被证伪。这个脚本让 git 自己
# 回答每一条，输出 tests/fixtures/gitignore-cases.txt 供 Zig 测试读取。
set -euo pipefail

out="$(cd "$(dirname "$0")/.." && pwd)/tests/fixtures/gitignore-cases.txt"
tmp="$(mktemp -d)"
trap 'rm -rf "$tmp"' EXIT

cd "$tmp"
git init -q .
git config core.excludesfile /dev/null

# 每组：一份 .gitignore 内容，加上一批待判定的路径。
# 覆盖 gitignore 手册里所有会咬人的规则。
cases=(
  # 名称=组名; IGNORE=规则; PATHS=路径（d: 前缀表示目录）
  "basename|*.log|a.log b.log a/b.log a/b/c.log notes.md"
  "anchored_root|/build|build d:build a/build a/build/x.o"
  "anchored_mid|doc/*.md|doc/a.md doc/sub/a.md other/a.md"
  "dir_only|logs/|d:logs logs/a.txt d:a/logs a/logs/b.txt"
  "dir_only_vs_file|logs/|logs"
  "negation|*.md%!keep.md|a.md keep.md sub/keep.md sub/a.md"
  "negation_order|!keep.md%*.md|a.md keep.md"
  "double_star_lead|**/tmp|tmp a/tmp a/b/tmp a/tmpx"
  "double_star_trail|build/**|build/a build/a/b d:build other/build/a"
  "double_star_mid|a/**/b|a/b a/x/b a/x/y/b b a/b/c"
  "question|?.md|a.md ab.md sub/a.md"
  "charclass|[abc].md|a.md b.md d.md sub/a.md"
  "charclass_range|x[0-9].md|x1.md x9.md xa.md"
  "charclass_neg|[!a]bc|abc bbc cbc"
  "escaped_star|a\\\\*b|a*b axb"
  "comment|# not a rule%*.tmp|a.tmp '#'"
  "trailing_space|*.md   |a.md b.txt"
  "nested_negate_in_ignored_dir|build/%!build/keep.md|build/keep.md build/other.md d:build"
  "star_no_cross|a/*/c|a/b/c a/b/x/c"
  "deep_anchored|/a/b|a/b a/b/c x/a/b"
  "both|*.md%!/README.md|README.md sub/README.md a.md"
  "reinclude_dir|*%!*/%!*.md|a.md d:sub sub/b.md a.txt"
  "double_star_only|**|a a/b d:a"
  "anchor_vs_basename|a/b%c|a/b x/a/b c x/c a/b/c"
  "dir_then_negate_file|node_modules/%!node_modules/keep.md|node_modules/keep.md d:node_modules"
  "negate_then_dir|!keep/%build/|d:keep d:build build/x keep/x"
  "trailing_slash_root|/logs/|d:logs logs/a d:a/logs"
  "star_dot|*.|a. a.md ab."
  "leading_star_slash|*/tmp|a/tmp tmp a/b/tmp"
  "escaped_bang|\\\\!important|'!important' important"
  "middle_double_star_deep|src/**/test|src/test src/a/test src/a/b/test test"
  "class_in_dir|log[0-9]/|d:log1 d:logx log1/a.txt"
)

: > "$out"
{
  echo "# 由 scripts/gen-gitignore-cases.sh 生成，oracle 是 git check-ignore。"
  echo "# 手改无意义——重新生成即可。格式：group / rules / 每行一个 path=verdict"
  echo "# git version: $(git --version | awk '{print $3}')"
  echo
} >> "$out"

for c in "${cases[@]}"; do
  IFS='|' read -r name rules paths <<< "$c"
  printf '%s\n' "$rules" | tr '%' '\n' > rules.txt
  echo "GROUP $name" >> "$out"
  while IFS= read -r r; do echo "RULE $r" >> "$out"; done < rules.txt
  # 路径必须真实存在。git 判定「是不是目录」靠文件系统，而 --no-index 加一个
  # 不存在的路径会给出不同答案——目录用例恰好是最难推理、最需要 oracle 的那些。
  # 第一版没建路径，于是 oracle 在 `*` + `!*/` + `!*.md` 这组上说 ignore，
  # 而真实 git（以及正确的实现）说 keep。按那个假读数改实现会把对的改坏。
  rm -rf tree && mkdir -p tree
  for p in $paths; do
    path="${p#d:}"; path="${path#\'}"; path="${path%\'}"
    # 先建全部目录，文件在第二轮建：同一组里 `d:build` 和 `build/x` 都出现时，
    # 交错创建会撞上「同名的文件已存在」。
    case "$p" in
      d:*) mkdir -p "tree/$path";;
      *)   mkdir -p "tree/$(dirname "$path")";;
    esac
  done
  for p in $paths; do
    path="${p#d:}"; path="${path#\'}"; path="${path%\'}"
    case "$p" in
      d:*) printf 'x\n' > "tree/$path/.keep";;
      *)   [ -d "tree/$path" ] || printf 'x\n' > "tree/$path";;
    esac
  done
  cp rules.txt tree/.gitignore

  for p in $paths; do
    isdir=0; path="${p#d:}"; path="${path#\'}"; path="${path%\'}"
    case "$p" in d:*) isdir=1;; esac
    if git -C tree check-ignore -q "$path" 2>/dev/null; then v=ignore; else v=keep; fi
    echo "CASE ${isdir} ${path} ${v}" >> "$out"
  done
  echo >> "$out"
done

n=$(grep -c '^CASE ' "$out")
echo "生成 $n 条用例 → $out"
