# zkb 配方

`--where` / `--agg` / `--window` 是受限语法，覆盖日常八成的问题。剩下两成用
`zkb sql` —— **它不是"高级用法"，是逃生舱**：受限语法总有说不出来的东西，
而没有逃生舱的代价是 Agent 直接卡死。

下面每一条都在真实 schema 上跑过。

---

## 结构化数据（records）

### 先看有什么

```bash
zkb records                        # 所有类型和行数
zkb records expenses --schema      # 推断出的列类型
```

`--schema` 里的 `kind` 决定了这一列能做什么：`number` / `date` 能比较和聚合，
`enum` / `id` 能过滤，只有 `string` 进向量、能被语义检索到。

### 过滤

```bash
zkb records expenses --where "amount > 1000"
zkb records expenses --where "category = food AND amount > 500"
zkb records expenses --where "merchant LIKE '%コンビニ%'"
zkb records expenses --where "category IN (food, clothing)"
zkb records expenses --where "note IS NULL"
zkb records expenses --where "(category = food OR category = home) AND amount > 1000"
```

值里有空格或运算符就加引号：`--where "note = 'a > b'"`。

### 聚合

```bash
zkb records expenses --agg "sum(amount) by category"
zkb records expenses --agg "count(*)"
zkb records expenses --agg "avg(amount) by merchant"
zkb records expenses --agg "sum(amount) by category" --where "date >= 2026-08-01"
```

### 移动平均

```bash
zkb records weight --window "avg(kg) over 7 by date"
zkb records expenses --window "sum(amount) over 30 by date"
zkb records expenses --window "avg(amount) over 7 by date partition category"
```

`over N` 是"到当前行为止的 N 行"，不是"N 天"——按天记录时两者相同，一天多条时不同。

### 语义 + 精确同用

```bash
zkb records expenses --search "咖啡" --where "amount < 1000"
```

**先 SQL 过滤、再在结果集上做 KNN**。反过来（先取 top-k 再过滤）会漏结果，
而且漏得不可预测。

---

## 逃生舱：zkb sql

只读连接，只接受 `SELECT` / `WITH` / `EXPLAIN`，一次一条语句。
输出 TSV，加 `--json` 出 JSON。

### 时间序列：按月汇总

受限语法没有日期函数，这是第一类要下沉到 SQL 的问题。

```bash
zkb sql "
select substr(date, 1, 7) as month,
       category,
       sum(amount) as total
from rec_expenses
group by 1, 2
order by 1, 3 desc"
```

### 环比

```bash
zkb sql "
with m as (
  select substr(date,1,7) as month, sum(amount) total
  from rec_expenses group by 1
)
select month, total,
       total - lag(total) over (order by month) as delta,
       round(100.0 * (total - lag(total) over (order by month))
             / lag(total) over (order by month), 1) as pct
from m order by month"
```

### 占比

```bash
zkb sql "
select category,
       sum(amount) as total,
       round(100.0 * sum(amount) / (select sum(amount) from rec_expenses), 1) as pct
from rec_expenses group by 1 order by 2 desc"
```

### 分位数

SQLite 没有内置 `percentile`，但窗口函数够用：

```bash
zkb sql "
with r as (
  select amount,
         percent_rank() over (order by amount) as pr
  from rec_expenses
)
select min(amount) filter (where pr >= 0.5)  as p50,
       min(amount) filter (where pr >= 0.9)  as p90
from r"
```

### 两根时间轴：延迟了多久才记下来

`at` 是生效时间，`recorded_at` 是写下来的时间。差值就是"我隔了多久才记"。

```bash
zkb sql "
select key, value_txt, at, recorded_at,
       julianday(recorded_at) - julianday(at) as lag_days
from facts
where recorded_at != ''
order by lag_days desc"
```

### 跨类型 join

受限语法只认一个类型，这是第二类必须下沉的问题。

```bash
zkb sql "
select w.date, w.kg, e.total
from rec_weight w
left join (
  select date, sum(amount) total from rec_expenses group by 1
) e on e.date = w.date
order by w.date"
```

---

## 索引本身

索引也是普通表，可以直接查。排障时特别有用。

```bash
# 哪些文档最大（可能该拆）
zkb sql "select rel_path, chunk_count from docs order by chunk_count desc limit 10"

# 索引失败的文档和原因
zkb sql "select rel_path, index_error from docs where index_error is not null"

# 分块大小分布
zkb sql "
select count(*) n, round(avg(n_tokens)) avg, min(n_tokens) mn, max(n_tokens) mx
from chunks"

# 三张表是否一致（不一致意味着搜到的内容不存在）
zkb sql "
select (select count(*) from chunks)     as chunks,
       (select count(*) from fts_chunks) as fts,
       (select count(*) from vec_chunks) as vec"

# 某个命名空间的入链情况
zkb sql "
select d.rel_path, count(l.id) as inbound
from docs d left join links l on l.target_doc_id = d.id
where d.rel_path like 'projects/agent-rt/%'
group by 1 order by 2"
```

---

## 检索 trace

`ZKB_TRACE=1` 打开后，每次查询往 `~/.zkb/run/trace.jsonl` 追一行，
记下两路各自的排名。**调参之前先看这个**——融合后的分数说不出「这条是哪一路找到的」，
而每个调参决定都拴在这个问题上。

```bash
ZKB_TRACE=1 zkb search "混合检索怎么设计"

# 哪些查询是关键词路径救回来的（向量没找到但 FTS 找到了）
jq -r 'select(.hits[0].vec == null) | .q' ~/.zkb/run/trace.jsonl

# 被 tokenizer 判为不可匹配、因而没搜的词
jq -r 'select(.dropped | length > 0) | "\(.q)\t\(.dropped | join(","))"' ~/.zkb/run/trace.jsonl

# 慢查询
jq -r 'select(.ms > 200) | "\(.ms)ms\t\(.q)"' ~/.zkb/run/trace.jsonl
```

---

## 为什么不把这些做成命令

每一条都可以变成一个 flag，但那条路的终点是第二套 SQL 方言。判据在 REQ §8.1：
`records` 的边界是「存 + 过滤 + 聚合」，复式记账、对账、多币种换算超出去了，
该用专用工具。同理，能用一行 SQL 表达清楚的东西，不值得为它长出一个语法。

真到了回归分析、时间序列预测那一层，答案是 `zkb sql --json` 导出给 Python，
而不是把 zkb 撑大。
