"""Four-call original Qwen9B cross-database SQL *structure binding* pilot.

Source: previously exposed BIRD TRAIN 242 labeled SQL examples; target:
historical BIRD TRAIN validation64 on different physical DBs. This is
a physical 4-case plumbing pilot, not new generalization.
Freeze masks source-only SQL identifiers/literals, and retrieves by question
and coarse operation intent. Reference SQL is never loaded by the infer mode.
A=4 Qwen independent SQL; B=same first3 + Qwen JSON mapping of one
source structural skeleton; C=original4 moved to slot1 (position control).
Scored separately, only AFTER selection artifacts have been uploaded.
"""
from __future__ import annotations
import argparse
from collections import Counter
from hashlib import sha256
import json
from pathlib import Path
import random
import re
import sqlite3
import time

from evaluation.template_slot_probe import PROTOCOL as SELECT_PROTOCOL, seal_choices, score_sealed
from scripts.prepare_birdtrain_qwen9b_sft_source import fetch_manifest, fetch_database
from semantic_sql.direct import _SYSTEM, _strip_fence
from semantic_sql.execution import _open_readonly_connection, validate_readonly_sql, SQLExecutionError
from semantic_sql.providers import OllamaProvider

MODEL = "qwen3.5:9b-q4_K_M"
DIGEST = "56671c2ab9385f9cfcb404638e32cd62d88e3501d44822208363c010179a3c90"
TRAIN_SHA = "b6851fb0f2b5027833aff45a0b5e01326359b245be1c812edc66986d0df1d402"
VALID_SHA = "cc112305bc9de33a03cea704f2edecfdfb45e62cc953ae498cea27fbed93b506"
PILOT = "BIRD-TRAIN-EXPOSED-4DB-FOUR-CALL-BINDING-PILOT-V1"
N = 4
OPS = ("count", "sum", "avg", "max", "min", "distinct", "join", "left join",
       "group by", "having", "order by", "limit", "union", "except",
       "intersect", "exists", "not exists", "case when", "like",
       "between", "in (", "/")
INTENTS = (
    ("count", r"how many|number of|count of"),
    ("avg", r"average|mean"), ("sum", r"\btotal\b|\bsum\b"),
    ("order by", r"highest|lowest|most|least|top|largest|smallest|earliest|latest|first|last|rank"),
    ("limit", r"top|highest|lowest|largest|smallest|most|least|first|last"),
    ("/", r"percentage|percent|ratio|proportion|fraction"),
    ("group by", r"for each|per (?:year|month|category|state|customer|department|employee|student)|grouped by|by each"),
    ("distinct", r"distinct|unique|different"),
    ("between", r"between|from \d{4} to \d{4}"),
    ("like", r"contains|beginning with|starts with|ending with"),
)
KEYWORDS = set("select from as where on and or not in is null with union all case when then else end by group order limit offset join inner left outer right cross having distinct count sum min max avg desc asc like between intersect except exists cast strftime round coalesce ifnull true false".split())
TOK = re.compile(r"'(?:''|[^'])*'|\b[A-Za-z_][A-Za-z_0-9]*\b|\b\d+(?:\.\d+)?\b")


def checked(path: Path, expected: str) -> list[dict]:
    data = path.read_bytes()
    if sha256(data).hexdigest() != expected:
        raise ValueError("original BIRD source SHA changed")
    return [json.loads(s) for s in data.splitlines() if s.strip()]


def question(r: dict) -> str:
    return r["messages"][1]["content"].split("\nSchema context:", 1)[0].removeprefix("Question: ").strip()


def operations(sql: str) -> set[str]:
    s = sql.lower()
    return {o for o in OPS if re.search(r"(?<!\w)" + re.escape(o) + r"(?!\w)", s)}


def intent(q: str) -> set[str]:
    return {o for o, pattern in INTENTS if re.search(pattern, q.casefold())}


def requested_answer_kind(question_text: str) -> str:
    """Conservative answer-shape gate: abstain on unrecognized questions."""
    s = question_text.casefold()
    if re.search(r"percentage|percent|proportion|fraction|ratio", s):
        return "ratio"
    if re.search(r"how many years|years have passed|difference in years|time elapsed", s):
        return "elapsed"
    if re.search(r"how many|number of|count of", s):
        return "count"
    if re.search(r"life expectancy|average|mean|median|total (?:amount|revenue|cost)", s):
        return "metric"
    if re.search(r"what are|list |give (?:the )?names|which (?:customers|cities|employees|students|people|stores)|addresses", s):
        return "records"
    return "unknown"


def template_projection_kind(sql: str) -> str:
    """Classify SELECT projection only; ORDER BY arithmetic cannot satisfy ratio."""
    m = re.match(r"(?is)^\s*select\s+(.*?)\bfrom\b", sql)
    if not m:
        return "unknown"
    expression = m.group(1).casefold()
    if re.search(r"\bcount\s*\(", expression):
        return "count"
    if "/" in expression or "percent" in expression:
        return "ratio"
    if "julianday" in expression or "strftime" in expression or re.search(r"\s-\s", expression):
        return "elapsed"
    if re.search(r"\b(?:avg|sum|median|min|max)\s*\(", expression):
        return "metric"
    return "records"


def template_shape_compatible(question_text: str, sql: str) -> bool:
    requested = requested_answer_kind(question_text)
    projected = template_projection_kind(sql)
    if requested in {"unknown", "elapsed"}:
        return requested == "elapsed" and projected == "elapsed"
    if requested == "metric":
        return projected in {"metric", "records"}
    return requested == projected


def _schema_roles(context: str) -> dict[str, str]:
    lines = context.split("\n")
    found = {}
    for line in lines:
        matched = re.fullmatch(r"([A-Za-z_][\w]*)\(([^()]*)\)", line.strip())
        if matched:
            found[matched.group(1).casefold()] = "table"
            for col in matched.group(2).split(","):
                c = col.strip()
                if re.fullmatch(r"[A-Za-z_][\w]*", c) and c.casefold() not in found:
                    found[c.casefold()] = "column"
    return found


def mask_template(original: str, source_context: str) -> tuple[str, dict]:
    if not re.match(r"^\s*(select|with)\b", original, re.I) or original.count(";") > 1:
        raise ValueError("unsupported source query")
    roles = _schema_roles(source_context)
    if not roles:
        raise ValueError("source schema missing")
    names: dict[tuple[str,str], str] = {}
    meta = {}
    def substitute(m):
        raw = m.group()
        if raw.startswith("'"):
            v = raw[1:-1].replace("''", "'")
            if len(v) > 100:
                raise ValueError("large literal")
            key = ("value_string", v)
            if key not in names:
                names[key] = f"__V{len(meta)+1}__"
                meta[names[key]] = {"kind": "value_string", "source": v}
            return "'" + names[key] + "'"
        if re.fullmatch(r"\d+(?:\.\d+)?",raw):
            if raw in {"0", "1", "2", "5", "10", "100"}:
                return raw
            key = ("value_number", raw)
            if key not in names:
                names[key] = f"__V{len(meta)+1}__"
                meta[names[key]] = {"kind": "value_number", "source": raw}
            return names[key]
        low = raw.casefold()
        kind = roles.get(low)
        # A source identifier can name BOTH a table and a column. After a dot
        # it must be a column, never the table placeholder.
        if kind == "table" and original[:m.start()].rstrip().endswith("."):
            kind = "column"
        if kind not in {"table", "column"} or low in KEYWORDS:
            return raw
        key = (kind, low)
        if key not in names:
            names[key] = f"__{ 'T' if kind=='table' else 'C'}{len(meta)+1}__"
            meta[names[key]] = {"kind": kind, "source": raw}
        return names[key]
    masked = TOK.sub(substitute, original)
    if not any(p["kind"] == "table" for p in meta.values()) or \
       not any(p["kind"] == "column" for p in meta.values()) or len(meta) > 30:
        raise ValueError("template has no bindable identifiers or too many slots")
    return masked, meta


def retrieve(train: list[dict], targets: list[dict]) -> list[dict]:
    from sklearn.feature_extraction.text import TfidfVectorizer
    vocabulary = [question(x) for x in train]
    vectorizer = TfidfVectorizer(ngram_range=(1,2), stop_words="english",
                                 sublinear_tf=True).fit(vocabulary)
    sims = (vectorizer.transform([question(x) for x in targets]) @
            vectorizer.transform(vocabulary).T).toarray()
    out = []
    for j, target in enumerate(targets):
        wanted = intent(question(target))
        available = []
        for i, source in enumerate(train):
            if source["source_db_id"] == target["source_db_id"]:
                continue
            if not template_shape_compatible(question(target), source["messages"][2]["content"]):
                continue
            try:
                template, placeholders = mask_template(
                    source["messages"][2]["content"], source["messages"][1]["content"])
            except ValueError:
                continue
            actual = operations(source["messages"][2]["content"])
            penalty = sum(x in actual and x not in wanted for x in
                          ("count","avg","sum","/","group by","limit","order by"))
            score = float(sims[j,i]) + .44*len(wanted&actual) - .30*len(wanted-actual) - .10*penalty
            available.append((score, i, template, placeholders))
        if not available:
            out.append({
                "source_index": target["source_index"],
                "target_db": target["source_db_id"],
                "target_context": target["messages"][1]["content"],
                "question_hash": sha256(question(target).encode()).hexdigest(),
                "source_template_index": None, "source_db": None,
                "source_question": None, "skeleton": None,
                "placeholders": {}, "abstained": True,
            })
            continue
        _, idx, skeleton, slots = max(available, key=lambda x:(x[0],-x[1]))
        source = train[idx]
        out.append({
            "source_index": target["source_index"],
            "target_db": target["source_db_id"],
            "target_context": target["messages"][1]["content"],
            "question_hash": sha256(question(target).encode()).hexdigest(),
            "source_template_index": source["source_index"],
            "source_db": source["source_db_id"],
            "source_question": question(source),
            "skeleton": skeleton,
            "placeholders": slots,
            "abstained": False,
        })
    return out


def freeze(trainfile:Path,validfile:Path,out:Path) -> dict:
    train=checked(trainfile,TRAIN_SHA)
    val=checked(validfile,VALID_SHA)
    if len(train)!=242 or len(val)!=64:
        raise ValueError("original source cohort size drift")
    train_dbs={r["source_db_id"] for r in train}
    val_dbs={r["source_db_id"] for r in val}
    if train_dbs & val_dbs or len(val_dbs)!=4:
        raise ValueError("source and target physical DBs no longer isolated")
    picked=[]
    for db in sorted(val_dbs):
        options=[v for v in val if v["source_db_id"]==db]
        options.sort(key=lambda x:sha256(f"pilot20261119:{x['source_index']}".encode()).hexdigest())
        picked.append(options[0])
    retrieved=retrieve(train,picked)
    payload={"pilot":PILOT,"model":MODEL,"digest":DIGEST,
             "n":N,"source_train_sha256":TRAIN_SHA,"source_valid_sha256":VALID_SHA,
             "source_disjoint_in_train":True,"validation_original_labels_discarded":True,
             "cases":retrieved}
    payload["manifest_sha256"]=sha256(json.dumps(payload,sort_keys=True).encode()).hexdigest()
    if out.exists():raise FileExistsError("immutable freeze would be overwritten")
    out.write_text(json.dumps(payload,indent=2,sort_keys=True)+"\n")
    return payload


def _source_schema_names(target_context:str):
    roles=_schema_roles(target_context)
    return {n for n,k in roles.items() if k=="table"}, {n for n,k in roles.items() if k=="column"}


def bind(skeleton:str,slots:dict,raw_response:str,target_context:str) -> str:
    match=re.search(r"\{[\s\S]*\}",raw_response)
    if not match: return ""
    try: mapping=json.loads(match.group())
    except json.JSONDecodeError:return ""
    if not isinstance(mapping,dict) or set(mapping)!=set(slots):
        return ""
    tables, columns=_source_schema_names(target_context)
    sql=skeleton
    for placeholder,info in sorted(slots.items(),key=lambda x:-len(x[0])):
        value=mapping.get(placeholder)
        if not isinstance(value,str) or not value or len(value)>100:return ""
        kind=info["kind"]
        if kind in ("table","column"):
            if not re.fullmatch(r"[A-Za-z_]\w*",value):return ""
            if value.casefold() not in (tables if kind=="table" else columns):return ""
            replacement='"'+value.replace('"','""')+'"'
        elif kind=="value_string":
            if any(ord(c)<32 for c in value):return ""
            replacement=value.replace("'","''")
        else:
            if not re.fullmatch(r"-?\d+(?:\.\d+)?",value):return ""
            replacement=value
        sql=sql.replace(placeholder,replacement)
    if re.search(r"__[TCV]\d+__",sql): return ""
    return sql


def physical(db:Path,sql:str) -> dict:
    start=time.monotonic()
    conn=None
    try:
        query=validate_readonly_sql(sql)
        conn,_=_open_readonly_connection(db)
        conn.set_progress_handler(lambda: int(time.monotonic()-start>8),4000)
        cursor=conn.execute(query)
        rows=cursor.fetchmany(50001)
        if len(rows)>50000:raise ValueError("row cap")
        encoded=sorted(set(json.dumps(list(r),default=str,ensure_ascii=False,separators=(",",":")) for r in rows))
        set_hash=sha256(json.dumps(encoded).encode()).hexdigest()
        signature=sha256(json.dumps([len(cursor.description or []),set_hash]).encode()).hexdigest()
        return {"valid":True,"signature":signature,"set_hash":set_hash}
    except (SQLExecutionError,sqlite3.Error,ValueError,OSError):
        return {"valid":False,"signature":None,"set_hash":None}
    finally:
        if conn:
            conn.set_progress_handler(None,0)
            conn.close()


def infer(frozen_path:Path,dbroot:Path,index:int,out:Path) -> dict:
    from evaluation.run_bird import _ollama_model_digest
    data=json.loads(frozen_path.read_text())
    if data["pilot"]!=PILOT or data["digest"]!=DIGEST or data["n"]!=N:
        raise ValueError("source/model or freeze mismatch")
    if _ollama_model_digest("http://127.0.0.1:11434",MODEL)!=DIGEST:
        raise ValueError("original fixed Qwen3.5-9B digest changed")
    if index not in range(N):raise ValueError("four source cases only")
    row=data["cases"][index]
    db=dbroot/row["target_db"]/(row["target_db"]+".sqlite")
    if not db.is_file():raise FileNotFoundError("pinned BIRD physical DB not available")
    first=[]
    for seed in (42,43,44,45):
        p=OllamaProvider(model=MODEL,temperature=.7,seed=seed,num_predict=384,timeout_s=900)
        response=_strip_fence(p.complete_text(system=_SYSTEM,user=row["target_context"]))
        first.append(response)
    using_template = row.get("abstained") is False
    p=OllamaProvider(model=MODEL,temperature=.7,seed=46,num_predict=384,timeout_s=900)
    instructions=(
        "Bind the following SQL calculation skeleton to the CURRENT physical SQLite schema. "
        "Keep the SQL computation and arrangement; DO NOT write new SQL. "
        "The placeholders are source table, source column and source literal roles. "
        "Return ONLY one JSON object with exactly those placeholder keys; each value is a "
        "current table, current column, or target-question literal as appropriate. "
        "If impossible, return empty JSON. "
        "Do not copy source foreign identifiers into the target unless they really exist."
    )
    if using_template:
        user=(row["target_context"]+"\n\nSource question: "+row["source_question"]+
              "\nCalculation skeleton:\n"+row["skeleton"]+
              "\nSource placeholder roles:\n"+json.dumps(row["placeholders"],ensure_ascii=False))
        response=p.complete_text(system=instructions,user=user)
        bound=bind(row["skeleton"],row["placeholders"],response,row["target_context"])
    else:
        bound=""
    all_sql=first+[bound if using_template else first[3]]
    outcomes=[physical(db,sql) for sql in all_sql]
    candidates=[{"id":f"c{i+1}",**outcomes[i]} for i in range(4)]
    templ={"id":"template",**outcomes[4]} if using_template else None
    select_source={
        "protocol":SELECT_PROTOCOL,"manifest_sha256":data["manifest_sha256"],
        "model_digest":DIGEST,
        "cases":[{"case_hash":sha256(str(row["source_index"]).encode()).hexdigest(),
                  "db_id":row["target_db"],"baseline":candidates,"template":templ}],
    }
    sealed=seal_choices(select_source)
    report={"pilot":PILOT,"index":index,"case_id":row["source_index"],
            "db_id":row["target_db"],"model_digest":DIGEST,
            "n_model_calls_unique":5 if using_template else 4,"n_model_calls_per_arm":4,
            "original_direct_sql":first,
            "template_sql":bound,"template_mapping_complete":bool(bound),
            "template_abstained":not using_template,
            "n_direct_executable":sum(x["valid"] for x in outcomes[:4]),
            "template_executable":bool(using_template and outcomes[4]["valid"]),
            "sealed":sealed,"no_target_gold_read":True,"holdout300_unused":True}
    if out.exists(): raise FileExistsError("refuse to overwrite original model predictions")
    out.write_text(json.dumps(report,indent=2,ensure_ascii=False)+"\n")
    return report


def score(frozen:Path,generated:Path,validation:Path,dbroot:Path,out:Path) -> dict:
    data=json.loads(frozen.read_text());gen=json.loads(generated.read_text())
    rows=checked(validation,VALID_SHA)
    if gen["pilot"]!=PILOT or gen["model_digest"]!=DIGEST or not gen["no_target_gold_read"]:
        raise ValueError("candidate seal missing or model/source mismatch")
    row=data["cases"][gen["index"]]
    if gen["case_id"]!=row["source_index"] or gen["db_id"]!=row["target_db"]:
        raise ValueError("generated source identity mismatch")
    target=next(x for x in rows if x["source_index"]==row["source_index"])
    db=dbroot/row["target_db"]/(row["target_db"]+".sqlite")
    gold=physical(db,target["messages"][2]["content"])
    references={"manifest_sha256":data["manifest_sha256"],
                "cases":[{"case_hash":sha256(str(row["source_index"]).encode()).hexdigest(),
                          "scorable":gold["valid"],"set_hash":gold["set_hash"]}]}
    summary=score_sealed(gen["sealed"],references)
    result={"pilot":PILOT,"index":gen["index"],"case_id":row["source_index"],
            "target_db":row["target_db"],
            "n_original_calls_unique":gen["n_model_calls_unique"],
            "template_executable":gen["template_executable"],
            "template_abstained":gen["template_abstained"],
            "template_mapping_complete":gen["template_mapping_complete"],
            "score":summary,"no_dev500":True,"no_holdout300":True}
    if out.exists():raise FileExistsError("refuse score overwrite")
    out.write_text(json.dumps(result,indent=2,sort_keys=True)+"\n")
    return result


def aggregate(parts:list[Path],out:Path) -> dict:
    if len(parts)!=N:raise ValueError("need all four independent physical case outcomes")
    records=[json.loads(p.read_text()) for p in parts]
    if sorted(x["index"] for x in records)!=list(range(N)) or \
       len({x["target_db"] for x in records})!=N:
        raise ValueError("partial 4-database paired results")
    totals=Counter()
    for r in records:
        s=r["score"]
        totals["n_scorable"]+=s["n_scorable"]
        totals["n_censored"]+=s["n_unscorable"]
        totals["model_calls_unique"]+=r["n_original_calls_unique"]
        totals["template_abstained"]+=int(r["template_abstained"])
        totals["template_bound"]+=int(r["template_mapping_complete"])
        totals["template_executable"]+=int(r["template_executable"])
        totals["A_correct"]+=s["final_correct"]["A"]
        totals["B_correct"]+=s["final_correct"]["B"]
        totals["C_correct"]+=s["final_correct"]["C"]
        totals["A_oracle"]+=s["oracle_at_least_one_correct"]["A"]
        totals["B_oracle"]+=s["oracle_at_least_one_correct"]["B"]
        totals["B_vs_A_rescues"]+=s["B_vs_A_rescues"]
        totals["B_vs_A_regressions"]+=s["B_vs_A_regressions"]
        totals["B_vs_C_rescues"]+=s["B_vs_C_rescues"]
        totals["B_vs_C_regressions"]+=s["B_vs_C_regressions"]
    result={"status":"COMPLETE_PHYSICAL_QWEN9B_TEMPLATE_BINDING_4CASE_PILOT",
            "protocol":PILOT,"n":N,"n_db":N,
            "original_Qwen_model_calls_unique":totals["model_calls_unique"],
            "effective_four_calls_per_arm":True,
            **dict(totals),
            "no_dev500":True,"no_holdout300":True,
            "not_independent_generalization":True,
            "limitation":"Only 4 previously used BIRD TRAIN validation task identities. Feasibility only; no model promotion or claim of real new unseen-source improvement."}
    if out.exists():raise FileExistsError("refuse overwrite")
    out.write_text(json.dumps(result,indent=2,sort_keys=True)+"\n")
    return result


def main():
    p=argparse.ArgumentParser()
    p.add_argument("mode",choices=("freeze","infer","score","aggregate"))
    for key in ("train","validation","frozen","dbroot","generated","out"):
        p.add_argument("--"+key,type=Path)
    p.add_argument("--index",type=int)
    p.add_argument("--parts",type=Path,nargs="+")
    args=p.parse_args()
    if args.mode=="freeze":res=freeze(args.train,args.validation,args.out)
    elif args.mode=="infer":res=infer(args.frozen,args.dbroot,args.index,args.out)
    elif args.mode=="score":res=score(args.frozen,args.generated,args.validation,args.dbroot,args.out)
    else:res=aggregate(args.parts,args.out)
    print("TEMPLATE_BINDING_PILOT",json.dumps({
        k:v for k,v in res.items() if k not in ("cases","original_direct_sql","sealed","score")
    },sort_keys=True),flush=True)


if __name__=="__main__":
    main()
