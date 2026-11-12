"""Fresh A5 confirmation testbed: activity lifecycles + commentary joins.

These databases have a DIFFERENT relational topology from the initial A5
counterfactual study: one entity table, one lifecycle table, and two independent
one-to-many annotation tables. No previous A5 SQLs or reference labels used.
"""
from __future__ import annotations

from pathlib import Path
import sqlite3
from experiments.audit_spider_exec_strict import score_readonly_pair
from experiments.xiyan_mschema_frozen100 import build_mschema
from semantic_sql.catalog import DatabaseCatalog

DOMAINS=(
 {"db":"caseflow_holdout","person":"agents","item":"cases","tags":"case_flags",
  "notes":"case_notes","src":"opener_id","dst":"closer_id",
  "from_verb":"opened","to_verb":"closed"},
 {"db":"cargo_holdout","person":"dispatchers","item":"shipments","tags":"shipment_labels",
  "notes":"shipment_comments","src":"dispatcher_id","dst":"receiver_id",
  "from_verb":"dispatched","to_verb":"received"},
 {"db":"scholar_holdout","person":"scholars","item":"applications","tags":"application_tags",
  "notes":"application_notes","src":"submitter_id","dst":"evaluator_id",
  "from_verb":"submitted","to_verb":"evaluated"},
 {"db":"craft_holdout","person":"craftspeople","item":"commissions","tags":"commission_marks",
  "notes":"commission_memos","src":"producer_id","dst":"inspector_id",
  "from_verb":"produced","to_verb":"inspected"},
)


def make_cases(db:Path,domain:int)->list[dict]:
    if domain not in range(4):
        raise ValueError("exactly four separate confirmatory schemas")
    if db.exists():
        raise FileExistsError("immutable holdout source already exists")
    db.parent.mkdir(parents=True,exist_ok=True)
    a=DOMAINS[domain]
    p,i,t,n,src,dst=(a[k] for k in ("person","item","tags","notes","src","dst"))
    f,to=a["from_verb"],a["to_verb"]
    with sqlite3.connect(db) as con:
        con.executescript(f"""
            CREATE TABLE {p}(id INTEGER PRIMARY KEY, name TEXT UNIQUE NOT NULL);
            CREATE TABLE {i}(
              id INTEGER PRIMARY KEY,
              {src} INTEGER NOT NULL REFERENCES {p}(id),
              {dst} INTEGER NOT NULL REFERENCES {p}(id),
              state TEXT NOT NULL,
              points INTEGER NOT NULL);
            CREATE TABLE {t}(id INTEGER PRIMARY KEY,
              activity_id INTEGER NOT NULL REFERENCES {i}(id), tag TEXT NOT NULL);
            CREATE TABLE {n}(id INTEGER PRIMARY KEY,
              activity_id INTEGER NOT NULL REFERENCES {i}(id),
              author_id INTEGER NOT NULL REFERENCES {p}(id));
        """)
        con.executemany(f"INSERT INTO {p} VALUES (?,?)",
            [(1,"Ada"),(2,"Bea"),(3,"Cal"),(4,"Dee"),(5,"Eli"),(6,"Fay")])
        events=[
            (1,1,2,"complete",120),(2,1,3,"complete",80),
            (3,2,3,"pending",300),(4,4,2,"complete",150),
            (5,5,2,"complete",180),(6,1,2,"pending",120),
            (7,5,2,"pending",90),(8,5,4,"complete",140),
            (9,3,4,"complete",170),(10,5,2,"complete",75),
        ]
        con.executemany(f"INSERT INTO {i} VALUES (?,?,?,?,?)",events)
        tag_rows=[(1,"urgent"),(1,"urgent"),(2,"urgent"),(3,"urgent"),
            (4,"routine"),(5,"urgent"),(6,"routine"),(7,"urgent"),
            (8,"routine"),(9,"urgent"),(10,"routine")]
        con.executemany(f"INSERT INTO {t}(activity_id,tag) VALUES (?,?)",tag_rows)
        notes=[(1,2),(1,2),(1,3),(2,2),(3,3),(4,2),
               (5,1),(6,1),(7,1),(8,5),(9,5),(10,5),(5,2)]
        con.executemany(f"INSERT INTO {n}(activity_id,author_id) VALUES (?,?)",notes)
    cases=[
        (f"Who {f} the most {i}?",
         f"SELECT a.name FROM {p} a JOIN {i} b ON a.id=b.{src} "
         "GROUP BY a.id ORDER BY COUNT(*) DESC,a.name LIMIT 1"),
        (f"Who {to} the most {i}?",
         f"SELECT a.name FROM {p} a JOIN {i} b ON a.id=b.{dst} "
         "GROUP BY a.id ORDER BY COUNT(*) DESC,a.name LIMIT 1"),
        (f"How many {i} were {f} by Ada?",
         f"SELECT COUNT(*) FROM {i} b JOIN {p} a ON a.id=b.{src} WHERE a.name='Ada'"),
        (f"How many {i} were {to} by Bea?",
         f"SELECT COUNT(*) FROM {i} b JOIN {p} a ON a.id=b.{dst} WHERE a.name='Bea'"),
        (f"Which {p} have never {f} any {i}?",
         f"SELECT a.name FROM {p} a WHERE NOT EXISTS "
         f"(SELECT 1 FROM {i} b WHERE b.{src}=a.id) ORDER BY a.name"),
        (f"Which {p} have never {to} any {i}?",
         f"SELECT a.name FROM {p} a WHERE NOT EXISTS "
         f"(SELECT 1 FROM {i} b WHERE b.{dst}=a.id) ORDER BY a.name"),
        (f"How many distinct {i} carry an 'urgent' tag?",
         f"SELECT COUNT(DISTINCT b.id) FROM {i} b JOIN {t} g "
         "ON b.id=g.activity_id WHERE g.tag='urgent'"),
        (f"Which {p} {to} {i} that Ada and Eli both {f}?",
         f"SELECT a.name FROM {p} a JOIN {i} b ON a.id=b.{dst} "
         f"JOIN {p} c ON c.id=b.{src} WHERE c.name IN ('Ada','Eli') "
         "GROUP BY a.id HAVING COUNT(DISTINCT c.name)=2 ORDER BY a.name"),
        (f"Which {p} wrote notes on the greatest number of DISTINCT {i}?",
         f"SELECT a.name FROM {p} a JOIN {n} b ON a.id=b.author_id "
         "GROUP BY a.id ORDER BY COUNT(DISTINCT b.activity_id) DESC,a.name LIMIT 1"),
        (f"Which {p} {f} {i} with the highest average points?",
         f"SELECT a.name FROM {p} a JOIN {i} b ON a.id=b.{src} "
         "GROUP BY a.id ORDER BY AVG(b.points) DESC,a.name LIMIT 1"),
        (f"Which {p} {f} completed {i} with points strictly greater than 100?",
         f"SELECT DISTINCT a.name FROM {p} a JOIN {i} b ON a.id=b.{src} "
         "WHERE b.state='complete' AND b.points>100 ORDER BY a.name"),
        (f"Which {p} {to} at least two distinct urgent-tagged {i}?",
         f"SELECT a.name FROM {p} a JOIN {i} b ON a.id=b.{dst} "
         f"JOIN {t} g ON g.activity_id=b.id WHERE g.tag='urgent' "
         "GROUP BY a.id HAVING COUNT(DISTINCT b.id)>=2 ORDER BY a.name"),
    ]
    if len(cases)!=12:
        raise AssertionError("frozen holdout must have 12 cases")
    for j,(q,sql) in enumerate(cases):
        s=score_readonly_pair(db,sql,sql)
        if not s["bag_equal"] or s.get("truncated"):
            raise ValueError(f"broken holdout fixture {domain}:{j}")
    return [{"case_id":f"holdout-{domain}:{j}","question":q,"reference_sql":sql}
            for j,(q,sql) in enumerate(cases)]


def build_holdout_schema(db:Path,domain:int)->str:
    catalog=DatabaseCatalog.from_sqlite(db)
    return build_mschema(
        catalog=catalog,database_id=DOMAINS[domain]["db"],
        selected_tables=sorted(catalog.tables),descriptions={},
        max_sample_columns=0,sample_values=0)
