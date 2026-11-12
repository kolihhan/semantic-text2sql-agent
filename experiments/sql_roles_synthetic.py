"""Unseen synthetic role/grain SQL tasks; no labels ever reach SQL generation."""
from __future__ import annotations
import sqlite3
from pathlib import Path
from semantic_sql.catalog import DatabaseCatalog
from experiments.xiyan_mschema_frozen100 import build_mschema
from experiments.audit_spider_exec_strict import score_readonly_pair

DOMAINS=(
 ("members","endorsements","endorser_id","endorsed_id","applications","applicant_id","approver_id"),
 ("dispatchers","handoffs","sender_id","recipient_id","shipments","shipper_id","inspector_id"),
 ("artists","mentions","mentioner_id","mentioned_id","commissions","creator_id","reviewer_id"),
 ("researchers","citations","citing_id","cited_id","proposals","submitter_id","evaluator_id"),
)


def create_domain(db:Path,domain:int)->list[dict]:
    if domain not in range(4):
        raise ValueError("exactly four independent domains")
    if db.exists():
        raise FileExistsError("freshly minted synthetic DB must not exist")
    db.parent.mkdir(parents=True,exist_ok=True)
    p,r,s,d,t,u,v=DOMAINS[domain]
    labels=t+"_labels"
    with sqlite3.connect(db) as con:
        con.executescript(f"""
          CREATE TABLE {p}(id INTEGER PRIMARY KEY,name TEXT UNIQUE NOT NULL);
          CREATE TABLE {r}(id INTEGER PRIMARY KEY,{s} INTEGER REFERENCES {p}(id),
                          {d} INTEGER REFERENCES {p}(id));
          CREATE TABLE {t}(id INTEGER PRIMARY KEY,{u} INTEGER REFERENCES {p}(id),
                          {v} INTEGER REFERENCES {p}(id),amount INTEGER NOT NULL);
          CREATE TABLE {labels}(item_id INTEGER REFERENCES {t}(id),tag TEXT NOT NULL);
        """)
        con.executemany(f"INSERT INTO {p} VALUES (?,?)",
            [(1,"Ada"),(2,"Bea"),(3,"Cal"),(4,"Dee"),(5,"Eli"),(6,"Fay")])
        edge=[(1,2),(1,3),(1,4),(2,3),(3,2),(4,2),
              (5,2),(5,3),(5,4),(5,2)]
        con.executemany(f"INSERT INTO {r} VALUES (?,?,?)",
                        [(idx,*pair) for idx,pair in enumerate(edge,1)])
        con.executemany(f"INSERT INTO {t} VALUES (?,?,?,?)",
            [(1,1,2,120),(2,1,3,80),(3,2,3,300),
             (4,4,2,150),(5,5,2,180),(6,1,2,120)])
        con.executemany(f"INSERT INTO {labels} VALUES (?,?)",
            [(1,"urgent"),(2,"urgent"),(3,"urgent"),(3,"urgent"),
             (4,"routine"),(5,"urgent"),(6,"routine")])
    cases=[
      (f"Which {p} initiated the most {r}?",
       f"SELECT a.name FROM {p} a JOIN {r} b ON a.id=b.{s} "
       "GROUP BY a.id ORDER BY COUNT(*) DESC,a.name LIMIT 1"),
      (f"Which {p} received the most {r}?",
       f"SELECT a.name FROM {p} a JOIN {r} b ON a.id=b.{d} "
       "GROUP BY a.id ORDER BY COUNT(*) DESC,a.name LIMIT 1"),
      (f"How many {r} did Ada initiate?",
       f"SELECT COUNT(*) FROM {r} b JOIN {p} a ON a.id=b.{s} WHERE a.name='Ada'"),
      (f"How many {r} did Bea receive?",
       f"SELECT COUNT(*) FROM {r} b JOIN {p} a ON a.id=b.{d} WHERE a.name='Bea'"),
      (f"Which {p} received {r} from the most distinct senders?",
       f"SELECT a.name FROM {p} a JOIN {r} b ON a.id=b.{d} "
       f"GROUP BY a.id ORDER BY COUNT(DISTINCT b.{s}) DESC,a.name LIMIT 1"),
      (f"Which {p} never initiated any {r}?",
       f"SELECT a.name FROM {p} a WHERE NOT EXISTS "
       f"(SELECT 1 FROM {r} b WHERE b.{s}=a.id) ORDER BY a.name"),
      (f"Which {p} never received any {r}?",
       f"SELECT a.name FROM {p} a WHERE NOT EXISTS "
       f"(SELECT 1 FROM {r} b WHERE b.{d}=a.id) ORDER BY a.name"),
      (f"Which {p} received {r} from both Ada and Eli?",
       f"SELECT a.name FROM {p} a JOIN {r} b ON a.id=b.{d} "
       f"JOIN {p} c ON c.id=b.{s} WHERE c.name IN ('Ada','Eli') "
       "GROUP BY a.id HAVING COUNT(DISTINCT c.name)=2 ORDER BY a.name"),
      (f"Which {p} approved the most {t}?",
       f"SELECT a.name FROM {p} a JOIN {t} b ON a.id=b.{v} "
       "GROUP BY a.id ORDER BY COUNT(*) DESC,a.name LIMIT 1"),
      (f"Which {p} created {t} with the greatest combined amount?",
       f"SELECT a.name FROM {p} a JOIN {t} b ON a.id=b.{u} "
       "GROUP BY a.id ORDER BY SUM(b.amount) DESC,a.name LIMIT 1"),
      (f"How many distinct {t} have the 'urgent' tag?",
       f"SELECT COUNT(DISTINCT a.id) FROM {t} a JOIN {labels} b "
       "ON a.id=b.item_id WHERE b.tag='urgent'"),
      (f"Which {p} created {t} with 'urgent' tag and amount greater than 100?",
       f"SELECT DISTINCT a.name FROM {p} a JOIN {t} b ON a.id=b.{u} "
       f"JOIN {labels} c ON c.item_id=b.id "
       "WHERE c.tag='urgent' AND b.amount>100 ORDER BY a.name"),
    ]
    for i,(_,sql) in enumerate(cases):
        if not score_readonly_pair(db,sql,sql)["bag_equal"]:
            raise ValueError(f"invalid synthetic reference {domain}:{i}")
    return [{"case_id":f"{domain}:{idx}","question":q,"reference_sql":sql}
            for idx,(q,sql) in enumerate(cases)]


def mschema(db:Path,domain:int)->str:
    cat=DatabaseCatalog.from_sqlite(db)
    return build_mschema(catalog=cat,database_id=f"new_counterfactual_domain_{domain}",
                         selected_tables=sorted(cat.tables),descriptions={},
                         sample_values=0,max_sample_columns=0)
