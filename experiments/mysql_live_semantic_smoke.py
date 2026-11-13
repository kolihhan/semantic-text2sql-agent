"""Real MySQL 8.4 vs SQLite equivalence smoke for portable SQL patterns.

Ephemeral CI fixture only: never connect to user databases and never accept
untrusted generated SQL. BIRD TRAIN labels are NOT executed in this smoke.
"""
from __future__ import annotations

import os
import sqlite3

from experiments.semantic_sft_dialects import transfer_portable_sql

USERS = [(1,"Amy","MY"),(2,"Bob","SG"),(3,"Ann","MY")]
ORDERS = [(10,1,"paid",10),(11,1,"paid",20),(12,2,"paid",20),(13,3,"refunded",10)]

FIXTURES = [
    "CREATE TABLE users (id INTEGER PRIMARY KEY, name TEXT, country TEXT)",
    "CREATE TABLE orders (id INTEGER PRIMARY KEY, user_id INTEGER, status TEXT, amount INTEGER)",
]

PORTABLE_QUERIES = (
    "SELECT COUNT(DISTINCT u.id) FROM users AS u JOIN orders AS o ON u.id=o.user_id WHERE o.status='paid'",
    "SELECT u.country, SUM(o.amount) AS total FROM users AS u JOIN orders AS o ON u.id=o.user_id WHERE o.status='paid' GROUP BY u.country ORDER BY total DESC, u.country",
    "SELECT u.name FROM users AS u WHERE NOT EXISTS (SELECT 1 FROM orders AS o WHERE o.user_id=u.id AND o.status='paid') ORDER BY u.name",
    "SELECT COUNT(*) FROM orders WHERE amount >= 20",
    "SELECT u.country, MAX(o.amount) FROM users AS u JOIN orders AS o ON o.user_id=u.id GROUP BY u.country ORDER BY u.country",
)


def assert_mysql_matches_sqlite(mysql_connection):
    with sqlite3.connect(":memory:") as sq:
        for ddl in FIXTURES:
            sq.execute(ddl)
        sq.executemany("INSERT INTO users VALUES (?, ?, ?)",USERS)
        sq.executemany("INSERT INTO orders VALUES (?, ?, ?, ?)",ORDERS)
        cur=mysql_connection.cursor()
        for name in ("orders","users"):
            cur.execute("DROP TABLE IF EXISTS "+name)
        for ddl in FIXTURES:
            cur.execute(ddl.replace(" TEXT", " VARCHAR(80)"))
        cur.executemany("INSERT INTO users VALUES (%s, %s, %s)",USERS)
        cur.executemany("INSERT INTO orders VALUES (%s, %s, %s, %s)",ORDERS)
        mysql_connection.commit()
        scores=[]
        for original in PORTABLE_QUERIES:
            mysql_sql=transfer_portable_sql(original)
            if mysql_sql is None:
                raise ValueError(f"supported portable query rejected: {original}")
            sqlite_rows=sq.execute(original).fetchall()
            cur.execute(mysql_sql)
            mysql_rows=cur.fetchall()
            # Match both rows and ordering; collation differences avoided by
            # controlled, fixed-case fixture and explicit ORDER BY.
            if sqlite_rows!=mysql_rows:
                raise AssertionError(f"SQL meaning differs: {original}\nsqlite={sqlite_rows}\nmysql={mysql_rows}\ntranslated={mysql_sql}")
            scores.append({"query":original,"rows":len(sqlite_rows),"result_equal":True})
        cur.close()
        return scores


def main():
    import mysql.connector
    connection=mysql.connector.connect(
        host=os.environ.get("MYSQL_HOST","127.0.0.1"),
        port=int(os.environ.get("MYSQL_PORT","3306")),
        user=os.environ["MYSQL_USER"],
        password=os.environ["MYSQL_PASSWORD"],
        database=os.environ["MYSQL_DB"],
        connection_timeout=15,
    )
    try:
        scores=assert_mysql_matches_sqlite(connection)
        print("REAL_MYSQL_SQLITE_EQUIVALENT",{ "query_count":len(scores),"passed":len(scores) })
    finally:
        connection.close()


if __name__=="__main__":
    main()
