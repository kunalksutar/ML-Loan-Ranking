import sqlite3
conn = sqlite3.connect("experiments/mlflow/mlflow.db")
cur = conn.cursor()
cur.execute("SELECT version, name, source, run_id, status FROM model_versions ORDER BY version")
rows = cur.fetchall()
cols = [d[0] for d in cur.description]
print("model_versions:")
for r in rows:
    print(" ", dict(zip(cols, r)))
print()
cur.execute("SELECT * FROM registered_model_aliases")
rows = cur.fetchall()
cols = [d[0] for d in cur.description]
print("aliases:", rows)
conn.close()
