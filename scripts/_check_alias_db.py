import sqlite3
conn = sqlite3.connect("experiments/mlflow/mlflow.db")
cur = conn.cursor()
cur.execute("SELECT * FROM registered_model_aliases")
rows = cur.fetchall()
cols = [d[0] for d in cur.description]
print("registered_model_aliases columns:", cols)
for r in rows:
    print(" ", dict(zip(cols, r)))
conn.close()
