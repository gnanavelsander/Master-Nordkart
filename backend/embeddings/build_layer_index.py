import json
import os
import sys
from pathlib import Path




parent_dir = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(parent_dir))
from database import Database
from embeddings import embed_text



def main():
    db = Database(os.environ["PGCONN_STRING"])
    db.connect()
    
    rows = db.query("""
        SELECT table_schema, table_name
        FROM information_schema.tables
        WHERE table_schema NOT IN ('pg_catalog', 'information_schema')
          AND table_type = 'BASE TABLE';
    """)
    
    index = []
    for _, r in rows.iterrows():
        layer = f"{r['table_schema']}.{r['table_name']}"
        desc = f"GIS layer named {r['table_name'].replace('_', ' ')}"
        vec = embed_text(f"{layer}. {desc}")
        index.append({
            "layer": layer,
            "description": desc,
            "embedding": vec
        })
    
    db.disconnect()
    
    # Save to embeddings directory
    output_path = Path(__file__).parent / "layer_index.json"
    with open(output_path, "w") as f:
        json.dump(index, f)
    
    print(f"Index size: {len(index)}")
    print(f"Saved to: {output_path}")

if __name__ == "__main__":
    main()
