# inspect_chroma_v2.py
import chromadb
import json

CHROMA_PATH = "chroma_db_v2"
COLLECTION_NAME = "tmdb_movie_reviews"

client = chromadb.PersistentClient(path=CHROMA_PATH)
col = client.get_collection(COLLECTION_NAME)

print("✅ Collection:", COLLECTION_NAME)
print("✅ Count:", col.count())

# Pull a few items (ids + metadatas + docs)
res = col.get(limit=5, include=["metadatas", "documents"])

print("\n=== SAMPLE IDS ===")
print(res["ids"])

print("\n=== SAMPLE METADATA (first item) ===")
if res["metadatas"]:
    print(json.dumps(res["metadatas"][0], indent=2, ensure_ascii=False))
else:
    print("No metadatas found.")

print("\n=== SAMPLE DOCUMENT (first item snippet) ===")
if res["documents"]:
    doc = res["documents"][0] or ""
    print(doc[:300])
else:
    print("No documents found.")
