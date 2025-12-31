"""
chroma_inspect.py

Small utility to inspect a local Chroma DB used by the project.

Features:
- List collections
- Show collection count
- Print N sample documents with metadata
- Simple filter query by metadata (e.g. movie_title)
- Nearest-neighbor search by text (requires embeddings via GeminiHelper)

Usage examples:
    python chroma_inspect.py --path chroma_db_v2 --collection tmdb_movie_reviews --samples 5
    python chroma_inspect.py --path chroma_db_v2 --collection tmdb_movie_reviews --filter "movie_title=Inception"

Note: This script expects the same Chromadb/PersistentClient API used in the project.
"""
import json
from typing import Optional

import chromadb

from gemini_helper import GeminiHelper
from dotenv import load_dotenv
import os

load_dotenv()

# ---------------------------
# Hard-coded configuration
# ---------------------------
CHROMA_PATH = "chroma_db_v2"
COLLECTION_NAME = "tmdb_movie_reviews"
SAMPLES = 5
# Set FILTER to something like "movie_title=Inception" or None
FILTER = None
# Set NN_QUERY to a string to run nearest-neighbor search (requires GEMINI_API_KEYS)
NN_QUERY: Optional[str] = None

GEMINI_API_KEYS = os.getenv("GEMINI_API_KEYS", "").split("|||||") if os.getenv("GEMINI_API_KEYS") else []


def format_meta(m):
    try:
        return json.dumps(m, indent=2, ensure_ascii=False)
    except Exception:
        return str(m)


def main():
    client = chromadb.PersistentClient(path=CHROMA_PATH)

    # list collections
    try:
        collections = client.list_collections()
    except Exception:
        collections = None

    print("Chroma path:", CHROMA_PATH)
    if collections is not None:
        print("Collections:")
        for c in collections:
            print(" -", c)
    else:
        print("Could not list collections (older chromadb). Proceeding...")

    print("Inspecting collection:", COLLECTION_NAME)
    coll = client.get_or_create_collection(name=COLLECTION_NAME)

    # count
    try:
        count = coll.count()
    except Exception:
        try:
            all_items = coll.get(include=['metadatas', 'documents'])
            # fall back to document length if ids not available
            count = len(all_items.get('documents', []))
        except Exception:
            count = None

    print("Document count:", count)

    if FILTER:
        if "=" not in FILTER:
            print("FILTER must be in the form key=value")
        else:
            k, v = FILTER.split("=", 1)
            print(f"Running filter: {k} == {v}")
            try:
                res = coll.get(filter={k: v}, include=['metadatas', 'documents'])
                docs = res.get('documents', [])
                metas = res.get('metadatas', [])
                ids = res.get('ids')  # some chroma versions include ids, but not always
                total = len(docs)
                print(f"Matches: {total}")
                for i in range(min(total, SAMPLES)):
                    print('\n---')
                    display_id = ids[i] if ids else f"idx_{i}"
                    print('id:', display_id)
                    print('meta:', format_meta(metas[i] if i < len(metas) else {}))
                    print('doc:', docs[i][:400])
            except Exception as e:
                print('Filter query failed:', e)
    else:
        print(f"Printing up to {SAMPLES} sample documents:")
        try:
            res = coll.get(include=['metadatas', 'documents'])
            docs = res.get('documents', [])
            metas = res.get('metadatas', [])
            ids = res.get('ids')
            for i in range(min(len(docs), SAMPLES)):
                print('\n---')
                display_id = ids[i] if ids else f"idx_{i}"
                print('id:', display_id)
                print('meta:', format_meta(metas[i] if i < len(metas) else {}))
                print('doc:', docs[i][:400])
        except Exception as e:
            print('Could not fetch documents via coll.get():', e)

    if NN_QUERY:
        if not GEMINI_API_KEYS:
            print('\nNN_QUERY set but GEMINI_API_KEYS not found in env')
        else:
            print('\nRunning nearest-neighbor search for:', NN_QUERY)
            gem = GeminiHelper(api_keys=GEMINI_API_KEYS)
            try:
                emb = gem.get_embeddings(texts=NN_QUERY, model='text-embedding-004', task_type='RETRIEVAL_DOCUMENT', output_dimensionality=768)
            except Exception as e:
                print('Embedding error:', e)
                emb = None

            if emb is not None:
                try:
                    nn = coll.query(query_embeddings=[emb], n_results=5, include=['metadatas', 'documents', 'distances'])
                    # nn structure is lists-of-lists (one query -> results list)
                    q_ids = nn.get('ids')
                    docs = nn.get('documents', [[]])[0]
                    metas = nn.get('metadatas', [[]])[0]
                    dists = nn.get('distances', [[]])[0]
                    n_results = len(docs)
                    for idx in range(n_results):
                        print('\n---')
                        display_id = q_ids[0][idx] if q_ids and len(q_ids) > 0 and idx < len(q_ids[0]) else f"nn_idx_{idx}"
                        print('id:', display_id)
                        print('distance:', dists[idx] if idx < len(dists) else None)
                        print('meta:', format_meta(metas[idx] if idx < len(metas) else {}))
                        print('doc:', docs[idx][:400])
                except Exception as e:
                    print('NN query failed:', e)


if __name__ == '__main__':
    main()
