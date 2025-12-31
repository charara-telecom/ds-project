# consumer.py
import json
import os
import time
from datetime import datetime, timezone
from typing import Optional, Tuple, Dict, Any

import chromadb
from kafka import KafkaConsumer
from dotenv import load_dotenv

from gemini_helper import GeminiHelper

# ==========================
# ENV + CONFIG
# ==========================
load_dotenv(override=True)

GEMINI_API_KEYS = os.getenv("GEMINI_API_KEYS", "").split("|||||")
if not GEMINI_API_KEYS or GEMINI_API_KEYS == [""]:
    raise RuntimeError("GEMINI_API_KEYS not found in environment.")

KAFKA_TOPIC = "movie-reviews"
KAFKA_BOOTSTRAP = "localhost:9092"
KAFKA_GROUP_ID = "tmdb-review-consumer"

# Create a NEW DB if you want (recommended)
CHROMA_PATH = "chroma_db_v2"   # <-- change back to "chroma_db" if you want
COLLECTION_NAME = "tmdb_movie_reviews"

EMBED_MODEL = "text-embedding-004"
EMBED_DIM = 768

# ==========================
# HELPERS
# ==========================
def iso_z_to_epoch_seconds(iso_z: str) -> Optional[int]:
    """
    Convert '2025-12-01T08:51:50.212Z' -> epoch seconds (int).
    Returns None if parsing fails.
    """
    if not iso_z:
        return None
    try:
        dt = datetime.fromisoformat(iso_z.replace("Z", "+00:00"))
        return int(dt.timestamp())
    except Exception:
        return None


def now_iso_z() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def build_chroma_record(review: dict) -> Optional[Tuple[str, str, Dict[str, Any]]]:
    """
    Build (doc_id, content, metadata) for Chroma.
    Ensures metadata types are filter-friendly.
    """
    content = review.get("content")
    if not content:
        return None

    movie_id = review.get("movie_id")
    movie_title = review.get("movie_title")

    # TMDB review id may be in review["id"]
    review_id = review.get("id")

    # If missing IDs, fallback to a time-based unique key
    if movie_id is None:
        movie_id = "unknown_movie"
    if not review_id:
        review_id = f"noid_{int(time.time() * 1000)}"

    doc_id = f"{movie_id}_{review_id}"

    author = review.get("author")
    author_details = review.get("author_details") or {}

    rating = author_details.get("rating")
    created_at = review.get("created_at")
    updated_at = review.get("updated_at")

    # Prefer created_at; fallback to updated_at; fallback to now
    date_iso = created_at or updated_at or now_iso_z()
    date_ts = iso_z_to_epoch_seconds(date_iso) or int(time.time())

    metadata: Dict[str, Any] = {
        "movie_title": movie_title or "Unknown",
        "author": author or author_details.get("username") or "Unknown",
        "date_iso": date_iso,
        "date_ts": date_ts,  # ✅ numeric for comparisons
        # Optional debugging / lineage
        "source": "tmdb",
        "movie_id": int(movie_id) if str(movie_id).isdigit() else str(movie_id),
    }

    # Only add rating if it is numeric (avoid None / str issues)
    if rating is not None:
        try:
            metadata["rating"] = float(rating)
        except Exception:
            # ignore non-numeric ratings
            pass

    return doc_id, content, metadata


# ==========================
# INIT CLIENTS
# ==========================
gemini = GeminiHelper(api_keys=GEMINI_API_KEYS)

client = chromadb.PersistentClient(path=CHROMA_PATH)
collection = client.get_or_create_collection(name=COLLECTION_NAME)

consumer = KafkaConsumer(
    KAFKA_TOPIC,
    bootstrap_servers=KAFKA_BOOTSTRAP,
    auto_offset_reset="earliest",
    enable_auto_commit=True,
    group_id=KAFKA_GROUP_ID,
    value_deserializer=lambda v: json.loads(v.decode("utf-8")),
)

print(f"✅ Consumer started. Topic='{KAFKA_TOPIC}' | Chroma='{CHROMA_PATH}' | Collection='{COLLECTION_NAME}'")
print("Waiting for messages...")


# ==========================
# MAIN LOOP
# ==========================
try:
    for msg in consumer:
        review = msg.value

        rec = build_chroma_record(review)
        if rec is None:
            continue

        doc_id, content, metadata = rec

        # Debug metadata (you asked for this)
        print("\n------------------------------")
        print("Incoming review -> metadata:")
        print(json.dumps(metadata, indent=2, ensure_ascii=False))

        # 1) Embed
        try:
            embedding = gemini.get_embeddings(
                texts=content,
                model=EMBED_MODEL,
                task_type="RETRIEVAL_DOCUMENT",
                output_dimensionality=EMBED_DIM,
            )
        except Exception as e:
            print("❌ Embedding error:", e)
            continue

        # 2) Store
        try:
            collection.add(
                ids=[doc_id],
                documents=[content],
                metadatas=[metadata],
                embeddings=[embedding],
            )
            print(f"✅ Stored doc_id={doc_id} | movie='{metadata.get('movie_title')}' | rating={metadata.get('rating')} | date_ts={metadata.get('date_ts')}")
        except Exception as e:
            # Most common error here is "ID already exists"
            print("❌ Error adding to Chroma:", e)
            print("doc_id:", doc_id)

except KeyboardInterrupt:
    print("\n🛑 Stopping consumer...")

finally:
    consumer.close()
    print("✅ Consumer closed.")
