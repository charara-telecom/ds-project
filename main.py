import json
from typing import Any, Dict, Optional, List, Tuple
from datetime import datetime, timedelta, timezone

from langchain_community.vectorstores import Chroma

from gemini_helper import GeminiHelper  # your helper class

# load api keys from .env
from dotenv import load_dotenv
load_dotenv(override=True)
import os
GEMINI_API_KEYS = os.getenv("GEMINI_API_KEYS").split("|||||")

# -------- CONSTANTS --------
CHROMA_PATH = "chroma_db"
COLLECTION_NAME = "tmdb_movie_reviews"


# -------- METADATA EXTRACTION --------
def extract_filters(gemini: GeminiHelper, question: str) -> Dict[str, Any]:
    """
    Use Gemini to extract structured filters from the user question.
    We also handle recency like "last 2 days", "in the last week", etc.

    Expected JSON fields:
    - movie_title: string | null
    - min_rating: float | null
    - max_rating: float | null
    - days_back: int | null   (e.g. 2 for "last 2 days")
    """
    prompt = f"""
You are an assistant that extracts structured filters for searching movie reviews.

From the user's question, detect these fields:
- movie_title: exact title if clearly mentioned (otherwise null, don't guess).
- min_rating: minimum rating (0-10) if the user asks for high / above X ratings, else null.
- max_rating: maximum rating (0-10) if the user asks for below X ratings, else null.
- days_back: integer number of days for recency if the user asks for something like:
  "last 2 days", "in the last 3 days", "last week", "last few days".
  - For "last week" assume 7.
  - If no clear recency constraint, set days_back to null.

Return ONLY valid JSON. No explanation, no extra text.

Example 1:
User: "Show me reviews for Wicked with rating above 8 in the last 2 days"
Output:
{{
  "movie_title": "Wicked",
  "min_rating": 8.0,
  "max_rating": null,
  "days_back": 2
}}

Example 2:
User: "What do people think about Joker?"
Output:
{{
  "movie_title": "Joker",
  "min_rating": null,
  "max_rating": null,
  "days_back": null
}}

User question:
\"\"\"{question}\"\"\"
"""

    raw = gemini.answer_and_rotate(prompt).strip()

    # DEBUG: print raw LLM output for filters
    print("\n[DEBUG] Raw filter LLM output:")
    print(raw)

    # Try to isolate JSON in case the model adds extra text
    try:
        if not raw.startswith("{"):
            start = raw.find("{")
            end = raw.rfind("}") + 1
            raw = raw[start:end]
        data = json.loads(raw)
    except Exception:
        print("⚠️ Could not parse JSON from model. Raw output:")
        print(raw)
        return {}

    # Normalize keys / defaults
    return {
        "movie_title": data.get("movie_title"),
        "min_rating": data.get("min_rating"),
        "max_rating": data.get("max_rating"),
        "days_back": data.get("days_back"),
    }


def iso_utc_from_days_back(days: int) -> str:
    """
    Convert days_back into an ISO8601 UTC string like 2025-11-30T17:55:37.911Z
    to compare with your `date` metadata (stored in this format).
    """
    now = datetime.now(timezone.utc)
    since = now - timedelta(days=days)
    # Convert to ISO with 'Z' suffix
    iso = since.isoformat()
    if iso.endswith("+00:00"):
        iso = iso.replace("+00:00", "Z")
    return iso


def build_chroma_filter(meta: Dict[str, Any]) -> Dict[str, Any]:
    """
    Convert extracted metadata into a Chroma filter dict.

    Chroma's latest filter rules (very important):
    - Each dict can only contain ONE field or ONE logical operator ($and / $or).
    - For a field dict, only ONE operator per field is allowed.
      So instead of:
          {"rating": {"$gte": 8.0, "$lte": 10.0}}
      you must do:
          {"$and": [
              {"rating": {"$gte": 8.0}},
              {"rating": {"$lte": 10.0}}
          ]}
    """

    conditions: List[Dict[str, Any]] = []

    movie_title = meta.get("movie_title")
    if movie_title:
        # equality is allowed directly
        conditions.append({"movie_title": movie_title})

    min_rating = meta.get("min_rating")
    max_rating = meta.get("max_rating")

    if min_rating is not None:
        conditions.append({"rating": {"$gte": float(min_rating)}})

    if max_rating is not None:
        conditions.append({"rating": {"$lte": float(max_rating)}})

    days_back = meta.get("days_back")
    if isinstance(days_back, (int, float)) and days_back > 0:
        since_iso = iso_utc_from_days_back(int(days_back))
        # date >= since_iso  (date is stored as ISO8601, e.g. "2025-11-30T17:55:37.911Z")
        conditions.append({"date": {"$gte": since_iso}})

    # No conditions → no filter
    if not conditions:
        return {}

    # One condition → just return it as-is
    if len(conditions) == 1:
        return conditions[0]

    # Multiple conditions → combine with $and
    return {"$and": conditions}


# -------- RAG LOGIC --------
def build_context_from_docs(
    docs_and_scores: List[Tuple[Any, float]]
) -> str:
    """
    Turn LangChain Documents + scores into a text context for the LLM.
    Assumes metadata contains:
      - movie_title
      - rating
      - author
      - date
    """
    parts = []
    for i, (doc, score) in enumerate(docs_and_scores, start=1):
        meta = doc.metadata or {}
        parts.append(
            f"Review #{i} | score={score:.4f}\n"
            f"Movie: {meta.get('movie_title')} | "
            f"Rating: {meta.get('rating')} | "
            f"Author: {meta.get('author')} | "
            f"Date: {meta.get('date')}\n"
            f"Content: {doc.page_content}\n"
            f"{'-'*60}"
        )
    return "\n\n".join(parts)


def rag_answer(
    gemini: GeminiHelper,
    question: str,
    chroma_filter: Optional[Dict[str, Any]] = None,
    k: int = 5,
) -> str:
    """
    Full RAG:
    - get embedding for question
    - query Chroma via LangChain with filter (including date)
    - feed retrieved docs + question to Gemini to answer
    """
    # 1) Embed question
    query_emb = gemini.get_embeddings(
        texts=question,
        model="text-embedding-004",
        task_type="RETRIEVAL_QUERY",
    )

    # 2) LangChain Chroma wrapper (reuses existing collection on disk)
    vectorstore = Chroma(
        collection_name=COLLECTION_NAME,
        persist_directory=CHROMA_PATH,
        embedding_function=None,  # we provide embeddings directly
    )

    docs_and_scores = vectorstore.similarity_search_by_vector_with_relevance_scores(
    query_emb,
    k=k,
    filter=chroma_filter if chroma_filter else None,
)
    # DEBUG: print retrieved docs and metadata
    print(f"\n[DEBUG] Retrieved {len(docs_and_scores)} docs from Chroma")
    for i, (doc, score) in enumerate(docs_and_scores, start=1):
        meta = doc.metadata or {}
        print(f"[DEBUG][DOC {i}] score={score:.4f}")
        print(
            f"    movie_title={meta.get('movie_title')}, "
            f"rating={meta.get('rating')}, "
            f"author={meta.get('author')}, "
            f"date={meta.get('date')}"
        )
        snippet = doc.page_content[:200]
        print(f"    content_snippet={snippet!r}")
        if len(doc.page_content) > 200:
            print("    ... [truncated]")
        print()

    if not docs_and_scores:
        return "I couldn't find any relevant reviews in the database for your question."

    context = build_context_from_docs(docs_and_scores)

    # 3) Build RAG prompt
    rag_prompt = f"""
You are a helpful assistant answering questions using real TMDB movie reviews.

User question:
{question}

Here are some relevant reviews with metadata:
---------------------------------------------
{context}
---------------------------------------------

Using ONLY the information from these reviews, answer the user's question.
If the reviews are not enough, say that you don't have enough information.
"""

    # DEBUG: you can also print the full RAG prompt if needed:
    # print("\n[DEBUG] RAG prompt being sent to LLM:\n", rag_prompt)

    return gemini.answer_and_rotate(rag_prompt)


def main():
    # Init Gemini
    gemini = GeminiHelper(api_keys=GEMINI_API_KEYS)

    # Simple CLI loop
    while True:
        user_q = input("\n🧠 Ask about movie reviews (or 'exit'): ").strip()
        if not user_q or user_q.lower() in {"exit", "quit"}:
            break

        # 1) Extract metadata filters (including days_back)
        extracted = extract_filters(gemini, user_q)
        print("\n📌 Extracted metadata filters from model (parsed):")
        print(json.dumps(extracted, indent=2))

        chroma_filter = build_chroma_filter(extracted)
        if chroma_filter:
            print("\n📌 Chroma filter that will be applied:")
            print(json.dumps(chroma_filter, indent=2))
        else:
            print("\n📌 No filter extracted. Querying all reviews.")

        # 2) RAG answer
        print("\n🔎 Running RAG retrieval + answer...\n")
        answer = rag_answer(gemini, user_q, chroma_filter=chroma_filter, k=5)

        # DEBUG: print final answer with a clear tag
        print("\n[DEBUG] Final LLM answer:")
        print(answer)


if __name__ == "__main__":
    main()
