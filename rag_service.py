import json
from typing import Any, Dict, List, Optional, Tuple
from datetime import datetime, timedelta, timezone

from pydantic import BaseModel, Field

from langchain_community.vectorstores import Chroma
from langchain_core.embeddings import Embeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.runnables import RunnableLambda
from langchain_core.language_models.llms import LLM

from gemini_helper import GeminiHelper
import logging
import os

logger = logging.getLogger("rag_service")

# Avoid adding multiple handlers on Streamlit reruns
if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("[%(asctime)s] %(levelname)s: %(message)s"))
    logger.addHandler(handler)

logger.setLevel(os.getenv("RAG_LOG_LEVEL", "INFO").upper())
logger.propagate = False


# ==========================
# CONFIG
# ==========================
CHROMA_PATH = "chroma_db_v2"

# Reviews collection (filtered)
REVIEWS_COLLECTION_NAME = "tmdb_movie_reviews"

# Descriptions collection (NO filters)
DESCRIPTIONS_COLLECTION_NAME = "movie_descriptions"

EMBED_MODEL = "text-embedding-004"
EMBED_DIM = 768


# ==========================
# LangChain Embeddings wrapper (Gemini)
# ==========================
class GeminiEmbeddings(Embeddings):
    def __init__(self, gemini: GeminiHelper, model: str = EMBED_MODEL, dim: int = EMBED_DIM):
        self.gemini = gemini
        self.model = model
        self.dim = dim

    def embed_query(self, text: str) -> List[float]:
        return self.gemini.get_embeddings(
            texts=text,
            model=self.model,
            task_type="RETRIEVAL_QUERY",
            output_dimensionality=self.dim,
        )

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return self.gemini.get_embeddings(
            texts=texts,
            model=self.model,
            task_type="RETRIEVAL_DOCUMENT",
            output_dimensionality=self.dim,
        )


# ==========================
# LangChain LLM wrapper (Gemini)
# ==========================
class GeminiLLM(LLM):
    gemini: GeminiHelper

    class Config:
        arbitrary_types_allowed = True

    @property
    def _llm_type(self) -> str:
        return "gemini_helper"

    def _call(self, prompt: str, stop: Optional[List[str]] = None, **kwargs: Any) -> str:
        text = self.gemini.answer_and_rotate(prompt)
        if stop:
            for s in stop:
                if s in text:
                    text = text.split(s)[0]
        return text


# ==========================
# Utilities
# ==========================
def _extract_json(raw: str) -> Optional[dict]:
    if not raw:
        return None
    raw = raw.strip()

    # remove markdown fences if any
    if raw.startswith("```"):
        raw = raw.replace("```json", "").replace("```", "").strip()

    # keep only JSON object part
    if "{" in raw and "}" in raw:
        raw = raw[raw.find("{") : raw.rfind("}") + 1]

    try:
        return json.loads(raw)
    except Exception:
        return None


def _since_ts_from_days_back(days: int) -> int:
    now = datetime.now(timezone.utc)
    since = now - timedelta(days=int(days))
    return int(since.timestamp())


def _build_chroma_filter(meta: Dict[str, Any]) -> Dict[str, Any]:
    conditions = []

    if meta.get("movie_title"):
        conditions.append({"movie_title": meta["movie_title"]})

    if meta.get("min_rating") is not None:
        conditions.append({"rating": {"$gte": float(meta["min_rating"])}})

    if meta.get("max_rating") is not None:
        conditions.append({"rating": {"$lte": float(meta["max_rating"])}})

    if meta.get("days_back") is not None:
        try:
            since_ts = _since_ts_from_days_back(int(meta["days_back"]))
            conditions.append({"date_ts": {"$gte": since_ts}})
        except Exception:
            pass

    if not conditions:
        return {}
    return conditions[0] if len(conditions) == 1 else {"$and": conditions}


def _build_reviews_context(docs_and_scores: List[Tuple[Any, float]]) -> str:
    parts = []
    for i, (doc, score) in enumerate(docs_and_scores, 1):
        meta = doc.metadata or {}
        parts.append(
            f"Review #{i} | score={score:.4f}\n"
            f"Movie: {meta.get('movie_title')} | Rating: {meta.get('rating')} | "
            f"Date: {meta.get('date_iso')} | date_ts: {meta.get('date_ts')}\n"
            f"Content: {doc.page_content}\n"
            f"{'-'*60}"
        )
    return "\n\n".join(parts)


def _build_descriptions_context(docs_and_scores: List[Tuple[Any, float]]) -> str:
    parts = []
    for i, (doc, score) in enumerate(docs_and_scores, 1):
        meta = doc.metadata or {}
        parts.append(
            f"Description #{i} | score={score:.4f}\n"
            f"Movie: {meta.get('movie_title')}\n"
            f"Content: {doc.page_content}\n"
            f"{'-'*60}"
        )
    return "\n\n".join(parts)


# ==========================
# Output schema
# ==========================
class RAGOut(BaseModel):
    answer: str = Field(...)
    movies_referenced: List[str] = Field(default_factory=list)


# ==========================
# Prompts
# ==========================
FILTER_PROMPT = """You extract structured filters from movie questions.
Return ONLY JSON (no markdown) with exactly:
{{
  "movie_title": null or string,
  "min_rating": null or number,
  "max_rating": null or number,
  "days_back": null or integer
}}

Rules:
- "last year" => days_back=365
- "last 10 years" => days_back=3650
- If no time constraint => days_back=null
- If user asks "rated 10" => min_rating=10 and max_rating=10
- If movie title misspelled, correct it.

User question:
{question}
"""

RAG_PROMPT = ChatPromptTemplate.from_template(
    """Answer using ONLY the information below. don't look for answers outside the provided contexts. and always answer in a way that doesnt look like you have such context, just a normal answer.

Question:
{question}

Movie descriptions:
{descriptions_context}

Reviews:
{reviews_context}

Return ONLY valid JSON:
{{
  "answer": "string",
  "movies_referenced": ["string", "..."]
}}
"""
)


# ==========================
# Vectorstores (LangChain)
# ==========================
_VECTORSTORE_REVIEWS: Optional[Chroma] = None
_VECTORSTORE_DESCRIPTIONS: Optional[Chroma] = None


def _get_reviews_vectorstore(gemini: GeminiHelper) -> Chroma:
    global _VECTORSTORE_REVIEWS
    if _VECTORSTORE_REVIEWS is None:
        _VECTORSTORE_REVIEWS = Chroma(
            collection_name=REVIEWS_COLLECTION_NAME,
            persist_directory=CHROMA_PATH,
            embedding_function=GeminiEmbeddings(gemini),
        )
    return _VECTORSTORE_REVIEWS


def _get_descriptions_vectorstore(gemini: GeminiHelper) -> Chroma:
    global _VECTORSTORE_DESCRIPTIONS
    if _VECTORSTORE_DESCRIPTIONS is None:
        _VECTORSTORE_DESCRIPTIONS = Chroma(
            collection_name=DESCRIPTIONS_COLLECTION_NAME,
            persist_directory=CHROMA_PATH,
            embedding_function=GeminiEmbeddings(gemini),
        )
    return _VECTORSTORE_DESCRIPTIONS


# ==========================
# Build the LangChain pipeline (NO date-filter fallback)
# ==========================
def build_chain(gemini: GeminiHelper, k_reviews: int = 5, k_desc: int = 5):
    llm = GeminiLLM(gemini=gemini)
    parser = JsonOutputParser(pydantic_object=RAGOut)

    reviews_store = _get_reviews_vectorstore(gemini)
    desc_store = _get_descriptions_vectorstore(gemini)

    def extract_filters_step(inp: Dict[str, Any]) -> Dict[str, Any]:
        q = inp["question"]
        raw = gemini.answer_and_rotate(FILTER_PROMPT.format(question=q))
        data = _extract_json(raw) or {}
        filters = {
            "movie_title": data.get("movie_title"),
            "min_rating": data.get("min_rating"),
            "max_rating": data.get("max_rating"),
            "days_back": data.get("days_back"),
        }
        return {"question": q, "filters": filters}

    def retrieve_step(inp: Dict[str, Any]) -> Dict[str, Any]:
        q = inp["question"]
        filters = inp["filters"]

        # ---- Reviews retrieval (filtered; NO fallback) ----
        chroma_filter = _build_chroma_filter(filters)

        reviews_docs_and_scores = reviews_store.similarity_search_with_relevance_scores(
            q, k=k_reviews, filter=chroma_filter or None
        )

        logger.info("QUERY=%r", q)
        logger.info("FILTERS=%s", filters)
        logger.info("USED_REVIEWS_FILTER=%s", chroma_filter)
        logger.info("REVIEWS_RETRIEVED=%d", len(reviews_docs_and_scores))

        for i, (doc, score) in enumerate(reviews_docs_and_scores, 1):
            meta = doc.metadata or {}
            snippet = (doc.page_content or "").replace("\n", " ")[:160]
            logger.info(
                "  [reviews] #%d score=%.4f title=%r rating=%s date_ts=%s date_iso=%s snippet=%r",
                i,
                float(score),
                meta.get("movie_title"),
                meta.get("rating"),
                meta.get("date_ts"),
                meta.get("date_iso"),
                snippet,
            )

        # ---- Descriptions retrieval (NO filters) ----
        desc_docs_and_scores = desc_store.similarity_search_with_relevance_scores(
            q, k=k_desc, filter=None
        )

        logger.info("DESCRIPTIONS_RETRIEVED=%d", len(desc_docs_and_scores))
        for i, (doc, score) in enumerate(desc_docs_and_scores, 1):
            meta = doc.metadata or {}
            snippet = (doc.page_content or "").replace("\n", " ")[:160]
            logger.info(
                "  [descriptions] #%d score=%.4f title=%r snippet=%r",
                i,
                float(score),
                meta.get("movie_title"),
                snippet,
            )

        return {
            "question": q,
            "filters": filters,
            "used_filter": chroma_filter,
            "reviews_docs_and_scores": reviews_docs_and_scores,
            "desc_docs_and_scores": desc_docs_and_scores,
        }

    def prepare_prompt_vars(inp: Dict[str, Any]) -> Dict[str, Any]:
        reviews_ds = inp["reviews_docs_and_scores"]
        desc_ds = inp["desc_docs_and_scores"]

        inp["reviews_context"] = _build_reviews_context(reviews_ds) if reviews_ds else ""
        inp["descriptions_context"] = _build_descriptions_context(desc_ds) if desc_ds else ""
        return inp

    def llm_step(inp: Dict[str, Any]) -> Dict[str, Any]:
        reviews_ds = inp["reviews_docs_and_scores"]
        desc_ds = inp["desc_docs_and_scores"]

        # If nothing retrieved from both, return without LLM call
        if not reviews_ds and not desc_ds:
            return {
                "answer": "No relevant reviews or movie descriptions found in the database for this query/filters.",
                "movies_referenced": [],
                "filters": inp["filters"],
                "used_filter": inp["used_filter"],
                "retrieved_reviews": 0,
                "retrieved_descriptions": 0,
            }

        msg = RAG_PROMPT.format_messages(
            question=inp["question"],
            descriptions_context=inp["descriptions_context"],
            reviews_context=inp["reviews_context"],
        )
        prompt_text = "\n".join([m.content for m in msg])

        raw = llm.invoke(prompt_text)

        # parser may raise if model returns garbage -> fallback to best-effort JSON extraction
        try:
            out = parser.parse(raw)
        except Exception:
            out = _extract_json(raw) or {"answer": raw, "movies_referenced": []}

        # normalize to dict
        if isinstance(out, BaseModel):
            out = out.model_dump()

        answer = out.get("answer", "")
        movies = out.get("movies_referenced", [])
        if not isinstance(answer, str):
            answer = str(answer)
        if not isinstance(movies, list):
            movies = []

        return {
            "answer": answer.strip(),
            "movies_referenced": movies,
            "filters": inp["filters"],
            "used_filter": inp["used_filter"],
            "retrieved_reviews": len(reviews_ds),
            "retrieved_descriptions": len(desc_ds),
        }

    return (
        RunnableLambda(extract_filters_step)
        | RunnableLambda(retrieve_step)
        | RunnableLambda(prepare_prompt_vars)
        | RunnableLambda(llm_step)
    )


# ==========================
# Public function
# ==========================
_CHAIN = None
_CHAIN_CFG: Optional[Tuple[int, int]] = None  # (k_reviews, k_desc)


def answer_question(gemini: GeminiHelper, question: str, k: int = 5, k_desc: int = 5) -> Dict[str, Any]:
    """
    k      -> number of review docs
    k_desc -> number of description docs
    """
    global _CHAIN, _CHAIN_CFG

    cfg = (int(k), int(k_desc))
    if _CHAIN is None or _CHAIN_CFG != cfg:
        _CHAIN = build_chain(gemini, k_reviews=cfg[0], k_desc=cfg[1])
        _CHAIN_CFG = cfg

    return _CHAIN.invoke({"question": question})
