from google import genai
from google.genai import types as genai_types  # 👈 for EmbedContentConfig


class GeminiHelper:
    def __init__(self, api_keys: list[str], model: str = "gemini-2.5-flash"):
        if not api_keys:
            raise ValueError("API key not found")

        self._api_keys = api_keys
        self._idx = 0
        self.model_name = model
        self._client = None
        self._set_key(self._idx)

    def _set_key(self, idx: int):
        idx = idx % len(self._api_keys)
        self._idx = idx

        # New GenAI SDK: create a Client per key
        self._client = genai.Client(api_key=self._api_keys[idx])

    def answer_and_rotate(self, prompt: str) -> str:
        n = len(self._api_keys)

        while n:
            try:
                response = self._client.models.generate_content(
                    model=self.model_name,
                    contents=prompt,
                )
                return response.text
            except Exception as e:
                print("Permutating Key due to error:", e)
                n -= 1
                if n == 0:
                    raise Exception("All API keys Exhausted.")
                self._set_key(self._idx + 1)

    def get_embeddings(
        self,
        texts: str | list[str],
        model: str = "text-embedding-004",  # or "text-embedding-004" if you prefer
        task_type: str | None = None,        # e.g. "RETRIEVAL_DOCUMENT", "SEMANTIC_SIMILARITY"
        output_dimensionality: int | None = None,  # e.g. 768, 1536, 3072
    ):
        """
        Get embeddings for a string or list of strings.
        Returns:
          - list[float] if input is a single string
          - list[list[float]] if input is a list of strings
        """
        single_input = isinstance(texts, str)
        if single_input:
            texts = [texts]

        # Optional config
        config = None
        if task_type or output_dimensionality:
            config = genai_types.EmbedContentConfig(
                task_type=task_type,
                output_dimensionality=output_dimensionality,
            )

        n = len(self._api_keys)

        while n:
            try:
                result = self._client.models.embed_content(
                    model=model,
                    contents=texts,
                    config=config,
                )
                # result.embeddings is a list of embedding objects, each with .values (list[float])
                vectors = [e.values for e in result.embeddings]
                return vectors[0] if single_input else vectors
            except Exception as e:
                print("Permutating Key due to embedding error:", e)
                n -= 1
                if n == 0:
                    raise Exception("All API keys Exhausted for embeddings.")
                self._set_key(self._idx + 1)
    # ---------------- LIVE API SUPPORT ----------------

    async def live_text_once_async(
        self,
        prompt: str,
        model: str | None = None,
    ) -> str:
        """
        Send a single text prompt to a *live* model (e.g. gemini-2.0-flash-live-*)
        and return the streamed text response concatenated.

        This uses the Gemini Live API via client.aio.live.connect() in TEXT mode.
        """
        model_id = model or self.model_name

        # Configure live session to return TEXT (not audio)
        live_config = genai_types.LiveConnectConfig(
            response_modalities=[genai_types.Modality.TEXT]
        )

        chunks: list[str] = []

        async with self._client.aio.live.connect(
            model=model_id,
            config=live_config,
        ) as session:
            # Send user text as a turn
            content = genai_types.Content(
                role="user",
                parts=[genai_types.Part(text=prompt)],
            )

            await session.send_client_content(
                turns=content,
                turn_complete=True,
            )

            # Collect streamed responses
            async for message in session.receive():
                # For text-only mode, message.text should contain incremental text
                if getattr(message, "text", None):
                    chunks.append(message.text)

            return "".join(chunks)

    def live_text_once(
        self,
        prompt: str,
        model: str | None = None,
    ) -> str:
        """
        Synchronous wrapper around live_text_once_async.
        WARNING: Don't use this inside an already running asyncio event loop.
        """
        import asyncio

        return asyncio.run(self.live_text_once_async(prompt, model=model))
from langchain_core.embeddings import Embeddings

class GeminiEmbeddings(Embeddings):
    def __init__(self, gemini, model="text-embedding-004", dim=768):
        self.gemini = gemini
        self.model = model
        self.dim = dim

    def embed_query(self, text: str) -> list[float]:
        return self.gemini.get_embeddings(
            texts=text,
            model=self.model,
            task_type="RETRIEVAL_QUERY",
            output_dimensionality=self.dim,
        )

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self.gemini.get_embeddings(
            texts=texts,
            model=self.model,
            task_type="RETRIEVAL_DOCUMENT",
            output_dimensionality=self.dim,
        )
