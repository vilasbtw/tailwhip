import pickle
from pathlib import Path

from tailwhip.models.schema_document import SchemaDocument


MODEL_NAME = "paraphrase-multilingual-MiniLM-L12-v2"
CHROMA_DIR = "chroma"
BM25_FILE  = "bm25.pkl"
DOCS_FILE  = "documents.pkl"
BATCH_SIZE = 256


class IndexManager:
    """
    Manages the two search indexes:
      - ChromaDB -> vector search (semantic, PT-BR + EN)
      - BM25     -> lexical search (exact token matching)

    Everything is persisted under base_dir (default: ~/.tailwhip/).
    The sentence-transformers model is loaded once per instance (lazy singleton).

    Public methods:
        build(documents)     -> index from scratch
        refresh(documents)   -> re-index everything
        vector_search(query) -> list of doc_ids ranked by vector similarity
        bm25_search(query)   -> list of doc_ids ranked by BM25 score
        get_documents()      -> dict of doc_id -> SchemaDocument
    """

    DEFAULT_BASE_DIR = Path.home() / ".tailwhip"

    def __init__(self, base_dir: Path | None = None):
        self._base_dir = Path(base_dir) if base_dir else self.DEFAULT_BASE_DIR
        self._base_dir.mkdir(parents=True, exist_ok=True)

        self._chroma_dir = self._base_dir / CHROMA_DIR
        self._bm25_path  = self._base_dir / BM25_FILE
        self._docs_path  = self._base_dir / DOCS_FILE

        self._model      = None  # lazy loaded
        self._collection = None  # ChromaDB collection
        self._bm25       = None  # BM25Okapi instance
        self._documents: dict[str, SchemaDocument] = {}

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------

    def build(self, documents: list[SchemaDocument]) -> None:
        """Index all documents from scratch."""
        self._documents = {doc.doc_id: doc for doc in documents}
        self._build_chroma(documents)
        self._build_bm25(documents)
        self._persist_documents()

    def refresh(self, documents: list[SchemaDocument]) -> None:
        """Re-index everything. Equivalent to build() — the schema rarely changes."""
        self.build(documents)

    def vector_search(self, query: str, n_results: int = 50) -> list[str]:
        """Return doc_ids ranked by vector similarity."""
        self._ensure_loaded()
        model = self._get_model()
        embedding = model.encode([query], show_progress_bar=False).tolist()

        results = self._collection.query(
            query_embeddings=embedding,
            n_results=min(n_results, self._collection.count()),
        )
        return results["ids"][0] if results["ids"] else []

    def bm25_search(self, query: str, n_results: int = 50) -> list[str]:
        """Return doc_ids ranked by BM25 score."""
        self._ensure_loaded()
        from tailwhip.models.schema_document import tokenize_name

        tokens = tokenize_name(query).split()
        scores = self._bm25.get_scores(tokens)

        doc_ids = list(self._documents.keys())
        ranked  = sorted(
            zip(doc_ids, scores),
            key=lambda x: x[1],
            reverse=True,
        )
        return [doc_id for doc_id, score in ranked[:n_results] if score > 0]

    def get_documents(self) -> dict[str, SchemaDocument]:
        """Return all indexed documents."""
        self._ensure_loaded()
        return self._documents

    # ------------------------------------------------------------------
    # Private
    # ------------------------------------------------------------------

    def _get_model(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(MODEL_NAME)
        return self._model

    def _build_chroma(self, documents: list[SchemaDocument]) -> None:
        import chromadb

        client = chromadb.PersistentClient(path=str(self._chroma_dir))

        # Drop and recreate the collection to index from scratch
        try:
            client.delete_collection("tailwhip")
        except Exception:
            pass

        self._collection = client.create_collection(
            name="tailwhip",
            metadata={"hnsw:space": "cosine"},
        )

        model = self._get_model()
        texts   = [doc.to_embedding_text() for doc in documents]
        doc_ids = [doc.doc_id for doc in documents]

        # Embed in batches to avoid memory overflow
        for i in range(0, len(texts), BATCH_SIZE):
            batch_texts      = texts[i : i + BATCH_SIZE]
            batch_ids        = doc_ids[i : i + BATCH_SIZE]
            batch_embeddings = model.encode(
                batch_texts, show_progress_bar=False
            ).tolist()
            self._collection.add(
                ids=batch_ids,
                documents=batch_texts,
                embeddings=batch_embeddings,
            )

    def _build_bm25(self, documents: list[SchemaDocument]) -> None:
        from rank_bm25 import BM25Okapi
        from tailwhip.models.schema_document import tokenize_name

        corpus = [
            tokenize_name(doc.to_embedding_text()).split()
            for doc in documents
        ]
        self._bm25 = BM25Okapi(corpus)

        with open(self._bm25_path, "wb") as f:
            pickle.dump(self._bm25, f)

    def _persist_documents(self) -> None:
        with open(self._docs_path, "wb") as f:
            pickle.dump(self._documents, f)

    def _ensure_loaded(self) -> None:
        """Load indexes from disk if not already in memory."""
        if self._documents and self._bm25 and self._collection:
            return

        if not self._docs_path.exists() or not self._bm25_path.exists():
            raise RuntimeError(
                "Index not found. Run 'tailwhip index --file <csv>' first."
            )

        with open(self._docs_path, "rb") as f:
            self._documents = pickle.load(f)

        with open(self._bm25_path, "rb") as f:
            self._bm25 = pickle.load(f)

        import chromadb
        client = chromadb.PersistentClient(path=str(self._chroma_dir))
        self._collection = client.get_collection("tailwhip")
