"""What tune may vary, and the values worth trying by default."""

from app.evals.tune.models import TunableParam

TUNABLE_PARAMS = (
    TunableParam(name="CHAT_SOURCES", values=(3, 5, 8)),
    TunableParam(name="CHAT_CONTEXT_CHUNKS", values=(10, 15, 20, 30)),
    TunableParam(name="EXPAND_SECTIONS", values=(True, False)),
    TunableParam(name="RERANK_ENABLED", values=(True, False)),
    TunableParam(name="MIN_COSINE_SIMILARITY", values=(0.20, 0.30, 0.40)),
    TunableParam(name="MIN_RERANKER_RELEVANCE", values=(0.35, 0.45, 0.55)),
)
