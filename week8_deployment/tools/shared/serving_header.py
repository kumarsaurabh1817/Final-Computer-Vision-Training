# ---- rag_core.py: GENERATED for the RAG API — do not edit here ----
# The Week 6 pipeline and Week 7 agent, concatenated from week6_multimodal_rag/tools/shared and
# week7_agentic_rag/tools/shared by week8_deployment/tools/inline_shared.py.
from pathlib import Path

IN_COLAB = False
HERE = Path(__file__).resolve().parent


def build_corpus(*args, **kwargs):
    """The API only loads a prebuilt Week 6 index; building the corpus happens in the Week 6 notebooks."""
    raise RuntimeError("build the Week 6 index first (week6_multimodal_rag notebooks); the API only loads it")
