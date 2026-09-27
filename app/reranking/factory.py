from app.core.config import Settings
from app.reranking.base import Reranker
from app.reranking.cross_encoder import CrossEncoderReranker


def create_reranker(settings: Settings) -> Reranker:
    if settings.reranker_provider == "cross_encoder":
        return CrossEncoderReranker(
            model_name=settings.reranker_model_name,
            revision=settings.reranker_model_revision,
            device=settings.reranker_device,
            batch_size=settings.reranker_batch_size,
            max_length=settings.reranker_max_length,
            cache_folder=settings.reranker_cache_directory,
        )
    raise ValueError(f"unsupported reranker provider: {settings.reranker_provider}")
