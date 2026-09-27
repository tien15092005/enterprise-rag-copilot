from app.core.config import Settings
from app.embeddings.base import EmbeddingProvider
from app.embeddings.sentence_transformers import SentenceTransformerEmbeddingProvider


def create_embedding_provider(settings: Settings) -> EmbeddingProvider:
    if settings.embedding_provider == "sentence_transformers":
        return SentenceTransformerEmbeddingProvider(
            model_name=settings.embedding_model_name,
            revision=settings.embedding_model_revision,
            dimension=settings.embedding_dimension,
            batch_size=settings.embedding_batch_size,
            device=settings.embedding_device,
            normalize=settings.embedding_normalize,
            cache_folder=settings.embedding_cache_directory,
        )
    raise ValueError(f"unsupported embedding provider: {settings.embedding_provider}")
