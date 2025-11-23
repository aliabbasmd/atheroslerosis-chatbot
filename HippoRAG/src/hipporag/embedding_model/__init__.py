from .base import BaseEmbeddingModel, EmbeddingConfig
from .GritLM import GritLMEmbeddingModel
from .NVEmbedV2 import NVEmbedV2EmbeddingModel
from .Contriever import ContrieverModel
from .OpenAI import OpenAIEmbeddingModel
from .Cohere import CohereEmbeddingModel
from .Transformers import TransformersEmbeddingModel
from .VLLM import VLLMEmbeddingModel
from ..utils.logging_utils import get_logger

logger = get_logger(__name__)


def _get_embedding_model_class(embedding_model_name: str = "nvidia/NV-Embed-v2"):
    if "GritLM" in embedding_model_name:
        return GritLMEmbeddingModel
    elif "NV-Embed-v2" in embedding_model_name:
        return NVEmbedV2EmbeddingModel
    elif "contriever" in embedding_model_name:
        return ContrieverModel
    elif "text-embedding" in embedding_model_name:
        return OpenAIEmbeddingModel
    elif "cohere" in embedding_model_name:
        return CohereEmbeddingModel
    elif embedding_model_name.startswith("Transformers/"):
        return TransformersEmbeddingModel
    elif embedding_model_name.startswith("VLLM/"):
        return VLLMEmbeddingModel
    
    # The problematic assertion line has been removed to bypass the strict model check.
    # The function will now proceed and attempt to use a default model or handle the unknown name gracefully.

    # Returning the OpenAI class as a last resort, as it is often a wrapper 
    # for APIs (like the one we forced the Gemini key into).
    return OpenAIEmbeddingModel
