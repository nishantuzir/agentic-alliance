from enum import StrEnum, auto
from typing import TypeAlias


class Provider(StrEnum):
    OPENAI = auto()
    DEEPSEEK = auto()
    ANTHROPIC = auto()
    GOOGLE = auto()
    GROQ = auto()
    AWS = auto()
    FAKE = auto()
    AZUREOPENAI = auto()
    OLLAMA = auto()

class AzureOpenAIModelName(StrEnum):
    """https://azure.microsoft.com/en-us/products/ai-services/openai-service"""

    AZURE_GPT_4O_MINI = "azure-gpt-4o-mini"
    AZURE_GPT_4O = "azure-gpt-4o"

class OpenAIModelName(StrEnum):
    """https://platform.openai.com/docs/models/gpt-4o"""

    GPT_4O_MINI = "gpt-4o-mini"
    GPT_4O = "gpt-4o"


class DeepseekModelName(StrEnum):
    """https://api-docs.deepseek.com/quick_start/pricing"""

    DEEPSEEK_CHAT = "deepseek-chat"


class AnthropicModelName(StrEnum):
    """https://docs.anthropic.com/en/docs/about-claude/models#model-names"""

    HAIKU_3 = "claude-3-haiku"
    HAIKU_35 = "claude-3.5-haiku"
    SONNET_35 = "claude-3.5-sonnet"


class GoogleModelName(StrEnum):
    """https://ai.google.dev/gemini-api/docs/models/gemini"""

    GEMINI_15_FLASH = "gemini-1.5-flash"


class GroqModelName(StrEnum):
    """https://console.groq.com/docs/models"""

    LLAMA_31_8B = "groq-llama-3.1-8b"
    LLAMA_33_70B = "groq-llama-3.3-70b"

    LLAMA_GUARD_3_8B = "groq-llama-guard-3-8b"


class AWSModelName(StrEnum):
    """https://docs.aws.amazon.com/bedrock/latest/userguide/models-supported.html"""

    BEDROCK_HAIKU = "bedrock-3.5-haiku"

class OllamaModelName(StrEnum):
    """https://ollama.com/search"""

    DEEPSEEK_R1_1_5B = "deepseek-r1:1.5b"
    DEEPSEEK_R1_14B = "deepseek-r1:14b"
    PHI4_14B = "phi4:7b"
    WIZARDLM2_7B = "wizardlm2:7b"
    QWEN2_5_CODER_14B = "qwen2.5-coder:14b"
    LLAMA_GUARD3_8B = "llama-guard3:8b"
    LLAMA3_2_VISION_11B = "llama3.2-vision:11b"
    SQLCODER_15B = "sqlcoder:15b"
    QWEN2_5_CODER_7B = "qwen2.5-coder:7b"
    QWEN2_5_1_5B = "qwen2.5:1.5b"
    QWEN2_5_0_5B = "qwen2.5:0.5b"
    QWEN2_5_14B = "qwen2.5:14b"
    MATHSTRAL_7B = "mathstral:7b"
    QWEN2_MATH_1_5B = "qwen2-math:1.5b"
    MISTRALV0_3_7B = "mistralv0.3:7b"
    LLAMA3_2_3B = "llama3.2:3b"
    LLAMA3_1_8B = "llama3.1:8b"



class FakeModelName(StrEnum):
    """Fake model for testing."""

    FAKE = "fake"


AllModelEnum: TypeAlias = (
    OpenAIModelName
    | DeepseekModelName
    | AnthropicModelName
    | GoogleModelName
    | GroqModelName
    | AWSModelName
    | FakeModelName
    | AzureOpenAIModelName
    | OllamaModelName
)
