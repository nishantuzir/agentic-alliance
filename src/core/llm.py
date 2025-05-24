from functools import cache
from typing import TypeAlias

from langchain_anthropic import ChatAnthropic
from langchain_aws import ChatBedrock
from langchain_community.chat_models import FakeListChatModel
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_groq import ChatGroq
from langchain_ollama import ChatOllama
from langchain_openai import AzureChatOpenAI, ChatOpenAI

from core.settings import settings
from schema.models import (
    AllModelEnum,
    AnthropicModelName,
    AWSModelName,
    AzureOpenAIModelName,
    DeepseekModelName,
    FakeModelName,
    GoogleModelName,
    GroqModelName,
    OllamaModelName,
    OpenAIModelName,
)

_MODEL_TABLE = {
    AzureOpenAIModelName.AZURE_GPT_4O_MINI: "azure-gpt-4o-mini",
    AzureOpenAIModelName.AZURE_GPT_4O: "azure-gpt-4o",

    OpenAIModelName.GPT_4O_MINI: "gpt-4o-mini",
    OpenAIModelName.GPT_4O: "gpt-4o",

    DeepseekModelName.DEEPSEEK_CHAT: "deepseek-chat",

    AnthropicModelName.HAIKU_3: "claude-3-haiku-20240307",
    AnthropicModelName.HAIKU_35: "claude-3-5-haiku-latest",
    AnthropicModelName.SONNET_35: "claude-3-5-sonnet-latest",

    GoogleModelName.GEMINI_15_FLASH: "gemini-1.5-flash",

    GroqModelName.LLAMA_31_8B: "llama-3.1-8b-instant",
    GroqModelName.LLAMA_33_70B: "llama-3.3-70b-versatile",
    GroqModelName.LLAMA_GUARD_3_8B: "llama-guard-3-8b",

    AWSModelName.BEDROCK_HAIKU: "anthropic.claude-3-5-haiku-20241022-v1:0",

    OllamaModelName.DEEPSEEK_R1_1_5B: "deepseek-r1:1.5b",
    OllamaModelName.DEEPSEEK_R1_14B: "deepseek-r1:14b",
    OllamaModelName.PHI4_14B: "phi4:latest",
    OllamaModelName.WIZARDLM2_7B: "wizardlm2:7b",
    OllamaModelName.QWEN2_5_CODER_14B: "qwen2.5-coder:14b",
    OllamaModelName.LLAMA_GUARD3_8B: "llama-guard3:latest",
    OllamaModelName.LLAMA3_2_VISION_11B: "llama3.2-vision:latest",
    OllamaModelName.SQLCODER_15B: "sqlcoder:15b",
    OllamaModelName.QWEN2_5_CODER_7B: "qwen2.5-coder:7b",
    OllamaModelName.QWEN2_5_1_5B: "qwen2.5:1.5b",
    OllamaModelName.QWEN2_5_0_5B: "qwen2.5:0.5b",
    OllamaModelName.QWEN2_5_14B: "qwen2.5:14b",
    OllamaModelName.MATHSTRAL_7B: "mathstral:latest",
    OllamaModelName.QWEN2_MATH_1_5B: "qwen2-math:1.5b",
    OllamaModelName.MISTRALV0_3_7B: "mistralv0.3",
    OllamaModelName.LLAMA3_2_3B: "llama3.2:latest",
    OllamaModelName.LLAMA3_1_8B: "llama3.1:latest",

    FakeModelName.FAKE: "fake",
}

ModelT: TypeAlias = ChatOpenAI | ChatAnthropic | ChatGoogleGenerativeAI | ChatGroq | ChatBedrock | AzureChatOpenAI | ChatOllama


@cache
def get_model(model_name: AllModelEnum, /) -> ModelT:
    # NOTE: models with streaming=True will send tokens as they are generated
    # if the /stream endpoint is called with stream_tokens=True (the default)
    api_model_name = _MODEL_TABLE.get(model_name)
    if not api_model_name:
        raise ValueError(f"Unsupported model: {model_name}")

    if model_name in AzureOpenAIModelName:
        return AzureChatOpenAI(
            model=api_model_name,
            deployment_name=settings.AZURE_OPENAI_DEPLOYMENT_NAME,
            api_key=settings.AZURE_OPENAI_API_KEY,
            azure_endpoint=settings.AZURE_OPENAI_ENDPOINT,
            api_version=settings.AZURE_OPENAI_API_VERSION,
            temperature=0.5,
            streaming=True
        )
    
    if model_name in OpenAIModelName:
        return ChatOpenAI(model=api_model_name, temperature=0.5, streaming=True)
    
    if model_name in DeepseekModelName:
        return ChatOpenAI(
            model=api_model_name,
            temperature=0.5,
            streaming=True,
            openai_api_base="https://api.deepseek.com",
            openai_api_key=settings.DEEPSEEK_API_KEY,
        )
    
    if model_name in AnthropicModelName:
        return ChatAnthropic(model=api_model_name, temperature=0.5, streaming=True)
    
    if model_name in GoogleModelName:
        return ChatGoogleGenerativeAI(model=api_model_name, temperature=0.5, streaming=True)
    
    if model_name in GroqModelName:
        if model_name == GroqModelName.LLAMA_GUARD_3_8B:
            return ChatGroq(model=api_model_name, temperature=0.0)
        return ChatGroq(model=api_model_name, temperature=0.5)
    
    if model_name in AWSModelName:
        return ChatBedrock(model_id=api_model_name, temperature=0.5)
    
    if model_name in AWSModelName:
        return ChatBedrock(model_id=api_model_name, temperature=0.5)
    
    if model_name in OllamaModelName:
        if settings.OLLAMA_BASE_URL:
            chat_ollama = ChatOllama(model=api_model_name, temperature=0.5, base_url=settings.OLLAMA_BASE_URL)
        else:
            chat_ollama = ChatOllama(model=settings.api_model_name, temperature=0.5)
        return chat_ollama
    
    if model_name in FakeModelName:
        return FakeListChatModel(responses=["This is a test response from the fake model."])
