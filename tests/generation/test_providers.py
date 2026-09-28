import httpx
import pytest

from app.agent.provider import AgentMessage, AgentProvider, AgentToolDefinition
from app.core.config import Settings
from app.generation.errors import LLMConfigurationError
from app.generation.providers.factory import create_llm_provider
from app.generation.providers.ollama import OllamaProvider
from app.generation.providers.openai_compatible import OpenAICompatibleProvider


def test_no_provider_is_valid_for_non_generation_paths() -> None:
    assert create_llm_provider(Settings(llm_provider=None)) is None


def test_openai_compatible_provider_construction() -> None:
    provider = create_llm_provider(
        Settings(
            llm_provider="openai_compatible",
            openai_api_key="placeholder-test-key",
            openai_model="test-model",
        )
    )
    assert isinstance(provider, OpenAICompatibleProvider)
    assert provider.model_name == "test-model"


def test_ollama_provider_construction() -> None:
    provider = create_llm_provider(Settings(llm_provider="ollama", ollama_model="small-test"))
    assert isinstance(provider, OllamaProvider)
    assert provider.model_name == "small-test"


def test_selected_provider_requires_configuration() -> None:
    with pytest.raises(LLMConfigurationError):
        create_llm_provider(Settings(llm_provider="openai_compatible"))


@pytest.mark.asyncio
async def test_openai_compatible_structured_tool_call() -> None:
    async def handler(request: httpx.Request) -> httpx.Response:
        payload = request.read().decode()
        assert '"tools"' in payload
        return httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "message": {
                            "content": None,
                            "tool_calls": [
                                {
                                    "id": "call-1",
                                    "function": {
                                        "name": "list_document_versions",
                                        "arguments": (
                                            '{"logical_document_id":'
                                            '"00000000-0000-0000-0000-000000000001"}'
                                        ),
                                    },
                                }
                            ],
                        }
                    }
                ],
                "usage": {"prompt_tokens": 10, "completion_tokens": 2, "total_tokens": 12},
            },
        )

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    provider = OpenAICompatibleProvider(
        base_url="https://example.test/v1",
        api_key="test",
        model="test-model",
        timeout_seconds=1,
        client=client,
    )
    assert isinstance(provider, AgentProvider)

    decision = await provider.decide(
        system_prompt="system",
        messages=[AgentMessage(role="user", content="list versions")],
        tools=[
            AgentToolDefinition(
                "list_document_versions", "list", {"type": "object", "properties": {}}
            )
        ],
        temperature=0,
        max_output_tokens=100,
    )

    assert decision.tool_call is not None
    assert decision.tool_call.name == "list_document_versions"
    assert decision.token_usage is not None and decision.token_usage.total_tokens == 12
    await client.aclose()


@pytest.mark.asyncio
async def test_ollama_structured_final_answer() -> None:
    async def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "message": {"content": "Grounded final answer.", "tool_calls": []},
                "prompt_eval_count": 8,
                "eval_count": 4,
            },
        )

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    provider = OllamaProvider(
        base_url="http://example.test",
        model="test-model",
        timeout_seconds=1,
        client=client,
    )
    assert isinstance(provider, AgentProvider)

    decision = await provider.decide(
        system_prompt="system",
        messages=[AgentMessage(role="user", content="answer")],
        tools=[],
        temperature=0,
        max_output_tokens=100,
    )

    assert decision.final_answer == "Grounded final answer."
    assert decision.token_usage is not None and decision.token_usage.total_tokens == 12
    await client.aclose()
