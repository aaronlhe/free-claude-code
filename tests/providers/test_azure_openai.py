"""Tests for Azure OpenAI provider."""

from contextlib import asynccontextmanager
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from providers.azure_openai import AzureOpenAIProvider
from providers.base import ProviderConfig


class MockMessage:
    def __init__(self, role, content):
        self.role = role
        self.content = content


class MockRequest:
    def __init__(self, **kwargs):
        self.model = "gpt-4o"
        self.messages = [MockMessage("user", "Hello")]
        self.max_tokens = 100
        self.temperature = 0.5
        self.top_p = 0.9
        self.system = "System prompt"
        self.stop_sequences = None
        self.tools = []
        self.thinking = MagicMock()
        self.thinking.enabled = True
        for key, value in kwargs.items():
            setattr(self, key, value)


@pytest.fixture
def azure_config():
    return ProviderConfig(
        api_key="test-azure-key",
        rate_limit=10,
        rate_window=60,
        enable_thinking=True,
    )


@pytest.fixture(autouse=True)
def mock_rate_limiter():
    """Mock the global rate limiter to prevent waiting."""

    @asynccontextmanager
    async def _slot():
        yield

    with (
        patch("providers.openai_compat.GlobalRateLimiter") as mock_rl,
        patch("providers.azure_openai.client.GlobalRateLimiter") as mock_rl2,
    ):
        instance = MagicMock()

        async def _passthrough(fn, *args, **kwargs):
            return await fn(*args, **kwargs)

        instance.execute_with_retry = AsyncMock(side_effect=_passthrough)
        instance.concurrency_slot.side_effect = _slot
        mock_rl.get_scoped_instance.return_value = instance
        mock_rl2.get_scoped_instance.return_value = instance
        yield instance


@pytest.fixture
def azure_provider(azure_config):
    with patch("providers.azure_openai.client.AsyncAzureOpenAI"):
        return AzureOpenAIProvider(
            azure_config,
            azure_endpoint="https://myresource.openai.azure.com",
            azure_deployment="gpt-4o",
            azure_api_version="2025-04-01-preview",
            azure_api_key="test-azure-key",
        )


def test_init_with_api_key(azure_config):
    """Provider initializes with API key auth."""
    with patch("providers.azure_openai.client.AsyncAzureOpenAI") as mock_azure:
        provider = AzureOpenAIProvider(
            azure_config,
            azure_endpoint="https://myresource.openai.azure.com",
            azure_deployment="gpt-4o",
            azure_api_version="2025-04-01-preview",
            azure_api_key="test-azure-key",
        )
        assert provider._api_key == "test-azure-key"
        assert provider._base_url == "https://myresource.openai.azure.com"
        assert provider._azure_deployment == "gpt-4o"
        assert provider._azure_credential is None
        mock_azure.assert_called_once()
        call_kwargs = mock_azure.call_args.kwargs
        assert call_kwargs["api_key"] == "test-azure-key"
        assert call_kwargs["azure_endpoint"] == "https://myresource.openai.azure.com"
        assert call_kwargs["azure_deployment"] == "gpt-4o"
        assert call_kwargs["api_version"] == "2025-04-01-preview"
        assert "azure_ad_token_provider" not in call_kwargs


def test_init_with_default_credentials(azure_config):
    """Provider initializes with DefaultAzureCredential when no API key."""
    mock_credential = MagicMock()
    mock_token_provider = MagicMock()

    with (
        patch("providers.azure_openai.client.AsyncAzureOpenAI") as mock_azure,
        patch(
            "azure.identity.aio.DefaultAzureCredential",
            return_value=mock_credential,
        ),
        patch(
            "azure.identity.aio.get_bearer_token_provider",
            return_value=mock_token_provider,
        ),
    ):
        provider = AzureOpenAIProvider(
            azure_config,
            azure_endpoint="https://myresource.openai.azure.com",
            azure_deployment="gpt-4o",
            azure_api_version="2025-04-01-preview",
            azure_api_key="",
        )
        assert provider._azure_credential is mock_credential
        call_kwargs = mock_azure.call_args.kwargs
        assert "api_key" not in call_kwargs
        assert call_kwargs["azure_ad_token_provider"] is mock_token_provider


def test_init_with_custom_scope(azure_config):
    """Custom scope is passed to get_bearer_token_provider."""
    custom_scope = "https://custom.scope/.default"
    mock_credential = MagicMock()

    with (
        patch("providers.azure_openai.client.AsyncAzureOpenAI"),
        patch(
            "azure.identity.aio.DefaultAzureCredential",
            return_value=mock_credential,
        ) as mock_cred_cls,
        patch(
            "azure.identity.aio.get_bearer_token_provider",
        ) as mock_token_fn,
    ):
        AzureOpenAIProvider(
            azure_config,
            azure_endpoint="https://myresource.openai.azure.com",
            azure_deployment="gpt-4o",
            azure_api_version="2025-04-01-preview",
            azure_scope=custom_scope,
            azure_api_key="",
        )
        mock_cred_cls.assert_called_once()
        mock_token_fn.assert_called_once_with(mock_credential, custom_scope)


def test_init_raises_on_missing_azure_identity(azure_config):
    """ImportError is raised when azure-identity is not installed and no API key."""
    with (
        patch("providers.azure_openai.client.AsyncAzureOpenAI"),
        patch.dict("sys.modules", {"azure.identity.aio": None}),
        pytest.raises(ImportError, match="azure-identity"),
    ):
        AzureOpenAIProvider(
            azure_config,
            azure_endpoint="https://myresource.openai.azure.com",
            azure_deployment="gpt-4o",
            azure_api_version="2025-04-01-preview",
            azure_api_key="",
        )


def test_build_request_body_basic(azure_provider):
    """Basic request body conversion works."""
    req = MockRequest()
    body = azure_provider._build_request_body(req)

    assert body["model"] == "gpt-4o"
    assert "max_completion_tokens" in body


def test_build_request_body_global_disable_blocks_reasoning(azure_config):
    """Thinking disabled globally blocks reasoning_content replay."""
    disabled_config = ProviderConfig(
        api_key="test-azure-key",
        rate_limit=10,
        rate_window=60,
        enable_thinking=False,
    )
    with patch("providers.azure_openai.client.AsyncAzureOpenAI"):
        provider = AzureOpenAIProvider(
            disabled_config,
            azure_endpoint="https://myresource.openai.azure.com",
            azure_deployment="gpt-4o",
            azure_api_version="2025-04-01-preview",
            azure_api_key="test-key",
        )
    req = MockRequest()
    body = provider._build_request_body(req)

    roles = [m.get("role") for m in body.get("messages", [])]
    assert "assistant_reasoning_content" not in roles


def test_build_request_body_max_completion_tokens_preferred(azure_provider):
    """max_completion_tokens takes precedence over max_tokens."""
    with patch(
        "providers.azure_openai.request.build_base_request_body"
    ) as mock_convert:
        mock_convert.return_value = {
            "model": "gpt-4o",
            "messages": [{"role": "user", "content": "x"}],
            "max_completion_tokens": 77,
            "max_tokens": 999,
        }
        body = azure_provider._build_request_body(MockRequest())

    assert body["max_completion_tokens"] == 77
    assert "max_tokens" not in body


def test_build_request_body_extra_body(azure_provider):
    """extra_body from request is preserved."""
    req = MockRequest(extra_body={"metadata": {"user": "u1"}})
    body = azure_provider._build_request_body(req)

    eb = body.get("extra_body")
    assert isinstance(eb, dict)
    assert eb.get("metadata") == {"user": "u1"}


@pytest.mark.asyncio
async def test_list_model_ids_returns_deployment(azure_provider):
    """list_model_ids returns the configured deployment name."""
    model_ids = await azure_provider.list_model_ids()
    assert model_ids == frozenset({"gpt-4o"})


@pytest.mark.asyncio
async def test_stream_response_text(azure_provider):
    """Text content deltas are emitted as text blocks."""
    req = MockRequest()

    mock_chunk = MagicMock()
    mock_chunk.choices = [
        MagicMock(
            delta=MagicMock(
                content="Hello from Azure!",
                reasoning_content=None,
                tool_calls=None,
            ),
            finish_reason="stop",
        )
    ]
    mock_chunk.usage = MagicMock(completion_tokens=5, prompt_tokens=10)

    async def mock_stream():
        yield mock_chunk

    with patch.object(
        azure_provider._client.chat.completions, "create", new_callable=AsyncMock
    ) as mock_create:
        mock_create.return_value = mock_stream()

        events = [event async for event in azure_provider.stream_response(req)]

        assert any(
            '"text_delta"' in event and "Hello from Azure!" in event for event in events
        )


@pytest.mark.asyncio
async def test_cleanup_closes_credential(azure_config):
    """cleanup() closes the Azure credential when using DefaultAzureCredential."""
    mock_credential = MagicMock()
    mock_credential.close = AsyncMock()

    with (
        patch("providers.azure_openai.client.AsyncAzureOpenAI") as mock_azure,
        patch(
            "azure.identity.aio.DefaultAzureCredential",
            return_value=mock_credential,
        ),
        patch("azure.identity.aio.get_bearer_token_provider"),
    ):
        mock_client = MagicMock()
        mock_client.close = AsyncMock()
        mock_azure.return_value = mock_client

        provider = AzureOpenAIProvider(
            azure_config,
            azure_endpoint="https://myresource.openai.azure.com",
            azure_deployment="gpt-4o",
            azure_api_version="2025-04-01-preview",
            azure_api_key="",
        )

        await provider.cleanup()
        mock_client.close.assert_awaited_once()
        mock_credential.close.assert_awaited_once()


@pytest.mark.asyncio
async def test_cleanup_no_credential(azure_provider):
    """cleanup() works when using API key auth (no credential to close)."""
    azure_provider._client = MagicMock()
    azure_provider._client.close = AsyncMock()

    await azure_provider.cleanup()
    azure_provider._client.close.assert_awaited_once()


# ---- Registry integration tests ----


def test_registry_factory_validates_endpoint():
    """Factory raises when AZURE_OPENAI_ENDPOINT is empty."""
    from providers.exceptions import AuthenticationError
    from providers.registry import create_provider

    with (
        patch("config.settings.Settings"),
        pytest.raises(AuthenticationError, match="AZURE_OPENAI_ENDPOINT"),
    ):
        settings = MagicMock()
        settings.azure_openai_endpoint = ""
        settings.azure_openai_deployment = "gpt-4o"
        settings.azure_openai_api_key = ""
        settings.azure_openai_api_version = "2025-04-01-preview"
        settings.azure_openai_scope = "https://cognitiveservices.azure.com/.default"
        settings.azure_openai_proxy = ""
        settings.provider_rate_limit = 10
        settings.provider_rate_window = 60
        settings.provider_max_concurrency = 5
        settings.http_read_timeout = 120.0
        settings.http_write_timeout = 10.0
        settings.http_connect_timeout = 10.0
        settings.enable_model_thinking = True
        settings.log_raw_sse_events = False
        settings.log_api_error_tracebacks = False
        create_provider("azure_openai", settings)


def test_registry_factory_validates_deployment():
    """Factory raises when AZURE_OPENAI_DEPLOYMENT is empty."""
    from providers.exceptions import AuthenticationError
    from providers.registry import create_provider

    with pytest.raises(AuthenticationError, match="AZURE_OPENAI_DEPLOYMENT"):
        settings = MagicMock()
        settings.azure_openai_endpoint = "https://myresource.openai.azure.com"
        settings.azure_openai_deployment = ""
        settings.azure_openai_api_key = "test-key"
        settings.azure_openai_api_version = "2025-04-01-preview"
        settings.azure_openai_scope = "https://cognitiveservices.azure.com/.default"
        settings.azure_openai_proxy = ""
        settings.provider_rate_limit = 10
        settings.provider_rate_window = 60
        settings.provider_max_concurrency = 5
        settings.http_read_timeout = 120.0
        settings.http_write_timeout = 10.0
        settings.http_connect_timeout = 10.0
        settings.enable_model_thinking = True
        settings.log_raw_sse_events = False
        settings.log_api_error_tracebacks = False
        create_provider("azure_openai", settings)
