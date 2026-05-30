"""Azure OpenAI provider implementation (OpenAI-compatible chat completions).

Supports two authentication modes:
1. API key via ``AZURE_OPENAI_API_KEY``
2. Azure Default Credentials (``azure-identity``) — used when no API key is set
"""

from __future__ import annotations

from typing import Any

import httpx
from loguru import logger
from openai import AsyncAzureOpenAI

from providers.base import BaseProvider, ProviderConfig
from providers.openai_compat import OpenAIChatTransport
from providers.rate_limit import GlobalRateLimiter

from .request import build_request_body


class AzureOpenAIProvider(OpenAIChatTransport):
    """Azure OpenAI API using Azure-hosted deployments."""

    def __init__(
        self,
        config: ProviderConfig,
        *,
        azure_endpoint: str,
        azure_deployment: str,
        azure_api_version: str,
        azure_scope: str = "https://cognitiveservices.azure.com/.default",
        azure_api_key: str = "",
    ):
        # Skip OpenAIChatTransport.__init__ — we need AsyncAzureOpenAI, not AsyncOpenAI.
        BaseProvider.__init__(self, config)

        self._provider_name = "AZURE_OPENAI"
        self._api_key = azure_api_key
        self._base_url = azure_endpoint.rstrip("/")
        self._azure_deployment = azure_deployment
        self._azure_credential: Any = None

        self._global_rate_limiter = GlobalRateLimiter.get_scoped_instance(
            "azure_openai",
            rate_limit=config.rate_limit,
            rate_window=config.rate_window,
            max_concurrency=config.max_concurrency,
        )

        timeout = httpx.Timeout(
            config.http_read_timeout,
            connect=config.http_connect_timeout,
            read=config.http_read_timeout,
            write=config.http_write_timeout,
        )

        http_client = None
        if config.proxy:
            http_client = httpx.AsyncClient(proxy=config.proxy, timeout=timeout)

        azure_kwargs: dict[str, Any] = {
            "azure_endpoint": azure_endpoint,
            "azure_deployment": azure_deployment,
            "api_version": azure_api_version,
            "max_retries": 0,
            "timeout": timeout,
        }
        if http_client is not None:
            azure_kwargs["http_client"] = http_client

        if azure_api_key:
            azure_kwargs["api_key"] = azure_api_key
            logger.info("AZURE_OPENAI: using API key authentication")
        else:
            try:
                from azure.identity.aio import (
                    DefaultAzureCredential,
                    get_bearer_token_provider,
                )
            except ImportError as exc:
                raise ImportError(
                    "azure-identity is required for Azure Default Credential auth. "
                    "Install it with: uv pip install azure-identity"
                ) from exc
            credential = DefaultAzureCredential()
            token_provider = get_bearer_token_provider(credential, azure_scope)
            azure_kwargs["azure_ad_token_provider"] = token_provider
            self._azure_credential = credential
            logger.info(
                "AZURE_OPENAI: using DefaultAzureCredential (scope={})", azure_scope
            )

        self._client = AsyncAzureOpenAI(**azure_kwargs)

    async def cleanup(self) -> None:
        """Release HTTP client and Azure credential resources."""
        await super().cleanup()
        if self._azure_credential is not None:
            await self._azure_credential.close()

    async def list_model_ids(self) -> frozenset[str]:
        """Return the configured deployment name as the available model."""
        return frozenset({self._azure_deployment})

    def _build_request_body(
        self, request: Any, thinking_enabled: bool | None = None
    ) -> dict:
        return build_request_body(
            request,
            thinking_enabled=self._is_thinking_enabled(request, thinking_enabled),
        )
