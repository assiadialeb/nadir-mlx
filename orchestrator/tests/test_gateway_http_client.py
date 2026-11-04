"""Tests for the shared gateway httpx client lifecycle."""

from __future__ import annotations

from unittest import IsolatedAsyncioTestCase
from unittest.mock import AsyncMock, MagicMock, patch

from orchestrator.gateway import http_client


class GatewayHttpClientLifecycleTests(IsolatedAsyncioTestCase):
    def tearDown(self) -> None:
        http_client._shared_client = None

    def test_get_http_client_creates_shared_instance(self) -> None:
        http_client._shared_client = None
        with patch("orchestrator.gateway.http_client.httpx.AsyncClient") as mock_cls:
            mock_cls.return_value = MagicMock(is_closed=False)
            first = http_client.get_http_client()
            second = http_client.get_http_client()
        self.assertIs(first, second)
        mock_cls.assert_called_once()

    async def test_lifespan_starts_and_shuts_down_client(self) -> None:
        http_client._shared_client = None
        fake = AsyncMock()
        fake.is_closed = False
        fake.aclose = AsyncMock()
        with patch(
            "orchestrator.gateway.http_client.httpx.AsyncClient",
            return_value=fake,
        ):
            async with http_client.gateway_lifespan(MagicMock()):
                self.assertIs(http_client._shared_client, fake)
            fake.aclose.assert_awaited_once()
            self.assertIsNone(http_client._shared_client)
