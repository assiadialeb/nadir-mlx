"""Tests for gateway middleware helpers."""

from __future__ import annotations

from django.test import SimpleTestCase
from fastapi import FastAPI
from starlette.middleware.cors import CORSMiddleware

from orchestrator.gateway.middleware import (
    cors_allow_credentials,
    install_gateway_middleware,
    parse_cors_origins,
)


class GatewayMiddlewareTests(SimpleTestCase):
    def test_parse_cors_origins(self) -> None:
        self.assertEqual(
            parse_cors_origins(" https://a.example ,https://b.example, "),
            ["https://a.example", "https://b.example"],
        )
        self.assertEqual(parse_cors_origins(""), [])

    def test_cors_allow_credentials_never_true(self) -> None:
        self.assertFalse(cors_allow_credentials(["*"]))
        self.assertFalse(cors_allow_credentials(["https://app.example"]))
        self.assertFalse(cors_allow_credentials([]))

    def test_install_cors_collapses_star_and_disables_credentials(self) -> None:
        app = FastAPI()
        install_gateway_middleware(app, cors_origins=["https://a.example", "*"])
        cors = [
            middleware.cls
            for middleware in app.user_middleware
            if middleware.cls is CORSMiddleware
        ]
        self.assertEqual(len(cors), 1)
        kwargs = next(
            middleware.kwargs
            for middleware in app.user_middleware
            if middleware.cls is CORSMiddleware
        )
        self.assertEqual(kwargs["allow_origins"], ["*"])
        self.assertFalse(kwargs["allow_credentials"])

    def test_cors_middleware_is_outermost(self) -> None:
        app = FastAPI()
        install_gateway_middleware(app, cors_origins=["https://app.example"])
        self.assertEqual(app.user_middleware[0].cls, CORSMiddleware)
