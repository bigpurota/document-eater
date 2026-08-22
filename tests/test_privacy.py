from __future__ import annotations

import socket

import pytest

from document_eater.privacy import (
    STRICT_OFFLINE_ENVIRONMENT,
    _is_loopback_address,
    _require_loopback,
)


def test_strict_offline_environment_disables_runtime_downloads_and_telemetry():
    assert STRICT_OFFLINE_ENVIRONMENT == {
        "DOCUMENT_EATER_STRICT_OFFLINE": "1",
        "UV_OFFLINE": "1",
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
        "HF_DATASETS_OFFLINE": "1",
        "HF_HUB_DISABLE_TELEMETRY": "1",
        "DO_NOT_TRACK": "1",
    }


@pytest.mark.parametrize(
    "address",
    [
        ("127.0.0.1", 8080),
        ("::1", 8080, 0, 0),
        ("localhost", 8080),
        "/tmp/document-eater.sock",
    ],
)
def test_offline_guard_allows_only_local_addresses(address):
    assert _is_loopback_address(address)
    _require_loopback(address)


@pytest.mark.parametrize(
    "address",
    [("8.8.8.8", 443), ("192.168.1.10", 8080), ("inference.example.com", 443)],
)
def test_offline_guard_rejects_non_loopback_addresses(address):
    with pytest.raises(OSError, match="Strict offline"):
        _require_loopback(address)


def test_guard_covers_the_socket_apis_used_by_python_http_clients():
    assert callable(socket.create_connection)
    assert callable(socket.socket.connect)
    assert callable(socket.socket.connect_ex)
    assert callable(socket.getaddrinfo)
    assert callable(socket.socket.sendto)
