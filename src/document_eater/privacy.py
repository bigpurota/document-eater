from __future__ import annotations

import ipaddress
import os
import socket
from typing import Any

STRICT_OFFLINE_ENVIRONMENT = {
    "DOCUMENT_EATER_STRICT_OFFLINE": "1",
    "UV_OFFLINE": "1",
    "HF_HUB_OFFLINE": "1",
    "TRANSFORMERS_OFFLINE": "1",
    "HF_DATASETS_OFFLINE": "1",
    "HF_HUB_DISABLE_TELEMETRY": "1",
    "DO_NOT_TRACK": "1",
}

_LOOPBACK_NAMES = {"localhost", "localhost.localdomain"}
_guard_installed = False
_original_create_connection = socket.create_connection
_original_connect = socket.socket.connect
_original_connect_ex = socket.socket.connect_ex
_original_getaddrinfo = socket.getaddrinfo
_original_sendto = socket.socket.sendto


def strict_offline_requested() -> bool:
    return os.environ.get("DOCUMENT_EATER_STRICT_OFFLINE", "").casefold() in {
        "1",
        "true",
        "yes",
        "on",
    }


def _is_loopback_address(address: Any) -> bool:
    if isinstance(address, str):
        # AF_UNIX sockets use filesystem paths and never leave the machine.
        return True
    if not isinstance(address, tuple) or not address:
        return False
    host = str(address[0]).rstrip(".").casefold()
    if host in _LOOPBACK_NAMES:
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


def _require_loopback(address: Any) -> None:
    if not _is_loopback_address(address):
        raise OSError(
            "Strict offline mode blocked a non-loopback network connection. "
            "Use the explicitly enabled remote profile only when external processing is approved."
        )


def enable_strict_offline() -> None:
    """Disable model/package networking and fail closed on Python TCP connections."""
    global _guard_installed
    for name, value in STRICT_OFFLINE_ENVIRONMENT.items():
        os.environ[name] = value
    if _guard_installed:
        return

    def guarded_create_connection(address: Any, *args: Any, **kwargs: Any):
        _require_loopback(address)
        return _original_create_connection(address, *args, **kwargs)

    def guarded_connect(sock: socket.socket, address: Any):
        _require_loopback(address)
        return _original_connect(sock, address)

    def guarded_connect_ex(sock: socket.socket, address: Any):
        _require_loopback(address)
        return _original_connect_ex(sock, address)

    def guarded_getaddrinfo(host: Any, port: Any, *args: Any, **kwargs: Any):
        _require_loopback((host, port))
        return _original_getaddrinfo(host, port, *args, **kwargs)

    def guarded_sendto(sock: socket.socket, data: Any, *args: Any):
        if not args:
            raise OSError("Strict offline mode blocked a datagram without a local address")
        _require_loopback(args[-1])
        return _original_sendto(sock, data, *args)

    socket.create_connection = guarded_create_connection
    socket.socket.connect = guarded_connect
    socket.socket.connect_ex = guarded_connect_ex
    socket.getaddrinfo = guarded_getaddrinfo
    socket.socket.sendto = guarded_sendto
    _guard_installed = True
