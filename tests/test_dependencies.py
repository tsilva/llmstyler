from __future__ import annotations

import asyncio
from importlib.metadata import version

import aiohttp
import pytest
from packaging.version import Version

from llmstyler.restyle import parse_restyle_response, selected_row_indexes


async def _serve_once(response: bytes) -> tuple[asyncio.AbstractServer, str]:
    async def handler(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        try:
            await reader.readuntil(b"\r\n\r\n")
            writer.write(response)
            await writer.drain()
        finally:
            writer.close()
            await writer.wait_closed()

    server = await asyncio.start_server(handler, "127.0.0.1", 0)
    port = server.sockets[0].getsockname()[1]
    return server, f"http://127.0.0.1:{port}/"


def test_aiohttp_contains_malformed_response_parser_fix() -> None:
    assert Version(version("aiohttp")) >= Version("3.14.3")


def test_malformed_chunked_response_is_rejected_without_parser_crash() -> None:
    async def scenario() -> None:
        response = (
            b"HTTP/1.1 200 OK\r\n"
            b"Transfer-Encoding: chunked\r\n"
            b"Connection: close\r\n\r\n"
            b"FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF\r\ninvalid\r\n0\r\n\r\n"
        )
        server, url = await _serve_once(response)
        try:
            async with aiohttp.ClientSession() as session:
                with pytest.raises((aiohttp.ClientPayloadError, aiohttp.ClientResponseError)):
                    async with session.get(url) as result:
                        await result.read()
        finally:
            server.close()
            await server.wait_closed()

    asyncio.run(scenario())


def test_valid_chunked_response_and_style_logic_remain_intact() -> None:
    async def scenario() -> None:
        response = (
            b"HTTP/1.1 200 OK\r\n"
            b"Transfer-Encoding: chunked\r\n"
            b"Connection: close\r\n\r\n"
            b"5\r\nhello\r\n0\r\n\r\n"
        )
        server, url = await _serve_once(response)
        try:
            async with aiohttp.ClientSession() as session, session.get(url) as result:
                assert await result.text() == "hello"
        finally:
            server.close()
            await server.wait_closed()

    asyncio.run(scenario())
    rows = [{"restyle": True}, {"restyle": False}, {"restyle": True}]
    assert selected_row_indexes(rows, sample=None, seed=7) == {0, 2}
    assert parse_restyle_response('{"restyled_content": "  polished answer  "}') == "polished answer"
