"""
Real-time monitoring server for TelegramLangBot.

Provides an HTTP server with SSE (Server-Sent Events) for a live
dashboard showing incoming requests, agent reviews, user activity,
conversation history, and logs.
"""

from __future__ import annotations

import datetime
import json
import logging
import queue
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

logger = logging.getLogger(__name__)


def _load_html() -> str:
    html_path = Path(__file__).parent / "monitor.html"

    if html_path.exists():
        return html_path.read_text(encoding="utf-8")

    return (
        "<!DOCTYPE html><html><head><title>Monitor</title></head>"
        "<body><h1>Monitor HTML not found</h1></body></html>"
    )


class _SSEClient:
    __slots__ = ("queue",)

    def __init__(self, wfile: Any) -> None:
        self.queue: queue.Queue[str] = queue.Queue(maxsize=500)


class MonitorServer:
    """HTTP server that serves a live monitoring dashboard via SSE."""

    def __init__(self, host: str = "0.0.0.0", port: int = 8080) -> None:
        self._host = host
        self._port = port
        self._httpd: ThreadingHTTPServer | None = None
        self._thread: threading.Thread | None = None
        self._clients: list[_SSEClient] = []
        self._lock = threading.Lock()
        self._html = _load_html()
        self._conversations: dict[int, list[dict[str, Any]]] = {}
        self._token_usage: dict[int, dict[str, Any]] = {}
        self._user_restrictions: dict[int, dict[str, Any]] = {}
        self._token_history: list[dict[str, Any]] = []
        self._agent_llm_history: list[dict[str, Any]] = []
        self._agent_step_history: list[dict[str, Any]] = []

    def start(self) -> None:
        """Start the monitor server in a background thread."""

        if self._httpd is not None:
            logger.warning("Monitor server is already running")
            return

        html_content = self._html
        srv = self

        class _Handler(BaseHTTPRequestHandler):
            def log_message(self, format: str, *args: Any) -> None:
                pass

            def handle_error(
                self,
                request: Any,
                client_address: Any,
            ) -> None:
                exc_type, _, _ = sys.exc_info()

                if exc_type in (
                    ConnectionAbortedError,
                    ConnectionResetError,
                    BrokenPipeError,
                ):
                    return

                super().handle_error(request, client_address)

            def handle_one_request(self) -> None:
                try:
                    super().handle_one_request()
                except (
                    ConnectionAbortedError,
                    ConnectionResetError,
                    BrokenPipeError,
                ):
                    return

            def _send_json(
                self,
                data: Any,
                code: int = 200,
            ) -> None:
                try:
                    self.send_response(code)
                    self.send_header(
                        "Content-Type",
                        "application/json",
                    )
                    self.send_header(
                        "Access-Control-Allow-Origin",
                        "*",
                    )
                    self.send_header(
                        "Cache-Control",
                        "no-cache",
                    )
                    self.end_headers()

                    body = json.dumps(
                        data,
                        ensure_ascii=False,
                        indent=2,
                        default=str,
                    )

                    self.wfile.write(body.encode("utf-8"))
                    self.wfile.flush()

                except (
                    ConnectionAbortedError,
                    ConnectionResetError,
                    BrokenPipeError,
                ):
                    pass

            def do_GET(self) -> None:
                parsed = urlparse(self.path)
                path = parsed.path
                qs = parse_qs(parsed.query)

                if path == "/" or path == "/index.html":
                    try:
                        self.send_response(200)
                        self.send_header(
                            "Content-Type",
                            "text/html; charset=utf-8",
                        )
                        self.send_header(
                            "Cache-Control",
                            "no-cache",
                        )
                        self.end_headers()

                        self.wfile.write(
                            html_content.encode("utf-8"),
                        )
                        self.wfile.flush()

                    except (
                        ConnectionAbortedError,
                        ConnectionResetError,
                        BrokenPipeError,
                    ):
                        pass

                elif path == "/events":
                    client = _SSEClient(self.wfile)

                    try:
                        self.send_response(200)
                        self.send_header(
                            "Content-Type",
                            "text/event-stream; charset=utf-8",
                        )
                        self.send_header(
                            "Cache-Control",
                            "no-cache, no-store, must-revalidate",
                        )
                        self.send_header(
                            "Connection",
                            "keep-alive",
                        )
                        self.send_header(
                            "Access-Control-Allow-Origin",
                            "*",
                        )
                        self.send_header(
                            "X-Accel-Buffering",
                            "no",
                        )
                        self.end_headers()
                        self.wfile.flush()

                        srv._add_client(client)

                        try:
                            self.wfile.write(b'event: connected\ndata: {"type":"connected"}\n\n')
                            self.wfile.flush()

                            while True:
                                try:
                                    event = client.queue.get(
                                        timeout=15,
                                    )

                                    self.wfile.write(
                                        event.encode("utf-8"),
                                    )
                                    self.wfile.flush()

                                except queue.Empty:
                                    self.wfile.write(
                                        b": keepalive\n\n",
                                    )
                                    self.wfile.flush()

                        except (
                            BrokenPipeError,
                            ConnectionResetError,
                            ConnectionAbortedError,
                            OSError,
                        ):
                            pass

                        finally:
                            srv._remove_client(client)

                    except (
                        BrokenPipeError,
                        ConnectionResetError,
                        ConnectionAbortedError,
                        OSError,
                    ):
                        srv._remove_client(client)

                elif path == "/api/history":
                    try:
                        cid_str = qs.get("chat_id", [""])[0]
                        uid_str = qs.get("user_id", [""])[0]

                        with srv._lock:
                            if cid_str:
                                data = list(
                                    srv._conversations.get(
                                        int(cid_str),
                                        [],
                                    )
                                )

                            elif uid_str:
                                uid = int(uid_str)
                                filtered: dict[
                                    int,
                                    list[dict[str, Any]],
                                ] = {}

                                for chat_id, msgs in srv._conversations.items():
                                    matching = [m for m in msgs if m.get("user_id") == uid]

                                    if matching:
                                        filtered[chat_id] = matching

                                data = filtered

                            else:
                                data = {
                                    chat_id: list(msgs)
                                    for chat_id, msgs in srv._conversations.items()
                                }

                        self._send_json(data)

                    except (
                        ConnectionAbortedError,
                        ConnectionResetError,
                        BrokenPipeError,
                    ):
                        pass

                elif path == "/api/restrictions":
                    with srv._lock:
                        data = {
                            user_id: dict(restriction)
                            for user_id, restriction in srv._user_restrictions.items()
                        }

                    self._send_json(data)

                elif path == "/api/tokens":
                    with srv._lock:
                        data = {
                            "per_user": {
                                user_id: dict(usage) for user_id, usage in srv._token_usage.items()
                            },
                            "history": list(srv._token_history[-100:]),
                        }

                    self._send_json(data)

                elif path == "/api/agent_llm":
                    with srv._lock:
                        data = {
                            "history": list(srv._agent_llm_history[-200:]),
                            "count": len(srv._agent_llm_history),
                        }

                    self._send_json(data)

                elif path == "/api/agent_steps":
                    with srv._lock:
                        data = {
                            "history": list(srv._agent_step_history[-500:]),
                            "count": len(srv._agent_step_history),
                        }

                    self._send_json(data)

                else:
                    try:
                        self.send_response(404)
                        self.end_headers()
                    except (
                        ConnectionAbortedError,
                        ConnectionResetError,
                        BrokenPipeError,
                    ):
                        pass

            def do_POST(self) -> None:
                try:
                    parsed = urlparse(self.path)

                    length = int(
                        self.headers.get(
                            "Content-Length",
                            0,
                        )
                    )

                    body = self.rfile.read(length) if length else b"{}"

                    try:
                        data = json.loads(body)
                    except Exception:
                        data = {}

                    if parsed.path == "/api/restrict":
                        uid = data.get("user_id")
                        action = data.get(
                            "action",
                            "toggle",
                        )
                        limit = int(
                            data.get(
                                "token_limit",
                                0,
                            )
                        )

                        if uid is not None:
                            uid = int(uid)

                            with srv._lock:
                                if action == "toggle":
                                    cur = srv._user_restrictions.get(
                                        uid,
                                        {
                                            "blocked": False,
                                            "token_limit": 0,
                                        },
                                    )

                                    cur["blocked"] = not cur.get(
                                        "blocked",
                                        False,
                                    )

                                    srv._user_restrictions[uid] = cur

                                elif action == "set_limit":
                                    r = srv._user_restrictions.setdefault(
                                        uid,
                                        {
                                            "blocked": False,
                                            "token_limit": 0,
                                        },
                                    )

                                    r["token_limit"] = limit

                                elif action == "remove":
                                    srv._user_restrictions.pop(
                                        uid,
                                        None,
                                    )

                                restrictions = {
                                    user_id: dict(restriction)
                                    for user_id, restriction in srv._user_restrictions.items()
                                }

                        else:
                            with srv._lock:
                                restrictions = {
                                    user_id: dict(restriction)
                                    for user_id, restriction in srv._user_restrictions.items()
                                }

                        self._send_json(
                            {
                                "ok": True,
                                "restrictions": restrictions,
                            }
                        )

                    else:
                        try:
                            self.send_response(404)
                            self.end_headers()
                        except (
                            ConnectionAbortedError,
                            ConnectionResetError,
                            BrokenPipeError,
                        ):
                            pass

                except (
                    ConnectionAbortedError,
                    ConnectionResetError,
                    BrokenPipeError,
                    OSError,
                ):
                    pass

            def do_OPTIONS(self) -> None:
                try:
                    self.send_response(200)
                    self.send_header(
                        "Access-Control-Allow-Origin",
                        "*",
                    )
                    self.send_header(
                        "Access-Control-Allow-Methods",
                        "GET, POST, OPTIONS",
                    )
                    self.send_header(
                        "Access-Control-Allow-Headers",
                        "Content-Type",
                    )
                    self.send_header(
                        "Cache-Control",
                        "no-cache",
                    )
                    self.end_headers()

                except (
                    ConnectionAbortedError,
                    ConnectionResetError,
                    BrokenPipeError,
                ):
                    pass

        self._httpd = ThreadingHTTPServer(
            (self._host, self._port),
            _Handler,
        )

        self._httpd.daemon_threads = True

        self._thread = threading.Thread(
            target=self._httpd.serve_forever,
            daemon=True,
        )

        self._thread.start()

        logger.info(
            "Monitor server started on http://%s:%s",
            self._host,
            self._port,
        )

    def stop(self) -> None:
        """Stop the monitor server."""

        if self._httpd:
            self._httpd.shutdown()
            self._httpd.server_close()
            self._httpd = None

        if self._thread:
            self._thread.join(timeout=2)
            self._thread = None

        with self._lock:
            self._clients.clear()

    def _add_client(self, client: _SSEClient) -> None:
        with self._lock:
            self._clients.append(client)

    def _remove_client(self, client: _SSEClient) -> None:
        with self._lock:
            if client in self._clients:
                self._clients.remove(client)

    def _broadcast(
        self,
        event_type: str,
        data: dict[str, Any],
    ) -> None:
        payload = json.dumps(
            data,
            ensure_ascii=False,
            default=str,
        )

        message = f"event: {event_type}\n" f"data: {payload}\n\n"

        with self._lock:
            dead: list[_SSEClient] = []

            for client in self._clients:
                try:
                    client.queue.put_nowait(message)
                except queue.Full:
                    dead.append(client)

            for client in dead:
                if client in self._clients:
                    self._clients.remove(client)

    def send_request(
        self,
        chat_id: int,
        user_id: int,
        message: str,
        username: str = "",
    ) -> None:
        """Log an incoming user request."""

        self._store_conversation(
            chat_id,
            "user",
            message,
            user_id=user_id,
        )

        self._broadcast(
            "request",
            {
                "type": "request",
                "chat_id": chat_id,
                "user_id": user_id,
                "username": username,
                "message": message,
                "timestamp": datetime.datetime.now().isoformat(),
            },
        )

    def send_response(
        self,
        chat_id: int,
        response: str,
        response_length: int,
        cot_mode: bool = False,
    ) -> None:
        """Log a bot response."""

        self._store_conversation(
            chat_id,
            "assistant",
            response,
        )

        self._broadcast(
            "response",
            {
                "type": "response",
                "chat_id": chat_id,
                "response": response,
                "response_length": response_length,
                "cot_mode": cot_mode,
                "timestamp": datetime.datetime.now().isoformat(),
            },
        )

    def send_review(
        self,
        chat_id: int,
        passed: bool,
        requires_rewrite: bool,
        agent_reviews: list[dict[str, Any]],
        rewrite_feedback: str = "",
    ) -> None:
        """Log agent review results."""

        self._broadcast(
            "review",
            {
                "type": "review",
                "chat_id": chat_id,
                "passed": passed,
                "requires_rewrite": requires_rewrite,
                "agent_reviews": agent_reviews,
                "rewrite_feedback": rewrite_feedback,
                "timestamp": datetime.datetime.now().isoformat(),
            },
        )

    def send_error(
        self,
        chat_id: int,
        error: str,
    ) -> None:
        """Log an error event."""

        self._broadcast(
            "error_event",
            {
                "type": "error",
                "chat_id": chat_id,
                "error": error,
                "timestamp": datetime.datetime.now().isoformat(),
            },
        )

    def send_clear(
        self,
        chat_id: int,
    ) -> None:
        """Log a memory clear event."""

        self._broadcast(
            "clear",
            {
                "type": "clear",
                "chat_id": chat_id,
                "timestamp": datetime.datetime.now().isoformat(),
            },
        )

    def _store_conversation(
        self,
        chat_id: int,
        role: str,
        content: str,
        user_id: int = 0,
    ) -> None:
        with self._lock:
            convs = self._conversations.setdefault(
                chat_id,
                [],
            )

            convs.append(
                {
                    "role": role,
                    "content": content,
                    "user_id": user_id,
                    "timestamp": datetime.datetime.now().isoformat(),
                }
            )

            if len(convs) > 500:
                self._conversations[chat_id] = convs[-500:]

    def is_user_blocked(
        self,
        user_id: int,
    ) -> bool:
        with self._lock:
            r = self._user_restrictions.get(
                user_id,
                {},
            )

            if r.get("blocked", False):
                return True

            limit = r.get(
                "token_limit",
                0,
            )

            if limit > 0:
                used = self._token_usage.get(
                    user_id,
                    {},
                ).get(
                    "total_tokens",
                    0,
                )

                if used >= limit:
                    return True

            return False

    def send_agent_step(
        self,
        chat_id: int,
        agent_name: str,
        review_type: str,
        score: str,
        feedback: str,
        suggested_fix: str | None,
        step_number: int,
        total_steps: int,
        original_response: str,
    ) -> None:
        entry = {
            "chat_id": chat_id,
            "agent_name": agent_name,
            "review_type": review_type,
            "score": score,
            "feedback": feedback,
            "suggested_fix": suggested_fix,
            "step_number": step_number,
            "total_steps": total_steps,
            "original_response": original_response[:300],
            "timestamp": datetime.datetime.now().isoformat(),
        }

        with self._lock:
            self._agent_step_history.append(entry)

            if len(self._agent_step_history) > 500:
                self._agent_step_history = self._agent_step_history[-500:]

        self._broadcast(
            "agent_step",
            {
                "type": "agent_step",
                **entry,
            },
        )

    def send_agent_llm_activity(
        self,
        chat_id: int,
        agent_name: str,
        review_type: str,
        llm_score: str,
        llm_feedback: str,
        duration_ms: float,
        rule_based_score: str,
    ) -> None:
        entry = {
            "chat_id": chat_id,
            "agent_name": agent_name,
            "review_type": review_type,
            "llm_score": llm_score,
            "llm_feedback": llm_feedback,
            "duration_ms": round(
                duration_ms,
                1,
            ),
            "rule_based_score": rule_based_score,
            "timestamp": datetime.datetime.now().isoformat(),
        }

        with self._lock:
            self._agent_llm_history.append(entry)

            if len(self._agent_llm_history) > 500:
                self._agent_llm_history = self._agent_llm_history[-500:]

        self._broadcast(
            "agent_llm",
            {
                "type": "agent_llm",
                **entry,
            },
        )

    def send_token_usage(
        self,
        chat_id: int,
        user_id: int,
        prompt_tokens: int,
        completion_tokens: int,
        elapsed_ms: int,
    ) -> None:
        total = prompt_tokens + completion_tokens

        tps = (
            round(
                total / (elapsed_ms / 1000),
                1,
            )
            if elapsed_ms > 0
            else 0
        )

        with self._lock:
            entry = self._token_usage.setdefault(
                user_id,
                {
                    "total_tokens": 0,
                    "total_requests": 0,
                    "prompt_tokens": 0,
                    "completion_tokens": 0,
                    "last_used": "",
                },
            )

            entry["total_tokens"] += total
            entry["total_requests"] += 1
            entry["prompt_tokens"] += prompt_tokens
            entry["completion_tokens"] += completion_tokens
            entry["last_used"] = datetime.datetime.now().isoformat()

            hist = {
                "chat_id": chat_id,
                "user_id": user_id,
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": total,
                "tokens_per_second": tps,
                "elapsed_ms": elapsed_ms,
                "timestamp": datetime.datetime.now().isoformat(),
            }

            self._token_history.append(hist)

            if len(self._token_history) > 500:
                self._token_history = self._token_history[-500:]

            per_user = {uid: dict(usage) for uid, usage in self._token_usage.items()}

        self._broadcast(
            "tokens",
            {
                "type": "tokens",
                "chat_id": chat_id,
                "user_id": user_id,
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": total,
                "tokens_per_second": tps,
                "elapsed_ms": elapsed_ms,
                "per_user": per_user,
                "timestamp": datetime.datetime.now().isoformat(),
            },
        )
