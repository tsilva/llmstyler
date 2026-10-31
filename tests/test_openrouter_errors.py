import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

from llmstyler.restyle import chat_completion


def test_http_failure_reports_status_and_redacts_key(monkeypatch):
    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            self.send_response(401)
            self.end_headers()
            self.wfile.write(json.dumps({"error": {"message": "Invalid test-secret"}}).encode())

        def log_message(self, *args):
            pass

    server = HTTPServer(("127.0.0.1", 0), Handler)
    worker = threading.Thread(target=server.serve_forever, daemon=True)
    worker.start()
    try:
        with pytest.raises(RuntimeError) as error:
            chat_completion(
                {
                    "model": "test",
                    "retries": 0,
                    "base_url": f"http://127.0.0.1:{server.server_port}",
                },
                [{"role": "user", "content": "Hello"}],
                api_key="test-secret",
            )
        assert "HTTP 401: Invalid [redacted]" in str(error.value)
        assert "test-secret" not in str(error.value)
        assert "Traceback" not in str(error.value)
    finally:
        server.shutdown()
        server.server_close()
        worker.join()
