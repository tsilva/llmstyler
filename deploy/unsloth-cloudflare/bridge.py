"""Exchange verified Cloudflare Access identity for an ordinary Studio session.

Run with Studio's Python environment, bound only to loopback. No vendor files or
passwords are modified. The Cloudflare issuer, audience, owner, and host are pinned.
"""

import json
import os
import re
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import jwt

STUDIO = Path(os.environ["STUDIO_PACKAGE"])
HOST = os.environ["STUDIO_PUBLIC_HOST"]
ORIGIN = f"https://{HOST}"
ISSUER = os.environ["CLOUDFLARE_ACCESS_ISSUER"]
AUDIENCE = os.environ["CLOUDFLARE_ACCESS_AUDIENCE"]
OWNER = os.environ["CLOUDFLARE_ACCESS_OWNER"]
KEYS = jwt.PyJWKClient(f"{ISSUER}/cdn-cgi/access/certs", timeout=5)
sys.path.insert(0, str(STUDIO / "backend"))


def verify_identity(headers, keys=KEYS):
    """Reject absent, forged, expired, wrong-application, or wrong-owner tokens."""
    if headers.get("Host") != HOST:
        raise ValueError("Unexpected host")
    token = headers.get("Cf-Access-Jwt-Assertion", "")
    if not token or len(token) > 16384:
        raise ValueError("Missing Access identity")
    claims = jwt.decode(
        token,
        keys.get_signing_key_from_jwt(token).key,
        algorithms=["RS256"],
        audience=AUDIENCE,
        issuer=ISSUER,
        options={"require": ["exp", "iat", "sub", "email", "aud", "iss"]},
    )
    if claims["email"] != OWNER or claims.get("type") != "app":
        raise ValueError("Owner identity required")
    return claims


def verify_login_request(headers):
    # Custom header forces a browser preflight for cross-origin attempts. This
    # server offers no CORS permissions, and rejects foreign Origin explicitly.
    if headers.get("Origin") != ORIGIN:
        raise ValueError("Same-origin login required")
    if headers.get("X-Studio-Cloudflare-Login") != "1":
        raise ValueError("Explicit login header required")


def frontend():
    html = (STUDIO / "frontend/dist/index.html").read_text()
    pattern = r'<script type="module" crossorigin src="(/assets/[^"\s]+\.js)"></script>'
    match = re.search(pattern, html)
    if match is None or html.count(match.group(0)) != 1:
        raise ValueError("Studio frontend entry changed; refusing automatic login")
    return re.sub(
        pattern,
        '<script type="module" src="/cloudflare-auth/bootstrap.js"></script>',
        html,
        count=1,
    ), match.group(1)


def session():
    from auth import create_access_token, create_refresh_token, storage

    username = storage.DEFAULT_ADMIN_USERNAME
    account = storage.get_account(username)
    record = storage.get_user_and_secret(username)
    if account is None or not account.is_owner or record is None or record[3]:
        raise ValueError("Studio owner must complete initial setup first")
    # Use Studio's standard subject tokens and credential generation, preserving
    # normal refresh, password rotation, and account permission semantics.
    secret = record[2]
    return {
        "access_token": create_access_token(username, secret=secret),
        "refresh_token": create_refresh_token(username, secret=secret),
        "must_change_password": False,
        "account_id": account.account_id,
    }


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *args):
        # Access assertions, cookies, and Studio sessions must never reach logs.
        pass

    def respond(self, code, body, content_type="text/plain; charset=utf-8"):
        data = body.encode()
        self.send_response(code)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("X-Frame-Options", "DENY")
        self.send_header("Content-Security-Policy", "frame-ancestors 'none'")
        self.end_headers()
        self.wfile.write(data)

    def do_POST(self):
        if self.path != "/cloudflare-auth/session":
            return self.respond(404, "Not found")
        try:
            verify_login_request(self.headers)
            verify_identity(self.headers)
        except Exception:
            return self.respond(403, "Verified Cloudflare owner identity required")
        try:
            payload = session()
        except Exception:
            return self.respond(503, "Studio session unavailable")
        self.respond(200, json.dumps(payload), "application/json")

    def do_GET(self):
        try:
            verify_identity(self.headers)
        except Exception:
            return self.respond(403, "Verified Cloudflare owner identity required")
        try:
            html, entry = frontend()
        except Exception:
            return self.respond(503, "Studio frontend changed; check bridge compatibility")
        if self.path == "/cloudflare-auth/bootstrap.js":
            script = """
try {
  const response = await fetch('/cloudflare-auth/session', {
    method: 'POST', credentials: 'same-origin',
    headers: {'X-Studio-Cloudflare-Login': '1'},
  });
  if (!response.ok) throw new Error('Cloudflare sign-in failed');
  const session = await response.json();
  localStorage.setItem('unsloth_auth_token', session.access_token);
  localStorage.setItem('unsloth_auth_refresh_token', session.refresh_token);
  localStorage.setItem('unsloth_auth_must_change_password', 'false');
  if (!localStorage.getItem('unsloth_auth_session_mark')) {
    localStorage.setItem('unsloth_auth_session_mark', `${Date.now()}.${Math.random()}`);
  }
  await import(ENTRY);
} catch {
  document.body.textContent = 'Automatic Studio sign-in failed. Refresh to retry.';
}
""".replace("ENTRY", json.dumps(entry))
            return self.respond(200, script, "text/javascript; charset=utf-8")
        if self.path.startswith("/cloudflare-auth/"):
            return self.respond(404, "Not found")
        self.respond(200, html, "text/html; charset=utf-8")


if __name__ == "__main__":
    frontend()  # Fail before listening if an upgrade changes the frontend entry.
    ThreadingHTTPServer(("127.0.0.1", 8889), Handler).serve_forever()
