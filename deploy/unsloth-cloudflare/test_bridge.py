"""Run with Studio's Python: python -m unittest discover -s <this directory>."""

import os
import time
import unittest
from types import SimpleNamespace

os.environ.setdefault("STUDIO_PACKAGE", "/unused")
os.environ.setdefault("STUDIO_PUBLIC_HOST", "studio.example.com")
os.environ.setdefault("CLOUDFLARE_ACCESS_ISSUER", "https://team.cloudflareaccess.com")
os.environ.setdefault("CLOUDFLARE_ACCESS_AUDIENCE", "application-tag")
os.environ.setdefault("CLOUDFLARE_ACCESS_OWNER", "owner@example.com")

import bridge
import jwt
from cryptography.hazmat.primitives.asymmetric import rsa


class IdentityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
        cls.foreign_key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
        cls.keys = SimpleNamespace(
            get_signing_key_from_jwt=lambda token: SimpleNamespace(key=cls.key.public_key())
        )

    def headers(self, **changes):
        now = int(time.time())
        claims = {
            "iss": bridge.ISSUER, "aud": [bridge.AUDIENCE], "sub": "owner-id",
            "email": bridge.OWNER, "type": "app", "iat": now, "exp": now + 60,
        }
        claims.update(changes)
        return {"Host": bridge.HOST, "Cf-Access-Jwt-Assertion": jwt.encode(
            claims, self.key, algorithm="RS256", headers={"kid": "test-key"}
        )}

    def test_valid_owner(self):
        self.assertEqual(bridge.verify_identity(self.headers(), self.keys)["email"], bridge.OWNER)

    def test_wrong_owner_application_issuer_and_expiry(self):
        cases = [dict(email="stranger@example.com"), dict(aud="wrong-app"),
                 dict(iss="https://foreign.example"), dict(exp=int(time.time()) - 1),
                 dict(nbf=int(time.time()) + 60), dict(type="service"), dict(email=None)]
        for case in cases:
            with self.subTest(case=case), self.assertRaises(Exception):
                bridge.verify_identity(self.headers(**case), self.keys)

    def test_forged_missing_and_wrong_host(self):
        headers = self.headers()
        claims = jwt.decode(headers["Cf-Access-Jwt-Assertion"], options={"verify_signature": False})
        headers["Cf-Access-Jwt-Assertion"] = jwt.encode(claims, self.foreign_key, algorithm="RS256")
        for candidate in [headers, {"Host": bridge.HOST}, {**self.headers(), "Host": "foreign.example"}]:
            with self.subTest(), self.assertRaises(Exception):
                bridge.verify_identity(candidate, self.keys)

    def test_unsigned_token(self):
        claims = jwt.decode(self.headers()["Cf-Access-Jwt-Assertion"], options={"verify_signature": False})
        headers = {"Host": bridge.HOST, "Cf-Access-Jwt-Assertion": jwt.encode(claims, key="", algorithm="none")}
        with self.assertRaises(Exception):
            bridge.verify_identity(headers, self.keys)

    def test_cross_origin_login(self):
        valid = {"Origin": bridge.ORIGIN, "X-Studio-Cloudflare-Login": "1"}
        bridge.verify_login_request(valid)
        for candidate in [{}, {**valid, "Origin": "https://foreign.example"}, {"Origin": bridge.ORIGIN}]:
            with self.subTest(), self.assertRaises(ValueError):
                bridge.verify_login_request(candidate)


if __name__ == "__main__":
    unittest.main()
