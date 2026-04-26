from base64 import b64encode

from app.auth import BasicAuthConfig, is_authorized


def test_auth_disabled_without_password() -> None:
    assert is_authorized("", "", BasicAuthConfig(username="arthur", password=""))


def test_auth_requires_exact_credentials() -> None:
    config = BasicAuthConfig(username="arthur", password="secret")
    assert is_authorized("arthur", "secret", config)
    assert not is_authorized("arthur", "wrong", config)


def test_basic_header_fixture_is_valid() -> None:
    token = b64encode(b"arthur:secret").decode("ascii")
    assert token == "YXJ0aHVyOnNlY3JldA=="
