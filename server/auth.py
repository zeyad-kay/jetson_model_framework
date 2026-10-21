import hmac


def authenticate(key: str, private_key: str) -> bool:
    """Checks the client's `key` against the server's `private_key`.

    Compares in constant time. An empty private key never authenticates.
    """
    if not private_key or not isinstance(key, str):
        return False
    return hmac.compare_digest(key.encode("utf-8"), private_key.encode("utf-8"))
