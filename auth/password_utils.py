"""
password_utils.py
Provides secure password generation and verification utilities.
Updated to use passlib's pure‑Python pbkdf2_sha256 algorithm
for consistency with user_manager.py.
"""

import secrets
import string
from passlib.hash import pbkdf2_sha256


def generate_system_password(length: int = 12) -> str:
    """
    Generate a random password consisting of letters, digits,
    and punctuation characters.

    Args:
        length (int): Desired password length (minimum 8).

    Returns:
        str: Randomly generated password.
    """
    if length < 8:
        raise ValueError("Password length should be at least 8 characters.")

    alphabet = string.ascii_letters + string.digits + string.punctuation
    return "".join(secrets.choice(alphabet) for _ in range(length))


def hash_password(plain_password: str) -> str:
    """
    Hash a password using passlib's pbkdf2_sha256 implementation.

    Args:
        plain_password (str): User's plain‑text password.

    Returns:
        str: Hashed password including salt and parameters.
    """
    if not plain_password:
        raise ValueError("Password cannot be empty.")
    return pbkdf2_sha256.hash(plain_password)


def verify_password(plain_password: str, hashed_password: str) -> bool:
    """
    Verify that a plain password matches its pbkdf2_sha256 hash.

    Args:
        plain_password (str): Password provided by user.
        hashed_password (str): Stored pbkdf2_sha256 hash.

    Returns:
        bool: True if password matches, False otherwise.
    """
    try:
        return pbkdf2_sha256.verify(plain_password, hashed_password)
    except (ValueError, Exception):
        # Handle corrupted or incompatible hash formats gracefully
        return False


# ---------------------------------------------------------------------
#  Stand‑alone quick check
# ---------------------------------------------------------------------
if __name__ == "__main__":
    pwd = generate_system_password(12)
    print(f"Generated password: {pwd}")
    hashed = hash_password(pwd)
    print(f"Hashed password: {hashed}")
    print("Verification result:", verify_password(pwd, hashed))