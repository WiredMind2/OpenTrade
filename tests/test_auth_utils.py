"""
Unit tests for authentication utilities.
"""
import pytest
import sqlite3
import tempfile
import os
from datetime import datetime, timedelta
from unittest.mock import patch, MagicMock
from jose import JWTError
from backend.auth_utils import (
    hash_password,
    verify_password,
    create_access_token,
    create_refresh_token,
    verify_token,
    hash_token,
    generate_reset_token,
    get_user_from_token,
    store_session_token,
    invalidate_session_token,
    invalidate_all_user_sessions,
    check_role_access,
    log_user_activity,
    TokenData,
    SECRET_KEY,
    ALGORITHM
)


@pytest.mark.unit
class TestAuthUtils:
    """Test authentication utility functions."""

    def test_hash_password(self):
        """Test password hashing."""
        password = "test_password_123"
        hashed = hash_password(password)

        # Hash should be different from plain password
        assert hashed != password
        # Hash should be a string
        assert isinstance(hashed, str)
        # Hash should be non-empty
        assert len(hashed) > 0

    def test_verify_password(self):
        """Test password verification."""
        password = "test_password_123"
        hashed = hash_password(password)

        # Correct password should verify
        assert verify_password(password, hashed) is True

        # Wrong password should not verify
        assert verify_password("wrong_password", hashed) is False

        # Empty password should not verify
        assert verify_password("", hashed) is False

    def test_create_access_token(self):
        """Test access token creation."""
        data = {"user_id": 1, "email": "test@example.com", "role": "admin"}
        token = create_access_token(data)

        # Token should be a string
        assert isinstance(token, str)
        assert len(token) > 0

        # Token should contain dots (JWT format)
        assert "." in token

    def test_create_access_token_with_expiry(self):
        """Test access token creation with custom expiry."""
        data = {"user_id": 1, "email": "test@example.com", "role": "admin"}
        expires_delta = timedelta(minutes=30)
        token = create_access_token(data, expires_delta)

        assert isinstance(token, str)
        assert len(token) > 0

    def test_create_refresh_token(self):
        """Test refresh token creation."""
        data = {"user_id": 1, "email": "test@example.com", "role": "admin"}
        token = create_refresh_token(data)

        # Token should be a string
        assert isinstance(token, str)
        assert len(token) > 0

        # Token should contain dots (JWT format)
        assert "." in token

    def test_verify_token_valid(self):
        """Test verification of valid token."""
        data = {"user_id": 1, "email": "test@example.com", "role": "admin"}
        token = create_access_token(data)

        result = verify_token(token)
        assert result is not None
        assert isinstance(result, TokenData)
        assert result.user_id == 1
        assert result.email == "test@example.com"
        assert result.role == "admin"

    def test_verify_token_invalid(self):
        """Test verification of invalid token."""
        # Completely invalid token
        result = verify_token("invalid.token.here")
        assert result is None

        # Empty token
        result = verify_token("")
        assert result is None

        # Token with missing required fields
        data = {"user_id": 1}  # Missing email and role
        token = create_access_token(data)
        result = verify_token(token)
        assert result is None

    def test_verify_token_expired(self):
        """Test verification of expired token."""
        data = {"user_id": 1, "email": "test@example.com", "role": "admin"}
        # Create token that expires immediately
        expires_delta = timedelta(seconds=-1)
        token = create_access_token(data, expires_delta)

        result = verify_token(token)
        assert result is None

    def test_hash_token(self):
        """Test token hashing."""
        token = "test_token_123"
        hashed = hash_token(token)

        # Hash should be different from token
        assert hashed != token
        # Hash should be a string
        assert isinstance(hashed, str)
        # Hash should be 64 characters (SHA256 hex)
        assert len(hashed) == 64
        # Hash should be consistent
        assert hash_token(token) == hashed

    def test_generate_reset_token(self):
        """Test reset token generation."""
        token1 = generate_reset_token()
        token2 = generate_reset_token()

        # Tokens should be strings
        assert isinstance(token1, str)
        assert isinstance(token2, str)

        # Tokens should be non-empty
        assert len(token1) > 0
        assert len(token2) > 0

        # Tokens should be different (very high probability)
        assert token1 != token2

    def test_get_user_from_token_valid(self, temp_db):
        """Test getting user from valid token."""
        # Create test user and session
        conn = sqlite3.connect(temp_db)
        conn.execute("""
            CREATE TABLE users (
                id INTEGER PRIMARY KEY,
                email TEXT,
                name TEXT,
                role TEXT,
                avatar TEXT,
                is_active INTEGER,
                last_login TEXT
            )
        """)
        conn.execute("""
            CREATE TABLE user_sessions (
                id INTEGER PRIMARY KEY,
                user_id INTEGER,
                token_hash TEXT,
                expires_at TEXT
            )
        """)

        # Insert test user
        conn.execute("""
            INSERT INTO users (email, name, role, avatar, is_active, last_login)
            VALUES (?, ?, ?, ?, ?, ?)
        """, ("test@example.com", "Test User", "admin", "avatar.png", 1, datetime.utcnow().isoformat()))

        user_id = conn.execute("SELECT last_insert_rowid()").fetchone()[0]
        conn.commit()
        conn.close()

        # Create token and store session
        data = {"user_id": user_id, "email": "test@example.com", "role": "admin"}
        token = create_access_token(data)
        store_session_token(user_id, token, temp_db)
        conn.close()

        # Test getting user from token
        user = get_user_from_token(token, temp_db)
        assert user is not None
        assert user["id"] == user_id
        assert user["email"] == "test@example.com"
        assert user["name"] == "Test User"
        assert user["role"] == "admin"
        assert user["is_active"] is True

    def test_get_user_from_token_invalid(self, temp_db):
        """Test getting user from invalid token."""
        # Create required tables to avoid errors
        conn = sqlite3.connect(temp_db)
        conn.execute("""
            CREATE TABLE users (
                id INTEGER PRIMARY KEY,
                email TEXT,
                name TEXT,
                role TEXT,
                avatar TEXT,
                is_active INTEGER,
                last_login TEXT
            )
        """)
        conn.execute("""
            CREATE TABLE user_sessions (
                id INTEGER PRIMARY KEY,
                user_id INTEGER,
                token_hash TEXT,
                expires_at TEXT
            )
        """)
        conn.commit()
        conn.close()

        # Test with invalid token
        user = get_user_from_token("invalid_token", temp_db)
        assert user is None

        # Test with valid token format but no session
        data = {"user_id": 1, "email": "test@example.com", "role": "admin"}
        token = create_access_token(data)
        user = get_user_from_token(token, temp_db)
        assert user is None

    def test_store_session_token(self, temp_db):
        """Test storing session token."""
        # Create required table
        conn = sqlite3.connect(temp_db)
        conn.execute("""
            CREATE TABLE user_sessions (
                id INTEGER PRIMARY KEY,
                user_id INTEGER,
                token_hash TEXT,
                expires_at TEXT
            )
        """)
        conn.commit()
        conn.close()

        user_id = 1
        token = "test_token_123"

        # Should not raise exception
        store_session_token(user_id, token, temp_db)

        # Verify token was stored
        conn = sqlite3.connect(temp_db)
        cur = conn.cursor()
        cur.execute("SELECT user_id, token_hash FROM user_sessions WHERE user_id = ?", (user_id,))
        row = cur.fetchone()
        conn.close()

        assert row is not None
        assert row[0] == user_id
        assert row[1] == hash_token(token)

    def test_invalidate_session_token(self, temp_db):
        """Test invalidating session token."""
        # Create required table
        conn = sqlite3.connect(temp_db)
        conn.execute("""
            CREATE TABLE user_sessions (
                id INTEGER PRIMARY KEY,
                user_id INTEGER,
                token_hash TEXT,
                expires_at TEXT
            )
        """)
        conn.commit()
        conn.close()

        user_id = 1
        token = "test_token_123"

        # Store token first
        store_session_token(user_id, token, temp_db)

        # Verify it exists
        conn = sqlite3.connect(temp_db)
        cur = conn.cursor()
        cur.execute("SELECT COUNT(*) FROM user_sessions WHERE user_id = ?", (user_id,))
        count_before = cur.fetchone()[0]
        conn.close()

        assert count_before == 1

        # Invalidate token
        invalidate_session_token(token, temp_db)

        # Verify it was removed
        conn = sqlite3.connect(temp_db)
        cur = conn.cursor()
        cur.execute("SELECT COUNT(*) FROM user_sessions WHERE user_id = ?", (user_id,))
        count_after = cur.fetchone()[0]
        conn.close()

        assert count_after == 0

    def test_invalidate_all_user_sessions(self, temp_db):
        """Test invalidating all sessions for a user."""
        # Create required table
        conn = sqlite3.connect(temp_db)
        conn.execute("""
            CREATE TABLE user_sessions (
                id INTEGER PRIMARY KEY,
                user_id INTEGER,
                token_hash TEXT,
                expires_at TEXT
            )
        """)
        conn.commit()
        conn.close()

        user_id = 1
        tokens = ["token1", "token2", "token3"]

        # Store multiple tokens (each call deletes previous ones for the user)
        for i, token in enumerate(tokens):
            store_session_token(user_id, token, temp_db)
            # Verify count after each insertion (should be 1 since each call deletes previous)
            conn = sqlite3.connect(temp_db)
            cur = conn.cursor()
            cur.execute("SELECT COUNT(*) FROM user_sessions WHERE user_id = ?", (user_id,))
            current_count = cur.fetchone()[0]
            conn.close()
            assert current_count == 1, f"Expected 1 session after storing token {i+1}, got {current_count}"

        # Verify final count (should be 1 since each store_session_token deletes previous sessions)
        conn = sqlite3.connect(temp_db)
        cur = conn.cursor()
        cur.execute("SELECT COUNT(*) FROM user_sessions WHERE user_id = ?", (user_id,))
        count_before = cur.fetchone()[0]
        conn.close()

        assert count_before == 1

        # Invalidate all sessions
        invalidate_all_user_sessions(user_id, temp_db)

        # Verify they were removed
        conn = sqlite3.connect(temp_db)
        cur = conn.cursor()
        cur.execute("SELECT COUNT(*) FROM user_sessions WHERE user_id = ?", (user_id,))
        count_after = cur.fetchone()[0]
        conn.close()

        assert count_after == 0

    def test_check_role_access(self):
        """Test role access checking."""
        # Test valid access
        assert check_role_access("admin", "viewer") is True
        assert check_role_access("admin", "analyst") is True
        assert check_role_access("admin", "trader") is True
        assert check_role_access("admin", "admin") is True

        assert check_role_access("trader", "viewer") is True
        assert check_role_access("trader", "analyst") is True
        assert check_role_access("trader", "trader") is True
        assert check_role_access("trader", "admin") is False

        assert check_role_access("analyst", "viewer") is True
        assert check_role_access("analyst", "analyst") is True
        assert check_role_access("analyst", "trader") is False
        assert check_role_access("analyst", "admin") is False

        assert check_role_access("viewer", "viewer") is True
        assert check_role_access("viewer", "analyst") is False
        assert check_role_access("viewer", "trader") is False
        assert check_role_access("viewer", "admin") is False

        # Test invalid roles
        assert check_role_access("invalid_role", "viewer") is False
        assert check_role_access("viewer", "invalid_role") is True  # invalid required role defaults to level 0, viewer (1) >= 0

    def test_log_user_activity(self, temp_db):
        """Test logging user activity."""
        # Create activity table
        conn = sqlite3.connect(temp_db)
        conn.execute("""
            CREATE TABLE user_activity (
                id INTEGER PRIMARY KEY,
                user_id INTEGER,
                action TEXT,
                details TEXT,
                ip_address TEXT,
                user_agent TEXT,
                timestamp TEXT DEFAULT CURRENT_TIMESTAMP
            )
        """)
        conn.commit()
        conn.close()

        user_id = 1
        action = "login"
        details = "User logged in successfully"
        ip_address = "192.168.1.1"
        user_agent = "Mozilla/5.0"

        # Should not raise exception
        log_user_activity(user_id, action, details, temp_db, ip_address, user_agent)

        # Verify activity was logged
        conn = sqlite3.connect(temp_db)
        cur = conn.cursor()
        cur.execute("SELECT user_id, action, details, ip_address, user_agent FROM user_activity WHERE user_id = ?", (user_id,))
        row = cur.fetchone()
        conn.close()

        assert row is not None
        assert row[0] == user_id
        assert row[1] == action
        assert row[2] == details
        assert row[3] == ip_address
        assert row[4] == user_agent

    def test_token_data_model(self):
        """Test TokenData Pydantic model."""
        # Valid data
        token_data = TokenData(user_id=1, email="test@example.com", role="admin")
        assert token_data.user_id == 1
        assert token_data.email == "test@example.com"
        assert token_data.role == "admin"
        assert token_data.exp is None

        # With expiry
        exp_time = datetime.utcnow()
        token_data_with_exp = TokenData(user_id=1, email="test@example.com", role="admin", exp=exp_time)
        assert token_data_with_exp.exp == exp_time

    @patch('backend.auth_utils.jwt.encode')
    def test_create_access_token_encoding_error(self, mock_encode):
        """Test handling of JWT encoding errors."""
        mock_encode.side_effect = Exception("Encoding failed")

        data = {"user_id": 1, "email": "test@example.com", "role": "admin"}

        with pytest.raises(Exception):
            create_access_token(data)

    @patch('backend.auth_utils.jwt.decode')
    def test_verify_token_decoding_error(self, mock_decode):
        """Test handling of JWT decoding errors."""
        mock_decode.side_effect = JWTError("Decoding failed")

        result = verify_token("invalid.token")
        assert result is None