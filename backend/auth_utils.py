"""
Authentication utilities for JWT token management and password hashing.
"""
import os
import hashlib
import secrets
from datetime import datetime, timedelta
from typing import Optional, Dict, Any
from passlib.context import CryptContext
from jose import JWTError, jwt
from pydantic import BaseModel
import sqlite3

# Password hashing context
pwd_context = CryptContext(schemes=["pbkdf2_sha256"], deprecated="auto")

# JWT configuration
SECRET_KEY = os.getenv("JWT_SECRET_KEY", "your-secret-key-change-in-production")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 60  # 1 hour
REFRESH_TOKEN_EXPIRE_DAYS = 7  # 7 days


class TokenData(BaseModel):
    """Data stored in JWT token."""
    user_id: int
    email: str
    role: str
    exp: Optional[datetime] = None


def hash_password(password: str) -> str:
    """Hash a password using PBKDF2-SHA256."""
    return pwd_context.hash(password)


def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verify a password against its hash."""
    return pwd_context.verify(plain_password, hashed_password)


def create_access_token(data: Dict[str, Any], expires_delta: Optional[timedelta] = None) -> str:
    """Create a JWT access token."""
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)

    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt


def create_refresh_token(data: Dict[str, Any]) -> str:
    """Create a JWT refresh token."""
    to_encode = data.copy()
    expire = datetime.utcnow() + timedelta(days=REFRESH_TOKEN_EXPIRE_DAYS)
    to_encode.update({"exp": expire, "type": "refresh"})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt


def verify_token(token: str) -> Optional[TokenData]:
    """Verify and decode a JWT token."""
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        user_id: int = payload.get("user_id")
        email: str = payload.get("email")
        role: str = payload.get("role")

        if user_id is None or email is None or role is None:
            return None

        return TokenData(user_id=user_id, email=email, role=role)
    except JWTError:
        return None


def hash_token(token: str) -> str:
    """Create a hash of a token for database storage."""
    return hashlib.sha256(token.encode()).hexdigest()


def generate_reset_token() -> str:
    """Generate a secure password reset token."""
    return secrets.token_urlsafe(32)


def get_user_from_token(token: str, db_path: str) -> Optional[Dict[str, Any]]:
    """Get user data from a valid token."""
    token_data = verify_token(token)
    if not token_data:
        return None

    # Check if token exists in database and is not expired
    conn = sqlite3.connect(db_path)
    try:
        cur = conn.cursor()
        cur.execute("""
            SELECT u.id, u.email, u.name, u.role, u.avatar, u.is_active, u.last_login
            FROM users u
            JOIN user_sessions s ON u.id = s.user_id
            WHERE s.token_hash = ? AND s.expires_at > datetime('now')
            AND u.is_active = 1
        """, (hash_token(token),))

        row = cur.fetchone()
        if row:
            return {
                "id": row[0],
                "email": row[1],
                "name": row[2],
                "role": row[3],
                "avatar": row[4],
                "is_active": bool(row[5]),
                "last_login": row[6]
            }
    finally:
        conn.close()

    return None


def store_session_token(user_id: int, token: str, db_path: str) -> None:
    """Store a session token in the database."""
    conn = sqlite3.connect(db_path)
    try:
        cur = conn.cursor()
        expires_at = datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)

        # Remove any existing sessions for this user
        cur.execute("DELETE FROM user_sessions WHERE user_id = ?", (user_id,))

        # Insert new session
        cur.execute("""
            INSERT INTO user_sessions (user_id, token_hash, expires_at)
            VALUES (?, ?, ?)
        """, (user_id, hash_token(token), expires_at.isoformat()))

        conn.commit()
    finally:
        conn.close()


def invalidate_session_token(token: str, db_path: str) -> None:
    """Invalidate a session token."""
    conn = sqlite3.connect(db_path)
    try:
        cur = conn.cursor()
        cur.execute("DELETE FROM user_sessions WHERE token_hash = ?", (hash_token(token),))
        conn.commit()
    finally:
        conn.close()


def invalidate_all_user_sessions(user_id: int, db_path: str) -> None:
    """Invalidate all sessions for a user."""
    conn = sqlite3.connect(db_path)
    try:
        cur = conn.cursor()
        cur.execute("DELETE FROM user_sessions WHERE user_id = ?", (user_id,))
        conn.commit()
    finally:
        conn.close()


def check_role_access(user_role: str, required_role: str) -> bool:
    """Check if user role has access to required role."""
    role_hierarchy = {
        "viewer": 1,
        "analyst": 2,
        "trader": 3,
        "admin": 4
    }

    user_level = role_hierarchy.get(user_role, 0)
    required_level = role_hierarchy.get(required_role, 0)

    return user_level >= required_level


def log_user_activity(user_id: int, action: str, details: Optional[str], db_path: str,
                      ip_address: Optional[str] = None, user_agent: Optional[str] = None) -> None:
    """Log user activity."""
    conn = sqlite3.connect(db_path)
    try:
        cur = conn.cursor()
        cur.execute("""
            INSERT INTO user_activity (user_id, action, details, ip_address, user_agent)
            VALUES (?, ?, ?, ?, ?)
        """, (user_id, action, details, ip_address, user_agent))
        conn.commit()
    finally:
        conn.close()