import pytest
import sqlite3
import os
import tempfile
from unittest.mock import patch, MagicMock
from backend.data_processing import connect_to_database, execute_db_query


def test_database_connection_success():
    """Test successful connection to existing database."""
    # Create a temporary database file
    with tempfile.NamedTemporaryFile(suffix='.sqlite', delete=False) as tmp_file:
        db_path = tmp_file.name

    try:
        # Test connection function
        db_conn = connect_to_database(db_path)
        assert db_conn is not None
        db_conn.close()
    finally:
        # Cleanup
        if os.path.exists(db_path):
            os.remove(db_path)


def test_database_connection_failure():
    """Test failure with a path that cannot be created."""
    if os.name == 'nt':
        invalid_path = 'Z:\\non_existent_drive\\invalid.db'
    else:
        invalid_path = '/dev/null/invalid.db'  # /dev/null is a file, not a dir

    with pytest.raises(sqlite3.OperationalError):
        connect_to_database(invalid_path)


def test_database_connection_invalid_path():
    """Test with empty or invalid path."""
    with pytest.raises(ValueError, match="Database path must be a non-empty string"):
        connect_to_database("")

    with pytest.raises(ValueError, match="Database path must be a non-empty string"):
        connect_to_database(None)


def test_db_operational_success():
    """Test successful database query execution."""
    # Mock successful query
    with patch('sqlite3.connect') as mock_connect:
        mock_conn = MagicMock()
        mock_connect.return_value = mock_conn
        mock_cursor = MagicMock()
        mock_conn.execute.return_value = mock_cursor
        mock_cursor.fetchall.return_value = [('result1',), ('result2',)]

        result = execute_db_query('SELECT * FROM test_table')
        assert result is not None
        assert result == [('result1',), ('result2',)]


def test_db_operational_failure():
    """Test database operational error handling."""
    # Mock operational error
    with patch('sqlite3.connect') as mock_connect:
        mock_conn = MagicMock()
        mock_connect.return_value = mock_conn
        mock_conn.execute.side_effect = sqlite3.OperationalError("Simulated failure")

        with pytest.raises(sqlite3.OperationalError):
            execute_db_query('SELECT * FROM test_table')