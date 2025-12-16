import pytest
from unittest.mock import patch
from backend.auth_utils import validate_api_key


def test_valid_api_key():
    # Test with valid key
    valid_key = 'valid_api_key_123'
    with patch('backend.config.API_KEY', valid_key):
        assert validate_api_key(valid_key) is True


def test_invalid_api_key():
    # Test with invalid key
    invalid_key = 'invalid_key'
    with patch('backend.config.API_KEY', 'valid_api_key_123'):
        with pytest.raises(ValueError, match='Invalid API key'):
            validate_api_key(invalid_key)