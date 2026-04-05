"""Tests for email validation utility functions."""

import pytest
from utils.validate import validate_email


class TestValidateEmail:
    """Test suite for validate_email function."""
    
    def test_valid_email(self):
        """Test that a valid email address returns True."""
        assert validate_email("user@example.com") is True
        assert validate_email("test.user@domain.co.uk") is True
        assert validate_email("name+tag@company.org") is True
        assert validate_email("user123@test-domain.com") is True
    
    def test_missing_at_symbol(self):
        """Test that an email missing @ symbol returns False."""
        assert validate_email("userexample.com") is False
        assert validate_email("user.example.com") is False
        assert validate_email("user#example.com") is False
    
    def test_missing_domain(self):
        """Test that an email missing domain returns False."""
        assert validate_email("user@") is False
        assert validate_email("user@domain") is False
        assert validate_email("user@.com") is False
    
    def test_empty_string(self):
        """Test that an empty string returns False."""
        assert validate_email("") is False
    
    def test_none_value(self):
        """Test that None value returns False."""
        assert validate_email(None) is False
    
    def test_invalid_formats(self):
        """Test various invalid email formats."""
        assert validate_email("@example.com") is False  # Missing local part
        assert validate_email("user@@example.com") is False  # Double @
        assert validate_email("user @example.com") is False  # Space in email
        assert validate_email("user@example") is False  # Missing TLD
        assert validate_email("user@.example.com") is False  # Domain starts with dot
    
    def test_edge_cases(self):
        """Test edge cases for email validation."""
        assert validate_email("a@b.co") is True  # Minimal valid email
        assert validate_email("user@sub.domain.example.com") is True  # Subdomain
        assert validate_email("user_name@example.com") is True  # Underscore
        assert validate_email("user-name@example.com") is True  # Hyphen
