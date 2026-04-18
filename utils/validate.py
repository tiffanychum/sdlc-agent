"""Email validation utility functions."""

import re


def validate_email(email: str) -> bool:
    """
    Validate an email address using a regex pattern.
    
    Args:
        email: The email address string to validate.
        
    Returns:
        True if the email is valid, False otherwise.
        
    Examples:
        >>> validate_email("user@example.com")
        True
        >>> validate_email("invalid.email")
        False
        >>> validate_email("")
        False
    """
    if not email or not isinstance(email, str):
        return False
    
    # RFC 5322 compliant email regex pattern (simplified but robust)
    # Pattern breakdown:
    # - Local part: alphanumeric, dots, hyphens, underscores, plus signs
    # - @ symbol (required)
    # - Domain: alphanumeric, dots, hyphens
    # - TLD: at least 2 characters
    # Domain part must not start or end with a dot or hyphen
    pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9]([a-zA-Z0-9.-]*[a-zA-Z0-9])?\.[a-zA-Z]{2,}$'
    
    return bool(re.match(pattern, email))
