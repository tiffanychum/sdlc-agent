"""Simple pytest test suite for demonstration."""


def test_addition():
    """Test that addition works correctly."""
    assert 1 + 1 == 2
    assert 2 + 3 == 5


def test_string_operations():
    """Test basic string operations."""
    assert "hello".upper() == "HELLO"
    assert "WORLD".lower() == "world"
    assert "hello" + " " + "world" == "hello world"


def test_list_operations():
    """Test basic list operations."""
    numbers = [1, 2, 3]
    assert len(numbers) == 3
    assert sum(numbers) == 6
    numbers.append(4)
    assert numbers == [1, 2, 3, 4]
