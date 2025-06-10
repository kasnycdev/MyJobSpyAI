"""
Example unit tests for the MyJobSpyAI project.
"""


def test_example():
    """Test that 1 + 1 equals 2."""
    assert 1 + 1 == 2


class TestExampleClass:
    """Example test class with multiple test methods."""

    def test_one_plus_one(self):
        """Test that 1 + 1 equals 2 in a class method."""
        assert 1 + 1 == 2

    def test_string_concatenation(self):
        """Test string concatenation."""
        assert "hello" + " " + "world" == "hello world"


# Example of a test that will be marked as slow
import pytest


@pytest.mark.slow
def test_slow():
    """This is a slow test that will be skipped by default."""
    import time

    time.sleep(2)  # Simulate a slow test
    assert True
