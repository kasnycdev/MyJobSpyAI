import pytest
from unittest.mock import patch, MagicMock, call
from myjobspyai.analysis.analyzer import BaseAnalyzer
from typing import List, Dict, Optional, Any
import json
from tenacity import RetryError
from ollama import ResponseError, Client

class MockLLMResponse:
    def __init__(self, content, tool_calls=None):
        self.choices = [
            {
                "message": {
                    "content": content,
                    "role": "assistant",
                    "tool_calls": tool_calls or []
                }
            }
        ]
        self.usage = {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0
        }
        
    def __getitem__(self, key):
        if key == 'choices':
            return self.choices
        elif key == 'usage':
            return self.usage
        raise KeyError(key)

class TestBaseAnalyzerRetry:
    @pytest.fixture
    def mock_settings(self):
        """Mock settings for testing."""
        return {
            'llm_provider': 'ollama',
            'ollama': {
                'base_url': 'http://localhost:11434',
                'model': 'llama3.1:latest',
                'request_timeout': 30,
                'max_retries': 3,
                'initial_retry_delay': 0.1,
                'max_retry_delay': 1.0
            }
        }

    @pytest.fixture
    def analyzer(self, mock_settings):
        """Create a test instance of BaseAnalyzer."""
        with patch('analysis.analyzer.settings', mock_settings):
            analyzer = BaseAnalyzer(provider_config_key='ollama')
            analyzer.provider = "ollama"
            analyzer.sync_client = MagicMock()
            analyzer.sync_client.chat = MagicMock()
            analyzer.sync_client.chat.completions = MagicMock()
            analyzer.sync_client.chat.completions.create = MagicMock()
            analyzer.async_client = MagicMock()
            
            # Set provider configuration
            analyzer.provider_config = mock_settings['ollama']
            
            # Set mock flag to skip connection check
            analyzer._mock_client = True
            
            return analyzer

    def test_successful_call(self, analyzer):
        """Test successful call without retries."""
        # Setup mock
        mock_response = {
            'choices': [
                {
                    'message': {
                        'content': 'Test response'
                    }
                }
            ]
        }
        analyzer.sync_client.chat.completions.create.return_value = mock_response
        
        # Call the method
        result = analyzer._call_llm("test prompt", "test task")
        
        # Verify result and call count
        assert result == "Test response"
        assert analyzer.sync_client.chat.completions.create.call_count == 1
        assert analyzer.sync_client.chat.completions.create.call_args_list == [
            call(model='llama3.1:latest', messages=[{'role': 'user', 'content': 'test prompt'}], temperature=0.7, max_tokens=1000, timeout=30)
        ]

    @patch('time.sleep')
    def test_retry_on_timeout(self, mock_sleep, analyzer):
        """Test that we retry on timeout errors."""
        # Setup mock to fail with timeout once, then succeed
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = 'test response'
        
        # First call raises TimeoutError, second call returns mock_response
        analyzer.sync_client.chat.completions.create.side_effect = [
            TimeoutError("Request timed out"),
            mock_response
        ]
        
        # Call the method
        result = analyzer._call_llm("test prompt", "test task")
        
        # Verify retry attempts
        assert analyzer.sync_client.chat.completions.create.call_count == 2
        assert analyzer.sync_client.chat.completions.create.call_args_list == [
            call(model='llama3.1:latest', messages=[{'role': 'user', 'content': 'test prompt'}], temperature=0.7, max_tokens=1000, timeout=30),
            call(model='llama3.1:latest', messages=[{'role': 'user', 'content': 'test prompt'}], temperature=0.7, max_tokens=1000, timeout=30)
        ]

    @patch('time.sleep')
    def test_give_up_after_max_retries(self, mock_sleep, analyzer):
        """Test that we give up after max retries."""
        # Setup mock to fail with TimeoutError three times
        analyzer.sync_client.chat.completions.create.side_effect = [
            TimeoutError("Request timed out"),
            TimeoutError("Request timed out"),
            TimeoutError("Request timed out")
        ]
        
        # Call the method and expect RetryError
        with pytest.raises(RetryError) as exc_info:
            analyzer._call_llm("test prompt", "test task")
            
        # Verify we got the expected exception
        assert isinstance(exc_info.value, RetryError)
        
        # Verify we tried the max number of times (1 initial + 2 retries = 3 total)
        assert analyzer.sync_client.chat.completions.create.call_count == 3
        
        # Verify the correct call arguments
        expected_call = call(
            model='llama3.1:latest',
            messages=[{'role': 'user', 'content': 'test prompt'}],
            temperature=0.7,
            max_tokens=1000,
            timeout=30
        )
        
        # Verify all calls were made with the same arguments
        assert analyzer.sync_client.chat.completions.create.call_args_list == [expected_call] * 3
        
        # Verify sleep was called twice (after first and second failures)
        assert mock_sleep.call_count == 2
        
        # Verify sleep was called with increasing backoff times
        assert mock_sleep.call_args_list[0][0][0] == 0.1  # First backoff
        assert mock_sleep.call_args_list[1][0][0] == 0.2  # Second backoff (exponential)

    def test_do_not_retry_non_retriable_errors(self, analyzer):
        """Test that we don't retry on non-retriable errors."""
        # Setup mock to always fail with a non-retriable error
        analyzer.sync_client.chat.completions.create.side_effect = ValueError("Invalid input")
        
        with pytest.raises(ValueError):
            analyzer._call_llm("test prompt", "test task")
        
        # Verify no retries
        analyzer.sync_client.chat.completions.create.assert_called_once()

    @patch('time.sleep')
    def test_exponential_backoff(self, mock_sleep, analyzer):
        """Test that backoff time increases exponentially."""
        # Setup mock to fail multiple times
        mock_response = {
            'choices': [
                {
                    'message': {
                        'content': 'test response'
                    }
                }
            ]
        }
        analyzer.sync_client.chat.completions.create.side_effect = [
            ResponseError("Timeout"),
            mock_response
        ]
        
        # Call the method
        analyzer._call_llm("test prompt", "test task")
        
        # Verify backoff times
        assert mock_sleep.call_count == 1
        assert mock_sleep.call_args_list[0][0][0] == 0.1  # Initial delay
        
        # Verify retry attempts
        assert analyzer.sync_client.chat.completions.create.call_count == 2
