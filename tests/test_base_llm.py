"""Tests for BaseLLMClient."""

from unittest.mock import MagicMock, patch

import pytest

from src.llm.base import BaseLLMClient
from src.utils.token_counter import TokenCounter


class FakeLLM(BaseLLMClient):
    """Concrete implementation for testing."""

    def __init__(self, responses=None):
        super().__init__("fake-model", 0.7, 4096)
        self._responses = list(responses or ["default response"])
        self._call_count = 0

    def complete(self, prompt, **kwargs):
        idx = min(self._call_count, len(self._responses) - 1)
        self._call_count += 1
        return self._responses[idx]

    def chat(self, messages, **kwargs):
        return self.complete(messages[-1]["content"])

    def get_available_models(self):
        return []


class TestCompleteChunked:
    """Tests for the complete_chunked method on BaseLLMClient."""

    def test_short_text_single_call(self):
        """Text that fits in one call should use a single LLM call."""
        llm = FakeLLM(["summary result"])
        result = llm.complete_chunked("Summarize:\n{text}", "short text")
        assert result == "summary result"
        assert llm._call_count == 1

    def test_empty_response_raises(self):
        """Should raise RuntimeError when LLM returns empty for short text."""
        llm = FakeLLM([""])
        with pytest.raises(RuntimeError, match="LLM returned empty response"):
            llm.complete_chunked("Summarize:\n{text}", "short text")

    def test_long_text_splits_and_merges(self):
        """Long text should be split into chunks, processed, and merged."""
        llm = FakeLLM([
            "# Part 1 result",
            "# Part 2 result",
            "# Merged result",
        ])
        # Mock the token counter to force chunking
        mock_counter = MagicMock()
        mock_counter.count.side_effect = lambda text: len(text)
        mock_counter.split_by_tokens.return_value = ["chunk1", "chunk2"]
        llm._token_counter = mock_counter

        result = llm.complete_chunked("Process:\n{text}", "x" * 200_000)

        assert llm._call_count == 3  # 2 chunks + 1 merge
        assert result == "# Merged result"

    def test_all_chunks_empty_raises(self):
        """Should raise if all chunks return empty."""
        llm = FakeLLM([""])
        mock_counter = MagicMock()
        mock_counter.count.side_effect = lambda text: len(text)
        mock_counter.split_by_tokens.return_value = ["chunk1", "chunk2"]
        llm._token_counter = mock_counter

        with pytest.raises(RuntimeError, match="empty response for all chunks"):
            llm.complete_chunked("Process:\n{text}", "x" * 200_000)

    def test_merge_empty_falls_back_to_concat(self):
        """If merge call returns empty, should concatenate chunk results."""
        llm = FakeLLM([
            "Part 1 output",
            "Part 2 output",
            "",  # merge returns empty
        ])
        mock_counter = MagicMock()
        mock_counter.count.side_effect = lambda text: len(text)
        mock_counter.split_by_tokens.return_value = ["chunk1", "chunk2"]
        llm._token_counter = mock_counter

        result = llm.complete_chunked("Process:\n{text}", "x" * 200_000)

        assert "Part 1 output" in result
        assert "Part 2 output" in result

    def test_single_chunk_result_no_merge(self):
        """If only one chunk has a result, skip merge step."""
        llm = FakeLLM([
            "",          # chunk 1 empty
            "Only result",
        ])
        mock_counter = MagicMock()
        mock_counter.count.side_effect = lambda text: len(text)
        mock_counter.split_by_tokens.return_value = ["chunk1", "chunk2"]
        llm._token_counter = mock_counter

        result = llm.complete_chunked("Process:\n{text}", "x" * 200_000)

        assert result == "Only result"
        assert llm._call_count == 2  # 2 chunks, no merge


class TestModelNameProperty:
    """Test that model_name setter re-creates TokenCounter."""

    def test_initial_model(self):
        llm = FakeLLM()
        assert llm.model_name == "fake-model"
        assert isinstance(llm._token_counter, TokenCounter)

    def test_setter_updates_counter(self):
        llm = FakeLLM()
        old_counter = llm._token_counter
        llm.model_name = "new-model"
        assert llm.model_name == "new-model"
        assert llm._token_counter is not old_counter
