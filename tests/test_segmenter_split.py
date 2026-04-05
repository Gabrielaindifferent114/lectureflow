"""Tests for segmenter oversized group splitting."""

from src.core.segmenter import SemanticSegmenter, _MAX_SEGMENT_CHARS


class TestResolveGroupsSplitting:
    """Test that resolve_groups splits oversized groups."""

    def setup_method(self):
        self.segmenter = SemanticSegmenter.__new__(SemanticSegmenter)
        self.segmenter.chunk_size = 2

    def test_small_group_no_split(self):
        segments = [
            ("short text", 0.0, 5.0),
            ("more text", 5.0, 5.0),
        ]
        chunk_groups = [[0]]  # one group with chunk 0 (segments 0-1)
        results = self.segmenter.resolve_groups(segments, chunk_groups)
        assert len(results) == 1
        assert "short text" in results[0]["text"]
        assert "more text" in results[0]["text"]

    def test_oversized_group_splits(self):
        # Create segments where combined text exceeds _MAX_SEGMENT_CHARS
        big_text = "x" * (_MAX_SEGMENT_CHARS // 2 + 1000)
        segments = [
            (big_text, 0.0, 10.0),
            (big_text, 10.0, 10.0),
            ("small", 20.0, 5.0),
            ("small", 25.0, 5.0),
        ]
        chunk_groups = [[0, 1]]  # one group spanning chunks 0-1 (all 4 segments)
        results = self.segmenter.resolve_groups(segments, chunk_groups)

        # Should split into at least 2 sub-segments
        assert len(results) >= 2
        # Each result's text should be within the limit
        for r in results:
            assert len(r["text"]) <= _MAX_SEGMENT_CHARS + len(big_text)  # at most one segment over

    def test_timestamps_preserved_after_split(self):
        big_text = "x" * (_MAX_SEGMENT_CHARS + 100)
        segments = [
            (big_text, 0.0, 10.0),
            (big_text, 10.0, 10.0),
        ]
        chunk_groups = [[0]]  # one chunk with 2 segments
        results = self.segmenter.resolve_groups(segments, chunk_groups)

        assert results[0]["start_time"] == 0.0
        if len(results) > 1:
            assert results[1]["start_time"] == 10.0
            assert results[1]["end_time"] == 20.0

    def test_empty_group_skipped(self):
        segments = [("text", 0.0, 5.0)]
        chunk_groups = [[], [0]]
        results = self.segmenter.resolve_groups(segments, chunk_groups)
        assert len(results) == 1
