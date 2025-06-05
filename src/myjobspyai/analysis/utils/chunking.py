"""Text chunking utilities for processing large documents."""

from typing import Iterator, List


def chunk_text(
    text: str,
    max_chunk_size: int = 2000,
    overlap: int = 100,
    separators: List[str] = None,
) -> Iterator[str]:
    """Split text into overlapping chunks.

    Args:
        text: The text to split into chunks.
        max_chunk_size: Maximum size of each chunk in characters.
        overlap: Number of characters to overlap between chunks.
        separators: List of separator strings to try when splitting.

    Yields:
        Chunks of text with the specified maximum size and overlap.
    """
    if not text:
        return

    if separators is None:
        separators = ["\n\n", ". ", "! ", "? ", " ", ""]

    start = 0
    text_length = len(text)

    while start < text_length:
        # Calculate the end position for this chunk
        end = min(start + max_chunk_size, text_length)

        if end == text_length:
            # We've reached the end of the text
            yield text[start:end]
            break

        # Try to find a good split point
        split_pos = -1
        for sep in separators:
            if not sep:  # If no separator is found, we'll split at max_chunk_size
                split_pos = end
                break

            # Look for the last occurrence of the separator in the chunk
            pos = text.rfind(sep, start, end)
            if pos != -1 and pos > start + (max_chunk_size // 2):
                split_pos = pos + len(sep)
                break

        if split_pos == -1:
            # No good split point found, just split at max_chunk_size
            split_pos = end

        chunk = text[start:split_pos].strip()
        if chunk:  # Only yield non-empty chunks
            yield chunk

        # Move the start position, accounting for overlap
        start = max(start + 1, split_pos - overlap)
