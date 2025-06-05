def chunk_text(text: str, chunk_size: int, chunk_overlap: int) -> list[str]:
    """
    Splits text into fixed-size chunks with overlap.

    Args:
        text: The input text string.
        chunk_size: The desired size of each chunk (in characters).
        chunk_overlap: The number of characters to overlap between chunks.

    Returns:
        A list of text chunks.
    """
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if chunk_overlap < 0:
        raise ValueError("chunk_overlap must be non-negative")
    if chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be less than chunk_size")

    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start : min(end, len(text))])
        start += chunk_size - chunk_overlap
    return chunks


if __name__ == "__main__":
    # Example Usage:
    sample_text = "This is a long piece of text that needs to be chunked for processing. We want to split it into smaller pieces to manage the input size for a language model. Overlap helps maintain context between chunks."
    chunk_size = 50
    chunk_overlap = 10

    text_chunks = chunk_text(sample_text, chunk_size, chunk_overlap)

    print(f"Original Text Length: {len(sample_text)}")
    print(f"Chunk Size: {chunk_size}")
    print(f"Chunk Overlap: {chunk_overlap}")
    print(f"Number of Chunks: {len(text_chunks)}")
    for i, chunk in enumerate(text_chunks):
        print(f"Chunk {i+1} (Length: {len(chunk)}):\n---\n{chunk}\n---")
