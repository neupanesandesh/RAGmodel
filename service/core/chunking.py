from typing import List, Tuple
import re
from service.logging_config import get_component_logger

# Initialize logger for chunking module
logger = get_component_logger("chunking")

def estimate_tokens(text: str) -> int:
    word_count = len(text.split())
    # Convert to token estimate (1 token ≈ 0.75 words)
    return int(word_count / 0.75)

def split_into_paragraphs(text: str) -> List[str]:
    paragraphs = re.split(r'\n\s*\n', text)
    # Clean up and filter empty paragraphs
    return [p.strip() for p in paragraphs if p.strip()]

def create_overlapping_chunks(text: str, chunk_size: int, overlap: int) -> List[str]:
    
    words = text.split()
    # Convert token sizes to approximate word counts
    chunk_words = int(chunk_size * 0.75)
    overlap_words = int(overlap * 0.75)
    
    chunks = []
    start = 0
    
    while start < len(words):
        end = start + chunk_words
        chunk = ' '.join(words[start:end])
        chunks.append(chunk)
        
        # Move start forward by chunk_size - overlap
        start += (chunk_words - overlap_words)
        
        # Break if we've covered the text
        if end >= len(words):
            break
    
    return chunks

def chunk_text(text: str) -> List[Tuple[str, int]]:
    # Clean the input text
    text = text.strip()
    if not text:
        logger.warning("Empty text provided for chunking")
        return []

    # Estimate total tokens
    total_tokens = estimate_tokens(text)

    logger.debug(
        f"Chunking text: {total_tokens} estimated tokens",
        extra={"text_length": len(text), "estimated_tokens": total_tokens}
    )

    # Strategy 1: Small text - keep as single chunk
    if total_tokens < 500:
        logger.debug("Using strategy: single chunk (small text)")
        return [(text, 0)]

    # Strategy 2: Medium text - split at paragraphs
    elif total_tokens < 1500:
        paragraphs = split_into_paragraphs(text)
        chunks = []
        current_chunk = []
        current_tokens = 0
        
        for para in paragraphs:
            para_tokens = estimate_tokens(para)
            
            # If single paragraph is too large, it becomes its own chunk
            if para_tokens > 400:
                if current_chunk:
                    chunks.append(' '.join(current_chunk))
                    current_chunk = []
                    current_tokens = 0
                chunks.append(para)
                continue
            
            # If adding this paragraph exceeds target, save current chunk
            if current_tokens + para_tokens > 400 and current_chunk:
                chunks.append(' '.join(current_chunk))
                # Keep last paragraph's final sentences for overlap
                last_para = current_chunk[-1]
                last_sentences = '. '.join(last_para.split('. ')[-2:]) if '. ' in last_para else ''
                current_chunk = [last_sentences, para] if last_sentences else [para]
                current_tokens = estimate_tokens(' '.join(current_chunk))
            else:
                current_chunk.append(para)
                current_tokens += para_tokens
        
        # Add remaining chunk
        if current_chunk:
            chunks.append(' '.join(current_chunk))

        result = [(chunk, idx) for idx, chunk in enumerate(chunks)]
        logger.debug(
            f"Using strategy: paragraph-based (medium text) - created {len(result)} chunks",
            extra={"chunks_count": len(result), "strategy": "paragraph"}
        )
        return result

    # Strategy 3: Large text - sliding window with overlap
    else:
        chunks = create_overlapping_chunks(
            text=text,
            chunk_size=400,  # Target 400 tokens per chunk
            overlap=100       # 100 token overlap for continuity
        )
        result = [(chunk, idx) for idx, chunk in enumerate(chunks)]
        logger.debug(
            f"Using strategy: sliding window (large text) - created {len(result)} chunks",
            extra={"chunks_count": len(result), "strategy": "sliding_window"}
        )
        return result
    
# def validate_chunk_size(text: str, max_tokens: int = 2048) -> bool:
#     """
#     Validate that a text chunk is within Gemini's token limit.
    
#     Args:
#         text: Text chunk to validate
#         max_tokens: Maximum allowed tokens (default: 2048 for Gemini)
        
#     Returns:
#         True if chunk is valid, False otherwise
#     """
#     return estimate_tokens(text) <= max_tokens


# if __name__ == "__main__":
#     # Test with different text sizes
#
#     # Small text
#     small = "This is a short piece of text that should stay as one chunk."
#     print(f"Small text ({estimate_tokens(small)} tokens):")
#     print(f"  Chunks: {len(chunk_text(small))}\n")
#
#     # Medium text - Make it actually medium sized (500-1500 tokens)
#     medium_paragraph = """
#     This is a substantial paragraph of a medium-length document. It contains detailed information
#     about various topics including technology, science, and methodology. The paragraph explores
#     different concepts and provides comprehensive explanations that help readers understand
#     the subject matter thoroughly. We discuss implementation details, theoretical frameworks,
#     and practical applications that are relevant to the domain.
#     """
#
#     medium = (medium_paragraph + "\n\n") * 15  # Repeat to get ~700 tokens
#     print(f"Medium text ({estimate_tokens(medium)} tokens):")
#     chunks = chunk_text(medium)
#     print(f"  Chunks: {len(chunks)}")
#     for i, (chunk, idx) in enumerate(chunks):  # Show all chunks
#         print(f"  Chunk {idx}: {estimate_tokens(chunk)} tokens")
#     print()
#
#     # Large text - Keep the same
#     large = "This is a very long document. " * 300
#     print(f"Large text ({estimate_tokens(large)} tokens):")
#     chunks = chunk_text(large)
#     print(f"  Chunks: {len(chunks)}")
#     for i, (chunk, idx) in enumerate(chunks[:3]):  # Show first 3
#         print(f"  Chunk {idx}: {estimate_tokens(chunk)} tokens")