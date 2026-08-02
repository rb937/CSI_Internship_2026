"""
document_processor.py
Extracts text from PDF documents and splits it into overlapping chunks
suitable for embedding.
"""

from pypdf import PdfReader


def extract_text_from_pdf(file_path_or_buffer) -> str:
    """Extract raw text from a PDF file path or file-like object (e.g. Streamlit upload)."""
    reader = PdfReader(file_path_or_buffer)
    pages = []
    for page in reader.pages:
        text = page.extract_text() or ""
        pages.append(text)
    return "\n".join(pages)


def extract_text_from_txt(file_path_or_buffer) -> str:
    """Extract raw text from a .txt file path or file-like object (e.g. Streamlit upload)."""
    if hasattr(file_path_or_buffer, "read"):
        content = file_path_or_buffer.read()
        if isinstance(content, bytes):
            content = content.decode("utf-8", errors="ignore")
        return content
    with open(file_path_or_buffer, "r", encoding="utf-8", errors="ignore") as f:
        return f.read()


def extract_text(file_path_or_buffer, filename: str) -> str:
    """Dispatch to the correct extractor based on file extension."""
    ext = filename.lower().rsplit(".", 1)[-1]
    if ext == "pdf":
        return extract_text_from_pdf(file_path_or_buffer)
    elif ext == "txt":
        return extract_text_from_txt(file_path_or_buffer)
    else:
        raise ValueError(f"Unsupported file type: .{ext} (supported: .pdf, .txt)")


def chunk_text(text: str, chunk_size: int = 800, chunk_overlap: int = 150) -> list:
    """
    Split text into overlapping chunks by character count.
    Simple sliding-window splitter — avoids pulling in LangChain for a single utility.
    """
    text = text.strip()
    if not text:
        return []

    chunks = []
    start = 0
    text_length = len(text)

    while start < text_length:
        end = start + chunk_size
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        start += chunk_size - chunk_overlap

    return chunks


def process_document(file_path_or_buffer, filename: str, chunk_size: int = 800, chunk_overlap: int = 150) -> list:
    """Full pipeline: PDF/TXT -> raw text -> list of text chunks."""
    text = extract_text(file_path_or_buffer, filename)
    return chunk_text(text, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
