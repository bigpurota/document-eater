"""Private document ingestion and retrieval primitives."""

from .pdf import ingest_pdf, inspect_pdf

__all__ = ["inspect_pdf", "ingest_pdf"]
__version__ = "0.1.0"
