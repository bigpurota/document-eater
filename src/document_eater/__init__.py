"""Private document ingestion and retrieval primitives."""

from .ingest import discover_documents, ingest_document
from .pdf import ingest_pdf, inspect_pdf

__all__ = ["discover_documents", "ingest_document", "inspect_pdf", "ingest_pdf"]
__version__ = "0.1.0"
