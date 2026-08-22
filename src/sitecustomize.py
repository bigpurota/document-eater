"""Apply the Document Eater socket boundary to isolated Python runtimes."""

from document_eater.privacy import enable_strict_offline, strict_offline_requested

if strict_offline_requested():
    enable_strict_offline()
