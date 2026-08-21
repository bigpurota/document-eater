from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

from .audit import audit_corpus
from .index import index_artifacts, search
from .ingest import ingest_document
from .llm import (
    ABLITERATED_GENERATION,
    ABLITERATED_MODEL,
    BASE_GENERATION,
    BASE_MODEL,
    QwenClient,
    answer_question,
)
from .pdf import inspect_pdf
from .rag import (
    DEFAULT_EMBEDDING_MODEL,
    BgeM3Encoder,
    BgeM3Reranker,
    FastEmbedEncoder,
    HybridRetriever,
    index_dense,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="document-eater")
    commands = parser.add_subparsers(dest="command", required=True)

    inspect_cmd = commands.add_parser("inspect", help="inspect PDF text coverage without OCR")
    inspect_cmd.add_argument("pdf", type=Path)
    inspect_cmd.add_argument("--min-native-chars", type=int, default=40)

    ingest_cmd = commands.add_parser("ingest", help="extract a document and build its graph")
    ingest_cmd.add_argument("document", type=Path)
    ingest_cmd.add_argument("--output", type=Path, default=Path("artifacts"))
    ingest_cmd.add_argument("--ocr", choices=("auto", "never", "always"), default="auto")
    ingest_cmd.add_argument("--languages", default="rus+eng")
    ingest_cmd.add_argument("--dpi", type=int, default=300)
    ingest_cmd.add_argument("--min-native-chars", type=int, default=40)

    index_cmd = commands.add_parser("index", help="build the local FTS5 control index")
    index_cmd.add_argument("artifacts", type=Path)
    index_cmd.add_argument("--database", type=Path, default=Path("data/index.sqlite3"))
    index_cmd.add_argument("--max-chars", type=int, default=1800)
    index_cmd.add_argument("--hybrid", action="store_true")
    index_cmd.add_argument("--quality", action="store_true")
    index_cmd.add_argument("--embedding-model", default=DEFAULT_EMBEDDING_MODEL)
    index_cmd.add_argument("--embedding-cache", type=Path, default=Path("models/retrieval"))

    search_cmd = commands.add_parser("search", help="search the local control index")
    search_cmd.add_argument("query")
    search_cmd.add_argument("--database", type=Path, default=Path("data/index.sqlite3"))
    search_cmd.add_argument("--limit", type=int, default=10)
    search_cmd.add_argument("--mode", choices=("quality", "hybrid", "lexical"), default="quality")
    search_cmd.add_argument("--embedding-model", default=DEFAULT_EMBEDDING_MODEL)
    search_cmd.add_argument("--embedding-cache", type=Path, default=Path("models/retrieval"))

    ask_cmd = commands.add_parser("ask", help="retrieve evidence and ask a local Qwen endpoint")
    ask_cmd.add_argument("question")
    ask_cmd.add_argument("--database", type=Path, default=Path("data/index.sqlite3"))
    ask_cmd.add_argument("--base-url", default="http://127.0.0.1:8080/v1")
    ask_cmd.add_argument("--profile", choices=("base", "abliterated"), default="base")
    ask_cmd.add_argument("--model", default=None)
    ask_cmd.add_argument("--retrieval-limit", type=int, default=10)
    ask_cmd.add_argument("--evidence-chars", type=int, default=12000)
    ask_cmd.add_argument("--retrieval", choices=("quality", "hybrid", "lexical"), default="quality")
    ask_cmd.add_argument("--embedding-model", default=DEFAULT_EMBEDDING_MODEL)
    ask_cmd.add_argument("--embedding-cache", type=Path, default=Path("models/retrieval"))

    audit_cmd = commands.add_parser(
        "audit", help="ingest a document folder and build a requirement compliance report"
    )
    audit_cmd.add_argument("input", type=Path, help="supported document or directory")
    audit_cmd.add_argument("--output", type=Path, default=Path("audit-run"))
    audit_cmd.add_argument("--use-llm", action="store_true", help="verify with local Qwen")
    audit_cmd.add_argument("--base-url", default="http://127.0.0.1:8080/v1")
    audit_cmd.add_argument("--profile", choices=("base", "abliterated"), default="base")
    audit_cmd.add_argument("--model", default=None)
    audit_cmd.add_argument("--ocr", choices=("auto", "never", "always"), default="auto")
    audit_cmd.add_argument("--languages", default="rus+eng")
    audit_cmd.add_argument("--dpi", type=int, default=300)
    audit_cmd.add_argument("--evidence-limit", type=int, default=10)
    audit_cmd.add_argument(
        "--retrieval", choices=("quality", "hybrid", "lexical"), default="quality"
    )
    audit_cmd.add_argument("--embedding-model", default=DEFAULT_EMBEDDING_MODEL)
    audit_cmd.add_argument("--embedding-cache", type=Path, default=Path("models/retrieval"))
    return parser


def _qwen_client(args: argparse.Namespace) -> QwenClient:
    abliterated = args.profile == "abliterated"
    return QwenClient(
        args.base_url,
        args.model or (ABLITERATED_MODEL if abliterated else BASE_MODEL),
        generation_options=ABLITERATED_GENERATION if abliterated else BASE_GENERATION,
        use_system_prompt=not abliterated,
    )


def main() -> None:
    args = _parser().parse_args()
    if args.command == "inspect":
        result = inspect_pdf(args.pdf, min_native_chars=args.min_native_chars)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return
    if args.command == "index":
        result = index_artifacts(args.artifacts, args.database, max_chars=args.max_chars)
        if args.quality:
            encoder = BgeM3Encoder(args.embedding_cache)
            result["dense_chunks"] = int(index_dense(args.database, encoder)["chunks"])
        elif args.hybrid:
            encoder = FastEmbedEncoder(args.embedding_model, args.embedding_cache)
            result["dense_chunks"] = int(index_dense(args.database, encoder)["chunks"])
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return
    if args.command == "search":
        if args.mode == "quality":
            encoder = BgeM3Encoder(args.embedding_cache)
            hits = HybridRetriever(
                args.database, encoder, BgeM3Reranker(args.embedding_cache)
            ).search(args.query, limit=args.limit)
        elif args.mode == "hybrid":
            encoder = FastEmbedEncoder(args.embedding_model, args.embedding_cache)
            hits = HybridRetriever(args.database, encoder).search(args.query, limit=args.limit)
        else:
            hits = search(args.database, args.query, limit=args.limit)
        print(json.dumps([asdict(hit) for hit in hits], ensure_ascii=False, indent=2))
        return
    if args.command == "ask":
        client = _qwen_client(args)
        if args.retrieval == "quality":
            encoder = BgeM3Encoder(args.embedding_cache)
            retriever = HybridRetriever(args.database, encoder, BgeM3Reranker(args.embedding_cache))

            def searcher(_database: str, query: str, limit: int):
                return retriever.search(query, limit=limit)

        elif args.retrieval == "hybrid":
            encoder = FastEmbedEncoder(args.embedding_model, args.embedding_cache)
            retriever = HybridRetriever(args.database, encoder)

            def searcher(_database: str, query: str, limit: int):
                return retriever.search(query, limit=limit)

        else:
            searcher = search
        answer = answer_question(
            str(args.database),
            args.question,
            client,
            retrieval_limit=args.retrieval_limit,
            evidence_chars=args.evidence_chars,
            searcher=searcher,
            retrieval_mode=args.retrieval,
        )
        print(json.dumps(asdict(answer), ensure_ascii=False, indent=2))
        return
    if args.command == "audit":
        report = audit_corpus(
            args.input,
            args.output,
            client=_qwen_client(args) if args.use_llm else None,
            ocr=args.ocr,
            languages=args.languages,
            dpi=args.dpi,
            evidence_limit=args.evidence_limit,
            progress=lambda message: print(message, file=sys.stderr, flush=True),
            retrieval_mode=args.retrieval,
            embedding_model=args.embedding_model,
            embedding_cache=args.embedding_cache,
        )
        result = {
            "verification_mode": report.verification_mode,
            "retrieval_mode": report.retrieval_mode,
            "requirements": len(report.items),
            "summary": report.summary,
            "report": str(Path(report.run_directory) / "report.html"),
            "json": str(Path(report.run_directory) / "audit.json"),
            "csv": str(Path(report.run_directory) / "requirements.csv"),
        }
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return
    destination = ingest_document(
        args.document,
        args.output,
        ocr=args.ocr,
        languages=args.languages,
        dpi=args.dpi,
        min_native_chars=args.min_native_chars,
    )
    print(destination)


if __name__ == "__main__":
    main()
