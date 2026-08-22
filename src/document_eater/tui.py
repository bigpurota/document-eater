from __future__ import annotations

import gc
import json
import os
import sys
import webbrowser
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, TextIO
from urllib.parse import urlparse

from .audit import AuditReport, audit_corpus
from .index import search
from .llm import (
    ABLITERATED_GENERATION,
    ABLITERATED_MODEL,
    BASE_GENERATION,
    BASE_MODEL,
    REMOTE_ABLITERATED_GENERATION,
    REMOTE_BASE_GENERATION,
    QwenClient,
    answer_question,
)
from .privacy import enable_strict_offline
from .rag import (
    DEFAULT_EMBEDDING_MODEL,
    BgeM3Encoder,
    BgeM3Reranker,
    FastEmbedEncoder,
    HybridRetriever,
)

_LOOPBACK_HOSTS = {"127.0.0.1", "localhost", "::1"}


@dataclass
class TUISettings:
    source: Path
    workspace: Path
    base_url: str = "http://127.0.0.1:8080/v1"
    profile: str = "base"
    model: str | None = None
    retrieval: str = "quality"
    embedding_model: str = DEFAULT_EMBEDDING_MODEL
    embedding_cache: Path = Path("models/retrieval")
    allow_remote: bool = False
    strict_offline: bool = True
    api_key_env: str = "DOCUMENT_EATER_QWEN_API_KEY"
    timeout_seconds: int = 300
    color: bool = True

    @property
    def selected_model(self) -> str:
        if self.model:
            return self.model
        return ABLITERATED_MODEL if self.profile == "abliterated" else BASE_MODEL

    @property
    def remote(self) -> bool:
        return is_remote_endpoint(self.base_url)


def is_remote_endpoint(base_url: str) -> bool:
    parsed = urlparse(base_url)
    return bool(parsed.hostname and parsed.hostname not in _LOOPBACK_HOSTS)


class Terminal:
    def __init__(
        self,
        *,
        input_fn: Callable[[str], str] = input,
        output: TextIO = sys.stdout,
        color: bool = True,
    ) -> None:
        self.input_fn = input_fn
        self.output = output
        self.color = color and bool(getattr(output, "isatty", lambda: False)())

    def clear(self) -> None:
        if self.color:
            self.output.write("\033[2J\033[H")
            self.output.flush()

    def write(self, value: str = "") -> None:
        print(value, file=self.output, flush=True)

    def ask(self, prompt: str) -> str:
        return self.input_fn(prompt).strip()

    def pause(self) -> None:
        self.input_fn("\nEnter — вернуться в меню ")

    def style(self, value: str, code: str) -> str:
        return f"\033[{code}m{value}\033[0m" if self.color else value


class DocumentTUI:
    def __init__(self, settings: TUISettings, terminal: Terminal | None = None) -> None:
        self.settings = settings
        self.terminal = terminal or Terminal(color=settings.color)

    def run(self) -> None:
        if self.settings.strict_offline:
            enable_strict_offline()
        self._validate_paths()
        while True:
            self._dashboard()
            choice = self.terminal.ask("Выбор: ").casefold()
            if choice in {"q", "quit", "0"}:
                self.terminal.write("До встречи.")
                return
            actions = {
                "1": lambda: self._run_audit(use_llm=False),
                "2": lambda: self._run_audit(use_llm=True),
                "3": self._ask_documents,
                "4": self._open_report,
                "5": self._configure,
            }
            action = actions.get(choice)
            if action is None:
                self._message("Неизвестная команда.", error=True)
                continue
            try:
                action()
            except (OSError, RuntimeError, ValueError, json.JSONDecodeError) as exc:
                self._message(str(exc), error=True)

    def _validate_paths(self) -> None:
        if self.settings.strict_offline and self.settings.remote:
            raise ValueError("Строгий offline-режим запрещает внешний endpoint")
        source = self.settings.source.expanduser().resolve()
        workspace = self.settings.workspace.expanduser().resolve()
        if not source.exists():
            raise ValueError(f"Папка или документ не найден: {source}")
        if source == workspace or source.is_relative_to(workspace):
            raise ValueError("Workspace не может быть исходной папкой или её родителем")
        self.settings.source = source
        self.settings.workspace = workspace
        self.settings.embedding_cache = self.settings.embedding_cache.expanduser().resolve()

    def _dashboard(self) -> None:
        self.terminal.clear()
        title = self.terminal.style("DOCUMENT EATER", "1;36")
        endpoint_kind = (
            "ВНЕШНИЙ — RAG-фрагменты покидают Mac" if self.settings.remote else "локальный"
        )
        if self.settings.remote:
            endpoint_kind = self.terminal.style(endpoint_kind, "1;31")
        self.terminal.write(f"╭─ {title} ─────────────────────────────────────────╮")
        self.terminal.write(f"│ Документы : {self.settings.source}")
        self.terminal.write(f"│ Workspace : {self.settings.workspace}")
        self.terminal.write(f"│ Endpoint  : {self.settings.base_url} ({endpoint_kind})")
        self.terminal.write(
            "│ Network   : только loopback (строгий offline)"
            if self.settings.strict_offline
            else "│ Network   : внешний доступ явно разрешён"
        )
        self.terminal.write(f"│ Model     : {self.settings.selected_model}")
        self.terminal.write(f"│ RAG       : {self.settings.retrieval}")
        self._write_cached_status()
        self.terminal.write("╰────────────────────────────────────────────────────╯")
        self.terminal.write("  1  Подготовить корпус без Qwen")
        self.terminal.write("  2  Проверить требования через Qwen")
        self.terminal.write("  3  Задать вопрос документам")
        self.terminal.write("  4  Открыть HTML-отчёт")
        self.terminal.write("  5  Настройки")
        self.terminal.write("  0  Выход")

    def _write_cached_status(self) -> None:
        audit_path = self.settings.workspace / "audit.json"
        if not audit_path.is_file():
            self.terminal.write("│ Статус    : корпус ещё не подготовлен")
            return
        try:
            payload = json.loads(audit_path.read_text(encoding="utf-8"))
            summary = payload.get("summary") or {}
            rendered = " ".join(
                f"{key}:{summary.get(key, 0)}"
                for key in ("PASS", "PARTIAL", "FAIL", "UNKNOWN", "CONFLICT")
            )
            mode = payload.get("verification_mode", "unknown")
            self.terminal.write(f"│ Статус    : {mode} · {rendered}")
        except (OSError, ValueError, json.JSONDecodeError):
            self.terminal.write("│ Статус    : workspace повреждён или недописан")

    def _client(self) -> QwenClient:
        abliterated = self.settings.profile == "abliterated"
        if self.settings.remote:
            generation_options = (
                REMOTE_ABLITERATED_GENERATION if abliterated else REMOTE_BASE_GENERATION
            )
        else:
            generation_options = ABLITERATED_GENERATION if abliterated else BASE_GENERATION
        return QwenClient(
            self.settings.base_url,
            self.settings.selected_model,
            timeout_seconds=self.settings.timeout_seconds,
            allow_nonlocal_endpoint=self.settings.allow_remote,
            api_key=os.environ.get(self.settings.api_key_env),
            generation_options=generation_options,
            use_system_prompt=not abliterated,
        )

    def _run_audit(self, *, use_llm: bool) -> None:
        client = self._client() if use_llm else None
        self.terminal.clear()
        self.terminal.write(self.terminal.style("Аудит документов", "1;36"))
        if use_llm and self.settings.remote:
            self.terminal.write(
                self.terminal.style(
                    "Внешнему endpoint будут отправлены выбранные доказательства.", "1;31"
                )
            )
        try:
            report = audit_corpus(
                self.settings.source,
                self.settings.workspace,
                client=client,
                retrieval_mode=self.settings.retrieval,  # type: ignore[arg-type]
                embedding_model=self.settings.embedding_model,
                embedding_cache=self.settings.embedding_cache,
                progress=lambda message: self.terminal.write(f"  {message}"),
            )
            self._show_report_summary(report)
        finally:
            self._release_retrieval_memory()
        self.terminal.pause()

    def _show_report_summary(self, report: AuditReport) -> None:
        self.terminal.write()
        reused = " · использован кэш" if report.reused else ""
        self.terminal.write(
            self.terminal.style(f"Готово: {len(report.items)} требований{reused}", "1;32")
        )
        self.terminal.write(
            "  " + "  ".join(f"{status}:{count}" for status, count in report.summary.items())
        )
        self.terminal.write(f"  Отчёт: {Path(report.run_directory) / 'report.html'}")

    def _ask_documents(self) -> None:
        database = self.settings.workspace / "index.sqlite3"
        if not database.is_file():
            raise ValueError("Сначала подготовьте корпус командой 1 или 2")
        question = self.terminal.ask("Вопрос: ")
        if not question:
            return
        client = self._client()
        self.terminal.write("Локальный поиск доказательств…")
        if self.settings.retrieval == "quality":
            encoder = BgeM3Encoder(self.settings.embedding_cache)
            retriever = HybridRetriever(
                database, encoder, BgeM3Reranker(self.settings.embedding_cache)
            )

            def searcher(_database: str, query: str, limit: int):
                return retriever.search(query, limit=limit)

        elif self.settings.retrieval == "hybrid":
            encoder = FastEmbedEncoder(self.settings.embedding_model, self.settings.embedding_cache)
            retriever = HybridRetriever(database, encoder)

            def searcher(_database: str, query: str, limit: int):
                return retriever.search(query, limit=limit)

        else:
            searcher = search
        try:
            answer = answer_question(
                str(database),
                question,
                client,
                retrieval_limit=6,
                evidence_chars=12_000 if self.settings.remote else 8_000,
                searcher=searcher,
                retrieval_mode=self.settings.retrieval,
            )
        finally:
            self._release_retrieval_memory()
        self.terminal.write()
        self.terminal.write(self.terminal.style("Ответ", "1;36"))
        self.terminal.write(answer.content)
        if answer.evidence:
            self.terminal.write("\nИсточники:")
            for citation in answer.evidence:
                locations = [value for value in citation.get("locations", []) if value]
                location = "–".join(str(value) for value in locations) or str(citation.get("pages"))
                self.terminal.write(
                    f"  • {citation.get('document_id')} · {location} · {citation.get('chunk_id')}"
                )
        self.terminal.pause()

    def _open_report(self) -> None:
        report = self.settings.workspace / "report.html"
        if not report.is_file():
            raise ValueError("Отчёт ещё не создан")
        opened = webbrowser.open(report.resolve().as_uri())
        if not opened:
            self.terminal.write(f"Откройте вручную: {report}")
            self.terminal.pause()

    def _configure(self) -> None:
        while True:
            self.terminal.clear()
            self.terminal.write(self.terminal.style("Настройки", "1;36"))
            self.terminal.write(f"1  Endpoint: {self.settings.base_url}")
            self.terminal.write(f"2  Model: {self.settings.selected_model}")
            self.terminal.write(f"3  Profile: {self.settings.profile}")
            self.terminal.write(f"4  RAG: {self.settings.retrieval}")
            self.terminal.write(f"5  API key env: {self.settings.api_key_env}")
            self.terminal.write("0  Назад")
            choice = self.terminal.ask("Выбор: ").casefold()
            if choice in {"0", "q", "back"}:
                return
            if choice == "1":
                self._configure_endpoint()
            elif choice == "2":
                value = self.terminal.ask("Model ID: ")
                if value:
                    self.settings.model = value
            elif choice == "3":
                value = self.terminal.ask("base / abliterated: ").casefold()
                if value in {"base", "abliterated"}:
                    self.settings.profile = value
                    self.settings.model = None
            elif choice == "4":
                value = self.terminal.ask("quality / hybrid / lexical: ").casefold()
                if value in {"quality", "hybrid", "lexical"}:
                    self.settings.retrieval = value
            elif choice == "5":
                value = self.terminal.ask("Имя переменной окружения: ")
                if value:
                    self.settings.api_key_env = value

    def _configure_endpoint(self) -> None:
        value = self.terminal.ask("OpenAI-compatible base URL: ")
        if not value:
            return
        parsed = urlparse(value)
        if parsed.scheme not in {"http", "https"} or not parsed.hostname:
            raise ValueError("Endpoint должен быть корректным HTTP(S) URL")
        if parsed.username or parsed.password:
            raise ValueError("Не помещайте ключ в URL; используйте API key env")
        if is_remote_endpoint(value):
            if self.settings.strict_offline:
                raise ValueError(
                    "Строгий offline-режим запрещает внешний endpoint. "
                    "Перезапустите CLI с --allow-remote только после разрешения владельца данных."
                )
            if parsed.scheme != "https":
                raise ValueError("Внешний endpoint должен использовать HTTPS")
            confirmation = self.terminal.ask(
                "RAG-фрагменты покинут Mac. Введите REMOTE для подтверждения: "
            )
            if confirmation != "REMOTE":
                self.terminal.write("Изменение отменено")
                return
            self.settings.allow_remote = True
        else:
            self.settings.allow_remote = False
        self.settings.base_url = value.rstrip("/")

    def _message(self, message: str, *, error: bool = False) -> None:
        code = "1;31" if error else "1;32"
        self.terminal.write(self.terminal.style(message, code))
        self.terminal.pause()

    @staticmethod
    def _release_retrieval_memory() -> None:
        gc.collect()
        try:
            import torch

            if torch.backends.mps.is_available():
                torch.mps.empty_cache()
        except (ImportError, RuntimeError):
            pass


def run_tui(settings: TUISettings) -> None:
    try:
        DocumentTUI(settings).run()
    except (EOFError, KeyboardInterrupt):
        print("\nВыход.")
