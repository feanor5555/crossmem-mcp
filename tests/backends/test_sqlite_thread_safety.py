"""Thread-safety tests for SQLiteBackend.

Backs TODO 23.2 (`sqlite-thread-safe-connection`). The default
``sqlite3.connect()`` binds the connection to the creating thread, so the
moment FastMCP dispatches a tool call onto a worker thread the backend
crashes with "SQLite objects created in a thread can only be used in that
same thread". These tests pin the contract:

* Negative path: a *cross-thread* use of the backend must NOT raise the
  thread-affinity ``ProgrammingError``.
* Positive path: a stress run with several writers and readers terminates
  without exceptions and ends in a consistent state (no lost writes, no
  ``sqlite3`` errors leaking from worker threads).
"""

from __future__ import annotations

import sqlite3
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

import pytest

from crossmem.backends.sqlite_backend import SQLiteBackend
from crossmem.core.models import Document, Metadata, generate_content_hash


def _make_doc(doc_id: str, content: str = "thread-safe content") -> Document:
    return Document(
        id=doc_id,
        content=content,
        embedding=[0.1] * 384,
        metadata=Metadata(
            source_url=f"https://example.com/{doc_id}",
            title=f"doc {doc_id}",
            source_type="web",
            stored_at="2026-05-11T00:00:00Z",
            embedding_model="test-model",
            embedding_dim=384,
            namespace="default",
            tags=["python"],
            content_hash=generate_content_hash(content),
        ),
    )


class TestSQLiteBackendCrossThread:
    """Connection must not be pinned to its creating thread."""

    def test_store_from_other_thread_does_not_raise_thread_affinity_error(
        self, tmp_path: object
    ) -> None:
        db_file = tmp_path / "thread.db"  # type: ignore[operator]
        backend = SQLiteBackend(db_file)

        errors: list[BaseException] = []

        def worker() -> None:
            try:
                backend.store(_make_doc("doc-cross-thread"))
            except BaseException as exc:  # noqa: BLE001 - capture for assertion
                errors.append(exc)

        t = threading.Thread(target=worker)
        t.start()
        t.join()

        # The whole point: no ProgrammingError about thread affinity.
        thread_errors = [
            e
            for e in errors
            if isinstance(e, sqlite3.ProgrammingError) and "thread" in str(e).lower()
        ]
        assert thread_errors == [], (
            f"cross-thread use raised sqlite3 thread-affinity error: {errors!r}"
        )
        assert errors == [], f"cross-thread use raised: {errors!r}"

        # Sanity: the row landed in the DB and is readable from the main thread.
        assert backend.get_by_id("doc-cross-thread") is not None
        backend.close()

    def test_query_from_other_thread_does_not_raise(self, tmp_path: object) -> None:
        db_file = tmp_path / "thread-query.db"  # type: ignore[operator]
        backend = SQLiteBackend(db_file)
        backend.store(_make_doc("seed"))

        errors: list[BaseException] = []
        results: list[int] = []

        def worker() -> None:
            try:
                hits = backend.query_vector([0.1] * 384, top_k=5)
                results.append(len(hits))
            except BaseException as exc:  # noqa: BLE001
                errors.append(exc)

        t = threading.Thread(target=worker)
        t.start()
        t.join()

        assert errors == [], f"cross-thread query raised: {errors!r}"
        assert results == [1]
        backend.close()


class TestSQLiteBackendConcurrentStress:
    """Mixed writer/reader stress run must terminate without data loss."""

    def test_parallel_store_and_query_no_errors(self, tmp_path: object) -> None:
        db_file = tmp_path / "stress.db"  # type: ignore[operator]
        backend = SQLiteBackend(db_file)

        writers = 4
        per_writer = 25
        total = writers * per_writer

        def write(start: int) -> int:
            count = 0
            for i in range(per_writer):
                backend.store(
                    _make_doc(f"w{start}-d{i}", content=f"content-{start}-{i}")
                )
                count += 1
            return count

        def read(_unused: int) -> int:
            results = backend.query_vector([0.1] * 384, top_k=10)
            return len(results)

        with ThreadPoolExecutor(max_workers=writers + 2) as pool:
            write_futures = [pool.submit(write, w) for w in range(writers)]
            read_futures = [pool.submit(read, r) for r in range(8)]
            for fut in as_completed(write_futures + read_futures):
                # Re-raise any exception captured in the worker; assert below
                # if a thread-affinity error escaped.
                fut.result()

        # Every doc landed exactly once — no lost writes, no duplicates.
        conn = backend._get_conn()
        count = conn.execute("SELECT COUNT(*) FROM documents").fetchone()[0]
        assert count == total, (
            f"expected {total} rows after concurrent writes, got {count}"
        )
        backend.close()


class TestSQLiteBackendCheckSameThreadFlag:
    """Document the configuration contract used to enable threading."""

    def test_connection_allows_cross_thread_use(self, tmp_path: object) -> None:
        # The backend must build its connection with check_same_thread=False;
        # exposing that explicitly here guards against silent regressions.
        db_file = tmp_path / "flag.db"  # type: ignore[operator]
        backend = SQLiteBackend(db_file)
        conn = backend._get_conn()
        # sqlite3.Connection does not expose check_same_thread directly,
        # but a real cross-thread execute() is the canonical contract test.
        result: list[str] = []

        def run() -> None:
            try:
                conn.execute("SELECT 1").fetchone()
                result.append("ok")
            except sqlite3.ProgrammingError as exc:  # pragma: no cover - defensive
                result.append(f"err:{exc}")

        t = threading.Thread(target=run)
        t.start()
        t.join()
        assert result == ["ok"]
        backend.close()


@pytest.mark.parametrize("workers", [2, 4, 8])
def test_stress_matrix_no_data_loss(tmp_path: object, workers: int) -> None:
    """Parameterised stress run — multiple thread counts, same invariant."""
    db_file = tmp_path / f"stress-{workers}.db"  # type: ignore[operator]
    backend = SQLiteBackend(db_file)

    per_worker = 10

    def write(start: int) -> None:
        for i in range(per_worker):
            backend.store(_make_doc(f"m{workers}-{start}-{i}"))

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(write, w) for w in range(workers)]
        for fut in as_completed(futures):
            fut.result()

    conn = backend._get_conn()
    count = conn.execute("SELECT COUNT(*) FROM documents").fetchone()[0]
    assert count == workers * per_worker
    backend.close()
