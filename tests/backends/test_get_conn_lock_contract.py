"""Contract test for ``SQLiteBackend._get_conn`` locking.

Backs TODO 26.18 (`sqlite-get-conn-lock-contract`). The chosen contract
is variant (a): ``_get_conn`` acquires ``self._lock`` itself. The lock is
an ``RLock``, so re-entry from already-locked callers (``store``,
``query_vector``, ...) is harmless. This test pins the contract so a
future refactor cannot silently drop the acquisition.
"""

from __future__ import annotations

import threading

from crossmem.backends.sqlite_backend import SQLiteBackend


class _LockSpy:
    """Wrap an ``RLock`` so we can count ``__enter__`` calls."""

    def __init__(self) -> None:
        self._inner = threading.RLock()
        self.enter_count = 0
        self.exit_count = 0

    def __enter__(self) -> _LockSpy:
        self._inner.acquire()
        self.enter_count += 1
        return self

    def __exit__(self, exc_type, exc, tb) -> None:  # noqa: ANN001
        self.exit_count += 1
        self._inner.release()

    def acquire(self, *args: object, **kwargs: object) -> bool:
        return self._inner.acquire(*args, **kwargs)

    def release(self) -> None:
        self._inner.release()


class TestGetConnAcquiresLock:
    """``_get_conn`` must acquire ``self._lock`` itself (variant a)."""

    def test_get_conn_enters_lock_on_call(self, tmp_path: object) -> None:
        db_file = tmp_path / "lock-contract.db"  # type: ignore[operator]
        backend = SQLiteBackend(db_file)

        spy = _LockSpy()
        backend._lock = spy  # type: ignore[assignment]

        before = spy.enter_count
        backend._get_conn()
        after = spy.enter_count

        assert after > before, (
            "_get_conn must acquire self._lock (variant a contract); "
            f"enter_count went {before} -> {after}"
        )
        assert spy.exit_count == after, (
            "_get_conn must release the lock it acquired; "
            f"enter={after} exit={spy.exit_count}"
        )

        backend.close()

    def test_get_conn_is_reentrant_under_existing_lock(self, tmp_path: object) -> None:
        """Calling ``_get_conn`` while already holding the lock must not
        deadlock — the RLock contract is what makes variant (a) safe."""
        db_file = tmp_path / "reentrant.db"  # type: ignore[operator]
        backend = SQLiteBackend(db_file)

        with backend._lock:
            conn = backend._get_conn()
            assert conn is not None

        backend.close()
