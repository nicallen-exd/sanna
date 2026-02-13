"""Tests for ReceiptStore — SQLite persistence for reasoning receipts."""

import json
import os
import threading

import pytest

from sanna.store import ReceiptStore


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_receipt(
    *,
    receipt_id="r-001",
    trace_id="sanna-abc123",
    timestamp="2026-02-13T10:00:00+00:00",
    coherence_status="PASS",
    agent_name="my-agent",
    constitution_version="1.0.0",
    checks=None,
    halt_event=None,
):
    if checks is None:
        checks = [
            {"check_id": "C1", "name": "Context Contradiction", "passed": True,
             "severity": "critical", "evidence": None},
            {"check_id": "C2", "name": "Unmarked Inference", "passed": True,
             "severity": "warning", "evidence": None},
        ]
    doc_id = f"{agent_name}/{constitution_version}" if agent_name else None
    constitution_ref = {"document_id": doc_id, "policy_hash": "abc123"} if doc_id else None
    return {
        "receipt_id": receipt_id,
        "trace_id": trace_id,
        "timestamp": timestamp,
        "coherence_status": coherence_status,
        "checks": checks,
        "checks_passed": sum(1 for c in checks if c.get("passed")),
        "checks_failed": sum(1 for c in checks if not c.get("passed")),
        "constitution_ref": constitution_ref,
        "halt_event": halt_event,
    }


@pytest.fixture
def store(tmp_path):
    s = ReceiptStore(str(tmp_path / "test.db"))
    yield s
    s.close()


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestSchemaCreation:
    def test_creates_db_file(self, tmp_path):
        db = str(tmp_path / "new.db")
        s = ReceiptStore(db)
        assert os.path.exists(db)
        s.close()

    def test_schema_version(self, store):
        row = store._conn.execute("SELECT version FROM schema_version").fetchone()
        assert row["version"] == 1

    def test_idempotent_open(self, tmp_path):
        db = str(tmp_path / "idem.db")
        s1 = ReceiptStore(db); s1.close()
        s2 = ReceiptStore(db)
        assert s2._conn.execute("SELECT COUNT(*) FROM schema_version").fetchone()[0] == 1
        s2.close()


class TestSave:
    def test_save_returns_id(self, store):
        assert store.save(_make_receipt()) == "r-001"

    def test_round_trip(self, store):
        r = _make_receipt()
        store.save(r)
        results = store.query()
        assert len(results) == 1
        assert results[0]["receipt_id"] == "r-001"

    def test_generates_id_when_missing(self, store):
        r = _make_receipt()
        del r["receipt_id"]
        rid = store.save(r)
        assert isinstance(rid, str) and len(rid) == 16

    def test_multiple_saves(self, store):
        for i in range(5):
            store.save(_make_receipt(receipt_id=f"r-{i}", trace_id=f"t-{i}"))
        assert store.count() == 5


class TestQuery:
    def test_by_agent_id(self, store):
        store.save(_make_receipt(receipt_id="r1", agent_name="alpha"))
        store.save(_make_receipt(receipt_id="r2", agent_name="beta"))
        assert len(store.query(agent_id="alpha")) == 1

    def test_by_status(self, store):
        store.save(_make_receipt(receipt_id="r1", coherence_status="PASS"))
        store.save(_make_receipt(receipt_id="r2", coherence_status="FAIL"))
        assert len(store.query(status="FAIL")) == 1

    def test_by_trace_id(self, store):
        store.save(_make_receipt(receipt_id="r1", trace_id="t-aaa"))
        store.save(_make_receipt(receipt_id="r2", trace_id="t-bbb"))
        assert len(store.query(trace_id="t-aaa")) == 1

    def test_empty_db(self, store):
        assert store.query() == []

    def test_no_match(self, store):
        store.save(_make_receipt())
        assert store.query(agent_id="nonexistent") == []

    def test_since_filter(self, store):
        from datetime import datetime, timezone
        store.save(_make_receipt(receipt_id="old", timestamp="2026-01-01T00:00:00+00:00"))
        store.save(_make_receipt(receipt_id="new", timestamp="2026-02-15T00:00:00+00:00"))
        results = store.query(since=datetime(2026, 2, 1, tzinfo=timezone.utc))
        assert len(results) == 1 and results[0]["receipt_id"] == "new"


class TestCount:
    def test_count_all(self, store):
        for i in range(3):
            store.save(_make_receipt(receipt_id=f"r-{i}"))
        assert store.count() == 3

    def test_count_with_filter(self, store):
        store.save(_make_receipt(receipt_id="r1", coherence_status="PASS"))
        store.save(_make_receipt(receipt_id="r2", coherence_status="FAIL"))
        assert store.count(status="PASS") == 1


class TestContextManager:
    def test_enter_exit(self, tmp_path):
        with ReceiptStore(str(tmp_path / "cm.db")) as s:
            s.save(_make_receipt())
            assert s.count() == 1
        assert s._closed

    def test_close_idempotent(self, store):
        store.close()
        store.close()


class TestThreadSafety:
    def test_concurrent_saves(self, tmp_path):
        store = ReceiptStore(str(tmp_path / "threads.db"))
        errors = []
        def save_batch(start):
            try:
                for i in range(20):
                    store.save(_make_receipt(receipt_id=f"t{start}-{i}", trace_id=f"tr{start}-{i}"))
            except Exception as e:
                errors.append(e)
        threads = [threading.Thread(target=save_batch, args=(t,)) for t in range(5)]
        for t in threads: t.start()
        for t in threads: t.join()
        assert not errors
        assert store.count() == 100
        store.close()


class TestAutoCreation:
    def test_nested_dirs(self, tmp_path):
        db = str(tmp_path / "a" / "b" / "c" / "receipts.db")
        s = ReceiptStore(db)
        s.save(_make_receipt())
        assert s.count() == 1
        s.close()
