"""Tests for the demo session store."""

import time
import pandas as pd
import pytest

from app.models.demo_schemas import DatasetProfile, TreeState
from app.services.demo_sessions import DemoSession, SessionStore


@pytest.fixture
def store():
    return SessionStore(idle_timeout=2)  # 2-second timeout for fast tests


@pytest.fixture
def sample_df():
    return pd.DataFrame({"x": [1, 2, 3], "y": [4, 5, 6]})


@pytest.fixture
def sample_profile():
    return DatasetProfile(row_count=3, column_count=2, columns=[])


@pytest.fixture
def sample_tree():
    return TreeState()


class TestSessionStore:
    def test_create_and_get(self, store, sample_df, sample_profile, sample_tree):
        sid = store.create(sample_df, sample_profile, sample_tree)
        assert sid
        session = store.get(sid)
        assert session is not None
        assert session.session_id == sid
        assert len(session.df) == 3

    def test_unknown_session_returns_none(self, store):
        assert store.get("nonexistent") is None

    def test_session_isolation(self, store, sample_df, sample_profile, sample_tree):
        sid1 = store.create(sample_df, sample_profile, sample_tree)
        sid2 = store.create(sample_df, sample_profile, sample_tree)

        assert sid1 != sid2
        s1 = store.get(sid1)
        s2 = store.get(sid2)
        assert s1 is not s2

    def test_delete_removes_session(self, store, sample_df, sample_profile, sample_tree):
        sid = store.create(sample_df, sample_profile, sample_tree)
        assert store.delete(sid) is True
        assert store.get(sid) is None

    def test_delete_nonexistent_returns_false(self, store):
        assert store.delete("nonexistent") is False

    def test_idle_expiry(self, store, sample_df, sample_profile, sample_tree):
        sid = store.create(sample_df, sample_profile, sample_tree)
        # Wait for expiry (2 second timeout)
        time.sleep(2.5)
        assert store.get(sid) is None

    def test_touch_resets_expiry(self, store, sample_df, sample_profile, sample_tree):
        sid = store.create(sample_df, sample_profile, sample_tree)
        time.sleep(1)
        # Access session to reset timer
        session = store.get(sid)
        assert session is not None
        time.sleep(1)
        # Should still be valid
        session = store.get(sid)
        assert session is not None

    def test_update_tree(self, store, sample_df, sample_profile, sample_tree):
        sid = store.create(sample_df, sample_profile, sample_tree)
        new_tree = TreeState(active_branch_id="test_branch")
        assert store.update_tree(sid, new_tree) is True
        session = store.get(sid)
        assert session.tree.active_branch_id == "test_branch"

    def test_update_nonexistent_returns_false(self, store, sample_tree):
        assert store.update_tree("nonexistent", sample_tree) is False

    def test_list_sessions(self, store, sample_df, sample_profile, sample_tree):
        sid1 = store.create(sample_df, sample_profile, sample_tree)
        sid2 = store.create(sample_df, sample_profile, sample_tree)
        sessions = store.list_sessions()
        assert sid1 in sessions
        assert sid2 in sessions
