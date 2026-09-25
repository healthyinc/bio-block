"""In-memory session store for Hypothesis Lab demo."""

from __future__ import annotations

import asyncio
import logging
import time
import uuid
from typing import Any, Dict, List, Optional

import pandas as pd

from app.models.demo_schemas import (
    CandidateAnalysis,
    CandidateHypothesis,
    DatasetProfile,
    TreeState,
)

logger = logging.getLogger(__name__)

# Default idle timeout in seconds (30 minutes)
DEFAULT_IDLE_TIMEOUT = 30 * 60

# Cleanup interval in seconds
CLEANUP_INTERVAL = 60


class DemoSession:
    """Single exploration session."""

    __slots__ = (
        "session_id",
        "df",
        "profile",
        "tree",
        "hypotheses",
        "analyses",
        "attestations",
        "created_at",
        "last_accessed",
        "idle_timeout",
    )

    def __init__(
        self,
        session_id: str,
        df: pd.DataFrame,
        profile: DatasetProfile,
        tree: TreeState,
        idle_timeout: int = DEFAULT_IDLE_TIMEOUT,
    ):
        self.session_id = session_id
        self.df = df
        self.profile = profile
        self.tree = tree
        self.hypotheses: Dict[str, CandidateHypothesis] = {}
        self.analyses: Dict[str, CandidateAnalysis] = {}
        self.attestations: List[Dict] = []
        self.created_at = time.time()
        self.last_accessed = time.time()
        self.idle_timeout = idle_timeout

    def touch(self) -> None:
        """Reset the idle timer."""
        self.last_accessed = time.time()

    @property
    def is_expired(self) -> bool:
        return (time.time() - self.last_accessed) > self.idle_timeout


class SessionStore:
    """In-memory session store with idle expiry."""

    def __init__(self, idle_timeout: int = DEFAULT_IDLE_TIMEOUT):
        self._sessions: Dict[str, DemoSession] = {}
        self._idle_timeout = idle_timeout
        self._cleanup_task: Optional[asyncio.Task] = None

    # -- public --

    def create(
        self,
        df: pd.DataFrame,
        profile: DatasetProfile,
        tree: TreeState,
    ) -> str:
        """Create session, return ID."""
        session_id = uuid.uuid4().hex
        session = DemoSession(
            session_id=session_id,
            df=df,
            profile=profile,
            tree=tree,
            idle_timeout=self._idle_timeout,
        )
        self._sessions[session_id] = session
        logger.info("Created demo session %s", session_id)
        return session_id

    def get(self, session_id: str) -> Optional[DemoSession]:
        """Get session by ID or None if expired."""
        session = self._sessions.get(session_id)
        if session is None:
            return None
        if session.is_expired:
            self._remove(session_id)
            return None
        session.touch()
        return session

    def update_tree(self, session_id: str, tree: TreeState) -> bool:
        """Replace the tree state in an existing session."""
        session = self.get(session_id)
        if session is None:
            return False
        session.tree = tree
        return True

    def delete(self, session_id: str) -> bool:
        """Immediately delete a session and its data."""
        return self._remove(session_id)

    def list_sessions(self) -> List[str]:
        """Return IDs of all non-expired sessions (for admin/debug)."""
        self._sweep()
        return list(self._sessions.keys())

    # -- lifecycle --

    async def start_cleanup_loop(self) -> None:
        """Start the background task that sweeps expired sessions."""
        if self._cleanup_task is None or self._cleanup_task.done():
            self._cleanup_task = asyncio.create_task(self._cleanup_loop())
            logger.info("Demo session cleanup loop started (interval=%ds)", CLEANUP_INTERVAL)

    async def stop_cleanup_loop(self) -> None:
        if self._cleanup_task and not self._cleanup_task.done():
            self._cleanup_task.cancel()
            try:
                await self._cleanup_task
            except asyncio.CancelledError:
                pass
            logger.info("Demo session cleanup loop stopped.")

    # -- internals --

    def _remove(self, session_id: str) -> bool:
        session = self._sessions.pop(session_id, None)
        if session is not None:
            # free memory
            session.df = pd.DataFrame()
            logger.info("Deleted demo session %s", session_id)
            return True
        return False

    def _sweep(self) -> None:
        expired = [
            sid for sid, s in self._sessions.items() if s.is_expired
        ]
        for sid in expired:
            self._remove(sid)
        if expired:
            logger.info("Swept %d expired demo session(s).", len(expired))

    async def _cleanup_loop(self) -> None:
        while True:
            await asyncio.sleep(CLEANUP_INTERVAL)
            self._sweep()


# Module-level singleton — imported by the router.
session_store = SessionStore()
