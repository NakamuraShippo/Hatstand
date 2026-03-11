from __future__ import annotations

import logging
from pathlib import Path

from hatstand.core.chat_features import session_matches_query
from hatstand.domain.entities import ChatSession
from hatstand.infra.json_utils import load_json_file, write_json_file


class SessionStore:
    def __init__(self, sessions_dir: Path, logger: logging.Logger | None = None) -> None:
        self._sessions_dir = sessions_dir
        self._logger = logger or logging.getLogger(__name__)
        self._sessions_dir.mkdir(parents=True, exist_ok=True)

    def list_sessions(self, query: str = "", sort_by: str = "updated_desc") -> list[ChatSession]:
        sessions: list[ChatSession] = []
        for path in self._sessions_dir.glob("*.json"):
            try:
                payload = load_json_file(path)
            except (OSError, ValueError) as exc:
                self._logger.warning("Failed to read session file: %s (%s)", path, exc)
                continue
            if not payload:
                continue
            try:
                session = ChatSession.from_dict(payload)
            except (KeyError, TypeError, ValueError) as exc:
                self._logger.warning("Failed to parse session file: %s (%s)", path, exc)
                continue
            if session_matches_query(session, query):
                sessions.append(session)
        if sort_by == "title_asc":
            sessions.sort(key=lambda item: (item.title.lower(), item.updated_at))
        elif sort_by == "updated_asc":
            sessions.sort(key=lambda item: (item.updated_at, item.title.lower()))
        else:
            sessions.sort(key=lambda item: (item.updated_at, item.title.lower()), reverse=True)
        sessions.sort(key=lambda item: not item.pinned)
        return sessions

    def load_session(self, session_id: str) -> ChatSession:
        path = self._build_path(session_id)
        try:
            payload = load_json_file(path)
        except (OSError, ValueError) as exc:
            self._logger.warning("Failed to read session file: %s (%s)", path, exc)
            raise ValueError(f"Failed to load session '{session_id}'. The saved JSON is invalid.") from exc
        if not payload:
            raise FileNotFoundError(f"Session not found: {session_id}")
        session = ChatSession.from_dict(payload)
        self._logger.info("load_session | session_id=%s title=%s", session.session_id, session.title)
        return session

    def save_session(self, session: ChatSession) -> Path:
        session.touch()
        path = self._build_path(session.session_id)
        write_json_file(path, session.to_dict())
        self._logger.info("save_session | session_id=%s title=%s", session.session_id, session.title)
        return path

    def delete_session(self, session_id: str) -> bool:
        path = self._build_path(session_id)
        if not path.exists():
            return False
        path.unlink()
        self._logger.info("delete_session | session_id=%s", session_id)
        return True

    def _build_path(self, session_id: str) -> Path:
        return self._sessions_dir / f"{session_id}.json"
