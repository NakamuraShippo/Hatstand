from __future__ import annotations

import logging
from pathlib import Path

from hatstand.domain.entities import AppSettings
from hatstand.infra.json_utils import load_json_file, write_json_file


class SettingsRepository:
    def __init__(self, settings_path: Path, logger: logging.Logger | None = None) -> None:
        self._settings_path = settings_path
        self._logger = logger or logging.getLogger(__name__)

    def load(self) -> AppSettings:
        payload = load_json_file(self._settings_path)
        settings = AppSettings.from_dict(payload)
        self._logger.info("load_settings | path=%s", self._settings_path)
        return settings

    def save(self, settings: AppSettings) -> Path:
        write_json_file(self._settings_path, settings.to_dict())
        self._logger.info("save_settings | path=%s", self._settings_path)
        return self._settings_path
