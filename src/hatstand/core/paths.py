from __future__ import annotations

from pathlib import Path

from hatstand.domain.entities import AppPaths


def build_app_paths(root_dir: Path | None = None) -> AppPaths:
    project_root = root_dir or Path(__file__).resolve().parents[3]
    data_dir = project_root / "data"
    sessions_dir = data_dir / "sessions"
    log_dir = data_dir / "logs"
    return AppPaths(
        root_dir=project_root,
        data_dir=data_dir,
        sessions_dir=sessions_dir,
        log_dir=log_dir,
        settings_path=data_dir / "settings.json",
        log_path=log_dir / "app.log",
    )


def ensure_app_paths(paths: AppPaths) -> AppPaths:
    paths.data_dir.mkdir(parents=True, exist_ok=True)
    paths.sessions_dir.mkdir(parents=True, exist_ok=True)
    paths.log_dir.mkdir(parents=True, exist_ok=True)
    return paths


def resolve_runtime_paths(base_paths: AppPaths, app_data_path: str) -> AppPaths:
    if not app_data_path.strip():
        return base_paths
    data_dir = Path(app_data_path).expanduser()
    return AppPaths(
        root_dir=base_paths.root_dir,
        data_dir=data_dir,
        sessions_dir=data_dir / "sessions",
        log_dir=data_dir / "logs",
        settings_path=base_paths.settings_path,
        log_path=data_dir / "logs" / "app.log",
    )
