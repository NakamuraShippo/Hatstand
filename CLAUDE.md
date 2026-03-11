# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

出力言語: 日本語（識別子・コマンド・ライブラリ名は原文）

## プロジェクト概要

Hatstand — Flet ベースの Windows デスクトップアプリ。ローカル環境で LLM の読み込み・チャット・セッション管理・ベンチマークを行う。

## プロジェクト状況

- フェーズ: 安定化 / 機能拡張 / リファクタ準備 / リリース強化
- 優先事項: リグレッション抑制、save/config 互換性の維持、小さな変更を優先
- 避けること: 大規模アーキテクチャ書き換え、装飾的チャーン、明示依頼のないリポジトリ全体リネーム

## 参照優先順位

仕様が衝突した場合:

1. `src/` 実装
2. `Qwen35_Playground_DesignSpec.md`
3. `.codex/ARCHITECTURE.md`

## 作業ルール

- 1タスク ≤5 files、±300 lines
- 差分は最小限、レビューしやすく保つ
- 既存設計・命名規則を尊重する
- タスクに無関係なコードを変更しない
- 1タスク完了ごとに git commit → `/clear`

### 破壊的変更禁止

明示依頼がない限り変更しない:

- データ保存形式（セッション JSON）
- 設定キー（`data/settings.json`）
- 公開エントリポイント（`flet_app:main`）
- pyproject.toml の依存・ビルド設定

## 品質ゲート

```bash
python -m pytest tests/unit/ -q
```

局所変更では対象を絞って実行可。

## 停止条件

次の場合は作業を停止し `docs/CONTINUATION.md` へ状況を記録する:

1. 同一失敗が 3 回再現
2. import / 依存関係の衝突
3. テストが長時間失敗
4. 仕様外の副作用

## 出力契約

最終報告には必ず含める: 変更要約、変更ファイル一覧、実行した検証コマンド、未実施検証、残リスク。

## アーキテクチャ

レイヤードアーキテクチャ。各層の依存は上→下の一方向のみ。

```
UI (Flet)  →  Application (Service/Controller)  →  Backend (Transformers)
                    ↓                                      ↓
              Core (Session, Registry, Presets)     Domain (Entities, DTOs, Enums)
                    ↓
              Infra (Settings, Logging, JSON)
```

### 主要ディレクトリ (`src/hatstand/`)

| ディレクトリ | 責務 |
|---|---|
| `ui/tabs/` | 5画面: Chat, System Prompt, Models, Benchmark, Settings |
| `ui/widgets/` | カスタムウィジェット |
| `application/` | PlaygroundService（中央オーケストレータ）, AppController |
| `backends/` | BaseBackend プロトコル → TransformersBackend 実装 |
| `core/` | SessionStore, ModelRegistry, SystemPromptPresets |
| `domain/` | entities.py (ChatSession, ChatMessage等), dtos.py, enums.py |
| `infra/` | SettingsRepository, ログ設定, JSON ユーティリティ |
| `workers/` | バックグラウンドワーカー: load, generate, benchmark, warmup |

### エントリポイント

- `flet_app.py` — メインの Flet アプリクラス (FletPlaygroundApp)
- `app/main.py` — sys.path 調整を行うブートストラップ

### ランタイムデータ (`data/`)

- `settings.json` — アプリ設定
- `sessions/` — セッション JSON ファイル
- `logs/app.log` — ログ出力

### レイヤールール

- **Core**: ドメインロジックと内部状態管理。UI ウィジェットに依存してはならない
- **UI**: プレゼンテーションのみ。ビジネスロジックや永続化ロジックを含めない
- **Persistence**: save/load、設定シリアライズ、互換性維持。変更は慎重に
- **Backends**: 外部サービス・AI バックエンド。アダプタ境界を明確に保つ

## コマンド

```bash
# セットアップ
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -e .                # 基本インストール
pip install -e ".[quant]"       # 量子化(bitsandbytes)込み
pip install -e ".[dev]"         # pytest 込み

# 起動
boot.bat                        # 推奨
hatstand                        # pip エントリポイント
python app/main.py              # 直接実行

# テスト
python -m pytest tests/unit/ -q                              # ユニットテスト
python -m pytest tests/unit/test_session_store.py            # 単一ファイル
python -m pytest tests/unit/test_session_store.py -k "test_save"  # 単一テスト
python -m pytest tests/ -q                                   # 全テスト
```

## 自己改訂

`.claude/` 配下の設定改善が必要な場合は `/evolve` コマンドで改訂を提案できる。
改訂にはユーザー承認が必要。履歴は `.claude/EVOLUTION_LOG.md` に記録する。
破壊的変更禁止リスト・品質ゲート・停止条件の緩和は禁止。

## 参照ドキュメント

- 設計仕様: `Qwen35_Playground_DesignSpec.md`
- アーキテクチャ: `.codex/ARCHITECTURE.md`
- テストチェックリスト: `CheckList.md`
- 継続情報: `docs/CONTINUATION.md`
