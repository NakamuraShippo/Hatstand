# Hatstand

自分の PC でAIチャットができるデスクトップアプリです。インターネット上のサービスにデータを送ることなく、手元の環境だけで会話できます。

## できること

- ローカルの LLM を読み込んで会話する
- 会話を自動保存し、あとから続きを再開する
- セッションを Markdown / JSON でエクスポートする
- System Prompt を切り替えて応答スタイルを変える
- 複数モデルの簡易ベンチマークを実行する

## 動作環境

- Windows 10 / 11
- Python 3.11 以上 3.14 未満
- モデルサイズに応じた GPU メモリ（VRAM）またはシステムメモリ

> モデルが PC に保存されていない場合、初回の読み込み時に Hugging Face から自動ダウンロードされます。ストレージの空き容量にご注意ください。

## セットアップ

### 1. 仮想環境を作る

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### 2. インストールする

```powershell
pip install -r requirements.txt
```

量子化モデル（4bit / 8bit）も使いたい場合:

```powershell
pip install bitsandbytes>=0.45
```

### 3. 起動する

一番かんたんな方法 — `boot.bat` をダブルクリックしてください。

または、仮想環境を有効にした状態で:

```powershell
hatstand
```

## 使い方

### モデルを読み込む

1. **Models** タブを開く
2. モデルプリセットを選ぶ
3. **Load** を押す

初回はダウンロードが入るため、しばらく待ちます。

#### モデルを手動で配置する場合

オフライン環境などで自動ダウンロードが使えない場合は、モデルファイルを手動で配置できます。

| 方法 | 配置先 |
|---|---|
| Hugging Face キャッシュに置く | `%USERPROFILE%\.cache\huggingface\hub\` （デフォルト） |
| Settings で変更した場合 | **Settings** タブの **HF Cache Dir** に指定したフォルダ |
| 任意のローカルパスを直接指定 | モデル ID の代わりにフォルダの絶対パスを入力 |

Hugging Face キャッシュに手動配置する場合は、`huggingface-cli download` であらかじめダウンロードしたスナップショットをそのままコピーしてください。

### 会話する

1. **Chat** タブを開く
2. メッセージを入力して送信ボタンを押す

生成中は送信ボタンが停止ボタンに変わります。押すと途中で止められます。

### セッションを管理する

- 会話は自動保存されます
- 左側の **Session** パネルから過去の会話を開けます
- 各セッションのメニューから操作できます:
  - **Pinned** — お気に入りとして固定
  - **Duplicate** — 複製
  - **Delete** — 削除
  - **Export MD** / **Export JSON** — ファイルに書き出し

### System Prompt を変える

**System Prompt** タブから、AIの話し方や制約を切り替えられます。よく使う内容はプリセットとして保存しておけます。

### ベンチマークを取る

**Benchmark** タブで、複数のモデルプリセットを順に読み込んで性能を比較できます。結果は CSV でエクスポートできます。

### 設定を変える

**Settings** タブで以下を変更できます:

- Hugging Face キャッシュの保存先
- アプリデータの保存先
- ログレベル
- デフォルトのモデルプリセット
- テーマ（ライト / ダーク）

変更後は **Save Settings** を押してください。

## データの保存場所

デフォルトでは `data/` フォルダに保存されます。

| 種類 | 場所 |
|---|---|
| 設定 | `data/settings.json` |
| セッション | `data/sessions/` |
| ログ | `data/logs/app.log` |

Settings の **App Data Path** を設定すると、セッションとログの保存先を変更できます。

## 対応モデル形式

現在は **Transformers 形式**（safetensors / bin）のモデルに対応しています。Hugging Face で `AutoModelForCausalLM` として読み込める因果言語モデルであれば、Qwen 以外のモデルファミリーも利用できます。

> **GGUF 形式**には現在対応していません。今後のバージョンで対応予定です。

## うまくいかないとき

| 症状 | 対処 |
|---|---|
| モデルが読み込めない | VRAM / メモリ不足の可能性があります。より小さいモデルを試してください |
| 量子化モデルが使えない | `pip install -e ".[quant]"` で bitsandbytes を追加してください |
| 初回の読み込みが遅い | モデルのダウンロード中です。完了すれば次回からは速くなります |

## NakamuraShippo  
Patreon  : https://www.patreon.com/cw/NakamuraShippo  
X        : [@nakamurashippo](https://x.com/Nakamurashippo)


## ライセンス

[Apache License 2.0](LICENSE)
