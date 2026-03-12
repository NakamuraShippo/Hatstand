# Hatstand

[日本語READMEはこちら](https://github.com/NakamuraShippo/Hatstand/blob/main/README_JP.md)  

A desktop app for AI chat on your own PC. Have conversations entirely on your local machine — no data is sent to any online service.

## Features

- Load local LLMs and chat with them
- Auto-save conversations and resume them later
- Export sessions as Markdown or JSON
- Switch System Prompts to change response styles
- Run simple benchmarks across multiple models

## Requirements

- Windows 10 / 11
- Python 3.11 or later, below 3.14
- GPU memory (VRAM) or system memory appropriate for the model size

> If a model is not already saved on your PC, it will be automatically downloaded from Hugging Face on first load. Please ensure you have enough free storage.

## Setup

### 1. Create a virtual environment

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### 2. Install dependencies

```powershell
pip install -r requirements.txt
```

To use quantized models (4-bit / 8-bit):

```powershell
pip install bitsandbytes>=0.45
```

### 3. Launch

The easiest way — double-click `boot.bat`.

Or, with the virtual environment activated:

```powershell
hatstand
```

## Usage

### Loading a model

1. Open the **Models** tab
2. Select a model preset
3. Click **Load**

The first time may take a while due to downloading.

#### Placing models manually

If automatic downloading is unavailable (e.g. in an offline environment), you can place model files manually.

| Method | Location |
|---|---|
| Place in Hugging Face cache | `%USERPROFILE%\.cache\huggingface\hub\` (default) |
| Custom cache directory | The folder specified in **HF Cache Dir** on the **Settings** tab |
| Specify a local path directly | Enter the absolute folder path instead of a model ID |

When placing files manually in the Hugging Face cache, copy the snapshot as downloaded by `huggingface-cli download`.

### Chatting

1. Open the **Chat** tab
2. Type a message and click the send button

While generating, the send button changes to a stop button. Click it to stop generation mid-way.

### Managing sessions

- Conversations are auto-saved
- Open past conversations from the **Session** panel on the left
- Use the menu on each session for these actions:
  - **Pinned** — Pin as a favorite
  - **Duplicate** — Create a copy
  - **Delete** — Remove
  - **Export MD** / **Export JSON** — Export to file

### Changing the System Prompt

From the **System Prompt** tab, you can change how the AI responds and what constraints it follows. Frequently used prompts can be saved as presets.

### Running benchmarks

In the **Benchmark** tab, you can load multiple model presets in sequence to compare performance. Results can be exported as CSV.

### Changing settings

The **Settings** tab lets you change:

- Hugging Face cache directory
- App data directory
- Log level
- Default model preset
- Theme (Light / Dark)

Click **Save Settings** after making changes.

## Data storage

By default, data is stored in the `data/` folder.

| Type | Location |
|---|---|
| Settings | `data/settings.json` |
| Sessions | `data/sessions/` |
| Logs | `data/logs/app.log` |

You can change the storage location for sessions and logs by setting **App Data Path** in Settings.

## Supported model formats

Currently supports **Transformers format** (safetensors / bin) models. Any causal language model loadable via `AutoModelForCausalLM` on Hugging Face can be used, not just Qwen family models.

> **GGUF format** is not currently supported. Support is planned for a future release.

## Troubleshooting

| Symptom | Solution |
|---|---|
| Model fails to load | Likely insufficient VRAM / memory. Try a smaller model |
| Quantized models don't work | Install bitsandbytes with `pip install -e ".[quant]"` |
| First load is slow | The model is being downloaded. Subsequent loads will be faster |

## License
[Apache License 2.0](LICENSE)

## NakamuraShippo
X        : [@nakamurashippo](https://x.com/Nakamurashippo)

[![支援をお願いします](https://github.com/NakamuraShippo/Hatstand/blob/main/assets/sponsor_logo.png)](https://www.patreon.com/cw/NakamuraShippo)
