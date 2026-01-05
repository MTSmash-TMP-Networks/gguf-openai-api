# GGUF + HF OpenAI-like API (GPU, Multi-Model, Streaming)

Lokale, OpenAI-kompatible Chat-API auf Basis von **GGUF (llama-cpp-python)** *und* **Hugging Face Transformers (HF)** – optimiert für GPU, Multi-Model-Support und Streaming.

> Entwickelt von **Marek Templin**  
> im Auftrag der **TMP Networks**

---

## Features

- 🧠 **Zwei Backends**
  - **GGUF / llama-cpp-python** (offline, lokal, sehr performant – ideal für GGUF)
  - **HF / Transformers** (lokale HF-Modelle, optional GPU via `device_map="auto"`)
- 🔁 **OpenAI-kompatible API**
  - `POST /v1/chat/completions`
  - `GET /v1/models`
- 🎛️ **Multi-Model-Support (Auto-Discovery)**
  - alle `.gguf`-Dateien aus `GGUF_MODELS_DIR` werden erkannt
  - jeder Unterordner in `HF_MODELS_DIR` gilt als HF-Modell
  - Auswahl über `model` im Request
- 🚀 **GPU-Unterstützung**
  - GGUF via `LLAMA_N_GPU_LAYERS`
  - HF via CUDA + `device_map="auto"`
- 🎚️ **Sampling/Control Parameter per API**
  - `max_tokens`, `temperature`, `top_p`, `top_k`, `min_p`, `typical_p`
  - `repeat_penalty`, `seed`
  - `presence_penalty`, `frequency_penalty` *(GGUF nativ; HF via LogitsProcessor emuliert)*
  - `stop` (String oder Liste)
- 📡 **Streaming-Support** (`stream: true`) via Server-Sent-Events (SSE)
- 🧠 **Kontextbudget pro Request**
  - `ctx` (Alias: `context_window`, `n_ctx`) begrenzt die genutzte History im Request
  - Hinweis: `ctx` kann nur **≤** dem beim Modell gesetzten Kontextfenster sein
- 🩺 **Healthcheck**: `GET /health`
- ⚙️ **systemd-Service**-Konfiguration für Dauerbetrieb unter Linux

---

## Projektstruktur

Empfohlene Struktur:

```text
gguf-openai-api/
├─ main.py
├─ requirements.txt
├─ README.md
├─ config/
│  ├─ gguf-api.service.example
│  └─ gguf-api.env.example
├─ models_gguf/
│  └─ (hier liegen deine .gguf-Modelle, nicht ins Repo committen)
├─ models_hf/
│  └─ (hier liegen HF-Modelle als Unterordner, nicht ins Repo committen)
└─ .gitignore
````

---

## Voraussetzungen

* Linux (getestet unter Ubuntu/Debian-ähnlich)
* Python 3.11 (empfohlen)
* NVIDIA GPU mit aktuellem Treiber + CUDA Runtime (z. B. CUDA 12.x)
* Modelle:

  * **GGUF**: `.gguf` Dateien (z. B. LLaMA/Mistral Derivate)
  * **HF**: lokale Model-Folder (z. B. ein Ordner aus `transformers` / `snapshot_download`)

---

## Installation

### 1) Repository klonen

```bash
git clone https://github.com/MTSmash-TMP-Networks/gguf-openai-api.git
cd gguf-openai-api
```

### 2) Python-Abhängigkeiten installieren

Optional: virtuelles Environment anlegen

```bash
python3.11 -m venv .venv
source .venv/bin/activate
```

Dann:

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

> Hinweis: Für GPU solltest du `llama-cpp-python` als CUDA-Build installiert haben
> (z. B. via extra-index passend zu deiner CUDA-Version).
> HF nutzt CUDA über dein installiertes `torch`.

---

## Modelle vorbereiten

### GGUF

Lege `.gguf`-Dateien in den GGUF-Ordner (Default: `./models_gguf`), z. B.:

```text
models_gguf/
  EvaGPT-German-X-LlamaTok-DE-7B-f16.gguf
  Meta-Llama-3-8B-Instruct.Q4_K_M.gguf
```

Model-ID = Dateiname ohne `.gguf`.

### HF (Transformers)

Jeder Unterordner in `HF_MODELS_DIR` (Default: `./models_hf`) wird als Modell registriert:

```text
models_hf/
  mistral-7b-instruct/
    config.json
    tokenizer.json
    model.safetensors
    ...
```

Model-ID = Ordnername (z. B. `mistral-7b-instruct`).

---

## Konfiguration (Environment)

Standardwerte (sofern nicht per ENV überschrieben):

* `GGUF_MODELS_DIR=./models_gguf`
* `HF_MODELS_DIR=./models_hf`
* `CUDA_VISIBLE_DEVICES=0`
* `LLAMA_N_CTX=8192`
* `LLAMA_N_GPU_LAYERS=-1` (so viele Layers wie möglich auf die GPU)
* `LLAMA_N_BATCH=512`
* `LLAMA_N_THREADS=<CPU count>`

Beispiel:

```bash
export GGUF_MODELS_DIR=/pfad/zu/models_gguf
export HF_MODELS_DIR=/pfad/zu/models_hf
export LLAMA_N_CTX=8192
export LLAMA_N_GPU_LAYERS=-1
export LLAMA_N_BATCH=512
export LLAMA_N_THREADS=8
export CUDA_VISIBLE_DEVICES=0
export LLAMA_LOG_LEVEL=info
```

Beispiel-Env-Datei: `config/gguf-api.env.example`

---

## Server lokal starten

Im Projektordner:

```bash
python3.11 main.py
```

Der Server lauscht standardmäßig auf:

* `http://0.0.0.0:8023`

> Falls du in deiner Umgebung einen anderen Port nutzt: bitte `uvicorn.run(..., port=XXXX)` anpassen.

---

## API-Endpunkte

### `GET /health`

```bash
curl http://127.0.0.1:8023/health
```

Beispiel-Antwort:

```json
{
  "status": "ok",
  "default_model": "EvaGPT-German-X-LlamaTok-DE-7B-f16",
  "models": ["EvaGPT-German-X-LlamaTok-DE-7B-f16", "mistral-7b-instruct"]
}
```

---

### `GET /v1/models`

```bash
curl http://127.0.0.1:8023/v1/models
```

Beispiel:

```json
{
  "object": "list",
  "data": [
    {
      "id": "EvaGPT-German-X-LlamaTok-DE-7B-f16",
      "object": "model",
      "owned_by": "local",
      "root": "./models_gguf/EvaGPT-German-X-LlamaTok-DE-7B-f16.gguf",
      "backend": "gguf"
    },
    {
      "id": "mistral-7b-instruct",
      "object": "model",
      "owned_by": "local",
      "root": "./models_hf/mistral-7b-instruct",
      "backend": "hf"
    }
  ]
}
```

---

## `POST /v1/chat/completions`

OpenAI-kompatibler Chat-Endpunkt (inkl. System-Prompt über `messages`).

### Request-Body (Beispiel, ohne Streaming)

```json
{
  "model": "EvaGPT-German-X-LlamaTok-DE-7B-f16",
  "messages": [
    { "role": "system", "content": "Du bist ein hilfreicher Assistent." },
    { "role": "user", "content": "Erklär mir kurz, was ein GGUF-Modell ist." }
  ],
  "ctx": 4096,
  "max_tokens": 256,
  "temperature": 0.7,
  "top_p": 0.95,
  "top_k": 40,
  "typical_p": 0.95,
  "min_p": 0.05,
  "repeat_penalty": 1.1,
  "presence_penalty": 0.3,
  "frequency_penalty": 0.2,
  "seed": 123,
  "stop": ["</s>"],
  "stream": false
}
```

### Parameter-Übersicht

* `model` *(optional)*: Model-ID aus `/v1/models` – wenn weggelassen: Default-Modell
* `messages`: Konversation im OpenAI-Format (`system`, `user`, `assistant`)
* `ctx` *(optional)*: Kontextbudget pro Request (Alias: `context_window`, `n_ctx`)

  * begrenzt, wie viel History in den Prompt passt (wird automatisch gekürzt)
  * muss **≤** dem Modell-Kontextfenster sein
* `max_tokens` *(optional)*: maximale neue Tokens in der Antwort (wird intern sinnvoll begrenzt)
* Sampling/Control:

  * `temperature`, `top_p`, `top_k`, `typical_p`, `min_p`
  * `repeat_penalty`
  * `seed` (deterministischer Sampling-Seed, soweit Backend unterstützt)
* Penalties:

  * `presence_penalty`, `frequency_penalty`

    * **GGUF:** nativ (sofern llama.cpp-build es unterstützt)
    * **HF:** emuliert via LogitsProcessor (OpenAI-ähnliche Wirkung)
* `stop` *(optional)*: String oder Liste von Strings

  * HF: token-basierter Stop + sauberes Abschneiden
  * GGUF: wird an llama.cpp übergeben (wenn unterstützt)
* `stream`:

  * `false`: normale Antwort
  * `true`: Streaming via SSE

### Curl-Beispiel

```bash
curl -X POST http://127.0.0.1:8023/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "EvaGPT-German-X-LlamaTok-DE-7B-f16",
    "messages": [
      { "role": "user", "content": "Schreib mir einen kurzen deutschen Satz." }
    ],
    "ctx": 4096,
    "max_tokens": 128,
    "temperature": 0.7,
    "stream": false
  }'
```

---

## Nutzung mit dem offiziellen OpenAI-Python-Client

```bash
pip install openai
```

Beispiel:

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://127.0.0.1:8023/v1",
    api_key="not-needed",
)

resp = client.chat.completions.create(
    model="EvaGPT-German-X-LlamaTok-DE-7B-f16",
    messages=[
        {"role": "system", "content": "Du bist ein hilfreicher Assistent."},
        {"role": "user", "content": "Was ist der Unterschied zwischen CPU und GPU?"}
    ],
    # Zusätzliche Felder funktionieren, wenn dein Client sie durchlässt:
    extra_body={
        "ctx": 4096,
        "top_k": 40,
        "repeat_penalty": 1.1,
        "seed": 123,
    },
    max_tokens=256,
)

print(resp.choices[0].message.content)
```

### Streaming mit OpenAI-Client

```python
stream = client.chat.completions.create(
    model="EvaGPT-German-X-LlamaTok-DE-7B-f16",
    messages=[{"role": "user", "content": "Erzähl mir eine kurze Geschichte."}],
    stream=True,
    extra_body={"ctx": 4096, "seed": 123},
)

for chunk in stream:
    delta = chunk.choices[0].delta
    if delta and getattr(delta, "content", None):
        print(delta.content, end="", flush=True)
print()
```

> Hinweis: Manche OpenAI-Client-Versionen akzeptieren Custom-Parameter nur über `extra_body`.

---

## Betrieb als systemd-Service (Linux)

### 1) Env-Datei anlegen

```bash
sudo cp config/gguf-api.env.example /etc/default/gguf-api
sudo nano /etc/default/gguf-api
```

Beispiel:

```bash
GGUF_MODELS_DIR=/home/deinuser/gguf-openai-api/models_gguf
HF_MODELS_DIR=/home/deinuser/gguf-openai-api/models_hf
LLAMA_N_CTX=8192
LLAMA_N_GPU_LAYERS=-1
LLAMA_N_BATCH=512
LLAMA_N_THREADS=8
CUDA_VISIBLE_DEVICES=0
```

### 2) systemd-Unit anlegen

```bash
sudo cp config/gguf-api.service.example /etc/systemd/system/gguf-api.service
sudo nano /etc/systemd/system/gguf-api.service
```

Wichtig:

* `User=` & `Group=` anpassen
* `WorkingDirectory=` & `ExecStart=` anpassen (inkl. korrektem Port)

### 3) Service aktivieren und starten

```bash
sudo systemctl daemon-reload
sudo systemctl enable gguf-api
sudo systemctl start gguf-api
```

Status:

```bash
systemctl status gguf-api
```

Logs:

```bash
journalctl -u gguf-api -f
```

---

## Lizenz / Nutzung

Dieses Projekt wurde von **Marek Templin** im Auftrag der **TMP Networks** entwickelt,
um lokale GGUF- und HF-Modelle komfortabel über eine OpenAI-kompatible API bereitzustellen.
