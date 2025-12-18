# GGUF OpenAI-like API (GPU, Multi-Model)

Implementierung einer lokalen, OpenAI-kompatiblen Chat-API auf Basis von `llama-cpp-python` und GGUF-Modellen – optimiert für GPU und Multi-Model-Support.

> Entwickelt von **Marek Templin**  
> im Auftrag der **TMP Networks**

---

## Features

- 🧠 **GGUF-Modelle** (lokal, komplett offline)
- 🔁 **OpenAI-kompatible API**:
  - `POST /v1/chat/completions`
  - `GET /v1/models`
- 🎛️ **Multi-Model-Support**:
  - alle `.gguf`-Dateien aus einem Ordner werden automatisch erkannt
  - Auswahl über `model`-Parameter im Request
- 🚀 **GPU-Unterstützung** (CUDA) mit `llama-cpp-python`
- 🎚️ **Sampling-Parameter über API steuerbar**:
  - `max_tokens`, `temperature`, `top_p`, `stop`, `presence_penalty`, `frequency_penalty`
- 📡 **Streaming-Support** (`stream: true`) via Server-Sent-Events (SSE)
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
├─ models/
│  └─ (hier liegen deine .gguf-Modelle, nicht ins Repo committen)
└─ .gitignore
````

---

## Voraussetzungen

* Linux (getestet unter Ubuntu/Debian-ähnlich)
* Python 3.11 (empfohlen)
* NVIDIA GPU mit aktuellem Treiber + CUDA Runtime (z. B. CUDA 12.x)
* GGUF-Modelle (z. B. LLaMA 3, Mistral, deutsche Modelle etc.)

---

## Installation

### 1. Repository klonen

```bash
git clone https://github.com/MTSmash-TMP-Networks/gguf-openai-api.git
cd gguf-openai-api
```

### 2. Python-Abhängigkeiten installieren

Optional: virtuelles Environment anlegen:

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
> (z. B. via extra-index `cu124/cu121`).
> Das hängt von deiner CUDA-Version ab.

### 3. Modelle vorbereiten

Lege deine `.gguf`-Dateien in den `models/`-Ordner, z. B.:

```text
models/
  EvaGPT-German-X-LlamaTok-DE-7B-f16.gguf
  Meta-Llama-3-8B-Instruct.Q4_K_M.gguf
```

Beim Start des Servers werden alle `.gguf`-Dateien in diesem Ordner automatisch erkannt und unter ihrer Dateibasis (ohne `.gguf`) als Model-ID registriert.

Beispiele:

* Datei `EvaGPT-German-X-LlamaTok-DE-7B-f16.gguf` → Model-ID `EvaGPT-German-X-LlamaTok-DE-7B-f16`
* Datei `Meta-Llama-3-8B-Instruct.Q4_K_M.gguf` → Model-ID `Meta-Llama-3-8B-Instruct.Q4_K_M`

---

## Konfiguration (Environment)

Standardmäßig:

* `MODELS_DIR = ./models`
* `CUDA_VISIBLE_DEVICES = 0`
* `LLAMA_N_CTX = 4096`
* `LLAMA_N_GPU_LAYERS = -1` (so viele Layers wie möglich auf die GPU)
* `LLAMA_N_BATCH = 512`

Diese Werte können über Environment-Variablen angepasst werden, z. B.:

```bash
export GGUF_MODELS_DIR=/pfad/zu/models
export LLAMA_N_CTX=4096
export LLAMA_N_GPU_LAYERS=-1
export LLAMA_N_BATCH=512
export LLAMA_N_THREADS=8
export CUDA_VISIBLE_DEVICES=0
export LLAMA_LOG_LEVEL=info
```

Beispiel-Env-Datei findest du unter `config/gguf-api.env.example`.

---

## Server lokal starten

Im Projektordner:

```bash
python3.11 main.py
```

Standardmäßig lauscht der Server auf:

* **[http://127.0.0.1:8000](http://127.0.0.1:8000)**

---

## API-Endpunkte

### `GET /health`

Einfacher Healthcheck:

```bash
curl http://127.0.0.1:8000/health
```

Beispiel-Antwort:

```json
{
  "status": "ok",
  "default_model": "EvaGPT-German-X-LlamaTok-DE-7B-f16",
  "models": [
    "EvaGPT-German-X-LlamaTok-DE-7B-f16",
    "Meta-Llama-3-8B-Instruct.Q4_K_M"
  ]
}
```

---

### `GET /v1/models`

Liste aller verfügbaren Modelle (OpenAI-kompatibler Stil):

```bash
curl http://127.0.0.1:8000/v1/models
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
      "root": "./models/EvaGPT-German-X-LlamaTok-DE-7B-f16.gguf"
    },
    {
      "id": "Meta-Llama-3-8B-Instruct.Q4_K_M",
      "object": "model",
      "owned_by": "local",
      "root": "./models/Meta-Llama-3-8B-Instruct.Q4_K_M.gguf"
    }
  ]
}
```

---

### `POST /v1/chat/completions`

OpenAI-kompatibler Chat-Endpunkt.

#### Request-Body (Beispiel, ohne Streaming)

```json
{
  "model": "EvaGPT-German-X-LlamaTok-DE-7B-f16",
  "messages": [
    { "role": "system", "content": "Du bist ein hilfreicher Assistent." },
    { "role": "user", "content": "Erklär mir kurz, was ein GGUF-Modell ist." }
  ],
  "max_tokens": -1,
  "temperature": 0.7,
  "top_p": 0.95,
  "stream": false
}
```

* `model` *(optional)*: Model-ID (siehe `/v1/models`), wenn weggelassen → Default-Modell
* `messages`: Liste der Konversationsnachrichten im OpenAI-Format
* `max_tokens`:

  * `> 0`: maximale Anzahl Tokens für die Antwort
  * `-1` oder `null`: wird als `None` an `llama-cpp-python` übergeben, d. h. das Modell entscheidet selbst (bis EOS oder Kontextgrenze)
* `temperature`, `top_p`, `presence_penalty`, `frequency_penalty`, `stop` *(optional)*:

  * wenn nicht gesetzt → `llama-cpp-python` verwendet seine Defaults
* `stream`:

  * `false`: normale Antwort
  * `true`: Streaming via Server-Sent-Events (SSE)

#### Beispiel mit `curl`

```bash
curl -X POST http://127.0.0.1:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "EvaGPT-German-X-LlamaTok-DE-7B-f16",
    "messages": [
      { "role": "user", "content": "Schreib mir einen kurzen deutschen Satz." }
    ],
    "max_tokens": -1,
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
    base_url="http://127.0.0.1:8000/v1",
    api_key="not-needed",  # wird ignoriert
)

resp = client.chat.completions.create(
    model="EvaGPT-German-X-LlamaTok-DE-7B-f16",
    messages=[
        {"role": "system", "content": "Du bist ein hilfreicher Assistent."},
        {"role": "user", "content": "Was ist der Unterschied zwischen CPU und GPU?"}
    ],
    max_tokens=-1,
)

print(resp.choices[0].message.content)
```

### Streaming mit OpenAI-Client

```python
stream = client.chat.completions.create(
    model="EvaGPT-German-X-LlamaTok-DE-7B-f16",
    messages=[
        {"role": "user", "content": "Erzähl mir eine kurze Geschichte."}
    ],
    stream=True,
)

for chunk in stream:
    delta = chunk.choices[0].delta
    if delta.content:
        print(delta.content, end="", flush=True)
print()
```

---

## Betrieb als systemd-Service (Linux)

### 1. Env-Datei anlegen

Beispiel (angepasst von `config/gguf-api.env.example`):

```bash
sudo cp config/gguf-api.env.example /etc/default/gguf-api
sudo nano /etc/default/gguf-api
```

Dort Pfade & Werte anpassen, z. B.:

```bash
GGUF_MODELS_DIR=/home/deinuser/gguf-openai-api/models
LLAMA_N_CTX=4096
LLAMA_N_GPU_LAYERS=-1
LLAMA_N_BATCH=512
LLAMA_N_THREADS=8
```

### 2. systemd-Unit anlegen

```bash
sudo cp config/gguf-api.service.example /etc/systemd/system/gguf-api.service
sudo nano /etc/systemd/system/gguf-api.service
```

Wichtige Punkte:

* `User=` & `Group=` auf deinen Linux-User setzen
* `WorkingDirectory=` & `ExecStart=` ggf. anpassen

### 3. Service aktivieren und starten

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
um lokale GGUF-Modelle komfortabel über eine OpenAI-kompatible API bereitzustellen.

---


