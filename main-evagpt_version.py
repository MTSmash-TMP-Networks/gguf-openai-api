import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import time
import json
import glob
import threading
from typing import List, Literal, Optional, Dict, Union

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from llama_cpp import Llama

# --------------------------------------------------
# Environment / GPU
# --------------------------------------------------

# Nur GPU 0 benutzen (falls mehrere verfügbar sind)
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
# Mehr Logs von llama.cpp (info/debug/trace)
os.environ.setdefault("LLAMA_LOG_LEVEL", "info")

# --------------------------------------------------
# Konfiguration (ENV)
# --------------------------------------------------

# Ordner mit deinen GGUF-Modellen
MODELS_DIR = os.getenv("GGUF_MODELS_DIR", "./models")

# Kontextgröße (wird beim Laden der Modelle verwendet)
N_CTX = int(os.getenv("LLAMA_N_CTX", "8192"))

# CPU-Threads
N_THREADS = int(os.getenv("LLAMA_N_THREADS", str(os.cpu_count() or 4)))

# GPU-Layer (-1 = so viele wie möglich auf die GPU)
N_GPU_LAYERS = int(os.getenv("LLAMA_N_GPU_LAYERS", "-1"))

# Batch-Größe (Token pro Schritt)
N_BATCH = int(os.getenv("LLAMA_N_BATCH", "512"))

# --------------------------------------------------
# Model-Registry: alle .gguf im Ordner einsammeln
# --------------------------------------------------

MODEL_CONFIGS: Dict[str, dict] = {}

for path in glob.glob(os.path.join(MODELS_DIR, "*.gguf")):
    base = os.path.basename(path)
    model_id = os.path.splitext(base)[0]  # Dateiname ohne .gguf
    MODEL_CONFIGS[model_id] = {
        "path": path,
        "n_ctx": N_CTX,
        "n_threads": N_THREADS,
        "n_gpu_layers": N_GPU_LAYERS,
        "n_batch": N_BATCH,
    }

if not MODEL_CONFIGS:
    raise RuntimeError(
        f"Keine GGUF-Modelle in {MODELS_DIR} gefunden. "
        "Lege dort .gguf-Dateien ab oder setze GGUF_MODELS_DIR."
    )

# Default-Modell (falls im Request kein model gesetzt ist)
DEFAULT_MODEL_ID = os.getenv("GGUF_DEFAULT_MODEL", next(iter(MODEL_CONFIGS.keys())))

# Cache für Llama-Instanzen
MODEL_CACHE: Dict[str, Llama] = {}

# Per-Model Locks für Warteschlange
MODEL_LOCKS: Dict[str, threading.Lock] = {}
MODEL_LOCKS_GLOBAL_LOCK = threading.Lock()


def get_model_lock(model_id: str) -> threading.Lock:
    """
    Liefert einen Lock pro Modell. Damit werden Requests pro Modell
    seriell ausgeführt (Warteschlange statt Fehler).
    """
    with MODEL_LOCKS_GLOBAL_LOCK:
        lock = MODEL_LOCKS.get(model_id)
        if lock is None:
            lock = threading.Lock()
            MODEL_LOCKS[model_id] = lock
        return lock


def get_llama(model_id: Optional[str]) -> Llama:
    """
    Holt (oder erstellt) eine Llama-Instanz für die angegebene Model-ID.
    """
    if model_id is None:
        model_id = DEFAULT_MODEL_ID

    if model_id not in MODEL_CONFIGS:
        raise KeyError(model_id)

    if model_id not in MODEL_CACHE:
        cfg = MODEL_CONFIGS[model_id]
        MODEL_CACHE[model_id] = Llama(
            model_path=cfg["path"],
            n_ctx=cfg["n_ctx"],
            n_threads=cfg["n_threads"],
            n_gpu_layers=cfg["n_gpu_layers"],
            n_batch=cfg["n_batch"],
        )

    return MODEL_CACHE[model_id]


# --------------------------------------------------
# API-Schema (OpenAI-ähnlich)
# --------------------------------------------------

class ChatMessage(BaseModel):
    role: Literal["system", "user", "assistant"]
    content: str


class ChatCompletionRequest(BaseModel):
    model: Optional[str] = None  # Model-ID wie aus /v1/models
    messages: List[ChatMessage]

    # max_tokens bleibt im API-Contract, wird aber NICHT mehr direkt an llama-cpp übergeben
    max_tokens: Optional[int] = None

    # Alle Sampling-Parameter sind optional,
    # wenn None -> llama-cpp nimmt seine Defaults
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    presence_penalty: Optional[float] = None
    frequency_penalty: Optional[float] = None
    stop: Optional[Union[str, List[str]]] = None

    stream: bool = False


class ChatCompletionResponseChoiceMessage(BaseModel):
    role: Literal["assistant"]
    content: str


class ChatCompletionResponseChoice(BaseModel):
    index: int
    message: ChatCompletionResponseChoiceMessage
    finish_reason: Optional[str] = "stop"


class ChatCompletionUsage(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class ChatCompletionResponse(BaseModel):
    id: str
    object: Literal["chat.completion"]
    created: int
    model: str
    choices: List[ChatCompletionResponseChoice]
    usage: Optional[ChatCompletionUsage] = None


# --------------------------------------------------
# FastAPI-App
# --------------------------------------------------

app = FastAPI(title="GGUF OpenAI-like API (Multi-Model + GPU, Completion-based)")


@app.get("/v1/models")
def list_models():
    """
    Gibt alle verfügbaren Modelle zurück – OpenAI-Style.
    """
    return {
        "object": "list",
        "data": [
            {
                "id": model_id,
                "object": "model",
                "owned_by": "local",
                "root": cfg["path"],
            }
            for model_id, cfg in MODEL_CONFIGS.items()
        ],
    }


@app.get("/health")
def health():
    """
    Einfacher Healthcheck.
    """
    return {
        "status": "ok",
        "default_model": DEFAULT_MODEL_ID,
        "models": list(MODEL_CONFIGS.keys()),
    }


@app.post("/v1/chat/completions")
def chat_completions(body: ChatCompletionRequest):
    try:
        llm = get_llama(body.model)
    except KeyError:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Unbekanntes Modell: {body.model!r}. "
                f"Verfügbare Modelle: {list(MODEL_CONFIGS.keys())}"
            ),
        )

    model_id = body.model or DEFAULT_MODEL_ID
    lock = get_model_lock(model_id)

    if body.stream:
        return StreamingResponse(
            stream_chat(llm, body, model_id, lock),
            media_type="text/event-stream",
        )
    else:
        return non_stream_chat(llm, body, model_id, lock)


# --------------------------------------------------
# Prompt-Helfer (Template-Integration)
# --------------------------------------------------

def render_messages_to_prompt(messages: List[ChatMessage]) -> str:
    """
    Baut einen Prompt entsprechend dem Template:

    <|System|>
    {{ .System }}
    </s>
    {{ .History }}<|Benutzer|>
    {{ .Prompt }}
    </s>
    <|Assistentin|>

    - .System  = alle system-Nachrichten (konkateniert)
    - .History = alle user/assistant vor der letzten user-Nachricht
    - .Prompt  = letzte user-Nachricht (aktuelle Frage)
    """

    if not messages:
        return "<|System|>\n</s>\n<|Benutzer|>\n</s>\n<|Assistentin|>"

    # System-Teil: alle system-Messages zusammenführen
    system_parts = [m.content for m in messages if m.role == "system"]
    system_text = "\n".join(system_parts).strip()

    # letzte user-Message suchen (Prompt)
    last_user_idx: Optional[int] = None
    for i in range(len(messages) - 1, -1, -1):
        if messages[i].role == "user":
            last_user_idx = i
            break

    if last_user_idx is None:
        # kein user -> letzte Message als Prompt behandeln
        last_user_idx = len(messages) - 1

    prompt_msg = messages[last_user_idx].content

    # History: alle non-system-Messages vor der Prompt-Message
    history_msgs: List[ChatMessage] = [
        m for i, m in enumerate(messages)
        if i < last_user_idx and m.role != "system"
    ]

    history_lines: List[str] = []
    for m in history_msgs:
        if m.role == "user":
            history_lines.append("<|Benutzer|>")
            history_lines.append(m.content)
            history_lines.append("</s>")
        elif m.role == "assistant":
            history_lines.append("<|Assistentin|>")
            history_lines.append(m.content)
            history_lines.append("</s>")

    history_text = ""
    if history_lines:
        history_text = "\n".join(history_lines) + "\n"

    parts: List[str] = []

    # <|System|>
    parts.append("<|System|>")
    parts.append(system_text)
    parts.append("</s>")

    # {{ .History }}
    if history_text:
        parts.append(history_text.rstrip("\n"))

    # <|Benutzer|> + Prompt
    parts.append("<|Benutzer|>")
    parts.append(prompt_msg)
    parts.append("</s>")

    # <|Assistentin|> (Start für die Antwort)
    parts.append("<|Assistentin|>")

    return "\n".join(parts)


def build_minimal_messages(body: ChatCompletionRequest) -> List[ChatMessage]:
    """
    Fallback-History bei Kontextfehler:
    - alle system-Messages (konkateniert in .System)
    - plus letzte user-Message
      (wenn keine vorhanden, letzte Message als user interpretiert)
    """
    systems: List[ChatMessage] = []
    last_user: Optional[ChatMessage] = None

    for m in body.messages:
        if m.role == "system":
            systems.append(m)
        elif m.role == "user":
            last_user = m

    if last_user is None:
        if body.messages:
            last = body.messages[-1]
            last_user = ChatMessage(role="user", content=last.content)
        else:
            last_user = ChatMessage(role="user", content="")

    return systems + [last_user]


def is_context_error(e: Exception) -> bool:
    msg = str(e)
    return ("exceed context window" in msg) or ("Requested tokens" in msg)


# --------------------------------------------------
# Completion-Aufruf
# --------------------------------------------------

def build_completion_kwargs(body: ChatCompletionRequest, prompt: str, stream: bool):
    """
    Kwargs für llama_cpp.Llama.create_completion
    """

    # max_tokens von außen ignorieren (None => llama.cpp rechnet selber passend)
    llama_max_tokens = None

    kwargs = {
        "prompt": prompt,
        "max_tokens": llama_max_tokens,
        "stream": stream,
    }

    if body.temperature is not None:
        kwargs["temperature"] = body.temperature
    if body.top_p is not None:
        kwargs["top_p"] = body.top_p
    if body.stop is not None:
        kwargs["stop"] = body.stop

    return kwargs


def non_stream_chat(llm: Llama, body: ChatCompletionRequest, model_id: str, lock: threading.Lock):
    with lock:
        # 1. Versuch: komplette History im Template-Format
        prompt_full = render_messages_to_prompt(body.messages)
        kwargs = build_completion_kwargs(body, prompt_full, stream=False)

        try:
            result = llm.create_completion(**kwargs)
        except ValueError as e:
            if not is_context_error(e):
                raise

            # 2. Versuch: minimale History (System + letzte User)
            minimal_msgs = build_minimal_messages(body)
            prompt_min = render_messages_to_prompt(minimal_msgs)
            kwargs = build_completion_kwargs(body, prompt_min, stream=False)
            result = llm.create_completion(**kwargs)

    choice = result["choices"][0]
    text = choice.get("text", "")

    now = int(time.time())

    usage = result.get("usage", {}) or {}
    prompt_tokens = usage.get("prompt_tokens", 0)
    completion_tokens = usage.get("completion_tokens", len(text.split()))
    total_tokens = prompt_tokens + completion_tokens

    response = ChatCompletionResponse(
        id=result.get("id", f"chatcmpl-{now}"),
        object="chat.completion",
        created=now,
        model=model_id,
        choices=[
            ChatCompletionResponseChoice(
                index=0,
                message=ChatCompletionResponseChoiceMessage(
                    role="assistant",
                    content=text,
                ),
                finish_reason=choice.get("finish_reason", "stop"),
            )
        ],
        usage=ChatCompletionUsage(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
        ),
    )
    return response.model_dump()


def stream_chat(llm: Llama, body: ChatCompletionRequest, model_id: str, lock: threading.Lock):
    def event_stream():
        created = int(time.time())

        # 1. Versuch: komplette History im Template-Format
        prompt_full = render_messages_to_prompt(body.messages)
        prompt_to_use = prompt_full
        tried_minimal = False

        lock.acquire()
        try:
            while True:
                kwargs = build_completion_kwargs(body, prompt_to_use, stream=True)

                try:
                    iterator = llm.create_completion(**kwargs)
                except ValueError as e:
                    if not is_context_error(e):
                        raise

                    if tried_minimal:
                        raise

                    # 2. Versuch: minimale History
                    minimal_msgs = build_minimal_messages(body)
                    prompt_to_use = render_messages_to_prompt(minimal_msgs)
                    tried_minimal = True
                    continue

                # Wenn wir hier sind, haben wir einen Iterator
                try:
                    for chunk in iterator:
                        choice = chunk["choices"][0]
                        delta_text = choice.get("text", "")

                        data = {
                            "id": chunk.get("id", f"chatcmpl-{created}"),
                            "object": "chat.completion.chunk",
                            "created": created,
                            "model": model_id,
                            "choices": [
                                {
                                    "index": 0,
                                    "delta": {
                                        "content": delta_text,
                                    },
                                    "finish_reason": None,
                                }
                            ],
                        }
                        yield "data: " + json.dumps(data, ensure_ascii=False) + "\n\n"

                    # Abschließendes Stop-Chunk
                    stop_chunk = {
                        "id": f"chatcmpl-{created}",
                        "object": "chat.completion.chunk",
                        "created": created,
                        "model": model_id,
                        "choices": [
                            {
                                "index": 0,
                                "delta": {},
                                "finish_reason": "stop",
                            }
                        ],
                    }
                    yield "data: " + json.dumps(stop_chunk, ensure_ascii=False) + "\n\n"
                    break

                except ValueError as e:
                    # Kontextfehler während der Iteration
                    if not is_context_error(e) or tried_minimal:
                        raise

                    minimal_msgs = build_minimal_messages(body)
                    prompt_to_use = render_messages_to_prompt(minimal_msgs)
                    tried_minimal = True
                    continue

            # OpenAI-kompatibles Stream-Ende
            yield "data: [DONE]\n\n"
        finally:
            lock.release()

    return event_stream()


# --------------------------------------------------
# Start (für direkten Aufruf)
# --------------------------------------------------

if __name__ == "__main__":
    import uvicorn
    # Wichtig: workers=1, da das Modell im Prozess lebt
    uvicorn.run("main:app", host="0.0.0.0", port=8023, reload=False)

