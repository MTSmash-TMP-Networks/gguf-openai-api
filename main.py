import os
import time
import json
import glob
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

# Kontextgröße
N_CTX = int(os.getenv("LLAMA_N_CTX", "4096"))

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

    # Steuerung komplett über API:
    # - None oder -1 => wir geben None an llama-cpp weiter (automatisch)
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

app = FastAPI(title="GGUF OpenAI-like API (Multi-Model + GPU)")


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
            detail=f"Unbekanntes Modell: {body.model!r}. "
                   f"Verfügbare Modelle: {list(MODEL_CONFIGS.keys())}",
        )

    if body.stream:
        return StreamingResponse(
            stream_chat(llm, body),
            media_type="text/event-stream",
        )
    else:
        return non_stream_chat(llm, body)


# --------------------------------------------------
# Helper-Funktionen
# --------------------------------------------------

def build_generation_kwargs(body: ChatCompletionRequest, stream: bool):
    """
    Erzeugt ein kwargs-Dict für llama_cpp.Llama.create_chat_completion
    basierend auf den API-Parametern.
    """
    # max_tokens: -1 oder None -> None an llama-cpp übergeben
    max_tokens = body.max_tokens
    if max_tokens is None or max_tokens < 0:
        max_tokens = None

    kwargs = {
        "messages": [m.model_dump() for m in body.messages],
        "max_tokens": max_tokens,
        "stream": stream,
    }

    # Optional-Parameter nur setzen, wenn sie tatsächlich vom Client kommen
    if body.temperature is not None:
        kwargs["temperature"] = body.temperature
    if body.top_p is not None:
        kwargs["top_p"] = body.top_p
    if body.presence_penalty is not None:
        kwargs["presence_penalty"] = body.presence_penalty
    if body.frequency_penalty is not None:
        kwargs["frequency_penalty"] = body.frequency_penalty
    if body.stop is not None:
        kwargs["stop"] = body.stop

    return kwargs


def non_stream_chat(llm: Llama, body: ChatCompletionRequest):
    generation_kwargs = build_generation_kwargs(body, stream=False)

    result = llm.create_chat_completion(**generation_kwargs)

    choice = result["choices"][0]
    message = choice["message"]
    completion_text = message["content"]
    now = int(time.time())

    usage = result.get("usage", {}) or {}
    prompt_tokens = usage.get("prompt_tokens", 0)
    completion_tokens = usage.get("completion_tokens", len(completion_text.split()))
    total_tokens = prompt_tokens + completion_tokens

    model_id = body.model or DEFAULT_MODEL_ID

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
                    content=completion_text,
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


def stream_chat(llm: Llama, body: ChatCompletionRequest):
    generation_kwargs = build_generation_kwargs(body, stream=True)
    model_id = body.model or DEFAULT_MODEL_ID

    def event_stream():
        created = int(time.time())

        for chunk in llm.create_chat_completion(**generation_kwargs):
            data = {
                "id": chunk.get("id", f"chatcmpl-{created}"),
                "object": "chat.completion.chunk",
                "created": created,
                "model": model_id,
                "choices": chunk.get("choices", []),
            }
            yield "data: " + json.dumps(data, ensure_ascii=False) + "\n\n"

        yield "data: [DONE]\n\n"

    return event_stream()


# --------------------------------------------------
# Start (für direkten Aufruf)
# --------------------------------------------------

if __name__ == "__main__":
    import uvicorn
    # Wichtig: workers=1, da das Modell im Prozess lebt
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=False)

