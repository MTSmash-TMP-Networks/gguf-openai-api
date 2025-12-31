import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import time
import json
import glob
import threading
from uuid import uuid4
from typing import List, Literal, Optional, Dict, Union, Tuple

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from llama_cpp import Llama

# NEU: HF-Imports
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer

# --------------------------------------------------
# Environment / GPU
# --------------------------------------------------
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
os.environ.setdefault("LLAMA_LOG_LEVEL", "info")

# --------------------------------------------------
# Konfiguration (ENV)
# --------------------------------------------------
GGUF_MODELS_DIR = os.getenv("GGUF_MODELS_DIR", "./models_gguf")
HF_MODELS_DIR   = os.getenv("HF_MODELS_DIR", "./models_hf")

N_CTX = int(os.getenv("LLAMA_N_CTX", "8192"))
N_THREADS = int(os.getenv("LLAMA_N_THREADS", str(os.cpu_count() or 4)))
N_GPU_LAYERS = int(os.getenv("LLAMA_N_GPU_LAYERS", "-1"))
N_BATCH = int(os.getenv("LLAMA_N_BATCH", "512"))

DEFAULT_MAX_NEW_TOKENS = int(os.getenv("LLAMA_DEFAULT_MAX_TOKENS", "1024"))
CONTEXT_MARGIN_TOKENS = int(os.getenv("LLAMA_CONTEXT_MARGIN_TOKENS", "64"))
MIN_PROMPT_BUDGET_TOKENS = int(os.getenv("LLAMA_MIN_PROMPT_BUDGET_TOKENS", "256"))

STREAM_BUFFER_CHARS = int(os.getenv("LLAMA_STREAM_BUFFER_CHARS", "48"))
STREAM_FLUSH_INTERVAL_SEC = float(os.getenv("LLAMA_STREAM_FLUSH_INTERVAL_SEC", "0.04"))

# --------------------------------------------------
# Wrapper-Interface für beide Backend-Typen
# --------------------------------------------------
class BaseLLM:
    backend: Literal["gguf", "hf"]
    n_ctx: int

    def tokenize(self, text: str) -> List[int]:
        raise NotImplementedError

    def detokenize(self, tokens: List[int]) -> str:
        raise NotImplementedError

    def create_completion(self, *, prompt: str, max_tokens: int,
                          stream: bool = False,
                          temperature: Optional[float] = None,
                          top_p: Optional[float] = None,
                          presence_penalty: Optional[float] = None,
                          frequency_penalty: Optional[float] = None,
                          stop: Optional[List[str]] = None):
        raise NotImplementedError


# ---------------- GGUF Wrapper -------------------
class GGUFLLM(BaseLLM):
    def __init__(self, cfg: dict):
        self.backend = "gguf"
        self._llm = Llama(
            model_path=cfg["path"],
            n_ctx=cfg["n_ctx"],
            n_threads=cfg["n_threads"],
            n_gpu_layers=cfg["n_gpu_layers"],
            n_batch=cfg["n_batch"],
        )
        self.n_ctx = cfg["n_ctx"]

    def tokenize(self, text: str) -> List[int]:
        b = text.encode("utf-8")
        try:
            return self._llm.tokenize(b, add_bos=False, special=True)
        except TypeError:
            return self._llm.tokenize(b, add_bos=False)

    def detokenize(self, tokens: List[int]) -> str:
        out = self._llm.detokenize(tokens)
        if isinstance(out, str):
            return out
        return out.decode("utf-8", errors="ignore")

    def create_completion(self, *, prompt: str, max_tokens: int,
                          stream: bool = False,
                          temperature: Optional[float] = None,
                          top_p: Optional[float] = None,
                          presence_penalty: Optional[float] = None,
                          frequency_penalty: Optional[float] = None,
                          stop: Optional[List[str]] = None):
        kwargs = {
            "prompt": prompt,
            "max_tokens": max_tokens,
            "stream": stream,
        }
        if temperature is not None:
            kwargs["temperature"] = temperature
        if top_p is not None:
            kwargs["top_p"] = top_p
        if presence_penalty is not None:
            kwargs["presence_penalty"] = presence_penalty
        if frequency_penalty is not None:
            kwargs["frequency_penalty"] = frequency_penalty
        if stop is not None:
            kwargs["stop"] = stop
        return self._llm.create_completion(**kwargs)


# ---------------- HF Wrapper ---------------------
class HFLLM(BaseLLM):
    def __init__(self, cfg: dict):
        self.backend = "hf"
        model_path = cfg["path"]
        device = "cuda" if torch.cuda.is_available() else "cpu"

        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16 if device == "cuda" else torch.float32,
            device_map="auto" if device == "cuda" else None,
        )
        self.model.eval()

        self.n_ctx = cfg["n_ctx"]

    def tokenize(self, text: str) -> List[int]:
        return self.tokenizer(text, add_special_tokens=False).input_ids

    def detokenize(self, tokens: List[int]) -> str:
        return self.tokenizer.decode(tokens, skip_special_tokens=True)

    def create_completion(self, *, prompt: str, max_tokens: int,
                          stream: bool = False,
                          temperature: Optional[float] = None,
                          top_p: Optional[float] = None,
                          presence_penalty: Optional[float] = None,
                          frequency_penalty: Optional[float] = None,
                          stop: Optional[List[str]] = None):

        device = next(self.model.parameters()).device
        inputs = self.tokenizer(prompt, return_tensors="pt").to(device)

        gen_kwargs = {
            "max_new_tokens": max_tokens,
            "do_sample": True if (temperature is not None and temperature > 0) else False,
            "temperature": temperature if temperature is not None else 1.0,
            "top_p": top_p if top_p is not None else 1.0,
            "eos_token_id": self.tokenizer.eos_token_id,
        }

        if stop:
            # einfache Stop-Token-Unterstützung: nur erstes Stop-String als EOS-Token
            # Für komplexere Patterns bräuchte man Logits-Processor.
            # Hier nur Minimalversion: ignoriert, wenn kein Token-Match gefunden wird.
            pass

        if not stream:
            with torch.no_grad():
                out = self.model.generate(**inputs, **gen_kwargs)
            gen_tokens = out[0][inputs["input_ids"].shape[-1]:]
            text = self.tokenizer.decode(gen_tokens, skip_special_tokens=True)
            return {
                "choices": [{
                    "text": text,
                    "finish_reason": "stop",
                }]
            }

        # Streaming mit TextIteratorStreamer
        streamer = TextIteratorStreamer(self.tokenizer, skip_prompt=True, skip_special_tokens=True)
        gen_kwargs_stream = dict(inputs, **gen_kwargs, streamer=streamer)

        def _run():
            with torch.no_grad():
                self.model.generate(**gen_kwargs_stream)

        import threading as _threading
        t = _threading.Thread(target=_run)
        t.start()

        def iterator():
            for new_text in streamer:
                yield {
                    "choices": [{
                        "text": new_text,
                        "finish_reason": None,
                    }]
                }

        return iterator()

# --------------------------------------------------
# Model-Registry: GGUF + HF
# --------------------------------------------------
MODEL_CONFIGS: Dict[str, dict] = {}

# GGUF-Modelle
for path in glob.glob(os.path.join(GGUF_MODELS_DIR, "*.gguf")):
    base = os.path.basename(path)
    model_id = os.path.splitext(base)[0]
    MODEL_CONFIGS[model_id] = {
        "path": path,
        "n_ctx": N_CTX,
        "n_threads": N_THREADS,
        "n_gpu_layers": N_GPU_LAYERS,
        "n_batch": N_BATCH,
        "backend": "gguf",
    }

# HF-Modelle: jeder Unterordner im HF_MODELS_DIR ist ein Modell
if os.path.isdir(HF_MODELS_DIR):
    for entry in os.listdir(HF_MODELS_DIR):
        full = os.path.join(HF_MODELS_DIR, entry)
        if os.path.isdir(full):
            model_id = entry
            MODEL_CONFIGS[model_id] = {
                "path": full,
                "n_ctx": N_CTX,
                "backend": "hf",
            }

if not MODEL_CONFIGS:
    raise RuntimeError(
        f"Keine Modelle gefunden. "
        f"GGUF: {GGUF_MODELS_DIR} (.gguf) oder HF: {HF_MODELS_DIR} (Unterordner mit HF-Modellen) befüllen."
    )

DEFAULT_MODEL_ID = os.getenv("DEFAULT_MODEL", next(iter(MODEL_CONFIGS.keys())))

MODEL_CACHE: Dict[str, BaseLLM] = {}
MODEL_LOCKS: Dict[str, threading.Lock] = {}
MODEL_LOCKS_GLOBAL_LOCK = threading.Lock()

def get_model_lock(model_id: str) -> threading.Lock:
    with MODEL_LOCKS_GLOBAL_LOCK:
        lock = MODEL_LOCKS.get(model_id)
        if lock is None:
            lock = threading.Lock()
            MODEL_LOCKS[model_id] = lock
        return lock

def get_llama(model_id: Optional[str]) -> BaseLLM:
    if model_id is None:
        model_id = DEFAULT_MODEL_ID
    if model_id not in MODEL_CONFIGS:
        raise KeyError(model_id)
    if model_id not in MODEL_CACHE:
        cfg = MODEL_CONFIGS[model_id]
        if cfg["backend"] == "gguf":
            MODEL_CACHE[model_id] = GGUFLLM(cfg)
        else:
            MODEL_CACHE[model_id] = HFLLM(cfg)
    return MODEL_CACHE[model_id]

# --------------------------------------------------
# API-Schema (OpenAI-ähnlich)
# --------------------------------------------------
class ChatMessage(BaseModel):
    role: Literal["system", "user", "assistant"]
    content: str

class ChatCompletionRequest(BaseModel):
    model: Optional[str] = None
    messages: List[ChatMessage]
    max_tokens: Optional[int] = None
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
app = FastAPI(title="GGUF+HF OpenAI-like API")

@app.get("/v1/models")
def list_models():
    return {
        "object": "list",
        "data": [
            {
                "id": model_id,
                "object": "model",
                "owned_by": "local",
                "root": cfg["path"],
                "backend": cfg["backend"],
            }
            for model_id, cfg in MODEL_CONFIGS.items()
        ],
    }

@app.get("/health")
def health():
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
        headers = {
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        }
        return StreamingResponse(
            stream_chat(llm, body, model_id, lock),
            media_type="text/event-stream",
            headers=headers,
        )
    else:
        return non_stream_chat(llm, body, model_id, lock)

# --------------------------------------------------
# Prompt-Helfer
# --------------------------------------------------
def render_messages_to_prompt(messages: List[ChatMessage]) -> str:
    if not messages:
        return "<|System|>\n</s>\n<|Benutzer|>\n</s>\n<|Assistentin|>"
    system_parts = [m.content for m in messages if m.role == "system"]
    system_text = "\n".join(system_parts).strip()
    last_user_idx: Optional[int] = None
    for i in range(len(messages) - 1, -1, -1):
        if messages[i].role == "user":
            last_user_idx = i
            break
    if last_user_idx is None:
        last_user_idx = len(messages) - 1
    prompt_msg = messages[last_user_idx].content
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
    parts.append("<|System|>")
    parts.append(system_text)
    parts.append("</s>")
    if history_text:
        parts.append(history_text.rstrip("\n"))
    parts.append("<|Benutzer|>")
    parts.append(prompt_msg)
    parts.append("</s>")
    parts.append("<|Assistentin|>")
    return "\n".join(parts)

def is_context_error(e: Exception) -> bool:
    msg = str(e)
    return ("exceed context window" in msg) or ("Requested tokens" in msg)

# --------------------------------------------------
# Token-Helfer (nutzt Wrapper)
# --------------------------------------------------
def _tokenize(llm: BaseLLM, text: str) -> List[int]:
    return llm.tokenize(text)

def _detokenize(llm: BaseLLM, tokens: List[int]) -> str:
    return llm.detokenize(tokens)

def count_tokens(llm: BaseLLM, text: str) -> int:
    return len(_tokenize(llm, text))

def truncate_text_by_tokens(llm: BaseLLM, text: str, max_tokens: int, keep_end: bool) -> str:
    if max_tokens <= 0:
        return ""
    toks = _tokenize(llm, text)
    if len(toks) <= max_tokens:
        return text
    toks = toks[-max_tokens:] if keep_end else toks[:max_tokens]
    return _detokenize(llm, toks)

def _normalize_stop(stop: Optional[Union[str, List[str]]]) -> Optional[List[str]]:
    if stop is None:
        return None
    if isinstance(stop, str):
        return [stop]
    stop = [s for s in stop if s]
    return stop or None

def _requested_new_tokens(body: ChatCompletionRequest, n_ctx: int) -> int:
    req = body.max_tokens if body.max_tokens is not None else DEFAULT_MAX_NEW_TOKENS
    try:
        req = int(req)
    except Exception:
        req = DEFAULT_MAX_NEW_TOKENS
    req = max(0, req)
    max_reasonable = max(0, n_ctx - CONTEXT_MARGIN_TOKENS - MIN_PROMPT_BUDGET_TOKENS)
    req = min(req, max_reasonable)
    return req

def fit_messages_to_context(
    llm: BaseLLM,
    model_id: str,
    messages: List[ChatMessage],
    reserved_new_tokens: int,
) -> Tuple[List[ChatMessage], str, int, int]:
    n_ctx = MODEL_CONFIGS[model_id]["n_ctx"]
    prompt_budget = max(1, n_ctx - reserved_new_tokens - CONTEXT_MARGIN_TOKENS)
    if not messages:
        prompt = render_messages_to_prompt(messages)
        prompt_tokens = count_tokens(llm, prompt)
        available = max(0, n_ctx - prompt_tokens - 1)
        return messages, prompt, prompt_tokens, min(reserved_new_tokens, available)

    system_msgs = [m for m in messages if m.role == "system"]
    last_user_idx: Optional[int] = None
    for i in range(len(messages) - 1, -1, -1):
        if messages[i].role == "user":
            last_user_idx = i
            break
    if last_user_idx is None:
        last_user_idx = len(messages) - 1
    prompt_msg = messages[last_user_idx]
    if prompt_msg.role != "user":
        prompt_msg = ChatMessage(role="user", content=prompt_msg.content)
    history_msgs: List[ChatMessage] = [
        m for i, m in enumerate(messages)
        if i < last_user_idx and m.role != "system"
    ]
    trimmed_history = history_msgs[:]
    final_msgs = system_msgs + trimmed_history + [prompt_msg]
    prompt = render_messages_to_prompt(final_msgs)
    prompt_tokens = count_tokens(llm, prompt)
    while prompt_tokens > prompt_budget and trimmed_history:
        trimmed_history.pop(0)
        final_msgs = system_msgs + trimmed_history + [prompt_msg]
        prompt = render_messages_to_prompt(final_msgs)
        prompt_tokens = count_tokens(llm, prompt)
    if prompt_tokens > prompt_budget:
        overshoot = prompt_tokens - prompt_budget
        user_toks = _tokenize(llm, prompt_msg.content)
        keep_user = max(32, len(user_toks) - overshoot)
        prompt_msg = ChatMessage(
            role="user",
            content=truncate_text_by_tokens(llm, prompt_msg.content, keep_user, keep_end=True),
        )
        final_msgs = system_msgs + trimmed_history + [prompt_msg]
        prompt = render_messages_to_prompt(final_msgs)
        prompt_tokens = count_tokens(llm, prompt)
    if prompt_tokens > prompt_budget and system_msgs:
        system_text = "\n".join([m.content for m in system_msgs]).strip()
        overshoot = prompt_tokens - prompt_budget
        sys_toks = _tokenize(llm, system_text)
        keep_sys = max(32, len(sys_toks) - overshoot)
        new_system = truncate_text_by_tokens(llm, system_text, keep_sys, keep_end=False)
        system_msgs = [ChatMessage(role="system", content=new_system)]
        final_msgs = system_msgs + trimmed_history + [prompt_msg]
        prompt = render_messages_to_prompt(final_msgs)
        prompt_tokens = count_tokens(llm, prompt)
    available_new = max(0, n_ctx - prompt_tokens - 1)
    max_new_tokens_final = min(reserved_new_tokens, available_new)
    return final_msgs, prompt, prompt_tokens, max_new_tokens_final

# --------------------------------------------------
# Completion-Aufruf (nutzt jetzt BaseLLM.create_completion)
# --------------------------------------------------
def build_completion_kwargs(
    body: ChatCompletionRequest,
    prompt: str,
    stream: bool,
    max_tokens: int,
):
    kwargs = {
        "prompt": prompt,
        "max_tokens": max_tokens,
        "stream": stream,
        "temperature": body.temperature,
        "top_p": body.top_p,
        "presence_penalty": body.presence_penalty,
        "frequency_penalty": body.frequency_penalty,
        "stop": _normalize_stop(body.stop),
    }
    return kwargs

def non_stream_chat(llm: BaseLLM, body: ChatCompletionRequest, model_id: str, lock: threading.Lock):
    with lock:
        n_ctx = MODEL_CONFIGS[model_id]["n_ctx"]
        reserved = _requested_new_tokens(body, n_ctx)
        _final_msgs, prompt, prompt_tokens, max_new = fit_messages_to_context(
            llm=llm,
            model_id=model_id,
            messages=body.messages,
            reserved_new_tokens=reserved,
        )
        if max_new <= 0:
            text = ""
            finish_reason = "stop"
            completion_tokens = 0
        else:
            kwargs = build_completion_kwargs(body, prompt, stream=False, max_tokens=max_new)
            try:
                result = llm.create_completion(**kwargs)
            except ValueError as e:
                if not is_context_error(e):
                    raise
                minimal = [m for m in body.messages if m.role == "system"]
                last_user = next((m for m in reversed(body.messages) if m.role == "user"),
                                 ChatMessage(role="user", content=""))
                minimal.append(last_user)
                _final_msgs, prompt, prompt_tokens, max_new = fit_messages_to_context(
                    llm=llm, model_id=model_id, messages=minimal, reserved_new_tokens=reserved
                )
                kwargs = build_completion_kwargs(body, prompt, stream=False, max_tokens=max_new)
                result = llm.create_completion(**kwargs)
            choice = result["choices"][0]
            text = choice.get("text", "")
            finish_reason = choice.get("finish_reason", "stop")
            completion_tokens = len(_tokenize(llm, text))
    now = int(time.time())
    response = ChatCompletionResponse(
        id=f"chatcmpl-{now}-{uuid4().hex[:8]}",
        object="chat.completion",
        created=now,
        model=model_id,
        choices=[
            ChatCompletionResponseChoice(
                index=0,
                message=ChatCompletionResponseChoiceMessage(role="assistant", content=text),
                finish_reason=finish_reason,
            )
        ],
        usage=ChatCompletionUsage(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens,
        ),
    )
    return response.model_dump()

def stream_chat(llm: BaseLLM, body: ChatCompletionRequest, model_id: str, lock: threading.Lock):
    def sse(data: dict) -> str:
        return "data: " + json.dumps(data, ensure_ascii=False) + "\n\n"

    def event_stream():
        created = int(time.time())
        stream_id = f"chatcmpl-{created}-{uuid4().hex[:8]}"
        with lock:
            n_ctx = MODEL_CONFIGS[model_id]["n_ctx"]
            reserved = _requested_new_tokens(body, n_ctx)
            _final_msgs, prompt, _prompt_tokens, max_new = fit_messages_to_context(
                llm=llm,
                model_id=model_id,
                messages=body.messages,
                reserved_new_tokens=reserved,
            )
            yield sse({
                "id": stream_id,
                "object": "chat.completion.chunk",
                "created": created,
                "model": model_id,
                "choices": [{
                    "index": 0,
                    "delta": {"role": "assistant"},
                    "finish_reason": None,
                }],
            })
            if max_new <= 0:
                yield sse({
                    "id": stream_id,
                    "object": "chat.completion.chunk",
                    "created": created,
                    "model": model_id,
                    "choices": [{
                        "index": 0,
                        "delta": {},
                        "finish_reason": "stop",
                    }],
                })
                yield "data: [DONE]\n\n"
                return
            kwargs = build_completion_kwargs(body, prompt, stream=True, max_tokens=max_new)
            try:
                iterator = llm.create_completion(**kwargs)
            except ValueError as e:
                if not is_context_error(e):
                    raise
                minimal = [m for m in body.messages if m.role == "system"]
                last_user = next((m for m in reversed(body.messages) if m.role == "user"),
                                 ChatMessage(role="user", content=""))
                minimal.append(last_user)
                _final_msgs, prompt, _prompt_tokens, max_new = fit_messages_to_context(
                    llm=llm, model_id=model_id, messages=minimal, reserved_new_tokens=reserved
                )
                kwargs = build_completion_kwargs(body, prompt, stream=True, max_tokens=max_new)
                iterator = llm.create_completion(**kwargs)

            buf = ""
            last_flush = time.monotonic()
            try:
                for chunk in iterator:
                    choice = chunk["choices"][0]
                    delta = choice.get("text", "")
                    if not delta:
                        continue
                    buf += delta
                    now = time.monotonic()
                    if (
                        len(buf) >= STREAM_BUFFER_CHARS
                        or "\n" in buf
                        or (now - last_flush) >= STREAM_FLUSH_INTERVAL_SEC
                    ):
                        yield sse({
                            "id": stream_id,
                            "object": "chat.completion.chunk",
                            "created": created,
                            "model": model_id,
                            "choices": [{
                                "index": 0,
                                "delta": {"content": buf},
                                "finish_reason": None,
                            }],
                        })
                        buf = ""
                        last_flush = now
                if buf:
                    yield sse({
                        "id": stream_id,
                        "object": "chat.completion.chunk",
                        "created": created,
                        "model": model_id,
                        "choices": [{
                            "index": 0,
                            "delta": {"content": buf},
                            "finish_reason": None,
                        }],
                    })
                yield sse({
                    "id": stream_id,
                    "object": "chat.completion.chunk",
                    "created": created,
                    "model": model_id,
                    "choices": [{
                        "index": 0,
                        "delta": {},
                        "finish_reason": "stop",
                    }],
                })
                yield "data: [DONE]\n\n"
            except GeneratorExit:
                return

    return event_stream()

# --------------------------------------------------
# Start
# --------------------------------------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8023, reload=False, workers=1)
