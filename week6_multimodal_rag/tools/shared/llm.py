# ---- llm.py: shared helper code (canonical copy in week6_multimodal_rag/tools/shared/) ----
# One chat interface over two backends:
#     "local"       Qwen3-VL on the Colab GPU via transformers (default: no account, nothing leaves the VM)
#     "openrouter"  any vision model on openrouter.ai (OpenAI-compatible API; set OPENROUTER_API_KEY in Colab secrets)
# Messages use one format for both:
#     [{"role": "system" | "user" | "assistant",
#       "content": "text" or [{"type": "text", "text": ...}, {"type": "image", "image": <PIL.Image>}]}]
# Plus the Week 5 Day 3 JSON helpers: extract the first JSON object, validate with pydantic, repair once.
import base64
import io
import json
import re
import time

import requests
import torch
from PIL import Image
from pydantic import BaseModel, ValidationError


def fit_image(img: Image.Image, max_side: int | None) -> Image.Image:
    img = img.convert("RGB")
    if max_side and max(img.size) > max_side:
        s = max_side / max(img.size)
        img = img.resize((max(1, round(img.width * s)), max(1, round(img.height * s))), Image.BICUBIC)
    return img


def default_dtype() -> torch.dtype:
    """bf16 on Ampere+ (A100, L4), fp16 on T4 (no native bf16), fp32 on CPU."""
    if torch.cuda.is_available():
        return torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16
    return torch.float32


class LLM:
    def __init__(self, backend: str = "local", model_id: str | None = None, max_side: int = 1024):
        self.backend, self.max_side = backend, max_side
        self.calls, self.seconds, self.tokens_in, self.tokens_out = 0, 0.0, 0, 0
        if backend == "local":
            from transformers import AutoModelForImageTextToText, AutoProcessor
            self.model_id = model_id or "Qwen/Qwen3-VL-4B-Instruct"
            self.processor = AutoProcessor.from_pretrained(self.model_id)
            self.model = AutoModelForImageTextToText.from_pretrained(
                self.model_id, dtype=default_dtype(), device_map="auto", attn_implementation="sdpa").eval()
        elif backend == "openrouter":
            self.model_id = model_id or "qwen/qwen3-vl-8b-instruct"
            self.key = get_secret("OPENROUTER_API_KEY")
            if not self.key:
                raise RuntimeError("Set OPENROUTER_API_KEY in Colab secrets (or use backend='local').")
        else:
            raise ValueError(f"unknown backend {backend!r}")

    def _normalize(self, messages: list) -> list:
        out = []
        for m in messages:
            content = m["content"]
            if isinstance(content, str):
                content = [{"type": "text", "text": content}]
            parts = [{"type": "image", "image": fit_image(p["image"], self.max_side)} if p["type"] == "image" else p for p in content]
            out.append({"role": m["role"], "content": parts})
        return out

    @torch.inference_mode()
    def _local(self, msgs: list, max_new_tokens: int) -> str:
        inputs = self.processor.apply_chat_template(msgs, tokenize=True, add_generation_prompt=True, return_dict=True,
                                                    return_tensors="pt").to(self.model.device)
        out = self.model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
        new = out[:, inputs["input_ids"].shape[1]:]
        self.tokens_in += int(inputs["input_ids"].shape[1])
        self.tokens_out += int((new[0] != self.processor.tokenizer.pad_token_id).sum())
        return self.processor.batch_decode(new, skip_special_tokens=True)[0].strip()

    def _remote(self, msgs: list, max_new_tokens: int) -> str:
        def part(p):
            if p["type"] == "text":
                return p
            buf = io.BytesIO()
            p["image"].save(buf, format="PNG")
            return {"type": "image_url", "image_url": {"url": "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()}}
        body = {"model": self.model_id, "temperature": 0, "max_tokens": max_new_tokens,
                "messages": [{"role": m["role"], "content": [part(p) for p in m["content"]]} for m in msgs]}
        for attempt in range(4):
            r = requests.post("https://openrouter.ai/api/v1/chat/completions", json=body, timeout=180,
                              headers={"Authorization": f"Bearer {self.key}"})
            if r.status_code == 429 and attempt < 3:  # free tiers rate-limit: back off and retry
                time.sleep(5 * 2 ** attempt)
                continue
            r.raise_for_status()
            j = r.json()
            usage = j.get("usage") or {}
            self.tokens_in += usage.get("prompt_tokens", 0)
            self.tokens_out += usage.get("completion_tokens", 0)
            return (j["choices"][0]["message"]["content"] or "").strip()
        raise RuntimeError("OpenRouter kept rate-limiting")

    def chat(self, messages: list, max_new_tokens: int = 512) -> str:
        t0 = time.perf_counter()
        msgs = self._normalize(messages)
        text = self._local(msgs, max_new_tokens) if self.backend == "local" else self._remote(msgs, max_new_tokens)
        self.calls += 1
        self.seconds += time.perf_counter() - t0
        return text

    def ask(self, prompt: str, images: list = (), system: str | None = None, max_new_tokens: int = 512) -> str:
        content = [{"type": "image", "image": im} for im in images] + [{"type": "text", "text": prompt}]
        msgs = ([{"role": "system", "content": system}] if system else []) + [{"role": "user", "content": content}]
        return self.chat(msgs, max_new_tokens)

    def stats(self) -> dict:
        return {"calls": self.calls, "seconds": round(self.seconds, 1), "tokens_in": self.tokens_in, "tokens_out": self.tokens_out}


def extract_json(text: str) -> str | None:
    """The first balanced {...} object in a reply (tolerates ```json fences and chatter around it)."""
    text = re.sub(r"```(?:json)?", "", text)
    start = text.find("{")
    if start < 0:
        return None
    depth, in_str, esc = 0, False, False
    for i, ch in enumerate(text[start:], start):
        if in_str:
            esc = (ch == "\\") and not esc
            if ch == '"' and not esc:
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[start:i + 1]
    return None


def validate_json(text: str, schema: type[BaseModel]) -> tuple[dict | None, BaseModel | None, str | None]:
    blob = extract_json(text)
    if blob is None:
        return None, None, "no JSON object found in the reply"
    try:
        data = json.loads(blob)
    except json.JSONDecodeError as exc:
        return None, None, f"invalid JSON: {exc}"
    try:
        return data, schema.model_validate(data), None
    except ValidationError as exc:
        return data, None, "; ".join(f"{'.'.join(map(str, e['loc']))}: {e['msg']}" for e in exc.errors()[:6])


def ask_json(llm: "LLM", messages: list, schema: type[BaseModel], max_new_tokens: int = 512, repairs: int = 1):
    """Schema in the prompt -> validate -> send the error back (Week 5 'schema_repair'). Returns (obj | None, raw, attempts)."""
    schema_txt = ("\nReply with one JSON object only, valid against this JSON Schema:\n"
                  + json.dumps(schema.model_json_schema(), separators=(",", ":")))
    msgs = [dict(m) for m in messages]
    last = msgs[-1]
    last["content"] = (last["content"] + schema_txt) if isinstance(last["content"], str) else \
        list(last["content"]) + [{"type": "text", "text": schema_txt}]
    for attempt in range(1, repairs + 2):
        raw = llm.chat(msgs, max_new_tokens)
        _, obj, err = validate_json(raw, schema)
        if obj is not None or attempt > repairs:
            return obj, raw, attempt
        msgs = msgs + [{"role": "assistant", "content": raw},
                       {"role": "user", "content": f"That reply is not valid: {err}. Return the corrected JSON object only."}]
    return None, raw, attempt
