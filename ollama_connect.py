import asyncio
import json
import threading
import time
import urllib.request
from dataclasses import dataclass
from typing import Dict, List, Optional


@dataclass
class OllamaConfig:
    base_url: str
    model: str
    system_prompt: str
    options: Dict
    context_lines: int


class OllamaClient:
    def __init__(self, ollama_config: OllamaConfig) -> None:
        self.ollama_config = ollama_config

    def _build_user_prompt(self, context_english_lines: List[str], latest_english: str) -> str:
        context_block = "\n".join([f"- {line}" for line in context_english_lines if line.strip()])
        if context_block.strip():
            return (
                "Context (previous English captions):\n"
                f"{context_block}\n\n"
                "Translate the latest English sentence into Simplified Chinese:\n"
                f"{latest_english}"
            )
        return (
            "Translate the latest English sentence into Simplified Chinese:\n"
            f"{latest_english}"
        )

    async def stream_chat_translate(
        self,
        event_loop: asyncio.AbstractEventLoop,
        context_english_lines: List[str],
        latest_english: str,
        on_partial_text,
    ) -> str:
        url = f"{self.ollama_config.base_url.rstrip('/')}/api/chat"

        body = {
            "model": self.ollama_config.model,
            "stream": True,
            "messages": [
                {"role": "system", "content": self.ollama_config.system_prompt.strip()},
                {"role": "user", "content": self._build_user_prompt(context_english_lines, latest_english)},
            ],
            "options": self.ollama_config.options,
        }

        done_future: asyncio.Future[str] = event_loop.create_future()

        def worker() -> None:
            accumulated = ""
            try:
                req = urllib.request.Request(
                    url,
                    data=json.dumps(body).encode("utf-8"),
                    headers={"Content-Type": "application/json"},
                    method="POST",
                )

                with urllib.request.urlopen(req, timeout=300) as resp:
                    for raw_line in resp:
                        line = raw_line.decode("utf-8", errors="ignore").strip()
                        if not line:
                            continue

                        obj = json.loads(line)

                        piece = ""
                        if "message" in obj and isinstance(obj["message"], dict):
                            piece = obj["message"].get("content", "") or ""
                        elif "response" in obj:
                            piece = obj.get("response", "") or ""

                        if piece:
                            accumulated += piece
                            partial_text = accumulated.strip()
                            event_loop.call_soon_threadsafe(on_partial_text, partial_text)

                        if obj.get("done") is True:
                            break

                final_text = accumulated.strip()
                event_loop.call_soon_threadsafe(lambda: done_future.set_result(final_text))
            except Exception as e:
                event_loop.call_soon_threadsafe(lambda: done_future.set_result(f"[AI translate error] {e}"))

        threading.Thread(target=worker, daemon=True).start()
        return await done_future
