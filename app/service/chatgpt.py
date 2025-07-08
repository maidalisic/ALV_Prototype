from __future__ import annotations
import json, os
from typing import Optional, List
from openai import AsyncOpenAI, OpenAIError
from ..schemas import Anomaly


class ChatGPTAnalyser:
    def __init__(self, api_key: Optional[str] = None, model_name: str = "gpt-4o-mini"):
        key = api_key or os.getenv("OPENAI_API_KEY")
        if not key:
            raise RuntimeError(
                "Provide OpenAI key via header X-OpenAI-Key or env OPENAI_API_KEY"
            )
        self.client = AsyncOpenAI(api_key=key)
        self.model_name = model_name

    async def analyse(self, text: str) -> dict:
        prompt = (
            "You are a senior DevOps engineer. Return JSON with key 'anomalies' "
            "([{line_number:int, score:float, message:str}]). Respond ONLY JSON."
        )
        msgs = [
            {"role": "system", "content": prompt},
            {"role": "user", "content": text[:20_000]},
        ]
        try:
            resp = await self.client.chat.completions.create(
                model=self.model_name,
                messages=msgs,
                response_format={"type": "json_object"},
            )
        except OpenAIError as exc:
            raise RuntimeError(f"OpenAI API error: {exc}") from exc

        try:
            payload = json.loads(resp.choices[0].message.content)
            anomalies: List[Anomaly] = [
                Anomaly(**a) for a in payload.get("anomalies", [])
            ]
            return {"anomalies": anomalies, "model_used": f"{self.model_name} (OpenAI)"}
        except Exception as exc:
            raise RuntimeError("Invalid JSON from ChatGPT") from exc
