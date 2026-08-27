from __future__ import annotations

import json
import re
from typing import Any

from openai import OpenAI


class ValuationAI:
    def __init__(self, api_key: str, model: str = "gpt-5.6-luna"):
        if not api_key:
            raise ValueError("OPENAI_API_KEY가 없습니다.")
        self.client = OpenAI(api_key=api_key)
        self.model = model or "gpt-5.6-luna"

    @staticmethod
    def _parse(text: str) -> dict[str, Any]:
        text = (text or "").strip()
        try:
            return json.loads(text)
        except Exception:
            m = re.search(r"\{.*\}", text, re.S)
            if not m:
                raise RuntimeError("OpenAI 응답을 JSON으로 해석하지 못했습니다.")
            return json.loads(m.group(0))

    def review(self, features: dict[str, Any], valuation: dict[str, Any], filings: list[dict[str, Any]]) -> dict[str, Any]:
        system = """
너는 buy-side valuation analyst다.
입력된 숫자나 공시 사실을 변경하거나 새 숫자를 만들어내지 마라.
Python이 계산한 valuation 결과를 검토하고, 가정의 합리성/리스크/민감도를 설명하라.
공시 본문이 아닌 제목만 제공되면 본문을 읽었다고 주장하지 마라.
Valuation의 숫자는 입력된 계산 결과만 사용하라.
답변은 반드시 JSON 하나만 출력하라.
"""
        schema = {
            "verdict": "supportive|cautious|mixed|insufficient_data",
            "summary": "string",
            "valuation_strengths": ["string"],
            "valuation_risks": ["string"],
            "assumption_review": ["string"],
            "scenario_interpretation": ["string"],
            "what_would_change_value": ["string"],
            "data_gaps": ["string"],
            "confidence": "high|medium|low",
        }
        payload = {
            "required_output_schema": schema,
            "features": features,
            "valuation": valuation,
            "recent_filings": filings,
        }
        response = self.client.responses.create(
            model=self.model,
            input=[
                {"role": "system", "content": system},
                {"role": "user", "content": json.dumps(payload, ensure_ascii=False, default=str)},
            ],
        )
        return self._parse(getattr(response, "output_text", ""))
