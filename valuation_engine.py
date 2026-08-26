from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any, Dict, Optional
import math
import pandas as pd


def _num(v: Any) -> Optional[float]:
    try:
        if v is None:
            return None
        if isinstance(v, float) and math.isnan(v):
            return None
        return float(str(v).replace(",", "").replace("%", ""))
    except Exception:
        return None


def _feature(features: pd.DataFrame, name: str) -> Optional[float]:
    if features is None or features.empty:
        return None
    row = features.loc[features["feature"].astype(str).eq(name)]
    if row.empty:
        return None
    return _num(row.iloc[0].get("value"))


def _feature_map(features: pd.DataFrame) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    if features is None or features.empty:
        return out
    for _, r in features.iterrows():
        out[str(r.get("feature"))] = r.get("value")
    return out


@dataclass
class ValuationAssumptions:
    wacc: float = 9.0
    terminal_growth: float = 2.5
    forecast_years: int = 5
    fcf_growth: float = 5.0
    fair_per: float = 12.0
    fair_pbr: float = 1.5
    weight_dcf: float = 0.50
    weight_per: float = 0.25
    weight_pbr: float = 0.25


class ValuationEngine:
    """Transparent, assumption-driven valuation layer.

    This engine intentionally separates observed market multiples (Feature Engine)
    from fair multiples / intrinsic value assumptions used for valuation.
    """

    def __init__(self, assumptions: ValuationAssumptions | None = None):
        self.a = assumptions or ValuationAssumptions()

    def summarize_inputs(self, features: pd.DataFrame) -> Dict[str, Any]:
        fm = _feature_map(features)
        current_price = _num(fm.get("Current Price"))
        per = _num(fm.get("PER"))
        pbr = _num(fm.get("PBR"))
        roic = _num(fm.get("ROIC"))
        roe = _num(fm.get("ROE"))
        fcf = _num(fm.get("FCF"))
        market_cap = self._market_cap_to_krw(fm.get("Market Cap"))
        return {
            "current_price": current_price,
            "per": per,
            "pbr": pbr,
            "roic": roic,
            "roe": roe,
            "fcf": fcf,
            "market_cap_krw": market_cap,
            "wacc": self.a.wacc,
            "terminal_growth": self.a.terminal_growth,
        }

    @staticmethod
    def _market_cap_to_krw(v: Any) -> Optional[float]:
        if v is None:
            return None
        if isinstance(v, (int, float)):
            return float(v)
        s = str(v).strip().replace(",", "")
        try:
            if "조" in s:
                parts = s.split("조", 1)
                tr = float(parts[0].strip() or 0) * 1e12
                rest = parts[1].strip().replace("원", "")
                rest_num = float(rest) * 1e8 if rest else 0.0
                return tr + rest_num
            if "억" in s:
                return float(s.replace("억", "").strip()) * 1e8
            return float(s)
        except Exception:
            return None

    def _per_value(self, features: pd.DataFrame) -> Dict[str, Any]:
        fm = _feature_map(features)
        current_price = _num(fm.get("Current Price"))
        current_per = _num(fm.get("PER"))
        if current_price is None or current_per in (None, 0):
            return {"status": "unavailable"}
        eps = current_price / current_per
        fair_price = eps * self.a.fair_per
        return {
            "status": "ok",
            "eps_implied": eps,
            "fair_per": self.a.fair_per,
            "fair_price": fair_price,
            "upside_pct": (fair_price / current_price - 1) * 100,
        }

    def _pbr_value(self, features: pd.DataFrame) -> Dict[str, Any]:
        fm = _feature_map(features)
        current_price = _num(fm.get("Current Price"))
        current_pbr = _num(fm.get("PBR"))
        if current_price is None or current_pbr in (None, 0):
            return {"status": "unavailable"}
        bps = current_price / current_pbr
        fair_price = bps * self.a.fair_pbr
        return {
            "status": "ok",
            "bps_implied": bps,
            "fair_pbr": self.a.fair_pbr,
            "fair_price": fair_price,
            "upside_pct": (fair_price / current_price - 1) * 100,
        }

    def _dcf_value(self, features: pd.DataFrame) -> Dict[str, Any]:
        fm = _feature_map(features)
        current_price = _num(fm.get("Current Price"))
        fcf_million = _num(fm.get("FCF"))
        if fcf_million is None or fcf_million <= 0 or current_price is None:
            return {"status": "unavailable", "reason": "FCF 또는 현재가가 없습니다."}

        # Feature Engine FCF is in KRW million. Convert to KRW.
        fcf0 = fcf_million * 1e6
        wacc = self.a.wacc / 100.0
        g = self.a.terminal_growth / 100.0
        growth = self.a.fcf_growth / 100.0
        if wacc <= g:
            return {"status": "unavailable", "reason": "WACC가 영구성장률 이하입니다."}

        pv = 0.0
        fcfs = []
        for year in range(1, self.a.forecast_years + 1):
            fcf_t = fcf0 * ((1 + growth) ** year)
            fcfs.append(fcf_t)
            pv += fcf_t / ((1 + wacc) ** year)

        terminal = fcfs[-1] * (1 + g) / (wacc - g)
        pv_terminal = terminal / ((1 + wacc) ** self.a.forecast_years)
        enterprise_value = pv + pv_terminal

        # For a first transparent version we only convert EV to equity value when
        # a usable market cap is available. Without net debt data, use a market-cap
        # proxy and explicitly label the limitation instead of fabricating debt.
        market_cap = self._market_cap_to_krw(fm.get("Market Cap"))
        if market_cap is not None and market_cap > 0:
            equity_value = enterprise_value
            limitation = "순차입금 연결 전의 단순 DCF이므로 기업가치/주주가치 구분에 한계가 있습니다."
        else:
            equity_value = enterprise_value
            limitation = "시가총액이 없어 DCF 결과를 주주가치로 직접 환산하지 않았습니다."

        # Convert total equity value to per-share using listed shares when available.
        shares = _num(fm.get("Listed Shares"))
        fair_price = equity_value / shares if shares and shares > 0 else None
        return {
            "status": "ok",
            "fcf0_krw": fcf0,
            "fcf_growth": self.a.fcf_growth,
            "wacc": self.a.wacc,
            "terminal_growth": self.a.terminal_growth,
            "pv_explicit": pv,
            "pv_terminal": pv_terminal,
            "enterprise_value": enterprise_value,
            "fair_price": fair_price,
            "upside_pct": ((fair_price / current_price - 1) * 100) if fair_price else None,
            "limitation": limitation,
        }

    def calculate(self, features: pd.DataFrame) -> Dict[str, Any]:
        inputs = self.summarize_inputs(features)
        per_v = self._per_value(features)
        pbr_v = self._pbr_value(features)
        dcf_v = self._dcf_value(features)

        prices = []
        weights = []
        for result, weight in [(dcf_v, self.a.weight_dcf), (per_v, self.a.weight_per), (pbr_v, self.a.weight_pbr)]:
            if result.get("status") == "ok" and result.get("fair_price") is not None:
                prices.append(float(result["fair_price"]))
                weights.append(float(weight))
        fair_price = None
        if prices and sum(weights) > 0:
            # Renormalize weights when one method is unavailable.
            fair_price = sum(p * w for p, w in zip(prices, weights)) / sum(weights)

        current = inputs.get("current_price")
        upside = ((fair_price / current - 1) * 100) if fair_price and current else None

        roic = inputs.get("roic")
        economic_spread = (roic - self.a.wacc) if roic is not None else None

        return {
            "inputs": inputs,
            "per_valuation": per_v,
            "pbr_valuation": pbr_v,
            "dcf_valuation": dcf_v,
            "fair_value_per_share": fair_price,
            "upside_pct": upside,
            "economic_spread_roic_minus_wacc": economic_spread,
            "weights": {
                "DCF": self.a.weight_dcf,
                "PER": self.a.weight_per,
                "PBR": self.a.weight_pbr,
            },
            "assumptions": asdict(self.a),
            "caveats": [
                "Fair PER/Fair PBR은 가정값이므로 자동으로 최적화된 값이 아닙니다.",
                "DCF는 현재 FCF를 기준으로 단순 성장 가정을 적용합니다.",
                "순차입금이 연결되지 않은 경우 DCF는 주주가치로 직접 환산하지 않습니다.",
                "다음 단계에서 산업별 정상배수와 WACC 산정 고도화를 추가할 수 있습니다.",
            ],
        }
