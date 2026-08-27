from __future__ import annotations

from dataclasses import dataclass, asdict
import math
import re
from typing import Any, Optional

import numpy as np
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
    if features is None or features.empty or "feature" not in features.columns:
        return None
    rows = features.loc[features["feature"].astype(str).eq(name)]
    return _num(rows.iloc[0].get("value")) if not rows.empty else None


def _feature_map(features: pd.DataFrame) -> dict[str, Any]:
    return {str(r.get("feature")): r.get("value") for _, r in (features.iterrows() if features is not None and not features.empty else [])}


@dataclass
class ValuationAssumptions:
    # Manual overrides are optional. None means auto-estimate.
    wacc: float | None = None
    terminal_growth: float | None = None
    fcf_growth: float | None = None
    forecast_years: int = 5
    fair_per: float | None = None
    fair_pbr: float | None = None
    dcf_weight: float = 0.50
    per_weight: float = 0.25
    pbr_weight: float = 0.25
    risk_free_rate_override: float | None = None
    equity_risk_premium: float = 6.0
    beta_override: float | None = None
    cost_of_debt_override: float | None = None


class ValuationEngine:
    """Automatic valuation engine with explicit assumptions and audit trail.

    Observed multiples stay in Feature Engine. This layer estimates fair multiples,
    WACC, normalized FCF and intrinsic values, then combines available methods.
    """

    def __init__(self, assumptions: ValuationAssumptions | None = None):
        self.a = assumptions or ValuationAssumptions()

    @staticmethod
    def _market_cap_krw(v: Any) -> Optional[float]:
        if v is None:
            return None
        if isinstance(v, (int, float)):
            return float(v)
        s = str(v).strip().replace(",", "")
        try:
            total = 0.0
            m = re.search(r"([0-9]+(?:\.[0-9]+)?)\s*조", s)
            if m:
                total += float(m.group(1)) * 1e12
            m = re.search(r"([0-9]+(?:\.[0-9]+)?)\s*억", s)
            if m:
                total += float(m.group(1)) * 1e8
            return total if total > 0 else float(s)
        except Exception:
            return None

    def inputs(self, features: pd.DataFrame) -> dict[str, Any]:
        fm = _feature_map(features)
        current_price = _num(fm.get("Current Price"))
        market_cap = self._market_cap_krw(fm.get("Market Cap"))
        shares = _num(fm.get("Listed Shares"))
        if shares is None and current_price and market_cap:
            shares = market_cap / current_price
        return {
            "current_price": current_price,
            "per": _num(fm.get("PER")),
            "pbr": _num(fm.get("PBR")),
            "roic": _num(fm.get("ROIC")),
            "roe": _num(fm.get("ROE")),
            "revenue_growth": _num(fm.get("Revenue Growth")),
            "fcf_growth": _num(fm.get("FCF Growth")),
            "fcf": _num(fm.get("FCF")),
            "fcf_margin": _num(fm.get("FCF Margin")),
            "market_cap_krw": market_cap,
            "shares": shares,
            "net_debt_million": _num(fm.get("Net Debt")),
            "debt_million": _num(fm.get("Debt")),
            "equity_million": None,
            "beta": _num(fm.get("Beta")),
            "policy_rate": _num(fm.get("Policy Rate")),
            "ktb_10y": _num(fm.get("KTB 10Y")),
            "cpi": _num(fm.get("CPI Index")),
        }

    def auto_assumptions(self, features: pd.DataFrame) -> dict[str, Any]:
        x = self.inputs(features)
        rf = self.a.risk_free_rate_override
        if rf is None:
            rf = x.get("ktb_10y") or x.get("policy_rate") or 3.0

        beta = self.a.beta_override or x.get("beta") or 1.0
        ke = rf + beta * self.a.equity_risk_premium
        cod = self.a.cost_of_debt_override
        if cod is None:
            # Transparent proxy until company-specific debt pricing is available.
            cod = max(2.0, rf + 1.5)

        market_cap = x.get("market_cap_krw")
        net_debt_m = x.get("net_debt_million")
        net_debt_krw = net_debt_m * 1e6 if net_debt_m is not None else 0.0
        total_cap = (market_cap or 0) + max(net_debt_krw, 0)
        if total_cap > 0:
            we = (market_cap or 0) / total_cap
            wd = max(net_debt_krw, 0) / total_cap
        else:
            we, wd = 0.85, 0.15
        tax = 0.21
        wacc_auto = we * ke + wd * cod * (1 - tax)

        hist_growth = x.get("fcf_growth")
        rev_growth = x.get("revenue_growth")
        growth_candidates = [g for g in [hist_growth, rev_growth] if g is not None and np.isfinite(g)]
        base_growth = float(np.median(growth_candidates)) if growth_candidates else 5.0
        fcf_growth = float(np.clip(base_growth, -5.0, 12.0))

        terminal_growth = float(np.clip(min(max(2.0, fcf_growth * 0.5), max(2.5, rf - 0.5)), 1.5, 4.0))
        wacc = float(np.clip(wacc_auto, 6.0, 14.0))

        roe = x.get("roe")
        if roe is not None and roe > 0:
            fair_pbr = float(np.clip((roe / 100.0 - terminal_growth / 100.0) / max((ke / 100.0) - terminal_growth / 100.0, 0.01), 0.5, 5.0))
        else:
            fair_pbr = 1.5

        # Justified P/E from payout/(Ke-g), using sustainable payout = 1-g/ROE.
        if roe is not None and roe > terminal_growth and ke > terminal_growth:
            payout = max(0.05, min(0.95, 1 - (terminal_growth / roe)))
            fair_per = float(np.clip(payout / ((ke - terminal_growth) / 100.0), 5.0, 30.0))
        else:
            fair_per = 12.0

        return {
            "risk_free_rate": rf,
            "beta": beta,
            "cost_of_equity": ke,
            "cost_of_debt": cod,
            "equity_weight": we,
            "debt_weight": wd,
            "wacc": self.a.wacc if self.a.wacc is not None else wacc,
            "terminal_growth": self.a.terminal_growth if self.a.terminal_growth is not None else terminal_growth,
            "fcf_growth": self.a.fcf_growth if self.a.fcf_growth is not None else fcf_growth,
            "fair_per": self.a.fair_per if self.a.fair_per is not None else fair_per,
            "fair_pbr": self.a.fair_pbr if self.a.fair_pbr is not None else fair_pbr,
            "forecast_years": self.a.forecast_years,
        }

    def per_valuation(self, features: pd.DataFrame, aa: dict[str, Any]) -> dict[str, Any]:
        x = self.inputs(features)
        p = x.get("current_price")
        per = x.get("per")
        if p is None or per in (None, 0) or aa["fair_per"] is None:
            return {"status": "unavailable"}
        eps = p / per
        fair_price = eps * aa["fair_per"]
        return {"status": "ok", "eps": eps, "fair_per": aa["fair_per"], "fair_price": fair_price,
                "upside_pct": (fair_price / p - 1) * 100}

    def pbr_valuation(self, features: pd.DataFrame, aa: dict[str, Any]) -> dict[str, Any]:
        x = self.inputs(features)
        p = x.get("current_price")
        pbr = x.get("pbr")
        if p is None or pbr in (None, 0) or aa["fair_pbr"] is None:
            return {"status": "unavailable"}
        bps = p / pbr
        fair_price = bps * aa["fair_pbr"]
        return {"status": "ok", "bps": bps, "fair_pbr": aa["fair_pbr"], "fair_price": fair_price,
                "upside_pct": (fair_price / p - 1) * 100}

    def dcf_valuation(self, features: pd.DataFrame, aa: dict[str, Any]) -> dict[str, Any]:
        x = self.inputs(features)
        p = x.get("current_price")
        fcf_m = x.get("fcf")
        shares = x.get("shares")
        if fcf_m is None or fcf_m <= 0 or p is None or shares is None or shares <= 0:
            return {"status": "unavailable", "reason": "양(+)의 FCF, 현재가, 상장주식수가 모두 필요합니다."}
        wacc = aa["wacc"] / 100.0
        g = aa["terminal_growth"] / 100.0
        growth = aa["fcf_growth"] / 100.0
        if wacc <= g:
            return {"status": "unavailable", "reason": "WACC가 영구성장률보다 낮거나 같습니다."}
        fcf0 = fcf_m * 1e6
        pv = 0.0
        fcfs = []
        for yr in range(1, aa["forecast_years"] + 1):
            fcf_t = fcf0 * ((1 + growth) ** yr)
            fcfs.append(fcf_t)
            pv += fcf_t / ((1 + wacc) ** yr)
        terminal = fcfs[-1] * (1 + g) / (wacc - g)
        pv_terminal = terminal / ((1 + wacc) ** aa["forecast_years"])
        enterprise_value = pv + pv_terminal
        net_debt_m = x.get("net_debt_million")
        equity_value = enterprise_value - (net_debt_m * 1e6 if net_debt_m is not None else 0.0)
        fair_price = equity_value / shares
        return {
            "status": "ok",
            "fcf0_krw": fcf0,
            "forecast_fcfs": fcfs,
            "pv_explicit": pv,
            "pv_terminal": pv_terminal,
            "enterprise_value": enterprise_value,
            "net_debt_krw": net_debt_m * 1e6 if net_debt_m is not None else 0.0,
            "equity_value": equity_value,
            "fair_price": fair_price,
            "upside_pct": (fair_price / p - 1) * 100,
            "formula": "PV(5Y FCF) + PV(Terminal Value) - Net Debt, divided by Shares",
        }

    def scenario(self, features: pd.DataFrame, base: dict[str, Any], label: str, wacc_delta: float, growth_delta: float) -> dict[str, Any]:
        aa = dict(base)
        aa["wacc"] = max(5.0, min(16.0, base["wacc"] + wacc_delta))
        aa["fcf_growth"] = max(-10.0, min(15.0, base["fcf_growth"] + growth_delta))
        return {"label": label, "assumptions": aa, "dcf": self.dcf_valuation(features, aa), "per": self.per_valuation(features, aa), "pbr": self.pbr_valuation(features, aa)}

    def calculate(self, features: pd.DataFrame) -> dict[str, Any]:
        aa = self.auto_assumptions(features)
        per_v = self.per_valuation(features, aa)
        pbr_v = self.pbr_valuation(features, aa)
        dcf_v = self.dcf_valuation(features, aa)
        methods = [(dcf_v, self.a.dcf_weight), (per_v, self.a.per_weight), (pbr_v, self.a.pbr_weight)]
        prices, weights = [], []
        for r, w in methods:
            if r.get("status") == "ok" and r.get("fair_price") is not None:
                prices.append(float(r["fair_price"]))
                weights.append(float(w))
        fair_price = sum(p * w for p, w in zip(prices, weights)) / sum(weights) if prices and sum(weights) > 0 else None
        current = self.inputs(features).get("current_price")
        upside = (fair_price / current - 1) * 100 if fair_price is not None and current else None
        spread = self.inputs(features).get("roic")
        roic_spread = (spread - aa["wacc"]) if spread is not None else None
        scenarios = {
            "Bear": self.scenario(features, aa, "Bear", +1.5, -3.0),
            "Base": self.scenario(features, aa, "Base", 0.0, 0.0),
            "Bull": self.scenario(features, aa, "Bull", -1.0, +3.0),
        }
        return {
            "status": "ok" if fair_price is not None else "partial",
            "inputs": self.inputs(features),
            "auto_assumptions": aa,
            "per_valuation": per_v,
            "pbr_valuation": pbr_v,
            "dcf_valuation": dcf_v,
            "fair_value_per_share": fair_price,
            "upside_pct": upside,
            "economic_spread_roic_minus_wacc": roic_spread,
            "weights": {"DCF": self.a.dcf_weight, "PER": self.a.per_weight, "PBR": self.a.pbr_weight},
            "scenarios": scenarios,
            "audit": {
                "wacc_method": "CAPM + after-tax cost of debt + current capital structure proxy",
                "fair_per_method": "justified P/E using ROE, cost of equity and terminal growth",
                "fair_pbr_method": "justified P/B using ROE, cost of equity and terminal growth",
                "dcf_method": "5-year FCF projection + Gordon terminal value - net debt",
            },
            "manual_overrides": asdict(self.a),
        }
