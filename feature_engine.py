from __future__ import annotations

import math
import re
from typing import Any

import numpy as np
import pandas as pd


def _num(x):
    try:
        if x is None or (isinstance(x, float) and math.isnan(x)):
            return None
        return float(str(x).replace(",", "").replace("%", ""))
    except Exception:
        return None


def _parse_display_amount(x):
    """Parse Naver-style market cap text such as '340조 8,746' into KRW."""
    if x is None:
        return None
    s = str(x).strip().replace(",", "")
    if not s:
        return None
    total = 0.0
    found = False
    m = re.search(r"(-?\d+(?:\.\d+)?)조", s)
    if m:
        total += float(m.group(1)) * 1_0000_0000_0000
        found = True
    m = re.search(r"(-?\d+(?:\.\d+)?)억", s)
    if m:
        total += float(m.group(1)) * 100_000_000
        found = True
    if not found:
        m = re.search(r"-?\d+(?:\.\d+)?", s)
        if m:
            return float(m.group(0))
        return None
    return total


def _pick_account(df: pd.DataFrame, candidates):
    if df is None or df.empty:
        return None
    name_cols = [c for c in ["account_nm", "account_detail", "account_id"] if c in df.columns]
    for cand in candidates:
        for col in name_cols:
            mask = df[col].astype(str).str.contains(cand, case=False, na=False)
            sub = df[mask]
            if not sub.empty:
                for value_col in ["thstrm_amount", "thstrm_add_amount", "frmtrm_amount", "frmtrm_q_amount"]:
                    if value_col in sub.columns:
                        return _num(sub.iloc[0][value_col])
    return None


def _pick_account_pair(df: pd.DataFrame, candidates):
    """Return current and prior-period amounts from one matched account row."""
    if df is None or df.empty:
        return None, None
    name_cols = [c for c in ["account_nm", "account_detail", "account_id"] if c in df.columns]
    for cand in candidates:
        for col in name_cols:
            mask = df[col].astype(str).str.contains(cand, case=False, na=False)
            sub = df[mask]
            if sub.empty:
                continue
            row = sub.iloc[0]
            current = None
            prior = None
            for c in ["thstrm_amount", "thstrm_add_amount", "thstrm_q_amount"]:
                if c in row.index and _num(row[c]) is not None:
                    current = _num(row[c])
                    break
            for c in ["frmtrm_amount", "frmtrm_q_amount"]:
                if c in row.index and _num(row[c]) is not None:
                    prior = _num(row[c])
                    break
            return current, prior
    return None, None


def _growth(curr, prior):
    if curr is None or prior in (None, 0):
        return None
    return (curr / prior - 1) * 100


def _annual_return(history: pd.DataFrame, months: int):
    if history is None or history.empty or "close" not in history.columns:
        return None
    h = history.dropna(subset=["date", "close"]).copy()
    if h.empty:
        return None
    last = h.iloc[-1]
    target = last["date"] - pd.DateOffset(months=months)
    old = h.iloc[(h["date"] - target).abs().argmin()]
    return (float(last["close"]) / float(old["close"]) - 1) * 100


def _series_latest(series_obj: Any):
    """Support the current ECOS structure and the earlier normalized structure."""
    if not series_obj:
        return None, None, []
    if isinstance(series_obj, dict):
        rows = series_obj.get("data") or series_obj.get("rows") or []
    else:
        rows = series_obj if isinstance(series_obj, list) else []
    if not rows:
        return None, None, []
    clean = []
    for r in rows:
        if not isinstance(r, dict):
            continue
        val = _num(r.get("DATA_VALUE", r.get("value")))
        t = r.get("TIME", r.get("date", r.get("period")))
        if val is not None:
            clean.append((str(t) if t is not None else "", val))
    if not clean:
        return None, None, []
    clean.sort(key=lambda x: x[0])
    return clean[-1][1], clean[-1][0], clean


def _yoy_from_series(clean, current_period):
    if not clean or not current_period:
        return None
    cur = next((v for p, v in clean if p == current_period), None)
    if cur is None:
        cur = clean[-1][1]
        current_period = clean[-1][0]
    p = str(current_period)
    target = None
    if len(p) == 6 and p.isdigit():
        target = f"{int(p[:4]) - 1}{p[4:]}"
    elif len(p) == 8 and p.isdigit():
        target = f"{int(p[:4]) - 1}{p[4:]}"
    elif "Q" in p.upper():
        try:
            y, q = p.upper().split("Q")
            target = f"{int(y)-1}Q{q}"
        except Exception:
            pass
    if not target:
        return None
    prev = next((v for per, v in clean if per == target), None)
    if prev in (None, 0):
        return None
    return (cur / prev - 1) * 100


def build_feature_table(company_info: dict, financials: pd.DataFrame, market: dict | None, macro: dict | None, notices: pd.DataFrame, asof: pd.Timestamp) -> pd.DataFrame:
    rows = []
    f = financials.copy() if financials is not None else pd.DataFrame()

    revenue, revenue_prev = _pick_account_pair(f, ["매출액", "수익(매출액)", "Revenue"])
    op_income, op_income_prev = _pick_account_pair(f, ["영업이익", "영업이익(손실)", "OperatingIncome"])
    net_income, net_income_prev = _pick_account_pair(f, ["당기순이익", "당기순이익(손실)", "NetIncome"])
    equity = _pick_account(f, ["자본총계", "TotalEquity"])
    assets = _pick_account(f, ["자산총계", "TotalAssets"])
    current_assets = _pick_account(f, ["유동자산", "CurrentAssets"])
    current_liab = _pick_account(f, ["유동부채", "CurrentLiabilities"])
    cash = _pick_account(f, ["현금및현금성자산", "현금및현금성 자산", "CashAndCashEquivalents"])
    debt = _pick_account(f, ["차입금", "단기차입금", "장기차입금", "Borrowings"])
    receivable, receivable_prev = _pick_account_pair(f, ["매출채권", "AccountsReceivable"])
    inventory, inventory_prev = _pick_account_pair(f, ["재고자산", "Inventories"])
    cfo = _pick_account(f, ["영업활동현금흐름", "영업활동으로 인한 현금흐름", "NetCashProvidedByUsedInOperatingActivities"])
    capex = _pick_account(f, ["유형자산의 취득", "유형자산취득", "PurchaseOfPropertyPlantAndEquipment"])
    da = _pick_account(f, ["감가상각비", "감가상각 및 무형자산상각", "DepreciationAndAmortization", "감가상각"])
    interest = _pick_account(f, ["이자비용", "금융원가", "InterestExpense"])
    tax = _pick_account(f, ["법인세비용", "법인세비용(수익)", "IncomeTaxExpense"])

    current_price = (market or {}).get("current_price")
    per = (market or {}).get("per")
    pbr = (market or {}).get("pbr")
    market_cap_raw = (market or {}).get("market_cap")
    market_cap = _parse_display_amount(market_cap_raw)
    listed_shares = (market or {}).get("listed_shares")
    hist = (market or {}).get("history")

    ret_1m = _annual_return(hist, 1)
    ret_3m = _annual_return(hist, 3)
    ret_6m = _annual_return(hist, 6)
    ret_12m = _annual_return(hist, 12)

    vol = None
    dd = None
    avg_volume_20d = None
    latest_volume = None
    if hist is not None and not hist.empty and "close" in hist.columns:
        h = hist.dropna(subset=["close"]).copy()
        daily = h["close"].pct_change().dropna()
        if len(daily) >= 20:
            vol = daily.std() * np.sqrt(252) * 100
        peak = h["close"].cummax()
        dd = ((h["close"] / peak) - 1).min() * 100 if not h.empty else None
        if "volume" in h.columns:
            v = pd.to_numeric(h["volume"], errors="coerce").dropna()
            if not v.empty:
                latest_volume = float(v.iloc[-1])
                if len(v) >= 20:
                    avg_volume_20d = float(v.tail(20).mean())

    tax_rate = None
    if op_income not in (None, 0) and tax is not None:
        tax_rate = max(0.0, min(0.35, tax / op_income))
    nopat = op_income * (1 - tax_rate) if op_income is not None and tax_rate is not None else None
    invested_capital = equity + (debt or 0) - (cash or 0) if equity is not None else None
    roic = (nopat / invested_capital * 100) if nopat is not None and invested_capital not in (None, 0) else None
    roe = (net_income / equity * 100) if net_income is not None and equity not in (None, 0) else None
    roa = (net_income / assets * 100) if net_income is not None and assets not in (None, 0) else None
    op_margin = (op_income / revenue * 100) if op_income is not None and revenue not in (None, 0) else None
    current_ratio = (current_assets / current_liab) if current_assets is not None and current_liab not in (None, 0) else None
    debt_equity = (debt / equity * 100) if debt is not None and equity not in (None, 0) else None
    interest_cov = (op_income / interest) if op_income is not None and interest not in (None, 0) else None
    fcf = (cfo + capex) if cfo is not None and capex is not None else None
    cfo_ni = (cfo / net_income) if cfo is not None and net_income not in (None, 0) else None
    net_debt = (debt - cash) if debt is not None and cash is not None else None
    ebitda = (op_income + abs(da)) if op_income is not None and da is not None else None
    enterprise_value = (market_cap + net_debt) if market_cap is not None and net_debt is not None else None
    ev_ebitda = (enterprise_value / ebitda) if enterprise_value is not None and ebitda not in (None, 0) else None
    fcf_yield = (fcf / market_cap * 100) if fcf is not None and market_cap not in (None, 0) else None
    revenue_growth = _growth(revenue, revenue_prev)
    op_income_growth = _growth(op_income, op_income_prev)
    net_income_growth = _growth(net_income, net_income_prev)
    fcf_prev = None
    # Current-period comparative FCF can be approximated only when prior CFO/capex values are present in the same rows.
    cfo_curr, cfo_prev = _pick_account_pair(f, ["영업활동현금흐름", "영업활동으로 인한 현금흐름", "NetCashProvidedByUsedInOperatingActivities"])
    capex_curr, capex_prev = _pick_account_pair(f, ["유형자산의 취득", "유형자산취득", "PurchaseOfPropertyPlantAndEquipment"])
    if cfo_prev is not None and capex_prev is not None:
        fcf_prev = cfo_prev + capex_prev
    fcf_growth = _growth(fcf, fcf_prev)
    fcf_margin = (fcf / revenue * 100) if fcf is not None and revenue not in (None, 0) else None
    recv_growth = _growth(receivable, receivable_prev)
    inv_growth = _growth(inventory, inventory_prev)
    ar_sales_gap = (recv_growth - revenue_growth) if recv_growth is not None and revenue_growth is not None else None
    inv_sales_gap = (inv_growth - revenue_growth) if inv_growth is not None and revenue_growth is not None else None
    debt_growth = None

    def add(cat, feature, value, unit="", flag="OK", source="OpenDART"):
        rows.append({"category": cat, "feature": feature, "value": value, "unit": unit, "quality_flag": flag, "source": source})

    add("Value", "Current Price", current_price, "KRW", source="Naver Finance")
    add("Value", "PER", per, "x", "OK" if per is not None else "N/A", source="Naver Finance")
    add("Value", "PBR", pbr, "x", "OK" if pbr is not None else "N/A", source="Naver Finance")
    add("Value", "Market Cap", market_cap, "KRW", "OK" if market_cap is not None else "N/A", source="Naver Finance")
    add("Value", "Listed Shares", listed_shares, "shares", "INFO" if listed_shares else "N/A", source="Naver Finance")
    add("Value", "Net Debt", net_debt, "KRW million", "OK" if net_debt is not None else "N/A")
    add("Value", "Enterprise Value", enterprise_value, "KRW", "OK" if enterprise_value is not None else "N/A")
    add("Value", "EBITDA", ebitda, "KRW million", "OK" if ebitda is not None else "N/A")
    add("Value", "EV/EBITDA", ev_ebitda, "x", "OK" if ev_ebitda is not None else "N/A")
    add("Value", "FCF Yield", fcf_yield, "%", "OK" if fcf_yield is not None else "N/A", "OpenDART + Naver Finance")

    add("Quality", "ROIC", roic, "%")
    add("Quality", "ROE", roe, "%")
    add("Quality", "ROA", roa, "%")
    add("Quality", "Operating Margin", op_margin, "%")
    add("Quality", "Current Ratio", current_ratio, "x")
    add("Quality", "Debt / Equity", debt_equity, "%")
    add("Quality", "Interest Coverage", interest_cov, "x")
    add("Quality", "CFO / Net Income", cfo_ni, "x")
    add("Quality", "FCF", fcf, "KRW million")
    add("Quality", "FCF Margin", fcf_margin, "%")

    add("Growth", "Revenue Growth", revenue_growth, "%", "OK" if revenue_growth is not None else "PENDING")
    add("Growth", "Operating Profit Growth", op_income_growth, "%", "OK" if op_income_growth is not None else "PENDING")
    add("Growth", "Net Income Growth", net_income_growth, "%", "OK" if net_income_growth is not None else "PENDING")
    add("Growth", "FCF Growth", fcf_growth, "%", "OK" if fcf_growth is not None else "PENDING")

    add("Momentum", "1M Return", ret_1m, "%", source="Naver Finance")
    add("Momentum", "3M Return", ret_3m, "%", source="Naver Finance")
    add("Momentum", "6M Return", ret_6m, "%", source="Naver Finance")
    add("Momentum", "12M Return", ret_12m, "%", source="Naver Finance")
    add("Momentum", "Annualized Volatility", vol, "%", source="Naver Finance")
    add("Momentum", "Max Drawdown", dd, "%", source="Naver Finance")
    add("Momentum", "Latest Volume", latest_volume, "shares", source="Naver Finance")
    add("Momentum", "20D Average Volume", avg_volume_20d, "shares", source="Naver Finance")

    add("Risk / Accounting", "Receivable Growth vs Sales Growth", ar_sales_gap, "pp", "OK" if ar_sales_gap is not None else "PENDING")
    add("Risk / Accounting", "Inventory Growth vs Sales Growth", inv_sales_gap, "pp", "OK" if inv_sales_gap is not None else "PENDING")
    add("Risk / Accounting", "CFO - Net Income", (cfo - net_income) if cfo is not None and net_income is not None else None, "KRW million")
    add("Risk / Accounting", "Debt", debt, "KRW million")
    add("Risk / Accounting", "Net Debt / EBITDA", (net_debt / ebitda) if net_debt is not None and ebitda not in (None, 0) else None, "x")
    add("Risk / Accounting", "Recent High-Risk Filings", int(_count_high_risk_notices(notices)), "count", source="OpenDART")

    macro_rate = macro_rate_date = None
    macro_latest = {}
    if macro:
        series = macro.get("series", {}) or {}
        aliases = {
            "base_rate": "Policy Rate",
            "ktb_3y": "KTB 3Y",
            "ktb_10y": "KTB 10Y",
            "usdkrw": "USD/KRW",
            "cpi": "CPI",
            "gdp_real": "Real GDP",
        }
        for key, label in aliases.items():
            latest, period, clean = _series_latest(series.get(key))
            macro_latest[key] = (latest, period, clean)
            if key == "base_rate":
                macro_rate, macro_rate_date = latest, period
            unit = (series.get(key) or {}).get("unit", "") if isinstance(series.get(key), dict) else ""
            if latest is not None:
                add("Macro", label, latest, unit, source="ECOS")
        y10, _, _ = macro_latest.get("ktb_10y", (None, None, []))
        y3, _, _ = macro_latest.get("ktb_3y", (None, None, []))
        spread = (y10 - y3) if y10 is not None and y3 is not None else None
        add("Macro", "10Y-3Y Spread", spread, "%p", "OK" if spread is not None else "N/A", source="ECOS")
        for key in ["usdkrw", "cpi", "gdp_real"]:
            latest, period, clean = macro_latest.get(key, (None, None, []))
            yoy = _yoy_from_series(clean, period)
            if key == "usdkrw":
                add("Macro", "USD/KRW YoY", yoy, "%", "OK" if yoy is not None else "PENDING", source="ECOS")
            elif key == "cpi":
                add("Macro", "CPI YoY", yoy, "%", "OK" if yoy is not None else "PENDING", source="ECOS")
            elif key == "gdp_real":
                add("Macro", "Real GDP YoY", yoy, "%", "OK" if yoy is not None else "PENDING", source="ECOS")
        add("Macro", "Policy Rate Observation", macro_rate_date, "", source="ECOS")

    # Industry engine is intentionally deferred. Keep only the source classification here.
    industry_value = company_info.get("induty_code") or company_info.get("industry_code")
    add("Industry", "Industry / Sector", industry_value, "", source="OpenDART/company info")
    add("Company", "Fiscal Year", company_info.get("acc_mt"), "", source="OpenDART")
    add("Company", "Employees", company_info.get("emp_stdn_nb"), "people", source="OpenDART")

    return pd.DataFrame(rows)


def _count_high_risk_notices(notices: pd.DataFrame) -> int:
    if notices is None or notices.empty or "report_nm" not in notices.columns:
        return 0
    keywords = ["유상증자", "전환사채", "횡령", "배임", "소송", "감사의견", "관리종목", "상장폐지"]
    s = notices["report_nm"].astype(str)
    return int(s.str.contains("|".join(keywords), case=False, na=False).sum())
