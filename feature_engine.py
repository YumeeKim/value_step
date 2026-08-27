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
                        v = _num(sub.iloc[0][value_col])
                        if v is not None:
                            return v
    return None


def _pick_pair(df: pd.DataFrame, candidates):
    """Return current-period and prior-period amounts for growth features."""
    if df is None or df.empty:
        return None, None
    name_cols = [c for c in ["account_nm", "account_detail", "account_id"] if c in df.columns]
    value_cols = [c for c in ["thstrm_amount", "thstrm_add_amount", "frmtrm_amount", "frmtrm_q_amount"] if c in df.columns]
    for cand in candidates:
        for col in name_cols:
            mask = df[col].astype(str).str.contains(cand, case=False, na=False)
            sub = df[mask]
            if not sub.empty:
                row = sub.iloc[0]
                cur = _num(row.get("thstrm_amount"))
                if cur is None:
                    cur = _num(row.get("thstrm_add_amount"))
                prev = _num(row.get("frmtrm_amount"))
                if prev is None:
                    prev = _num(row.get("frmtrm_q_amount"))
                if cur is not None or prev is not None:
                    return cur, prev
    return None, None


def _growth(cur, prev):
    if cur is None or prev in (None, 0):
        return None
    return (cur / prev - 1) * 100


def _annual_return(history: pd.DataFrame, months: int):
    if history is None or history.empty or "close" not in history.columns:
        return None
    h = history.dropna(subset=["date", "close"]).copy()
    if h.empty:
        return None
    last = h.iloc[-1]
    target = last["date"] - pd.DateOffset(months=months)
    old = h.iloc[(h["date"] - target).abs().argmin()]
    if float(old["close"]) == 0:
        return None
    return (float(last["close"]) / float(old["close"]) - 1) * 100


def _parse_display_krw(v: Any) -> float | None:
    if v is None:
        return None
    if isinstance(v, (int, float)) and not (isinstance(v, float) and math.isnan(v)):
        return float(v)
    s = str(v).strip().replace(",", "")
    if not s:
        return None
    try:
        total = 0.0
        m = re.search(r"([0-9]+(?:\.[0-9]+)?)\s*조", s)
        if m:
            total += float(m.group(1)) * 1e12
        m = re.search(r"([0-9]+(?:\.[0-9]+)?)\s*억", s)
        if m:
            total += float(m.group(1)) * 1e8
        if total > 0:
            return total
        return float(re.search(r"[-+]?[0-9]+(?:\.[0-9]+)?", s).group(0))
    except Exception:
        return None


def _parse_shares(v: Any) -> float | None:
    if v is None:
        return None
    if isinstance(v, (int, float)) and not (isinstance(v, float) and math.isnan(v)):
        return float(v)
    s = str(v).strip().replace(",", "")
    m = re.search(r"[-+]?[0-9]+(?:\.[0-9]+)?", s)
    return float(m.group(0)) if m else None


def _macro_snapshot(macro: dict | None):
    latest = {}
    if not macro:
        return latest
    try:
        series = macro.get("series", {})
        for key in ["base_rate", "ktb_3y", "ktb_10y", "usdkrw", "cpi", "gdp_real"]:
            data = series.get(key, {})
            rows = data.get("data", data.get("rows", []))
            if isinstance(rows, list) and rows:
                valid = [r for r in rows if _num(r.get("DATA_VALUE")) is not None]
                if valid:
                    latest[key] = valid[-1]
    except Exception:
        pass
    return latest


def build_feature_table(company_info: dict, financials: pd.DataFrame, market: dict | None, macro: dict | None, notices: pd.DataFrame, asof: pd.Timestamp) -> pd.DataFrame:
    rows = []
    f = financials.copy() if financials is not None else pd.DataFrame()

    revenue, revenue_prev = _pick_pair(f, ["매출액", "수익(매출액)", "Revenue"])
    op_income, op_income_prev = _pick_pair(f, ["영업이익", "영업이익(손실)", "OperatingIncome"])
    net_income, net_income_prev = _pick_pair(f, ["당기순이익", "당기순이익(손실)", "NetIncome"])
    equity = _pick_account(f, ["자본총계", "TotalEquity"])
    assets = _pick_account(f, ["자산총계", "TotalAssets"])
    current_assets = _pick_account(f, ["유동자산", "CurrentAssets"])
    current_liab = _pick_account(f, ["유동부채", "CurrentLiabilities"])
    cash = _pick_account(f, ["현금및현금성자산", "현금및현금성 자산", "CashAndCashEquivalents"])
    debt = _pick_account(f, ["차입금", "단기차입금", "장기차입금", "Borrowings"])
    receivable, receivable_prev = _pick_pair(f, ["매출채권", "AccountsReceivable"])
    inventory, inventory_prev = _pick_pair(f, ["재고자산", "Inventories"])
    cfo, cfo_prev = _pick_pair(f, ["영업활동현금흐름", "영업활동으로 인한 현금흐름", "NetCashProvidedByUsedInOperatingActivities"])
    capex = _pick_account(f, ["유형자산의 취득", "유형자산취득", "PurchaseOfPropertyPlantAndEquipment"])
    interest = _pick_account(f, ["이자비용", "금융원가", "InterestExpense"])
    tax = _pick_account(f, ["법인세비용", "법인세비용(수익)", "IncomeTaxExpense"])
    depreciation = _pick_account(f, ["감가상각비", "감가상각", "Depreciation"])
    amortization = _pick_account(f, ["무형자산상각비", "상각비", "Amortization"])

    current_price = (market or {}).get("current_price")
    per = (market or {}).get("per")
    pbr = (market or {}).get("pbr")
    market_cap_raw = (market or {}).get("market_cap")
    listed_shares_raw = (market or {}).get("listed_shares")
    market_cap = _parse_display_krw(market_cap_raw)
    listed_shares = _parse_shares(listed_shares_raw)
    hist = (market or {}).get("history")

    ret_1m = _annual_return(hist, 1)
    ret_3m = _annual_return(hist, 3)
    ret_6m = _annual_return(hist, 6)
    ret_12m = _annual_return(hist, 12)

    vol = None
    dd = None
    if hist is not None and not hist.empty and "close" in hist.columns:
        h = hist.dropna(subset=["close"]).copy()
        daily = h["close"].pct_change().dropna()
        if len(daily) >= 20:
            vol = daily.std() * np.sqrt(252) * 100
        peak = h["close"].cummax()
        dd = ((h["close"] / peak) - 1).min() * 100 if not h.empty else None

    tax_rate = None
    if op_income not in (None, 0) and tax is not None:
        tax_rate = max(0.0, min(0.35, tax / op_income))
    elif op_income not in (None, 0):
        tax_rate = 0.21
    nopat = op_income * (1 - tax_rate) if op_income is not None else None
    invested_capital = equity + (debt or 0) - (cash or 0) if equity is not None else None
    roic = (nopat / invested_capital * 100) if nopat is not None and invested_capital not in (None, 0) else None
    roe = (net_income / equity * 100) if net_income is not None and equity not in (None, 0) else None
    roa = (net_income / assets * 100) if net_income is not None and assets not in (None, 0) else None
    op_margin = (op_income / revenue * 100) if op_income is not None and revenue not in (None, 0) else None
    current_ratio = current_assets / current_liab if current_assets is not None and current_liab not in (None, 0) else None
    debt_equity = debt / equity * 100 if debt is not None and equity not in (None, 0) else None
    interest_cov = op_income / interest if op_income is not None and interest not in (None, 0) else None
    fcf = (cfo + capex) if cfo is not None and capex is not None else None
    ebitda = (op_income + (depreciation or 0) + (amortization or 0)) if op_income is not None else None
    net_debt = (debt - cash) if debt is not None and cash is not None else None
    enterprise_value = (market_cap + net_debt) if market_cap is not None and net_debt is not None else None
    ev_ebitda = enterprise_value / ebitda if enterprise_value is not None and ebitda not in (None, 0) else None
    fcf_yield = (fcf * 1e6 / market_cap * 100) if fcf is not None and market_cap not in (None, 0) else None
    cfo_ni = cfo / net_income if cfo is not None and net_income not in (None, 0) else None

    macro_latest = _macro_snapshot(macro)
    policy_rate = _num(macro_latest.get("base_rate", {}).get("DATA_VALUE")) if macro_latest.get("base_rate") else None
    ktb3 = _num(macro_latest.get("ktb_3y", {}).get("DATA_VALUE")) if macro_latest.get("ktb_3y") else None
    ktb10 = _num(macro_latest.get("ktb_10y", {}).get("DATA_VALUE")) if macro_latest.get("ktb_10y") else None
    usdkrw = _num(macro_latest.get("usdkrw", {}).get("DATA_VALUE")) if macro_latest.get("usdkrw") else None
    cpi = _num(macro_latest.get("cpi", {}).get("DATA_VALUE")) if macro_latest.get("cpi") else None
    gdp = _num(macro_latest.get("gdp_real", {}).get("DATA_VALUE")) if macro_latest.get("gdp_real") else None
    spread = ktb10 - ktb3 if ktb10 is not None and ktb3 is not None else None

    def add(cat, feature, value, unit="", flag="OK", source="OpenDART"):
        if value is None:
            flag = "N/A" if flag == "OK" else flag
        rows.append({"category": cat, "feature": feature, "value": value, "unit": unit, "quality_flag": flag, "source": source})

    # Value / market-implied metrics
    add("Value", "Current Price", current_price, "KRW", source="Naver Finance")
    add("Value", "PER", per, "x", source="Naver Finance")
    add("Value", "PBR", pbr, "x", source="Naver Finance")
    add("Value", "Market Cap", market_cap, "KRW", "OK" if market_cap is not None else "N/A", "Naver Finance")
    add("Value", "Listed Shares", listed_shares, "shares", "OK" if listed_shares is not None else "N/A", "Naver Finance")
    add("Value", "Net Debt", net_debt, "KRW million")
    add("Value", "Enterprise Value", enterprise_value, "KRW")
    add("Value", "EBITDA", ebitda, "KRW million")
    add("Value", "EV/EBITDA", ev_ebitda, "x")
    add("Value", "FCF Yield", fcf_yield, "%")

    # Quality
    add("Quality", "ROIC", roic, "%")
    add("Quality", "ROE", roe, "%")
    add("Quality", "ROA", roa, "%")
    add("Quality", "Operating Margin", op_margin, "%")
    add("Quality", "Current Ratio", current_ratio, "x")
    add("Quality", "Debt / Equity", debt_equity, "%")
    add("Quality", "Interest Coverage", interest_cov, "x")
    add("Quality", "CFO / Net Income", cfo_ni, "x")
    add("Quality", "FCF", fcf, "KRW million")
    add("Quality", "FCF Margin", (fcf / revenue * 100) if fcf is not None and revenue not in (None, 0) else None, "%")

    # Growth
    add("Growth", "Revenue Growth", _growth(revenue, revenue_prev), "%")
    add("Growth", "Operating Profit Growth", _growth(op_income, op_income_prev), "%")
    add("Growth", "Net Income Growth", _growth(net_income, net_income_prev), "%")
    add("Growth", "FCF Growth", _growth(fcf, cfo_prev + capex if cfo_prev is not None and capex is not None else None), "%", "INFO")

    # Momentum / liquidity
    for name, value in [("1M Return", ret_1m), ("3M Return", ret_3m), ("6M Return", ret_6m), ("12M Return", ret_12m), ("Annualized Volatility", vol), ("Max Drawdown", dd)]:
        add("Momentum", name, value, "%", source="Naver Finance")
    latest_volume = None
    avg_volume_20d = None
    if hist is not None and not hist.empty and "volume" in hist.columns:
        v = pd.to_numeric(hist["volume"], errors="coerce").dropna()
        if not v.empty:
            latest_volume = float(v.iloc[-1])
            if len(v) >= 20:
                avg_volume_20d = float(v.tail(20).mean())
    add("Momentum", "Latest Volume", latest_volume, "shares", source="Naver Finance")
    add("Momentum", "20D Average Volume", avg_volume_20d, "shares", source="Naver Finance")

    # Accounting / risk
    add("Risk / Accounting", "Receivable Growth vs Sales Growth", None, "pp", "PENDING", "OpenDART comparison mapping")
    add("Risk / Accounting", "Inventory Growth vs Sales Growth", None, "pp", "PENDING", "OpenDART comparison mapping")
    add("Risk / Accounting", "CFO - Net Income", (cfo - net_income) if cfo is not None and net_income is not None else None, "KRW million")
    add("Risk / Accounting", "Debt", debt, "KRW million")
    add("Risk / Accounting", "Recent High-Risk Filings", int(_count_high_risk_notices(notices)), "count", source="OpenDART")

    # Macro: keep compact for investment use.
    add("Macro", "Policy Rate", policy_rate, "%", source="ECOS")
    add("Macro", "KTB 3Y", ktb3, "%", source="ECOS")
    add("Macro", "KTB 10Y", ktb10, "%", source="ECOS")
    add("Macro", "10Y - 3Y Spread", spread, "%p", source="ECOS")
    add("Macro", "USD/KRW", usdkrw, "KRW", source="ECOS")
    add("Macro", "CPI Index", cpi, "index", source="ECOS")
    add("Macro", "Real GDP", gdp, "index", source="ECOS")

    add("Industry", "Industry / Sector", company_info.get("induty_code") or company_info.get("industry_code"), "", source="OpenDART/company info")
    add("Company", "Fiscal Year", company_info.get("acc_mt"), "", source="OpenDART")
    add("Company", "Employees", company_info.get("emp_stdn_nb"), "people", source="OpenDART")

    return pd.DataFrame(rows)


def _count_high_risk_notices(notices: pd.DataFrame) -> int:
    if notices is None or notices.empty or "report_nm" not in notices.columns:
        return 0
    keywords = ["유상증자", "전환사채", "횡령", "배임", "소송", "감사의견", "관리종목", "상장폐지"]
    s = notices["report_nm"].astype(str)
    return int(s.str.contains("|".join(keywords), case=False, na=False).sum())
