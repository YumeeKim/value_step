from __future__ import annotations

import json
import os
import traceback
from datetime import date

import pandas as pd
import streamlit as st

from data_sources import OpenDARTClient, NaverFinanceClient, ECOSClient
from feature_engine import build_feature_table
from valuation_engine import ValuationEngine, ValuationAssumptions
from valuation_ai import ValuationAI

SECRET_ALIASES = {
    "OPENDART_API_KEY": ["OPENDART_API_KEY", "DART_API_KEY", "OPEN_DART_API_KEY"],
    "ECOS_API_KEY": ["ECOS_API_KEY", "BOK_ECOS_API_KEY"],
    "OPENAI_API_KEY": ["OPENAI_API_KEY", "OPENAI_KEY", "GPT_API_KEY"],
    "OPENAI_MODEL": ["OPENAI_MODEL"],
}
SECTION_NAMES = ("default", "api_keys", "secrets", "keys")


def _text(v):
    return "" if v is None else str(v).strip()


def load_settings():
    values = {}
    try:
        secrets = st.secrets
    except Exception:
        secrets = {}
    for canonical, aliases in SECRET_ALIASES.items():
        found = None
        for name in aliases:
            try:
                if name in secrets and _text(secrets[name]):
                    found = _text(secrets[name])
                    break
            except Exception:
                pass
        if not found:
            for section_name in SECTION_NAMES:
                try:
                    if section_name not in secrets:
                        continue
                    section = secrets[section_name]
                    for name in aliases:
                        if name in section and _text(section[name]):
                            found = _text(section[name])
                            break
                    if found:
                        break
                except Exception:
                    pass
        if not found:
            for name in aliases:
                v = _text(os.getenv(name))
                if v:
                    found = v
                    break
        if found:
            values[canonical] = found
    return values


def feature_dict(df: pd.DataFrame) -> dict:
    return {
        str(r.get("feature")): {"value": r.get("value"), "unit": r.get("unit"), "quality_flag": r.get("quality_flag"), "source": r.get("source"), "category": r.get("category")}
        for _, r in (df.iterrows() if df is not None and not df.empty else []) if r.get("feature")
    }


def compact_df(df, cols, n=30):
    if df is None or df.empty:
        return []
    use = [c for c in cols if c in df.columns]
    return df[use].head(n).to_dict(orient="records")


def _run_feature(company, year, dart_key, ecos_key):
    dart = OpenDARTClient(dart_key)
    corp = dart.resolve_company(company.strip())
    if not corp:
        raise RuntimeError("OpenDART에서 기업을 찾지 못했습니다.")
    financials = dart.get_financials(corp["corp_code"], year=year)
    notices = dart.search_filings(corp["corp_code"], bgn_de=date(year, 1, 1), end_de=date.today())
    company_info = dart.get_company_info(corp["corp_code"])
    market = None
    market_msg = ""
    try:
        market = NaverFinanceClient().get_snapshot_and_history(corp.get("stock_code"))
    except Exception as e:
        market_msg = f"네이버 증권 데이터 수집 실패: {e}"
    macro = None
    macro_msg = ""
    if ecos_key:
        try:
            macro = ECOSClient(ecos_key).get_macro_snapshot()
        except Exception as e:
            macro_msg = f"ECOS 데이터 수집 실패: {e}"
    else:
        macro_msg = "ECOS API Key가 없어 거시 Feature가 비어 있습니다."
    features = build_feature_table(company_info, financials, market, macro, notices, pd.Timestamp.today().normalize())
    dossier = {
        "company": {"corp_name": company_info.get("corp_name", company.strip()), "stock_code": corp.get("stock_code"), "corp_code": corp.get("corp_code"), "sector_code": company_info.get("induty_code") or company_info.get("industry_code")},
        "features": feature_dict(features),
        "recent_filings": compact_df(notices, ["rcept_dt", "report_nm", "flr_nm", "corp_name"]),
        "market": {"current_price": (market or {}).get("current_price")},
    }
    return {"corp": corp, "company_info": company_info, "financials": financials, "notices": notices, "market": market, "macro": macro, "features": features, "dossier": dossier, "market_msg": market_msg, "macro_msg": macro_msg}


st.set_page_config(page_title="기업가치 Agent", page_icon="📈", layout="wide")
st.title("📈 기업가치 분석 Agent")
st.caption("Step 1 Feature Engine → Step 2 Valuation Engine")
settings = load_settings()

with st.sidebar:
    st.header("API")
    dart_key = st.text_input("OpenDART API Key", value=settings.get("OPENDART_API_KEY", ""), type="password")
    ecos_key = settings.get("ECOS_API_KEY", "")
    openai_key = settings.get("OPENAI_API_KEY", "")
    model = settings.get("OPENAI_MODEL", "gpt-5.6-luna")
    st.write(f"OpenDART: {'감지됨' if dart_key else '없음'}")
    st.write(f"ECOS: {'감지됨' if ecos_key else '없음'}")
    st.write(f"OpenAI: {'감지됨' if openai_key else '없음'}")
    st.caption("Step 1에서는 GPT를 호출하지 않습니다. Step 2의 가정 검토를 요청할 때만 GPT를 사용합니다.")

company = st.text_input("분석할 기업명", placeholder="예: 삼성전자")
year = st.number_input("재무 기준연도", min_value=2015, max_value=date.today().year, value=max(2015, date.today().year - 1), step=1)

if "research" not in st.session_state:
    st.session_state.research = None
if "valuation" not in st.session_state:
    st.session_state.valuation = None

c1, c2, c3 = st.columns(3)
run_feature = c1.button("① Feature Engine 실행", type="primary", use_container_width=True)
go_valuation = c2.button("② Valuation Engine", use_container_width=True)
reset = c3.button("초기화", use_container_width=True)

if reset:
    st.session_state.research = None
    st.session_state.valuation = None
    st.rerun()

if run_feature:
    if not company.strip():
        st.error("기업명을 입력하세요.")
        st.stop()
    if not dart_key:
        st.error("OpenDART API Key가 필요합니다.")
        st.stop()
    try:
        with st.status("Step 1 데이터 수집 중...", expanded=True) as s:
            st.session_state.research = _run_feature(company, int(year), dart_key, ecos_key)
            st.session_state.valuation = None
            s.update(label="Step 1 완료", state="complete")
    except Exception as e:
        st.error(f"Feature Engine 실행 중 오류가 발생했습니다: {e}")
        with st.expander("개발용 오류 로그"):
            st.code(traceback.format_exc())

research = st.session_state.research
if go_valuation and research is None:
    st.warning("먼저 ① Feature Engine을 실행해 주세요.")

if research:
    if research.get("market_msg"):
        st.warning(research["market_msg"])
    if research.get("macro_msg"):
        st.warning(research["macro_msg"])
    fd = feature_dict(research["features"])
    m = research.get("market") or {}
    st.success(f"{research['company_info'].get('corp_name', company)} - Feature 완료")
    cols = st.columns(6)
    cols[0].metric("현재가", f"{m.get('current_price'):,.0f}원" if m.get("current_price") else "-")
    cols[1].metric("PER", f"{fd.get('PER', {}).get('value'):.2f}x" if isinstance(fd.get('PER', {}).get('value'), (int, float)) else "-")
    cols[2].metric("PBR", f"{fd.get('PBR', {}).get('value'):.2f}x" if isinstance(fd.get('PBR', {}).get('value'), (int, float)) else "-")
    cols[3].metric("ROIC", f"{fd.get('ROIC', {}).get('value'):.2f}%" if isinstance(fd.get('ROIC', {}).get('value'), (int, float)) else "-")
    cols[4].metric("FCF", f"{fd.get('FCF', {}).get('value'):,.0f}" if isinstance(fd.get('FCF', {}).get('value'), (int, float)) else "-")
    cols[5].metric("12M 수익률", f"{fd.get('12M Return', {}).get('value'):.2f}%" if isinstance(fd.get('12M Return', {}).get('value'), (int, float)) else "-")

    t1, t2 = st.tabs(["① Feature Engine", "② Valuation Engine"])
    with t1:
        st.dataframe(research["features"][["category", "feature", "value", "unit", "quality_flag", "source"]], use_container_width=True, hide_index=True)
        st.subheader("최근 공시")
        n = research.get("notices")
        if n is not None and not n.empty:
            st.dataframe(n[[c for c in ["rcept_dt", "report_nm", "flr_nm"] if c in n.columns]].head(30), use_container_width=True, hide_index=True)
    with t2:
        if st.session_state.valuation is None and go_valuation:
            st.session_state.valuation = ValuationEngine().calculate(research["features"])
        if st.session_state.valuation is None:
            st.info("'② Valuation Engine' 버튼을 누르세요.")
        else:
            v = st.session_state.valuation
            aa = v["auto_assumptions"]
            st.subheader("자동 산출 가정")
            ac = st.columns(6)
            ac[0].metric("WACC", f"{aa['wacc']:.2f}%")
            ac[1].metric("장기성장률", f"{aa['terminal_growth']:.2f}%")
            ac[2].metric("FCF 성장", f"{aa['fcf_growth']:.2f}%")
            ac[3].metric("Fair PER", f"{aa['fair_per']:.2f}x")
            ac[4].metric("Fair PBR", f"{aa['fair_pbr']:.2f}x")
            ac[5].metric("ROIC-WACC", f"{v['economic_spread_roic_minus_wacc']:.2f}%p" if v['economic_spread_roic_minus_wacc'] is not None else "-")

            c = st.columns(3)
            for idx, (title, key) in enumerate([("DCF", "dcf_valuation"), ("PER", "per_valuation"), ("PBR", "pbr_valuation")]):
                r = v[key]
                with c[idx]:
                    st.markdown(f"### {title}")
                    if r.get("status") == "ok":
                        st.metric("적정주가", f"{r['fair_price']:,.0f}원")
                        st.write(f"상승여력: {r['upside_pct']:.1f}%")
                    else:
                        st.warning(r.get("reason", "계산 불가"))

            st.divider()
            fv = v.get("fair_value_per_share")
            if fv is not None:
                st.metric("종합 적정주가", f"{fv:,.0f}원", f"상승여력 {v['upside_pct']:.1f}%" if v.get("upside_pct") is not None else None)

            st.subheader("Bear / Base / Bull")
            scenario_rows = []
            for name, sc in v["scenarios"].items():
                vals = [sc[m]["fair_price"] for m in ["dcf", "per", "pbr"] if sc[m].get("status") == "ok"]
                scenario_rows.append({"Scenario": name, "WACC": sc["assumptions"]["wacc"], "FCF Growth": sc["assumptions"]["fcf_growth"], "DCF": sc["dcf"].get("fair_price"), "PER": sc["per"].get("fair_price"), "PBR": sc["pbr"].get("fair_price")})
            st.dataframe(pd.DataFrame(scenario_rows), use_container_width=True, hide_index=True)

            with st.expander("고급: 수동 가정으로 다시 계산"):
                aa = v["auto_assumptions"]
                mw1, mw2, mw3 = st.columns(3)
                wacc = mw1.number_input("WACC", value=float(aa["wacc"]), min_value=1.0, max_value=20.0, step=0.25)
                tg = mw2.number_input("Terminal Growth", value=float(aa["terminal_growth"]), min_value=0.0, max_value=8.0, step=0.25)
                fg = mw3.number_input("FCF Growth", value=float(aa["fcf_growth"]), min_value=-20.0, max_value=20.0, step=0.5)
                if st.button("수동 가정으로 재계산"):
                    assumptions = ValuationAssumptions(wacc=wacc, terminal_growth=tg, fcf_growth=fg)
                    st.session_state.valuation = ValuationEngine(assumptions).calculate(research["features"])
                    st.rerun()

            if openai_key:
                st.subheader("🤖 GPT Valuation Review")
                if st.button("GPT로 Valuation 가정 검토"):
                    ai = ValuationAI(openai_key, model=model).review(feature_dict(research["features"]), v, compact_df(research.get("notices"), ["rcept_dt", "report_nm", "flr_nm"], 30))
                    st.json(ai)

    with st.expander("원본 JSON"):
        st.json({"features": feature_dict(research["features"]), "company": research["company_info"], "market": research["market"], "macro": research["macro"]})
