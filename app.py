from __future__ import annotations

from datetime import date
import os

import pandas as pd
import streamlit as st

from data_sources import OpenDARTClient, NaverFinanceClient, ECOSClient
from feature_engine import build_feature_table
from valuation_engine import ValuationEngine, ValuationAssumptions

SECRET_ALIASES = {
    "OPENDART_API_KEY": ["OPENDART_API_KEY", "DART_API_KEY", "OPEN_DART_API_KEY"],
    "ECOS_API_KEY": ["ECOS_API_KEY", "BOK_ECOS_API_KEY"],
    "OPENAI_API_KEY": ["OPENAI_API_KEY", "OPENAI_KEY", "GPT_API_KEY"],
}
SECTION_NAMES = ("default", "api_keys", "secrets", "keys")


def _text(value):
    return "" if value is None else str(value).strip()


def load_settings():
    values = {}
    try:
        secrets = st.secrets
    except Exception:
        secrets = {}

    for canonical, aliases in SECRET_ALIASES.items():
        value = None
        for name in aliases:
            try:
                if name in secrets and _text(secrets[name]):
                    value = _text(secrets[name])
                    break
            except Exception:
                pass
        if not value:
            for section_name in SECTION_NAMES:
                try:
                    if section_name not in secrets:
                        continue
                    section = secrets[section_name]
                    for name in aliases:
                        if name in section and _text(section[name]):
                            value = _text(section[name])
                            break
                    if value:
                        break
                except Exception:
                    pass
        if not value:
            for name in aliases:
                value = _text(os.getenv(name))
                if value:
                    break
        if value:
            values[canonical] = value
    return values


def feature_dict(df: pd.DataFrame) -> dict:
    if df is None or df.empty:
        return {}
    out = {}
    for row in df.to_dict(orient="records"):
        key = row.get("feature")
        if not key:
            continue
        out[str(key)] = {
            "value": row.get("value"),
            "unit": row.get("unit"),
            "quality_flag": row.get("quality_flag"),
            "source": row.get("source"),
            "category": row.get("category"),
        }
    return out


def clean_for_display(df: pd.DataFrame, limit: int = 100) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()
    return df.head(limit).copy()


def fmt_money(value):
    if value is None:
        return "-"
    try:
        return f"{float(value):,.0f}원"
    except Exception:
        return "-"


def fmt_pct(value):
    if value is None:
        return "-"
    try:
        return f"{float(value):.1f}%"
    except Exception:
        return "-"


st.set_page_config(page_title="기업 분석 Engine", page_icon="📈", layout="wide")
st.title("📈 기업 분석 Engine")
st.caption("Step 1 Feature Engine → Step 2 Valuation Engine")

# Step indicator
s1, s2, s3, s4 = st.columns(4)
s1.markdown("### 🟢 1. Feature Engine")
s2.markdown("### 🔵 2. Valuation Engine")
s3.markdown("### ⚪ 3. Decision Engine")
s4.markdown("### ⚪ 4. Backtest")

settings = load_settings()
with st.sidebar:
    st.header("API 설정")
    dart_key = st.text_input("OpenDART API Key", value=settings.get("OPENDART_API_KEY", ""), type="password")
    ecos_key = settings.get("ECOS_API_KEY", "")
    st.write(f"OpenDART: {'감지됨' if dart_key else '없음'}")
    st.write(f"ECOS: {'감지됨' if ecos_key else '없음'}")
    st.info("현재 Step 1~2에서는 GPT를 호출하지 않습니다.")

company = st.text_input("분석할 기업명", placeholder="예: 삼성전자", key="company_input")
year = st.number_input(
    "재무 기준연도",
    min_value=2015,
    max_value=date.today().year,
    value=max(2015, date.today().year - 1),
    step=1,
    key="analysis_year",
)

run_feature = st.button("① Feature Engine 실행", type="primary", use_container_width=True)

if run_feature:
    if not company.strip():
        st.error("기업명을 입력하세요.")
        st.stop()
    if not dart_key:
        st.error("OpenDART API Key를 Streamlit Cloud Secrets에 등록하세요.")
        st.stop()

    try:
        with st.status("Step 1 데이터 수집 및 Feature 계산 중...", expanded=True) as status:
            dart = OpenDARTClient(dart_key)
            corp = dart.resolve_company(company.strip())
            if not corp:
                status.update(label="기업을 찾지 못했습니다.", state="error")
                st.error("OpenDART에서 기업을 찾지 못했습니다.")
                st.stop()
            status.write("✓ OpenDART 기업코드 확인")

            financials = dart.get_financials(corp["corp_code"], year=year)
            notices = dart.search_filings(corp["corp_code"], bgn_de=date(year, 1, 1), end_de=date.today())
            company_info = dart.get_company_info(corp["corp_code"])
            status.write("✓ DART 재무/공시 수집")

            market = None
            market_error = None
            try:
                market = NaverFinanceClient().get_snapshot_and_history(corp.get("stock_code"))
                status.write("✓ 시장가격/거래량 수집")
            except Exception as exc:
                market_error = str(exc)
                status.write("⚠ 시장데이터 일부 실패")

            macro = None
            macro_error = None
            if ecos_key:
                try:
                    macro = ECOSClient(ecos_key).get_macro_snapshot()
                    status.write("✓ ECOS 거시데이터 수집")
                except Exception as exc:
                    macro_error = str(exc)
                    status.write("⚠ 거시데이터 일부 실패")
            else:
                macro_error = "ECOS API Key가 없습니다."

            features = build_feature_table(
                company_info,
                financials,
                market,
                macro,
                notices,
                pd.Timestamp.today().normalize(),
            )
            status.write("✓ Feature Engine 계산 완료")
            status.update(label="Step 1 완료", state="complete")

        st.session_state["analysis_ready"] = True
        st.session_state["company_name"] = company_info.get("corp_name", company.strip())
        st.session_state["corp"] = corp
        st.session_state["company_info"] = company_info
        st.session_state["financials"] = financials
        st.session_state["notices"] = notices
        st.session_state["market"] = market
        st.session_state["macro"] = macro
        st.session_state["features"] = features
        st.session_state["market_error"] = market_error
        st.session_state["macro_error"] = macro_error
        st.session_state.pop("valuation", None)

    except Exception as exc:
        st.error(f"Feature Engine 실행 중 오류가 발생했습니다: {exc}")
        st.exception(exc)

# Step 1 results persist across Streamlit reruns.
if st.session_state.get("analysis_ready"):
    features = st.session_state["features"]
    market = st.session_state.get("market") or {}
    macro = st.session_state.get("macro") or {}
    notices = st.session_state.get("notices")
    company_name = st.session_state.get("company_name", company)
    fd = feature_dict(features)

    if st.session_state.get("market_error"):
        st.warning(f"네이버 증권 데이터 일부 수집 실패: {st.session_state['market_error']}")
    if st.session_state.get("macro_error"):
        st.warning(f"ECOS 거시데이터 일부 수집 실패: {st.session_state['macro_error']}")

    st.success(f"{company_name} Feature Engine 완료")
    cols = st.columns(6)
    cols[0].metric("현재가", fmt_money(market.get("current_price")))
    cols[1].metric("PER", str(fd.get("PER", {}).get("value", "-")))
    cols[2].metric("PBR", str(fd.get("PBR", {}).get("value", "-")))
    cols[3].metric("ROIC", fmt_pct(fd.get("ROIC", {}).get("value")))
    cols[4].metric("ROE", fmt_pct(fd.get("ROE", {}).get("value")))
    cols[5].metric("FCF", f"{fd.get('FCF', {}).get('value'):,.0f}" if isinstance(fd.get('FCF', {}).get('value'), (int, float)) else "-")

    t1, t2, t3, t4 = st.tabs(["📊 Feature Table", "💰 Value", "📈 Market / Macro", "📄 DART 공시"])
    with t1:
        st.dataframe(clean_for_display(features, 250), use_container_width=True, hide_index=True)
    with t2:
        value_df = features[features["category"].astype(str).str.contains("Value", case=False, na=False)] if "category" in features.columns else pd.DataFrame()
        st.dataframe(clean_for_display(value_df, 100), use_container_width=True, hide_index=True)
    with t3:
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("### 시장 데이터")
            st.json({
                "current_price": market.get("current_price"),
                "market_cap": market.get("market_cap"),
                "per": market.get("per"),
                "pbr": market.get("pbr"),
                "listed_shares": market.get("listed_shares"),
            })
        with c2:
            st.markdown("### 주요 거시 Feature")
            macro_rows = features[features["category"].astype(str).eq("Macro")] if "category" in features.columns else pd.DataFrame()
            st.dataframe(clean_for_display(macro_rows, 50), use_container_width=True, hide_index=True)
    with t4:
        st.dataframe(clean_for_display(notices, 100), use_container_width=True, hide_index=True)

    st.divider()
    st.subheader("② Valuation Engine")
    st.caption("현재 시장 배수는 Feature로 사용하고, 적정 배수/DCF는 별도의 가정으로 계산합니다.")

    with st.expander("Valuation 가정 보기 / 수정", expanded=False):
        a1, a2, a3 = st.columns(3)
        wacc = a1.number_input("WACC (%)", 4.0, 20.0, 9.0, 0.25, key="v_wacc")
        tg = a1.number_input("Terminal Growth (%)", 0.0, 5.0, 2.5, 0.25, key="v_tg")
        years = a1.number_input("Forecast Years", 3, 10, 5, 1, key="v_years")
        fcf_growth = a2.number_input("FCF Growth (%)", -10.0, 30.0, 5.0, 0.5, key="v_fcf_growth")
        fair_per = a2.number_input("Fair PER (x)", 5.0, 40.0, 12.0, 0.5, key="v_fair_per")
        fair_pbr = a2.number_input("Fair PBR (x)", 0.3, 8.0, 1.5, 0.1, key="v_fair_pbr")
        weight_dcf = a3.number_input("DCF Weight", 0.0, 1.0, 0.50, 0.05, key="v_weight_dcf")
        weight_per = a3.number_input("PER Weight", 0.0, 1.0, 0.25, 0.05, key="v_weight_per")
        weight_pbr = a3.number_input("PBR Weight", 0.0, 1.0, 0.25, 0.05, key="v_weight_pbr")
        if abs((weight_dcf + weight_per + weight_pbr) - 1.0) > 1e-6:
            st.warning("가중치 합계가 100%가 아닙니다. 계산 시 사용 가능한 방법 기준으로 재정규화됩니다.")

    run_valuation = st.button("➡️ ② Valuation Engine 실행", use_container_width=True)
    if run_valuation:
        assumptions = ValuationAssumptions(
            wacc=float(wacc),
            terminal_growth=float(tg),
            forecast_years=int(years),
            fcf_growth=float(fcf_growth),
            fair_per=float(fair_per),
            fair_pbr=float(fair_pbr),
            weight_dcf=float(weight_dcf),
            weight_per=float(weight_per),
            weight_pbr=float(weight_pbr),
        )
        valuation = ValuationEngine(assumptions).calculate(features)
        st.session_state["valuation"] = valuation

    if st.session_state.get("valuation"):
        valuation = st.session_state["valuation"]
        st.success("Step 2 Valuation Engine 완료")

        vc = st.columns(5)
        vc[0].metric("현재가", fmt_money(valuation["inputs"].get("current_price")))
        vc[1].metric("종합 적정가", fmt_money(valuation.get("fair_value_per_share")))
        vc[2].metric("상승여력", fmt_pct(valuation.get("upside_pct")))
        vc[3].metric("ROIC - WACC", fmt_pct(valuation.get("economic_spread_roic_minus_wacc")))
        vc[4].metric("현재 PER", f"{valuation['inputs'].get('per'):.2f}x" if valuation['inputs'].get('per') is not None else "-")

        v1, v2, v3 = st.columns(3)
        with v1:
            st.markdown("### DCF")
            st.json(valuation.get("dcf_valuation", {}))
        with v2:
            st.markdown("### PER Valuation")
            st.json(valuation.get("per_valuation", {}))
        with v3:
            st.markdown("### PBR Valuation")
            st.json(valuation.get("pbr_valuation", {}))

        st.markdown("### 종합 적정가치")
        st.write(valuation.get("weights"))
        st.info("다음 단계에서는 이 결과를 바탕으로 매수/보류/매도 조건과 목표가격을 결정하는 Decision Engine을 연결합니다.")

        with st.expander("Valuation 상세 / 주의사항"):
            st.json({
                "inputs": valuation.get("inputs"),
                "assumptions": valuation.get("assumptions"),
                "caveats": valuation.get("caveats"),
            })
