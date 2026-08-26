from __future__ import annotations

import os
from datetime import date

import pandas as pd
import streamlit as st

from data_sources import OpenDARTClient, NaverFinanceClient, ECOSClient
from feature_engine import build_feature_table

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
                env_value = _text(os.getenv(name))
                if env_value:
                    found = env_value
                    break
        if found:
            values[canonical] = found
    return values


def secret_status(settings):
    return {k: bool(settings.get(k)) for k in SECRET_ALIASES}


def feature_dict(df: pd.DataFrame) -> dict:
    if df is None or df.empty:
        return {}
    out = {}
    for row in df.to_dict(orient="records"):
        key = row.get("feature")
        if key:
            out[key] = row
    return out


def compact_df(df: pd.DataFrame, columns, n=30):
    if df is None or df.empty:
        return []
    use = [c for c in columns if c in df.columns]
    return df[use].head(n).to_dict(orient="records")


def render_feature_cards(features: pd.DataFrame):
    fd = feature_dict(features)
    cols = st.columns(6)
    items = [
        ("현재가", "Current Price", "원"),
        ("PER", "PER", "x"),
        ("PBR", "PBR", "x"),
        ("ROIC", "ROIC", "%"),
        ("ROE", "ROE", "%"),
        ("12M 수익률", "12M Return", "%"),
    ]
    for col, (label, key, unit) in zip(cols, items):
        value = fd.get(key, {}).get("value")
        if value is None:
            text = "-"
        elif unit == "원":
            try:
                text = f"{float(value):,.0f}원"
            except Exception:
                text = str(value)
        elif unit == "%":
            text = f"{float(value):.2f}%"
        else:
            text = f"{float(value):.2f}x"
        col.metric(label, text)


st.set_page_config(page_title="기업분석 Agent", page_icon="📈", layout="wide")
st.title("📈 기업분석 Agent")
st.caption("Step 1 — Feature Engine. 데이터 수집과 정량 지표 계산까지만 실행합니다.")

# Workflow state
if "analysis_stage" not in st.session_state:
    st.session_state.analysis_stage = 1
if "analysis_payload" not in st.session_state:
    st.session_state.analysis_payload = None

# Step navigation
step_cols = st.columns(4)
steps = [
    ("1", "Feature Engine"),
    ("2", "Valuation Engine"),
    ("3", "Decision Engine"),
    ("4", "Backtest"),
]
for i, (num, label) in enumerate(steps, start=1):
    if i < st.session_state.analysis_stage:
        step_cols[i-1].success(f"✓ {num}. {label}")
    elif i == st.session_state.analysis_stage:
        step_cols[i-1].info(f"▶ {num}. {label}")
    else:
        step_cols[i-1].warning(f"○ {num}. {label}")

settings = load_settings()
with st.sidebar:
    st.header("API 상태")
    dart_key = st.text_input("OpenDART API Key", value=settings.get("OPENDART_API_KEY", ""), type="password")
    ecos_key = settings.get("ECOS_API_KEY", "")
    st.write(f"OpenDART: {'감지됨' if dart_key else '없음'}")
    st.write(f"ECOS: {'감지됨' if ecos_key else '없음'}")
    st.write(f"OpenAI/GPT: {'저장됨' if settings.get('OPENAI_API_KEY') else '없음'}")
    st.caption("GPT는 Step 1에서는 호출하지 않습니다. 키만 보관합니다.")

company = st.text_input("분석할 기업명", placeholder="예: 삼성전자")
year = st.number_input(
    "재무 기준연도",
    min_value=2015,
    max_value=date.today().year,
    value=max(2015, date.today().year - 1),
    step=1,
)

if st.session_state.analysis_stage == 1:
    run = st.button("① Feature Engine 실행", type="primary", use_container_width=True)
else:
    run = False

if run:
    if not company.strip():
        st.error("기업명을 입력하세요.")
        st.stop()
    if not dart_key:
        st.error("OpenDART API Key가 필요합니다.")
        st.stop()

    try:
        with st.status("Feature Engine 실행 중...", expanded=True) as status:
            dart = OpenDARTClient(dart_key)
            corp = dart.resolve_company(company.strip())
            if not corp:
                st.error("OpenDART에서 기업을 찾지 못했습니다.")
                st.stop()
            status.write("✓ 기업코드 확인")

            financials = dart.get_financials(corp["corp_code"], year=year)
            notices = dart.search_filings(corp["corp_code"], bgn_de=date(year, 1, 1), end_de=date.today())
            company_info = dart.get_company_info(corp["corp_code"])
            status.write("✓ OpenDART 재무/공시 수집")

            market = None
            market_msg = ""
            try:
                market = NaverFinanceClient().get_snapshot_and_history(corp.get("stock_code"))
                status.write("✓ Naver 시장가격/거래량 수집")
            except Exception as exc:
                market_msg = f"네이버 시장데이터 일부 실패: {exc}"
                status.write("⚠ 시장데이터 일부 실패")

            macro = None
            macro_msg = ""
            if ecos_key:
                try:
                    macro = ECOSClient(ecos_key).get_macro_snapshot()
                    status.write("✓ ECOS 거시데이터 수집")
                except Exception as exc:
                    macro_msg = f"ECOS 데이터 일부 실패: {exc}"
                    status.write("⚠ ECOS 일부 실패")
            else:
                macro_msg = "ECOS API Key가 없어 거시 Feature가 비어 있을 수 있습니다."

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

        st.session_state.analysis_payload = {
            "company": company_info,
            "corp": corp,
            "financials": financials,
            "notices": notices,
            "market": market,
            "macro": macro,
            "features": features,
            "market_msg": market_msg,
            "macro_msg": macro_msg,
            "year": year,
        }
        st.session_state.analysis_stage = 1

    except Exception as exc:
        st.error(f"Feature Engine 실행 중 오류가 발생했습니다: {exc}")
        st.exception(exc)

payload = st.session_state.analysis_payload
if payload:
    if payload.get("market_msg"):
        st.warning(payload["market_msg"])
    if payload.get("macro_msg"):
        st.warning(payload["macro_msg"])

    st.success(f"{payload['company'].get('corp_name', company)} — Step 1 Feature Engine 완료")
    render_feature_cards(payload["features"])

    t1, t2, t3, t4 = st.tabs(["📊 Feature Table", "📄 공시", "📈 시장/거시", "🧩 원본 데이터"])
    with t1:
        st.dataframe(payload["features"], use_container_width=True, hide_index=True)
    with t2:
        notices = payload["notices"]
        if notices is None or notices.empty:
            st.info("공시가 없습니다.")
        else:
            st.dataframe(notices, use_container_width=True, hide_index=True)
    with t3:
        market = payload["market"] or {}
        hist = market.get("history")
        if hist is not None and not hist.empty:
            cols = [c for c in ["date", "open", "high", "low", "close", "volume"] if c in hist.columns]
            st.dataframe(hist[cols].tail(30), use_container_width=True, hide_index=True)
        macro = payload["macro"] or {}
        series = macro.get("series", {}) if isinstance(macro, dict) else {}
        if series:
            st.markdown("### 거시 시계열")
            for key, label in [("base_rate", "기준금리"), ("market_rates", "시장금리")]:
                rows = series.get(key, [])
                if rows:
                    st.write(label)
                    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
    with t4:
        st.write("회사정보")
        st.json(payload["company"])
        st.write("시장 원본")
        st.json({k: v for k, v in (payload["market"] or {}).items() if k != "history"})
        st.write("거시 원본")
        st.json(payload["macro"] or {})

    st.divider()
    c1, c2 = st.columns(2)
    with c1:
        st.button("✅ Step 1 확정", use_container_width=True, disabled=True)
    with c2:
        if st.button("➡️ 다음: Valuation Engine", type="primary", use_container_width=True):
            st.session_state.analysis_stage = 2
            st.rerun()

if st.session_state.analysis_stage >= 2 and payload:
    st.divider()
    st.header("② Valuation Engine")
    st.info("다음 단계에서 구현합니다. 현재는 Step 1 결과를 보존한 상태입니다.")
    st.write("전달될 핵심 입력:")
    fd = feature_dict(payload["features"])
    st.write({
        "Current Price": fd.get("Current Price", {}).get("value"),
        "PER": fd.get("PER", {}).get("value"),
        "PBR": fd.get("PBR", {}).get("value"),
        "ROIC": fd.get("ROIC", {}).get("value"),
        "ROE": fd.get("ROE", {}).get("value"),
        "FCF": fd.get("FCF", {}).get("value"),
    })
