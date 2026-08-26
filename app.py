from __future__ import annotations

from datetime import date
import os

import pandas as pd
import streamlit as st

from data_sources import OpenDARTClient, NaverFinanceClient, ECOSClient
from feature_engine import build_feature_table

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

        # Top-level Streamlit Secrets
        for name in aliases:
            try:
                if name in secrets and _text(secrets[name]):
                    value = _text(secrets[name])
                    break
            except Exception:
                pass

        # Optional nested sections
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

        # Environment fallback
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
        out[key] = {
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
    out = df.copy()
    if len(out) > limit:
        out = out.head(limit)
    return out


st.set_page_config(page_title="기업 Feature Engine", page_icon="📊", layout="wide")
st.title("📊 기업 Feature Engine")
st.caption("Step 1 · OpenDART + Naver Finance + ECOS → 정량 Feature 계산")

settings = load_settings()

with st.sidebar:
    st.header("API 설정")
    dart_key = st.text_input(
        "OpenDART API Key",
        value=settings.get("OPENDART_API_KEY", ""),
        type="password",
    )
    ecos_key = settings.get("ECOS_API_KEY", "")
    openai_key = settings.get("OPENAI_API_KEY", "")

    st.write(f"OpenDART: {'감지됨' if dart_key else '없음'}")
    st.write(f"ECOS: {'감지됨' if ecos_key else '없음'}")
    st.write(f"OpenAI/GPT: {'감지됨' if openai_key else '없음'}")
    st.info("현재 Step 1에서는 GPT를 호출하지 않습니다. OpenAI 키는 이후 단계용으로만 저장할 수 있습니다.")

company = st.text_input("분석할 기업명", placeholder="예: 삼성전자")
year = st.number_input(
    "재무 기준연도",
    min_value=2015,
    max_value=date.today().year,
    value=max(2015, date.today().year - 1),
    step=1,
)

run = st.button("🚀 Feature Engine 실행", type="primary", use_container_width=True)

if run:
    if not company.strip():
        st.error("기업명을 입력하세요.")
        st.stop()
    if not dart_key:
        st.error("OpenDART API Key를 Streamlit Secrets에 등록하세요.")
        st.stop()

    try:
        with st.status("Step 1 데이터 수집 및 Feature 계산 중...", expanded=True) as status:
            # IMPORTANT: this preserves the known-good OpenDART client unchanged.
            dart = OpenDARTClient(dart_key)
            corp = dart.resolve_company(company.strip())
            if not corp:
                status.update(label="기업을 찾지 못했습니다.", state="error")
                st.error("OpenDART에서 기업을 찾지 못했습니다.")
                st.stop()
            status.write("✓ OpenDART 기업코드 확인")

            financials = dart.get_financials(corp["corp_code"], year=year)
            notices = dart.search_filings(
                corp["corp_code"],
                bgn_de=date(year, 1, 1),
                end_de=date.today(),
            )
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

        if market_error:
            st.warning(f"네이버 증권 데이터 일부 수집 실패: {market_error}")
        if macro_error:
            st.warning(f"ECOS 거시데이터 일부 수집 실패: {macro_error}")

        fd = feature_dict(features)
        corp_name = company_info.get("corp_name", company.strip())

        st.success(f"{corp_name} Feature Engine 완료")

        cols = st.columns(6)
        cols[0].metric("현재가", f"{market.get('current_price'):,.0f}원" if market and market.get("current_price") else "-")
        cols[1].metric("PER", str(fd.get("PER", {}).get("value", "-")))
        cols[2].metric("PBR", str(fd.get("PBR", {}).get("value", "-")))
        cols[3].metric("ROIC", f"{fd.get('ROIC', {}).get('value')}%" if fd.get("ROIC", {}).get("value") is not None else "-")
        cols[4].metric("ROE", f"{fd.get('ROE', {}).get('value')}%" if fd.get("ROE", {}).get("value") is not None else "-")
        cols[5].metric("FCF", f"{fd.get('FCF', {}).get('value'):,.0f}" if isinstance(fd.get("FCF", {}).get("value"), (int, float)) else "-")

        t1, t2, t3, t4, t5 = st.tabs([
            "📊 Feature Table",
            "💰 Value",
            "🏢 Quality/Growth",
            "📈 Market/Macro",
            "📄 DART 공시",
        ])

        with t1:
            st.dataframe(clean_for_display(features, 200), use_container_width=True, hide_index=True)

        with t2:
            value_df = features[features["category"].astype(str).str.contains("Value", case=False, na=False)] if not features.empty and "category" in features.columns else pd.DataFrame()
            st.dataframe(clean_for_display(value_df, 100), use_container_width=True, hide_index=True)

        with t3:
            if not features.empty and "category" in features.columns:
                q_df = features[features["category"].astype(str).str.contains("Quality|Growth|Risk|Accounting", case=False, na=False)]
            else:
                q_df = pd.DataFrame()
            st.dataframe(clean_for_display(q_df, 150), use_container_width=True, hide_index=True)

        with t4:
            left, right = st.columns(2)
            with left:
                st.markdown("### 시장 데이터")
                if market:
                    st.json({
                        "current_price": market.get("current_price"),
                        "market_cap": market.get("market_cap"),
                        "per": market.get("per"),
                        "pbr": market.get("pbr"),
                        "listed_shares": market.get("listed_shares"),
                    })
                else:
                    st.info("시장 데이터가 없습니다.")
            with right:
                st.markdown("### 거시 데이터")
                if macro:
                    st.json({"status": macro.get("status"), "series": macro.get("series")})
                else:
                    st.info("ECOS 데이터가 없습니다.")

        with t5:
            st.dataframe(clean_for_display(notices, 100), use_container_width=True, hide_index=True)

        st.divider()
        st.info("✅ Step 1 Feature Engine이 끝났습니다. 다음 단계는 Valuation Engine입니다.")

    except Exception as exc:
        st.error(f"Feature Engine 실행 중 오류가 발생했습니다: {exc}")
        st.exception(exc)
