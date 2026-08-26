from __future__ import annotations
import os
from datetime import date
import pandas as pd
import streamlit as st
from data_sources import OpenDARTClient, NaverFinanceClient, ECOSClient
from feature_engine import build_feature_table
from valuation_engine import ValuationEngine, ValuationAssumptions
from valuation_ai import ValuationAI

SECRET_ALIASES={'OPENDART_API_KEY':['OPENDART_API_KEY','DART_API_KEY','OPEN_DART_API_KEY'],'ECOS_API_KEY':['ECOS_API_KEY','BOK_ECOS_API_KEY'],'OPENAI_API_KEY':['OPENAI_API_KEY','OPENAI_KEY','GPT_API_KEY'],'OPENAI_MODEL':['OPENAI_MODEL']}
SECTION_NAMES=('default','api_keys','secrets','keys')
def _text(v): return '' if v is None else str(v).strip()
def load_settings():
    vals={}
    try: secrets=st.secrets
    except Exception: secrets={}
    for canonical,aliases in SECRET_ALIASES.items():
        val=None
        for n in aliases:
            try:
                if n in secrets and _text(secrets[n]): val=_text(secrets[n]); break
            except Exception: pass
        if not val:
            for sec in SECTION_NAMES:
                try:
                    if sec not in secrets: continue
                    block=secrets[sec]
                    for n in aliases:
                        if n in block and _text(block[n]): val=_text(block[n]); break
                    if val: break
                except Exception: pass
        if not val:
            for n in aliases:
                e=_text(os.getenv(n))
                if e: val=e; break
        if val: vals[canonical]=val
    return vals

def feature_dict(df):
    return {r.get('feature'):r for r in df.to_dict(orient='records')} if df is not None and not df.empty else {}
def fmt(r,d=2):
    v=r.get('value') if r else None
    if v is None: return '-'
    try: return f'{float(v):,.{d}f}'
    except Exception: return str(v)

st.set_page_config(page_title='기업가치 Agent',page_icon='📈',layout='wide')
st.title('📈 기업가치 분석 Agent')
st.caption('① Feature Engine → ② Valuation Engine')
settings=load_settings()
for k,v in [('features',None),('company_info',None),('notices',None),('market',None),('macro',None),('corp',None),('company',''),('valuation',None),('ai_review',None)]:
    if k not in st.session_state: st.session_state[k]=v
if 'step' not in st.session_state: st.session_state.step=1
with st.sidebar:
    st.header('API 상태')
    dart_key=st.text_input('OpenDART API Key',value=settings.get('OPENDART_API_KEY',''),type='password')
    ecos_key=settings.get('ECOS_API_KEY',''); openai_key=settings.get('OPENAI_API_KEY',''); openai_model=settings.get('OPENAI_MODEL','gpt-5.6')
    st.write(f"OpenDART: {'감지됨' if dart_key else '없음'}"); st.write(f"ECOS: {'감지됨' if ecos_key else '없음'}"); st.write(f"OpenAI: {'감지됨' if openai_key else '없음'}")
    step=st.radio('단계',['1. Feature Engine','2. Valuation Engine'],index=st.session_state.step-1)
    st.session_state.step=1 if step.startswith('1') else 2
company=st.text_input('분석 기업명',value=st.session_state.get('company',''),placeholder='예: 삼성전자')
year=st.number_input('재무 기준연도',2015,date.today().year,max(2015,date.today().year-1),1)
run_feature=st.button('① Feature Engine 실행',type='primary',use_container_width=True)
move=st.button('② Valuation Engine 이동',disabled=st.session_state.features is None,use_container_width=True)
if run_feature:
    if not company.strip(): st.error('기업명을 입력하세요.'); st.stop()
    if not dart_key: st.error('OpenDART API Key가 필요합니다.'); st.stop()
    try:
        with st.status('Feature Engine 실행 중...',expanded=True) as status:
            dart=OpenDARTClient(dart_key); corp=dart.resolve_company(company.strip())
            if not corp: raise RuntimeError('OpenDART에서 기업을 찾지 못했습니다.')
            status.write('✓ 기업코드 확인'); financials=dart.get_financials(corp['corp_code'],year=year); notices=dart.search_filings(corp['corp_code'],bgn_de=date(year,1,1),end_de=date.today()); info=dart.get_company_info(corp['corp_code']); status.write('✓ DART 재무/공시 수집')
            market=None
            try: market=NaverFinanceClient().get_snapshot_and_history(corp.get('stock_code')); status.write('✓ 시장가격/거래량 수집')
            except Exception as e: status.write(f'⚠ 시장데이터 실패: {e}')
            macro=None
            if ecos_key:
                try: macro=ECOSClient(ecos_key).get_macro_snapshot(); status.write('✓ ECOS 거시데이터 수집')
                except Exception as e: status.write(f'⚠ ECOS 실패: {e}')
            features=build_feature_table(info,financials,market,macro,notices,pd.Timestamp.today().normalize()); status.write('✓ Feature 계산 완료'); status.update(label='Feature Engine 완료',state='complete')
        st.session_state.update(features=features,company_info=info,notices=notices,market=market,macro=macro,corp=corp,company=company.strip(),valuation=None,ai_review=None,step=1); st.success('Step 1 완료. 이제 Step 2로 이동하세요.')
    except Exception as e:
        st.error(f'Feature Engine 실행 중 오류가 발생했습니다: {e}'); st.stop()
if move:
    st.session_state.step=2
if st.session_state.features is None:
    st.info('먼저 Step 1에서 기업을 분석하세요.'); st.stop()
features=st.session_state.features; fm=feature_dict(features)
if st.session_state.step==1:
    st.subheader(f"① Feature Engine — {st.session_state.company}")
    cs=st.columns(6)
    cs[0].metric('현재가',(fmt(fm.get('Current Price'),0)+'원') if fm.get('Current Price',{}).get('value') is not None else '-')
    cs[1].metric('PER',(fmt(fm.get('PER'))+'x') if fm.get('PER',{}).get('value') is not None else '-')
    cs[2].metric('PBR',(fmt(fm.get('PBR'))+'x') if fm.get('PBR',{}).get('value') is not None else '-')
    cs[3].metric('ROIC',(fmt(fm.get('ROIC'))+'%') if fm.get('ROIC',{}).get('value') is not None else '-')
    cs[4].metric('ROE',(fmt(fm.get('ROE'))+'%') if fm.get('ROE',{}).get('value') is not None else '-')
    cs[5].metric('FCF',fmt(fm.get('FCF'),0))
    st.dataframe(features,use_container_width=True,hide_index=True)
else:
    st.subheader(f"② Valuation Engine — {st.session_state.company}")
    st.caption('Python이 적정가치를 계산하고, GPT는 가치평가 가정을 검토합니다.')
    a=st.columns(4); wacc=a[0].number_input('WACC (%)',1.0,30.0,9.0,0.1); tg=a[1].number_input('Terminal Growth (%)',-5.0,8.0,2.5,0.1); fg=a[2].number_input('FCF Growth (%)',-20.0,50.0,5.0,0.5); fp=a[3].number_input('Fair PER (x)',1.0,80.0,12.0,0.5)
    b=st.columns(4); fpr=b[0].number_input('Fair PBR (x)',0.1,20.0,1.5,0.1); years=b[1].number_input('Forecast Years',3,10,5,1); dw=b[2].number_input('DCF Weight',0.0,1.0,0.50,0.05); pw=b[3].number_input('PER Weight',0.0,1.0,0.25,0.05); pbw=max(0.0,1.0-dw-pw); st.caption(f'PBR Weight = {pbw:.2f}')
    if st.button('📐 Valuation 계산',type='primary',use_container_width=True):
        ass=ValuationAssumptions(wacc_pct=wacc,terminal_growth_pct=tg,fcf_growth_pct=fg,forecast_years=int(years),fair_per=fp,fair_pbr=fpr,dcf_weight=dw,per_weight=pw,pbr_weight=pbw)
        st.session_state.valuation=ValuationEngine(ass).calculate(features); st.session_state.ai_review=None
    val=st.session_state.get('valuation')
    if val:
        c=st.columns(4); cur=val['inputs']['current_price']; fv=val['fair_value']; up=val['upside_pct']
        c[0].metric('현재가',f'{cur:,.0f}원' if cur else '-'); c[1].metric('종합 적정가',f'{fv:,.0f}원' if fv else '-'); c[2].metric('상승여력',f'{up:.1f}%' if up is not None else '-'); c[3].metric('ROIC-WACC',f"{val['roic_minus_wacc_pct']:.1f}%p" if val['roic_minus_wacc_pct'] is not None else '-')
        st.markdown('### Method별 가치'); st.dataframe(pd.DataFrame([['PER',val['per']['fair_price'],val['assumptions']['fair_per']],['PBR',val['pbr']['fair_price'],val['assumptions']['fair_pbr']],['DCF',val['dcf']['fair_price_proxy'],None]],columns=['방법','적정주가','배수']),use_container_width=True,hide_index=True)
        if val['dcf'].get('limitation'): st.warning(val['dcf']['limitation'])
        if openai_key:
            if st.button('🤖 GPT로 Valuation 가정 검토',use_container_width=True):
                try:
                    payload={k:v.get('value') for k,v in fm.items()}; st.session_state.ai_review=ValuationAI(openai_key,openai_model).review(st.session_state.company,payload,val)
                except Exception as e: st.error(f'GPT Valuation Review 실패: {e}')
        else: st.info('OPENAI_API_KEY가 없으면 GPT 검토를 사용할 수 없습니다.')
        r=st.session_state.get('ai_review')
        if r:
            st.info(r.get('overall_view','')); cc=st.columns(2)
            with cc[0]:
                st.markdown('**가정 검토**'); st.write('WACC:',r.get('wacc_assessment','')); st.write('Fair PER:',r.get('fair_per_assessment','')); st.write('Fair PBR:',r.get('fair_pbr_assessment','')); st.write('FCF Growth:',r.get('fcf_growth_assessment','')); st.write('Terminal Growth:',r.get('terminal_growth_assessment',''))
            with cc[1]:
                st.markdown('**강점 / 위험**'); [st.write('•',x) for x in r.get('valuation_strengths',[])]; [st.write('•',x) for x in r.get('valuation_risks',[])]
            st.caption(f"GPT confidence: {r.get('confidence','-')}")
