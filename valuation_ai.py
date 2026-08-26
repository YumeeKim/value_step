from __future__ import annotations
import json, re
from typing import Any
from openai import OpenAI
class ValuationAI:
    def __init__(self, api_key:str, model:str='gpt-5.6'):
        if not api_key: raise ValueError('OPENAI_API_KEY가 없습니다.')
        self.client=OpenAI(api_key=api_key); self.model=model or 'gpt-5.6'
    @staticmethod
    def _clean(text:str)->dict[str,Any]:
        text=(text or '').strip()
        try: return json.loads(text)
        except json.JSONDecodeError:
            m=re.search(r'\{.*\}', text, re.S)
            if not m: raise RuntimeError('OpenAI 응답을 JSON으로 해석하지 못했습니다.')
            return json.loads(m.group(0))
    def review(self, company:str, features:dict[str,Any], valuation:dict[str,Any])->dict[str,Any]:
        system='''너는 한국 상장기업의 기업가치 평가를 검토하는 buy-side valuation analyst다. 입력 숫자를 바꾸거나 만들지 말고, Python 계산 결과를 유지한 채 WACC, Fair PER, Fair PBR, FCF 성장률, terminal growth 가정의 적절성만 검토하라. 공시 본문을 받지 않았다면 내용을 지어내지 마라. JSON만 출력하라.'''
        user={'task':'정량 valuation 결과와 가정 검토','company':company,'features':features,'valuation':valuation,'output_schema':{'overall_view':'string','wacc_assessment':'string','fair_per_assessment':'string','fair_pbr_assessment':'string','fcf_growth_assessment':'string','terminal_growth_assessment':'string','valuation_strengths':['string'],'valuation_risks':['string'],'recommended_assumption_changes':{'wacc_pct':'number|null','fair_per':'number|null','fair_pbr':'number|null','fcf_growth_pct':'number|null','terminal_growth_pct':'number|null'},'confidence':'high|medium|low'}}
        r=self.client.responses.create(model=self.model,input=[{'role':'system','content':system},{'role':'user','content':json.dumps(user,ensure_ascii=False,default=str)}])
        return self._clean(getattr(r,'output_text',''))
