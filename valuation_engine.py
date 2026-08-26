from __future__ import annotations
from dataclasses import dataclass, asdict
from typing import Any, Optional
import math
import pandas as pd

def _num(v: Any) -> Optional[float]:
    try:
        if v is None: return None
        if isinstance(v, float) and math.isnan(v): return None
        return float(str(v).replace(',', '').replace('%', ''))
    except Exception: return None

def _map(features: pd.DataFrame):
    if features is None or features.empty: return {}
    return {str(r.get('feature')): r.get('value') for _, r in features.iterrows()}

@dataclass
class ValuationAssumptions:
    wacc_pct: float = 9.0
    terminal_growth_pct: float = 2.5
    fcf_growth_pct: float = 5.0
    forecast_years: int = 5
    fair_per: float = 12.0
    fair_pbr: float = 1.5
    dcf_weight: float = 0.50
    per_weight: float = 0.25
    pbr_weight: float = 0.25

class ValuationEngine:
    def __init__(self, assumptions=None): self.a = assumptions or ValuationAssumptions()
    def calculate(self, features: pd.DataFrame):
        fm=_map(features); cur=_num(fm.get('Current Price')); per=_num(fm.get('PER')); pbr=_num(fm.get('PBR')); roic=_num(fm.get('ROIC')); fcf=_num(fm.get('FCF')); shares=_num(fm.get('Listed Shares'))
        implied_eps = cur/per if cur is not None and per not in (None,0) else None
        per_price = implied_eps*self.a.fair_per if implied_eps is not None else None
        implied_bps = cur/pbr if cur is not None and pbr not in (None,0) else None
        pbr_price = implied_bps*self.a.fair_pbr if implied_bps is not None else None
        dcf_price = None; dcf_ev = None; limitation=None
        if fcf is not None and fcf>0 and shares and shares>0:
            w=self.a.wacc_pct/100; g=self.a.terminal_growth_pct/100; gr=self.a.fcf_growth_pct/100
            if w>g:
                base=fcf*1e6; pv=0.0; last=None
                for y in range(1,self.a.forecast_years+1):
                    last=base*((1+gr)**y); pv += last/((1+w)**y)
                term=last*(1+g)/(w-g); pv_term=term/((1+w)**self.a.forecast_years); dcf_ev=pv+pv_term; dcf_price=dcf_ev/shares; limitation='순차입금을 반영하지 않은 DCF Enterprise Value proxy입니다.'
        vals=[]; ws=[]
        for p,wt in [(dcf_price,self.a.dcf_weight),(per_price,self.a.per_weight),(pbr_price,self.a.pbr_weight)]:
            if p is not None and wt>0: vals.append(p); ws.append(wt)
        fair_value=sum(p*w for p,w in zip(vals,ws))/sum(ws) if vals and sum(ws)>0 else None
        upside=(fair_value/cur-1)*100 if fair_value is not None and cur not in (None,0) else None
        return {'inputs':{'current_price':cur,'current_per':per,'current_pbr':pbr,'roic':roic,'fcf_million':fcf,'listed_shares':shares},'per':{'fair_multiple':self.a.fair_per,'implied_eps':implied_eps,'fair_price':per_price},'pbr':{'fair_multiple':self.a.fair_pbr,'implied_bps':implied_bps,'fair_price':pbr_price},'dcf':{'fair_price_proxy':dcf_price,'enterprise_value_proxy':dcf_ev,'limitation':limitation},'fair_value':fair_value,'upside_pct':upside,'roic_minus_wacc_pct':(roic-self.a.wacc_pct) if roic is not None else None,'assumptions':asdict(self.a)}
