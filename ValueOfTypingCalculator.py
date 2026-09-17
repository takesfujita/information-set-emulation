"""Radius and squared-radius changes for compatible images of one locked target.
Uncapped squared radii are exact fiber minimax MSE under the theorem's conditions.
With an active cap, the result is only a capped reporting-score change.
"""
from dataclasses import dataclass
from math import isfinite

@dataclass
class RadiusState:
    label: str
    radius: float
    scale_cap: float | None=None

    def __post_init__(self):
        if not isfinite(self.radius) or self.radius<0:
            raise ValueError('Radius must be finite and nonnegative.')
        if self.scale_cap is not None and (not isfinite(self.scale_cap) or self.scale_cap<=0):
            raise ValueError('Cap must be finite and positive.')

    @property
    def truncated_radius(self):
        return self.radius if self.scale_cap is None else min(self.radius,self.scale_cap)

    @property
    def squared_radius_score(self):
        return self.truncated_radius**2

    @property
    def exact_minimax_mse(self):
        return self.radius**2

def value_of_typing(untyped,typed,risk_scale=False):
    if untyped.scale_cap!=typed.scale_cap:
        raise ValueError('Use the same reporting cap at both states.')
    if risk_scale:
        return untyped.squared_radius_score-typed.squared_radius_score
    return untyped.truncated_radius-typed.truncated_radius

def value_of_certification(current,refined,risk_scale=False):
    return value_of_typing(current,refined,risk_scale)

def value_per_cost(value,cost):
    if not isfinite(cost) or cost<=0:
        raise ValueError('Cost must be finite and positive.')
    return value/cost

if __name__=='__main__':
    k0,kt,kr=RadiusState('flat',.20),RadiusState('typed',.08),RadiusState('audit',.03)
    print('VoT radius',value_of_typing(k0,kt))
    print('VoT squared-radius score (uncapped: exact MSE reduction)',value_of_typing(k0,kt,True))
    print('VoC radius',value_of_certification(kt,kr))
    print('VoC squared-radius score (uncapped: exact MSE reduction)',value_of_certification(kt,kr,True))
