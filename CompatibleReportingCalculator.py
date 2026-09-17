"""Simultaneous compatible reporting for ONE locked scalar estimand.
Each row is a nominal functional under a retained role/world assignment.
The caller must justify world retention and simultaneous estimand-scale bounds.
The calculator supplies Bonferroni sampling critical values, not validation.
"""
from dataclasses import dataclass
from math import isfinite
from statistics import NormalDist
from typing import Iterable

@dataclass
class RoleEstimate:
    role: str
    estimate: float
    se: float
    bound_extract: float=0.
    bound_avail: float=0.
    bound_sep: float=0.
    bound_obs: float=0.
    bound_judge: float=0.
    bound_pos: float=0.

    @property
    def validation_bound(self):
        return sum((self.bound_extract,self.bound_avail,self.bound_sep,
                    self.bound_obs,self.bound_judge,self.bound_pos))

    def interval(self,z):
        bounds=[self.bound_extract,self.bound_avail,self.bound_sep,
                self.bound_obs,self.bound_judge,self.bound_pos]
        if not isfinite(self.estimate) or not isfinite(z) or z<=0:
            raise ValueError('Estimate must be finite and critical value positive.')
        if any(not isfinite(x) or x<0 for x in [self.se]+bounds):
            raise ValueError('Standard errors and component bounds must be finite and nonnegative.')
        width=z*self.se+self.validation_bound
        return self.estimate-width,self.estimate+width

def compatible_reporting_set(rows: Iterable[RoleEstimate], *, alpha=.05, alpha_validation=.01):
    rows=list(rows)
    if not rows or not 0<=alpha_validation<alpha<1:
        raise ValueError('Require nonempty assignments and 0 <= alpha_validation < alpha < 1.')
    alpha_sampling=alpha-alpha_validation
    z=NormalDist().inv_cdf(1-alpha_sampling/(2*len(rows)))
    intervals=[(r.role,*r.interval(z),r.validation_bound) for r in rows]
    union=[]
    for lo,hi in sorted((r[1],r[2]) for r in intervals):
        if union and lo<=union[-1][1]:
            union[-1][1]=max(union[-1][1],hi)
        else:
            union.append([lo,hi])
    lo,hi=union[0][0],union[-1][1]
    return dict(lower=lo,upper=hi,midpoint=(lo+hi)/2,reporting_radius=(hi-lo)/2,
                interval_union=union,role_intervals=intervals,critical_value=z,
                alpha_sampling=alpha_sampling,alpha_validation=alpha_validation,
                coverage_condition='Simultaneous nominal coverage and uniform validation bounds for every world of one locked target.')

if __name__=='__main__':
    example=[RoleEstimate('world_B_total_effect',.62,.05,bound_extract=.02),
             RoleEstimate('world_M_total_effect',.48,.06,bound_sep=.03)]
    print(compatible_reporting_set(example))
