"""Recovered bounded CPU iteration policy; no original-module dependency.

Coarse automatic multiplier is four in the ordinary environment. The public route schedules only the levels emitted by the cloud; helper-only
research limits must not reject ordinary public iteration inputs.
"""
import math
import numpy as np

QUALITY_LEVELS={'FAST':(.03,.03),'BALANCED':(.01,.01),'HIGH':(.001,.01),'VERY_HIGH':(.001,.001)}

def quality_thresholds(quality):
    return QUALITY_LEVELS[quality]

def schedule(count,*,jacobi_iters=4,auto_iters=False,auto_iters_cap=640,
             fmg_iters=None,refine_iters_ratio=4,stop_moving=.01,stop_moving_fine=.01,sparse_refinement=False,public_scalar_policy=False,bone_count=20):
    if count<1 or (not public_scalar_policy and count>3):raise ValueError('iteration policy supports at most three solved levels')
    if not 0<=jacobi_iters<=(4294967295 if public_scalar_policy else 256) or not 0<=auto_iters_cap<=(4294967295 if public_scalar_policy else 640) or refine_iters_ratio<1:raise ValueError('iteration policy research bound exceeded')
    if fmg_iters is not None and not 0<=fmg_iters<=(4294967295 if public_scalar_policy else 256):raise ValueError('fine iteration research bound exceeded')
    if not all(math.isfinite(v) for v in (stop_moving,stop_moving_fine)):raise ValueError('nonfinite stopping policies remain unqualified')
    # Native 0x8743c uses saturating u32 multiplication, not wrapping or
    # an unbounded Python integer. Refined levels keep the original cap.
    coarse=min(4294967295,4*auto_iters_cap) if auto_iters else jacobi_iters
    fine=auto_iters_cap if auto_iters else (max(1,jacobi_iters//refine_iters_ratio) if fmg_iters is None else max(1,fmg_iters))
    rows=[]
    for index in range(count):
        stop=None
        if auto_iters:
            fraction=np.float32(np.clip(stop_moving if index==0 else stop_moving_fine,0,1))
            # Intermediate sparse rows carry 16 slots while stopping counts
            # correspond to the full bone width. Final output uses the requested
            # fraction directly. The legacy helper default is its 20-bone case;
            # actual binding always supplies the real solve width.
            if 0<index<count-1 and sparse_refinement:
                fraction=np.float32(fraction*np.float32(bone_count/16))
            # CPU primitive admits fractions through one. Higher intermediate
            # values require separate qualification rather than silent clamping.
            if fraction>1:
                if not public_scalar_policy:raise ValueError('expanded stopping fraction above one remains unqualified')
                # The changed/active ratio is at most one, so a larger native
                # threshold has the same stopping decision as one.
                fraction=np.float32(1.)
            stop=(32,float(np.float32(.005)),float(fraction))
        rows.append(dict(budget=coarse if index==0 else fine,stop=stop))
    return rows
