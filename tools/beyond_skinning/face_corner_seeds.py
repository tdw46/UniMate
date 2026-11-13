"""Bounded paired-planar eyelid corner seed softening, default tuning.

Recovered opening plane (0x1319e8), nearest-face endpoint radius (0x90740),
and pre-diffusion strength stage (0x9ad08). Inputs are caller geometry only.
"""
import numpy as np
F=np.float32

def dot(a,b):return F(F(a[0]*b[0]+a[1]*b[1])+a[2]*b[2])
def mean(points):
    out=np.zeros(3,'f4')
    for p in points:out+=p/F(len(points))
    return out

def records(face,heads,scale):
    t=np.asarray(heads,'f4');result=[];roles=face['roles']
    face_ids=[i for i,r,s in roles if r.startswith('eyelid_') or r in ('brow','cheek')]
    for side in ('l','r'):
        corners=[i for i,r,s in roles if r=='eyelid_corner' and s==side]
        if not corners:continue
        if len(corners)<2:continue
        corners.sort(key=lambda i:abs(float(t[i,0])))
        upper=[i for i,r,s in roles if r=='eyelid_top' and s==side]
        lower=[i for i,r,s in roles if r=='eyelid_bot' and s==side]
        if not upper or not lower:continue
        best=F(-1);pair=None
        for j,i in enumerate(corners):
            for k in corners[j+1:]:
                d=t[i]-t[k];distance=F(np.sqrt(dot(d,d)))
                if distance>best:best=distance;pair=(i,k)
        if best<=0:continue
        a,b=t[list(pair)];d=b-a;length=best
        axis=d/length;delta=mean(t[upper])-mean(t[lower])
        normal=np.array([F(axis[1]*F(0)-axis[2]),F(axis[2]*F(0)+axis[0]*F(-0.)),F(axis[0]+axis[1]*F(-0.))],'f4')
        length=F(np.sqrt(F(normal[2]*normal[2]+F(normal[0]*normal[0]+normal[1]*normal[1]))))
        if length<F(.1):
            normal=delta-axis*dot(delta,axis);length=F(np.sqrt(dot(normal,normal)))
        if length<=F(1e-9):continue
        normal/=length
        if dot(delta,normal)<0:normal=-normal
        origin=(a+b)*F(.5)
        for i in corners:
            distance=min(F(np.sqrt(dot(t[i]-t[j],t[i]-t[j]))) for j in face_ids if j!=i)
            radius=F(distance*F(.6))
            result.append(np.concatenate([t[i]*F(scale),[F(radius*F(scale))],origin*F(scale),normal,[F(.35)]]).astype('f4'))
    return np.asarray(result,'f4').reshape(-1,11)

def strengths(points,labels,soft_records):
    p=np.asarray(points,'f4');valid=np.asarray(labels)!=0xffffffff;out=np.ones(len(p),'f4')
    for rec in soft_records:
        center,radius,origin,normal,minimum=rec[:3],rec[3],rec[4:7],rec[7:10],rec[10]
        if radius<=0:continue
        d=p-center;inside=valid & ((d[:,0]*d[:,0]+d[:,1]*d[:,1])+d[:,2]*d[:,2]<=F(radius*radius))
        d=p[inside]-origin
        projection=(d[:,0]*normal[0]+d[:,1]*normal[1])+d[:,2]*normal[2]
        fraction=np.minimum(F(1),np.abs(projection)/radius)
        out[inside]=np.minimum(out[inside],minimum+F(F(1)-minimum)*fraction)
    return out
