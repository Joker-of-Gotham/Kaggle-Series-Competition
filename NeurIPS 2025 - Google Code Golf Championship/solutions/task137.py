R=range
L=len
def p(a):
    n,m=L(a),L(a[0])
    o=[[0]*m for _ in R(n)]
    mp={}
    for i in R(n):
        for j in R(m):
            v=a[i][j]
            if v: mp.setdefault(v,[]).append((i,j))
    for v,ps in mp.items():
        ps.sort(); r0,c0=ps[len(ps)//2]
        k=0
        if len(ps)>1:
            d1=abs(ps[1][0]-ps[0][0]); d2=abs(ps[1][1]-ps[0][1])
            k=d1 or d2
        if k<=0:
            for i in R(L(ps)):
                for j in R(i+1,L(ps)):
                    d=max(abs(ps[i][0]-ps[j][0]),abs(ps[i][1]-ps[j][1]))
                    if d: k=d; break
                if k: break
        if k<=0: k=1
        t=max(r0,n-1-r0,c0,m-1-c0)//k
        for s in R(1,t+1):
            r1,r2=r0-s*k,r0+s*k; c1,c2=c0-s*k,c0+s*k
            if 0<=r1<n:
                for c in R(max(0,c1),min(m-1,c2)+1): o[r1][c]=v
            if 0<=r2<n:
                for c in R(max(0,c1),min(m-1,c2)+1): o[r2][c]=v
            lo,hi=max(0,r1),min(n-1,r2)
            if 0<=c1<m:
                for r in R(lo,hi+1): o[r][c1]=v
            if 0<=c2<m:
                for r in R(lo,hi+1): o[r][c2]=v
        o[r0][c0]=v
    return o