def p(g):
 t=lambda p,i:((lambda r,c:(c,-r) if i%4==1 else (-r,-c) if i%4==2 else (-c,r) if i%4==3 else (r,c))(*((p[0],-p[1]) if i>=4 else p)))
 R,C=len(g),len(g[0])
 V=[[0]*C for _ in range(R)];S=[]
 for rs in range(R):
  for cs in range(C):
   if g[rs][cs]!=0 and not V[rs][cs]:
    q=[(rs,cs)];V[rs][cs]=1;h=0;P=[];cols=set();a=None
    while h<len(q):
     r,c=q[h];h+=1;v=g[r][c];P.append((r,c));cols.add(v)
     if v==2: a=(r,c)
     for dr in(-1,0,1):
      for dc in(-1,0,1):
       if dr==0 and dc==0: continue
       nr,nc=r+dr,c+dc
       if 0<=nr<R and 0<=nc<C and g[nr][nc]!=0 and not V[nr][nc]:
        V[nr][nc]=1;q.append((nr,nc))
    if a:
     ar,ac=a;sk={(0,0)};pl={}
     for r,c in P:
      v=g[r][c];d=(r-ar,c-ac)
      if v==4: sk.add(d)
      if v in (1,3): pl[d]=v
     S.append({'a':a,'s':sk,'p':pl,'c':cols})
 src=None;T=[]
 for s in S:
  if src is None and (1 in s['c'] or 3 in s['c']): src=s
  if 2 in s['c'] and 4 in s['c']: T.append(s)
 if not src or not T: return g
 ng=[row[:] for row in g]
 for tgt in T:
  for i in range(8):
   if {t(p,i) for p in src['s']}==tgt['s']:
    for d,val in src['p'].items():
     rr,cc=t(d,i);ar,ac=tgt['a'];Rr,Cr=ar+rr,ac+cc
     if 0<=Rr<R and 0<=Cr<C: ng[Rr][Cr]=val
    break
 return ng
