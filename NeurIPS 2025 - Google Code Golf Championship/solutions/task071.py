def p(g):
 h,w=len(g),len(g[0]);d={}
 for r,R in enumerate(g):
  for c,v in enumerate(R):
   if v:d.setdefault(v,[]).append((r,c))
 def s(p):
  if not p:return 0
  P=set(p);r,c=zip(*P);a,b,C,D=min(r),max(r),min(c),max(c);return len(P)==(b-a+1)*(D-C+1)and P=={(R,c)for R in range(a,b+1)for c in range(C,D+1)}
 cs=sorted(d)
 if len(cs)<2:return g
 o,s_c=cs[0],cs[1]
 for c in cs:
  if s(d.get(c,[])):o,s_c=c,next(x for x in cs if x!=c);break
 S,O,b=set(d.get(s_c,[])),set(d.get(o,[])),set(d.get(s_c,[]));m=w*h+1
 if S and O:
  l=sum(c for _,c in O)/len(O)<sum(c for _,c in S)/len(S)
  for a in range(w*2-1):
   ln=a/2;mir=lambda c:a-c;b_s={p for p in S if(p[1]>ln if l else p[1]<ln)};o_a={p for p in S if p[1]==ln};o_v=S-b_s-o_a;m_s={(r,int(mir(c)))for r,c in b_s};f=b_s|m_s|o_a
   if o_v<=m_s and all(0<=c<w for _,c in m_s)and f-S<=O and len(f)-len(S)<m:
    C=not f
    if f:
     q,v=[next(iter(f))],{next(iter(f))};H=0
     while H<len(q):
      r,c=q[H];H+=1
      for dr in[-1,0,1]:
       for dc in[-1,0,1]:
        P=(r+dr,c+dc)
        if(dr or dc)and P in f and P not in v:v.add(P);q.append(P)
     C=len(v)==len(f)
    if C:m=len(f)-len(S);b=f
 G=[[0]*w for _ in range(h)]
 for r,c in b:G[r][c]=s_c
 return G