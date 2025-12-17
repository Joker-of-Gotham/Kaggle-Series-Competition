from collections import*
def p(g):
 h,w=len(g),len(g[0]);v,S=set(),[]
 for C in range(w):
  for R in range(h):
   if g[R][C] and (R,C) not in v:
    q,P,sc=deque([(R,C)]),{},0;v.add((R,C))
    while q:
     r,c=q.popleft();V=g[r][c];P[(r,c)]=V
     if V not in[0,5]:sc=V
     for dr,dc in[(0,1),(0,-1),(1,0),(-1,0)]:
      nr,nc=r+dr,c+dc
      if 0<=nr<h and 0<=nc<w and g[nr][nc]and(nr,nc)not in v:v.add((nr,nc));q.append((nr,nc))
    A=sorted([k for k,val in P.items()if val==5],key=lambda x:(x[1],x[0]));S.append([P,A,sc])
 C,T={},None
 for i,s in enumerate(S):
  P,A,sc=s;dr,dc=0,0
  if i==0:
   if A:T=A[-1]
  elif A and T:
   dr=T[0]-A[0][0];dc=T[1]+1-A[0][1]
   if A:T=(A[-1][0]+dr,A[-1][1]+dc)
  else:continue
  for(r,c),V in P.items():C[(r+dr,c+dc)]=sc if V==5 else V
 if not C:return[[0]*w for _ in range(h)]
 O=[[0]*(max(c for r,c in C)+1)for _ in range(h)]
 for(r,c),V in C.items():
  if 0<=r<h:O[r][c]=V
 return O