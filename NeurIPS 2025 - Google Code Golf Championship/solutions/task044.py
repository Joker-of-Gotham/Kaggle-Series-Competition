from collections import*
def p(g):
 h,w=len(g),len(g[0]);o=[r[:]for r in g];D,V=defaultdict(list),set()
 for r in range(h):
  for c in range(w):
   if(r,c)not in V:
    k=g[r][c];S,Q={(r,c)},deque([(r,c)]);V.add((r,c))
    while Q:
     y,x=Q.popleft()
     for dy,dx in[(0,1),(-1,0),(0,-1),(1,0)]:
      Y,X=y+dy,x+dx
      if 0<=Y<h and 0<=X<w and g[Y][X]==k and(Y,X)not in V:V.add((Y,X));Q.append((Y,X));S.add((Y,X))
    D[k].append(S)
 O,Q=set(),deque([(r,c)for r in range(h)for c in range(w)if g[r][c]!=5 and(r*c==0 or r==h-1 or c==w-1)]);O.update(Q)
 while Q:
  r,c=Q.popleft()
  for dy,dx in[(0,1),(-1,0),(0,-1),(1,0)]:
   Y,X=r+dy,c+dx
   if 0<=Y<h and 0<=X<w and g[Y][X]!=5 and(Y,X)not in O:O.add((Y,X));Q.append((Y,X))
 N=lambda s:frozenset(sorted((y-min(z[0]for z in s),x-min(z[1]for z in s))for y,x in s))
 M=[(s[0],N(s[0]),k)for k,s in D.items()if k not in(0,5)and len(s)<2 and s[0].issubset(O)]
 W=[(s,N(s))for s in D[0]if not s.issubset(O)]
 for sc,sg,sv in M:
  for i,(wc,wg)in enumerate(W):
   if sg==wg:
    for r,c in sc:o[r][c]=0
    for r,c in wc:o[r][c]=sv
    W.pop(i);break
 return o