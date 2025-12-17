def p(g,L=len,R=range):
 h,w=L(g),L(g[0])
 Z=[r[:] for r in g]
 for s in R(min([h,w]),1,-1):
  t=0
  for r in R(h):
   for c in R(w):
    X=g[r:r+s]
    X=[m[c:c+s][:] for m in X]
    if sum(X,[]).count(0)==s*s:
     t=1
     for i in R(r,r+s):
      for j in R(c,c+s):
       Z[i][j]=6
  if t:return Z
 return g