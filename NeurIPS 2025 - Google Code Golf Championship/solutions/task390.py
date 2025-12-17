def X(g):return list(zip(*g[::-1]))
def p(g,L=len,R=range):
 v=1
 for r in g:
  if r.count(2)>4:v=0
 if v:P=[[0,6],[1,5]]
 else:P=[[1,7],[2,6]]
 if v:
  g=X(g)
  for a,b in P:
   g[a]=g[b]
   g[-(a+2)]=g[-(b+2)]
   g[b]=g[-1]
   g[-(b+2)]=g[-1]
 else:
  for a,b in P:
   g[a]=g[b]
   g[-a]=g[-b]
   g[b]=g[0]
   g[-b]=g[0]
 if v:g=X(X(X(g)))
 return g