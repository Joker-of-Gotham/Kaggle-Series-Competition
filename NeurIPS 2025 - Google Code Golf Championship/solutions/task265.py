def p(g,L=len,R=range):
 h,w=L(g),L(g[0])
 for r in R(h-1):
  for c in R(w-1):
   C=g[r][c:c+2]+g[r+1][c:c+2]
   if C.count(0)==4:
    g[r][c]=2
    g[r][c+1]=2
    g[r+1][c]=2
    g[r+1][c+1]=2
   if C.count(0)==2 and C.count(2)==2:
    g[r][c]=2
    g[r][c+1]=2
    g[r+1][c]=2
    g[r+1][c+1]=2
 return g