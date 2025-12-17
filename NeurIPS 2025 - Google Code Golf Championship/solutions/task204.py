R=range
L=len
def p(g):
 h,w,C=L(g),L(g[0]),7
 for r in R(1,h-1):
  for c in R(1,w-1):
   if sum([g[r-1][c],g[r+1][c],g[r][c-1],g[r][c+1]])>1 and g[r][c]==0:
    g[r][c]=C
 for r in R(1,h-1):
  for c in R(1,w-1):
   if [g[r-1][c],g[r+1][c],g[r][c-1],g[r][c+1]].count(0)>0 and g[r][c]==7:
    g[r][c]=0
 return g