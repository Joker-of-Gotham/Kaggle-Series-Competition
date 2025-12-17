def X(g):return list(zip(*g[::-1]))
def p(g,L=len,R=range):
 t=[r[:] for r in g]
 for _ in R(4):
  g=X(g);t=[list(r) for r in X(t)]
  h,w=L(g),L(g[0])
  for r in R(h-1):
   for c in R(w-2):
    m=[i for i in R(w) if t[r][i]>0]
    if L(m)>0:
     if g[r][c]==1 and g[r][c+2]==1 and L(m)>3:t[r][c+1]=2
     if g[r][c]==1 and g[r+1][c+1]==1 and L(m)>3:t[r][c+1]=2
     if min(m)<c+1<max(m) and L(m)>3 and g[r][c+1]==0:t[r][c+1]=2
 h,w=L(g),L(g[0])
 for r in R(h):
  for c in R(w):
   if g[r][c]>0:t[r][c]=1
 return t