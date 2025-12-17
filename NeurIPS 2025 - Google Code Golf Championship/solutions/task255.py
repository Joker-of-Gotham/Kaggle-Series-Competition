R=range
L=len
P=lambda m:list(map(list,zip(*m[::-1])))
def p(g):
 C=max(sum(g,[]))
 for i in R(4):
  g=P(g)
  h,w=L(g),L(g[0])
  for r in R(h-1):
   for c in R(w-1):
    if g[r][c]==C:
     for y,x in [[0,0],[0,1],[1,0],[1,1]]:
      if g[r+y][c+x]==0:g[r+y][c+x]=10
 for i in R(4):
  g=P(g)   
  for r in R(h):
   M=sorted(set(g[r]))
   if M==[0] or M==[0,3]:
    g[r]=[3]*L(g[r])
 for i in R(4):
  g=P(g)    
  for r in R(h):
   if C not in g[r][:10] and 10 not in g[r][:10]:
    for c in R(w):
     if g[r][c]<1:g[r][c]=3
     else:break
 g=[[0 if c>9 else c for c in r] for r in g]
 return g