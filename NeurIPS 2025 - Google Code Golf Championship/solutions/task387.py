from itertools import *
L=len
R=range
def p(g):
 Z=[r[:] for r in g]
 C=sorted(set(sum(g,[])))[1:]
 d={C[0]:C[1],C[1]:C[0]}
 for i in range(4):
  g=list(map(list,zip(*g[::-1])))
  Z=list(map(list,zip(*Z[::-1])))
  h,w=L(g),L(g[0])
  for r in R(h):
   P=0
   for c in R(w):
    if g[r][c] in d:
     for x,y in list(product([0,1,-1],repeat=2)):
      if 0<r+y<h and 0<=c+x<w and not x==y==0:Z[r+y][c+x]=d[g[r][c]]
    if g[r][c] in d and P>0:P=0
    if g[r][c] in d and P==0 and g[r].index(g[r][c])<g[r].index(d[g[r][c]]):P=c+2
    if P>0 and P==c:Z[r][c]=5;P+=2
 return Z