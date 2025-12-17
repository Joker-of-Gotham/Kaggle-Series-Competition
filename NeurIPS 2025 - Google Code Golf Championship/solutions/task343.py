def p(g,L=len,R=range):
 h,w=L(g),L(g[0])
 C=0
 for c in R(w):
  if g[-1][c]==0:
   C=c;break
 #pattern start varies must compare
 C=[r[:C]+r[2:C]*20 for r in g]
 C=[r[:w] for r in C]
 return C