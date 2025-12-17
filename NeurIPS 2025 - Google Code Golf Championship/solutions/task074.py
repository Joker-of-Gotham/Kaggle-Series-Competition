def p(g):
 n=len(g);i=2;o=[[0]*n for _ in range(n)];C,I,N=(n+1)/2,(i+n-1)/2,{0,9}
 for r in range(n):
  for c in range(n):
   v=g[r][c]
   if v in N:
    O=r<i or c<i;E=C if O else I;R=lambda y,x:(y<i or x<i)if O else(y>=i and x>=i);ro,co=r-E,c-E;L=[(round(E-ro),c),(r,round(E-co)),(round(E+co),round(E+ro)),(round(E-co),round(E-ro))]
    v=next((g[int(y)][int(x)]for y,x in L if 0<=y<n and 0<=x<n and R(y,x)and g[int(y)][int(x)]not in N),0)
   o[r][c]=v
 for r in range(n):
  for c in range(r,n):
   a,b=o[r][c],o[c][r]
   if a and not b:o[c][r]=a
   elif b and not a:o[r][c]=b
 return o