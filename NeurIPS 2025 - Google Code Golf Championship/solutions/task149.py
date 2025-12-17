def p(j):
 A=range
 c=[[0]*3 for _ in A(3)]
 for E in A(3):
  for k in A(3):
   W=0
   for l in A(3):
    for J in A(3):
     if j[E*4+l][k*4+J]==6:W+=1
   c[E][k]=1 if W>=2 else 0
 return c