def p(m):
 b=range((f:=len(m)));c=m[2][2];r=eval(str((o:=[[x*(x!=c)for x in R]for R in m])));T=[*zip(*o)];u=lambda g:(g.index(d),f-g[::-1].index(d))
 for d in b:
  for l in b:
   if d in(g:=o[l]):a,e=u(g);r[l][a:e]=[d]*(e-a)
   if d in(g:=T[l]):
    for i in range(*u(g)):r[i][l]=d
 return[[(c,r[l][g])[m[l][g]!=c]for g in b]for l in b]