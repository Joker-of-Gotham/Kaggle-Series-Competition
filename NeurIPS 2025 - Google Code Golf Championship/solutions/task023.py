def p(g):
 h,w,m=len(g),len(g[0]),{}
 def f(t):
  if t in m:return m[t]
  i=''.join(t).find('5')
  if i<0:return 0,[]
  R,C=i//w,i%w;S=[]
  for s,c,th,tw,P in[(4,8,2,2,[(0,0),(1,0),(0,1),(1,1)]),(3,2,1,3,[(0,0),(0,1),(0,2)]),(3,2,3,1,[(0,0),(1,0),(2,0)])]:
   for y in range(th):
    for x in range(tw):
     r,k=R-y,C-x
     if 0<=r<=h-th and 0<=k<=w-tw:
      Q=[(r+Y,k+X)for Y,X in P]
      if all(t[Y][X]=='5'for Y,X in Q):
       L=[list(rw)for rw in t]
       for Y,X in Q:L[Y][X]='0'
       ss,sl=f(tuple(''.join(row)for row in L))
       if ss!=-1:S.append((s+ss,[(c,Q)]+sl))
  if not S:m[t]=(-1,[]);return m[t]
  m[t]=max(S);return m[t]
 _,sol=f(tuple(''.join(map(str,r))for r in g))
 o=[[c if c!=5 else 0 for c in r]for r in g]
 if sol:
  for c,Q in sol:
   for R,C in Q:o[R][C]=c
 return o