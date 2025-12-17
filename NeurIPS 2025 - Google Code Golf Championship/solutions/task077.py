def p(g):
 if not g or not g[0]:return g
 h,w=len(g),len(g[0]);B=next((c for r in g for c in r if c not in[0,2]),-1)
 if B<0:return g
 A,N=2,4;T=[r[:]for r in g];S=[{(-1,-1),(-1,0),(-1,1)},{(1,-1),(1,0),(1,1)},{(-1,-1),(0,-1),(1,-1)},{(-1,1),(0,1),(1,1)}]
 for r in range(h):
  for c in range(w):
   if g[r][c]==B:
    P={(dr,dc)for dr in[-1,0,1]for dc in[-1,0,1]if(dr or dc)and 0<=r+dr<h and 0<=c+dc<w and g[r+dr][c+dc]==A}
    if len(P)>1 and not any(P.issubset(s)for s in S):T[r][c]=N
 V=[[0]*w for _ in range(h)]
 for r in range(h):
  for c in range(w):
   if T[r][c]in[A,N]and not V[r][c]:
    q=[(r,c)];V[r][c]=1
    for R,C in q:
     for dr in[-1,0,1]:
      for dc in[-1,0,1]:
       if(dr or dc)and 0<=R+dr<h and 0<=C+dc<w and not V[R+dr][C+dc]and T[R+dr][C+dc]in[A,N]:V[R+dr][C+dc]=1;q.append((R+dr,C+dc))
    x,X=min(k[0]for k in q),max(k[0]for k in q);y,Y=min(k[1]for k in q),max(k[1]for k in q)
    for i in range(x,X+1):
     for j in range(y,Y+1):
      if T[i][j]==B:T[i][j]=N
 return T