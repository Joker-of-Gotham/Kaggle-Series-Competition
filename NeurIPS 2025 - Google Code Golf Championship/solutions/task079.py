def p(g):
 h,w=len(g),len(g[0]);D={}
 for C in sorted({c for r in g for c in r if c}):
  v,n=[[0]*w for _ in g],0
  for r in range(h):
   for c in range(w):
    if g[r][c]==C and not v[r][c]:
     n+=1;q=[(r,c)];v[r][c]=1
     while q:
      y,x=q.pop(0)
      for dy in[-1,0,1]:
       for dx in[-1,0,1]:
        if 0<=y+dy<h and 0<=x+dx<w and g[y+dy][x+dx]==C and not v[y+dy][x+dx]:v[y+dy][x+dx]=1;q.append((y+dy,x+dx))
  D[C]=n
 if not D:return g
 W=max(D,key=D.get)
 for r in range(h):
  for c in range(w):
   if g[r][c]==W:
    q,P,v=[(r,c)],[(r,c)],{(r,c)}
    while q:
     y,x=q.pop(0)
     for dy in[-1,0,1]:
      for dx in[-1,0,1]:
       ny,nx=y+dy,x+dx
       if 0<=ny<h and 0<=nx<w and g[ny][nx]==W and(ny,nx)not in v:v.add((ny,nx));P.append((ny,nx));q.append((ny,nx))
    my,My,mx,Mx=min(k[0]for k in P),max(k[0]for k in P),min(k[1]for k in P),max(k[1]for k in P)
    O=[[0]*(Mx-mx+1)for _ in range(My-my+1)]
    for y,x in P:O[y-my][x-mx]=W
    return O