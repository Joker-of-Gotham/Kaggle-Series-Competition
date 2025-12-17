def p(g):
 h,w=len(g),len(g[0]);o=[r[:]for r in g];D=((0,1),(0,-1),(1,0),(-1,0));v=set();B=[]
 for i in range(h):
  for j in range(w):
   if(i,j)in v:continue
   c=g[i][j];s={(i,j)};v.add((i,j));q=[(i,j)]
   while q:
    x,y=q.pop()
    for dx,dy in D:
     nx,ny=x+dx,y+dy
     if 0<=nx<h and 0<=ny<w and(nx,ny)not in v and g[nx][ny]==c:v.add((nx,ny));q.append((nx,ny));s.add((nx,ny))
   B.append((c,s))
 bg=max(B,key=lambda b:len(b[1]))[0];L=[b for b in B if len(b[1])>1 and b[0]!=bg];O=[b for b in B if len(b[1])==1];T=[]
 for b in O:
  p0=next(iter(b[1]));r,c=p0
  for Lb in L:
   nb=[(r+dx,c+dy)for dx,dy in D if 0<=r+dx<h and 0<=c+dy<w]
   if nb and all(n in Lb[1]for n in nb):T.append((p0,set(Lb[1])));break
 A=set().union(*(s for _,s in T))if T else set();tp={p for p,_ in T}
 brush={(i,j)for i in range(h)for j in range(w)if g[i][j]!=bg and(i,j)not in A|tp}
 if not brush or not T:return g
 def comps(s):
  R=[];r=set(s)
  while r:
   st=r.pop();q=[st];c={st}
   while q:
    x,y=q.pop()
    for dx,dy in D:
     nb=(x+dx,y+dy)
     if nb in r:r.remove(nb);c.add(nb);q.append(nb)
   R.append(c)
  return R
 bc=max(comps(brush),key=len);mr=sum(x for x,y in bc)/len(bc);mc=sum(y for x,y in bc)/len(bc);mx=-1;cen=None
 for x,y in bc:
  cc=sum(((x+dx,y+dy)in bc)for dx,dy in D)
  if cc>mx:mx,cen=cc,(x,y)
  elif cc==mx and (x-mr)**2+(y-mc)**2<(cen[0]-mr)**2+(cen[1]-mc)**2:cen=(x,y)
 bmap={(x-cen[0],y-cen[1]):g[x][y]for x,y in bc};placed=[]
 for p,S in T:
  tr,tc=p;mp={}
  for(dr,dc),col in bmap.items():
   nr,nc=tr+dr,tc+dc
   if 0<=nr<h and 0<=nc<w:mp[(nr,nc)]=col
  if not mp:continue
  for(nr,nc),col in mp.items():o[nr][nc]=col
  placed.append((p,S))
 f=[r[:]for r in o]
 for p,S in placed:
  tr,tc=p;ic=(lambda a,b:0<=a<h and 0<=b<w)if not S else(lambda a,b:(a,b)in S)
  if tr>1 and o[tr-1][tc]==o[tr-2][tc]:
   c=o[tr-1][tc]
   for rr in range(tr-1,-1,-1):
    if ic(rr,tc):f[rr][tc]=c
    else:break
  if tr+2<h and o[tr+1][tc]==o[tr+2][tc]:
   c=o[tr+1][tc]
   for rr in range(tr+1,h):
    if ic(rr,tc):f[rr][tc]=c
    else:break
  if tc>1 and o[tr][tc-1]==o[tr][tc-2]:
   c=o[tr][tc-1]
   for cc in range(tc-1,-1,-1):
    if ic(tr,cc):f[tr][cc]=c
    else:break
  if tc+2<w and o[tr][tc+1]==o[tr][tc+2]:
   c=o[tr][tc+1]
   for cc in range(tc+1,w):
    if ic(tr,cc):f[tr][cc]=c
    else:break
 for x,y in bc:f[x][y]=bg
 for i in range(h):
  for j in range(w):
   if g[i][j]==bg:f[i][j]=bg
 return f
