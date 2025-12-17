def p(g):
 h,w=len(g),len(g[0]);o=[r[:]for r in g];H=[-1]+[r for r,R in enumerate(g)if len(set(R))<2 and R[0]]+[h];V=[-1]+[c for c in range(w)if len({r[c]for r in g})<2 and g[0][c]]+[w]
 if len(H)<3 or len(V)<3:return g
 mh,mw=len(H)-1,len(V)-1;M=[[g[H[i]+1][V[j]+1]for j in range(mw)]for i in range(mh)];C=[c for r in M for c in r];bg=max(set(C),key=C.count);S={c:[]for c in set(C)if c!=bg}
 for r in range(mh):
  for c in range(mw):
   if M[r][c]in S:S[M[r][c]].append(((r,c),sum(1 for dr in(-1,0,1)for dc in(-1,0,1)if(dr or dc)and 0<=r+dr<mh and 0<=c+dc<mw and M[r+dr][c+dc]!=bg)))
 K={c for c,v in S.items()if v and max(s for _,s in v)>min(s for _,s in v)}
 if not K:return g
 c=max(K,key=lambda k:max(s for _,s in S[k])-min(s for _,s in S[k]));(tr,tc),_=max(S[c],key=lambda x:x[1])
 T=[[M[tr+dr][tc+dc]if 0<=tr+dr<mh and 0<=tc+dc<mw else bg for dc in(-1,0,1)]for dr in(-1,0,1)];T[1][1]=bg
 for r,c in(i[0]for i in S[c]):
  for dr in(-1,0,1):
   for dc in(-1,0,1):
    if 0<=r+dr<mh and 0<=c+dc<mw and M[r+dr][c+dc]==bg:M[r+dr][c+dc]=T[dr+1][dc+1]
 for i in range(mh):
  for j in range(mw):
   rs,re,cs,ce=H[i]+1,H[i+1],V[j]+1,V[j+1]
   for y in range(rs,re):o[y][cs:ce]=[M[i][j]]*(ce-cs)
 return o