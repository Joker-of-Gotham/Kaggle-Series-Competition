def p(g):
    h,w=len(g),len(g[0]);o={(r,c)for r in range(h)for c in range(w)if g[r][c]};a=[]
    for n in range(min(h,w)+1):
        for R in range(h-n+1):
            for C in range(w-n+1):
                ar,ac=2*R+n-1,2*C+n-1;pc,pp=set(),set()
                for r,c in o:
                    if(r,c)not in pp:
                        sr,sc=ar-r,ac-c
                        if not(0<=sr<h and 0<=sc<w):break
                        pp.add((r,c));pp.add((sr,sc))
                        if not g[sr][sc]:pc.add((sr,sc))
                else:
                    if all((ar-r,ac-c)in o for r,c in pc)and pc:a.append({'c':pc,'a':(ar,ac)})
    if not a:return[r[:]for r in g]
    s_s=[]
    lp=lambda p,s:tuple(tuple(1 if(p[0]+dr,p[1]+dc)in s else 0 for dc in[-1,0,1])for dr in[-1,0,1])
    rp=lambda p:tuple(zip(*p[::-1]))
    for s in a:
        ts=o|s['c'];ar,ac=s['a'];cr,cc=ar/2.,ac/2.;sc=0
        for r,c in ts:
            pts=[(int(round(cr+c-cc)),int(round(cc-r+cr))),(ar-r,ac-c),(int(round(cr-c+cc)),int(round(cc+r-cr)))]
            if all(pt in ts for pt in pts):
                p0=lp((r,c),ts);p1=rp(p0);p2=rp(p1);p3=rp(p2)
                if p1==lp(pts[0],ts)and p2==lp(pts[1],ts)and p3==lp(pts[2],ts):sc+=1
        s['s']=sc;s_s.append(s)
    tkc=sorted(s_s,key=lambda x:x['s'],reverse=True)[:10]
    bs=tkc and min(tkc,key=lambda x:(len(x['c']),-x['s']))
    og=[r[:]for r in g]
    if bs:
        for r,c in bs['c']:og[r][c]=2
    return og