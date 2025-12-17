def p(g,e=enumerate):
 for _ in'1234':b=max(i for i,r in e(g)if 2in r);d=min(i for i,r in e(g)if 8in r);g=[*zip(*(g,(g[b+1:d]+g[:b+1]+g[d:]))[d>b])][::-1]
 return g