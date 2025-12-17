R=range
L=len
def p(g):
 h,w=L(g),L(g[0])
 S=[-1,0,1]
 S=[[x,y] for x in S for y in S]
 for r in R(1,h-1):
  for c in R(1,w-1):
   if g[r][c]==0:
    M=[g[r+y][c+x] for y,x in S]
    if M.count(5)+M.count(2)>3:g[r][c]=2
 return g