def p(g):
 h,w=len(g),len(g[0]);o=[r[:]for r in g];P,s,d=set(),[],set()
 def S(r,c):
  n=1
  while all(r+n<h and c+i<w and g[r+n][c+i]==3 and r+i<h and c+n<w and g[r+i][c+n]==3 for i in range(n+1)):n+=1
  return n
 def D(r,c,n):
  for i in range(n):
   for j in range(n):
    if 0<=r+i<h and 0<=c+j<w:o[r+i][c+j]=8
 for r in range(h):
  for c in range(w):
   if g[r][c]==3 and(r,c)not in P and(r<1 or g[r-1][c]!=3)and(c<1 or g[r][c-1]!=3):
    n=S(r,c);s.append(((r,c),n));[P.add((r+i,c+j))for i in range(n)for j in range(n)]
 for i in range(len(s)):
  for j in range(i+1,len(s)):
   (r,c),n=s[i];(R,C),N=s[j]
   if n==N and i not in d and j not in d:
    A,B=0,0
    if R==r+n and C==c+n:A,B=(r-n,c+2*n),(R+n,C-2*n)
    elif R==r+n and C==c-n:A,B=(R-2*n,C-n),(r+2*n,c+n)
    elif r==R+n and c==C-n:A,B=(r-n,c-n),(R+n,C+n)
    if A:D(A[0],A[1],n);D(B[0],B[1],n);d.add(i);d.add(j)
 return o