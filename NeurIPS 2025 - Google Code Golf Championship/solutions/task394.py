L=len
R=range
E=enumerate
def p(g):
 Z=[r[:] for r in g]
 for i in range(4):
  g=list(map(list,zip(*g[::-1])))
  Z=list(map(list,zip(*Z[::-1])))
  h,w=L(g),L(g[0])
  if sum(Z,[]).count(0)>0:
   for i in R(-w,w):
    M=sum(g,[])
    C=(w*h)//2+i
    A=M[:C];B=M[C:]
    N=min([L(A),L(B)])
    T=sum([1 if A[j]==B[j] else 0 for j in R(N)])
    if T+max([A.count(0),B.count(0)])==N:
     for j in R(N):
      if A[j]==0 or B[j]==0:
       A[j]=B[j]=max([A[j],B[j]])
     M=A+B
     Z=[M[x*w:(x+1)*w] for x in R(h)]
 P=[[x,y] for y,r in E(g) for x,c in E(r) if c==0]
 f=sum(P,[]);x=f[::2];y=f[1::2]
 Z=Z[min(y):max(y)+1]
 Z=[r[min(x):max(x)+1][:] for r in Z]
 return Z