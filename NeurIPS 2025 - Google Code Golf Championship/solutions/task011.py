def p(j):
 A=range
 for c in A(3):
  for E in A(3):
   if sum(j[c*4+W][E*4+l]==0 for W in A(3)for l in A(3))==5:
    k=[[5 if i%4==3 or j%4==3 else 0 for j in A(11)]for i in A(11)]
    for W in A(3):
     for l in A(3):
      J=j[c*4+W][E*4+l]
      if J:
       for a in A(3):
        for C in A(3):k[W*4+a][l*4+C]=J
    return k