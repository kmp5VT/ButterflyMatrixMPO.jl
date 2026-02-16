#### The idea is maybe we can partition the probelm into sub problems which can be solved cheaply then 
## Apply rand SVD on each subproblem since it's over determined 
lambda = itensor(scalings, r2, i0,j0)
ii1 = 1
ii0 = 1
a = itensor(array(S1MPO)[:,:,ii1,ii0], i2, r1)
b = itensor(array(S2MPO)[:,ii1,:,ii0,:], r1, r2, j0)
c = itensor(array(S3MPO)[:,:,:,ii0,:], r2, r3, j1, j0)
d = itensor(array(S4MPO)[:,:,:,:], r3, j2, j1, j0)

ab = a * b
lam = itensor(array(lambda)[:,ii0,:], r2, j0)
lc = ITensorCPD.had_contract(c, lam, r2,j0)
cd = ITensorCPD.had_contract(lc, d, j0,j1)
abcd = ITensorCPD.had_contract(ab,cd, j0)



## Sanity check should be 0
norm(abcd.tensor- reshape(M, (2,2,2,2,2,2))[:,ii1,ii0,:,:,:])



## ALS for slices of S1. This allows you to solve each block in NlogN time.
## You have N / logN blocks but each block is independent so its parallelizable. 
## Question is can you use properties of waves to smoothly transform one answer to the next?
using ITensorCPD: had_contract
krp = had_contract(b, cd, j0)
#array(a * krp) - reshape(M, (2,2,2,2,2,2))[:,1,1,:,:,:] # sanity
MtKRP = itensor(reshape(M, (2,2,2,2,2,2))[:,ii1,ii0,:,:,:], i2, j2,j1,j0) * dag(krp)
gram = krp * dag(prime(krp, tags="r1"))

# |M (2 x 8) P(8, s) - A (2, r) X (r, 8) P(8,s) |
## Sanity check, should be zero.
sol = array(gram)  \ array(MtKRP)'
norm(sol' - array(a))

## Here I want to see how j's combine to make the update above
## ordered r1, r2, j0
b_j0_0 = itensor(array(b)[:,:,1], r1, r2)
b_j0_1 = itensor(array(b)[:,:,2], r1, r2)

## ordered r2, r3, j1, j0
λ = 0.01
c_j1_0_j0_0 = itensor(array(c)[:,:,1,1] + λ .* randn(4,2), r2, r3)
c_j1_0_j0_1 = itensor(array(c)[:,:,1,2] + λ .* randn(4,2), r2, r3)
c_j1_1_j0_0 = itensor(array(c)[:,:,2,1]+ λ .* randn(4,2), r2, r3)
c_j1_1_j0_1 = itensor(array(c)[:,:,2,2]+ λ .* randn(4,2), r2, r3)

d_j1_0_j0_0 = itensor(array(d)[:,:,1,1], r3, j2, )
d_j1_0_j0_1 = itensor(array(d)[:,:,1,2], r3, j2, )
d_j1_1_j0_0 = itensor(array(d)[:,:,2,1], r3, j2, )
d_j1_1_j0_1 = itensor(array(d)[:,:,2,2], r3, j2, )

lam_j0_0 = itensor(array(lam)[:,1], r2)
lam_j0_1 = itensor(array(lam)[:,2], r2)

krp00 = b_j0_0 * had_contract(c_j1_0_j0_0, lam_j0_0, r2) * d_j1_0_j0_0
array(krp00) - array(krp)[:,:,1,1]

## ordered j1, j0
krp10 = b_j0_0 * had_contract(c_j1_1_j0_0, lam_j0_0, r2) * d_j1_1_j0_0
array(krp10) - array(krp)[:,:,2,1]

krp01 = b_j0_1 * had_contract(c_j1_0_j0_1, lam_j0_1, r2) * d_j1_0_j0_1
array(krp01) - array(krp)[:,:,1,2]

krp11 = b_j0_1 * had_contract(c_j1_1_j0_1, lam_j0_1, r2) * d_j1_1_j0_1
array(krp11) - array(krp)[:,:,2,2]

norm(krp00)
norm(krp10)
norm(krp01)
norm(krp11)


array(krp)[:,:,2,1] .= krp10
array(krp)[:,:,1,2] .= krp01
array(krp)[:,:,2,2] .= krp11
array(krp)[:,:,1,1] .= krp00

krp.tensor
MtKRP_appx = itensor(reshape(M, (2,2,2,2,2,2))[:,ii1,ii0,:,:,:], i2, j2,j1,j0) * dag(krp)
gram_appx = krp * dag(prime(krp, tags="r1"))

MtKRP_appx1 = itensor(reshape(M, (2,2,2,2,2,2))[:,ii1,ii0,:,1,1], i2, j2) * dag(krp00)
MtKRP_appx2 = itensor(reshape(M, (2,2,2,2,2,2))[:,ii1,ii0,:,2,2], i2, j2) * dag(krp11)
MtKRP_appx3 = itensor(reshape(M, (2,2,2,2,2,2))[:,ii1,ii0,:,1,2], i2, j2) * dag(krp01)
MtKRP_appx4 = itensor(reshape(M, (2,2,2,2,2,2))[:,ii1,ii0,:,2,1], i2, j2) * dag(krp10)

norm(MtKRP_appx - (MtKRP_appx1 + MtKRP_appx2 + MtKRP_appx3 )) / norm(MtKRP_appx)
# gram_appx = krp00 * dag(prime(krp00, tags="r1"))

norm(MtKRP - MtKRP_appx) / norm(MtKRP)
norm(gram_appx - gram) / norm(gram)

## Sanity check, should be zero.
norm((array(gram_appx)  \ array(MtKRP_appx)')' - array(a))/ norm(a)

(reshape(array(krp), (2,8))' \ reshape(reshape(M, (2,2,2,2,2,2))[:,ii1,ii0,:,:,:], (2,8))')' - array(a)


##### Mode 2 ALS solver !
ii0 = 1
p = itensor(array(S1MPO)[:,:,:, ii0, ], i2, r1, i1)
q = itensor(array(S2MPO)[:,:,:, ii0, :], r1, i1, r2, j0)
r = itensor(array(S3MPO)[:,:,:, ii0, :], r2, r3, j1, j0)
s = itensor(array(S4MPO)[:,:,:,:], r3, j2, j1, j0)

lam = itensor(array(lambda)[:,ii0,:], r2, j0)
rls = ITensorCPD.had_contract(ITensorCPD.had_contract(r, lam, r2, j0), s, j1, j0)

array(q)[:,1,:,1]
# i0 x i1 x j2 x j1
ii1 = 1
jj0 = 1
p1 = itensor(array(p)[:,:,ii1], i2, r1)
rls1 = itensor(array(rls)[:,:,:,jj0], r2, j2, j1)
mtkrp = dag(p1) * itensor(reshape(M, (2,2,2, 2,2,2))[:,ii1,ii0,:,:,jj0], i2, j2, j1) * dag(rls1)

lhs = p1 * dag(prime(p1, tags="r1"))
rhs = rls1 * dag(prime(rls1, tags="r2"))

ans = array(lhs) \ diagm(ones(2)) * array(mtkrp) * (array(rhs) \ diagm(ones(4)))
ans - array(q)[:,ii1,:,jj0]

## only the left hand side of the problem has i1
## only right hand side has j2 and j1 so LHS solves for (i1 x r1)
