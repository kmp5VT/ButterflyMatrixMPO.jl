M = randn(8,8)

BMPO = RandomButteflyMatrixMPO(M);
l = log2(8)
@test length(BMPO) == l + 1 # l = 3 because 8 = 2^l length is l + 1
@test length(inds(BMPO)) == 2 * (l + 1)
@test prod(dim.(inds(BMPO))) == 8 * 8 

@test BMPO.hyperindsi[1] == inds(BMPO)[2:4]
@test BMPO.hyperindsj[1][] == inds(BMPO)[end]

Bmp = ButterFlyMatrixMPO.reconstruct_butterfly(BMPO)

@test commoninds(inds(BMPO), inds(Bmp),) == inds(BMPO)