using ButterFlyMatrixMPO
# using Main.ButterFlyMatrixMPO:  BFMatrixMPO, IndexSetFromLevels
using Pkg
using ButterFlyMatrixMPO: BFMatrixMPO, IndexSetFromLevels, reconstruct_butterfly
## Convert the MPO test to fit in the ButterFlyMatrixMPO structure
a,b,c,d = IndexSetFromLevels(8, 3)
i3, i2, i1, i0, j3, j2, j1, j0 = [a..., b[end:-1:1]...]

S1MPO = itensor(array(S1MPO), i3, r1, i2, i1, i0, j0)
S2MPO = itensor(array(S2MPO), r1, i2, r2, i1, i0, j1, j0)
S3MPO = itensor(array(S3MPO), r2, r3, j2, i1, j1, i0, j0)
S4MPO = itensor(array(S4MPO), r3, j3, j2, j1, j0, i0)
lambda = itensor(array(lambda), r2, i1, j1)

ITensorCPD.had_contract(S1MPO, S2MPO, c[1]..., d[1]...)
BMPO = BFMatrixMPO([S1MPO, S2MPO, ITensorCPD.had_contract(S3MPO, lambda, r2, j1, i1), S4MPO], 3, a, b, c, d, [r1, r2, r3]);
## Verify that the contractions work and that the decomposition can be contracted to form M.
    norm(reshape(array(ButterFlyMatrixMPO.reconstruct_butterfly(BMPO), inds(BMPO)), (8,8)) - M)

### Working on a general algorithm to solve LS problem for a single factor.
### Given a single factor, we will must form the left and right hand side environment tensors.
### These environments will be determined by the designation of the bits of the problem.
### The only "ignored" bits are those which have no copy indices (i.e. iₗ and jₗ). 
### Therefore, for each factor, we can always loop over all bit values. 
factor = 2
# slices = []
# slice_inds_order = []
# for i in 1:length(BMPO)
#     hyperindsi = commoninds(inds(BMPO[i]), keys(BMPO.bitsI))
#     hyperindsj = commoninds(inds(BMPO[i]), keys(BMPO.bitsJ))
#     indsposi = map(x->findfirst(inds(BMPO[i]), x), hyperindsi)
#     indsposj = map(x->findfirst(inds(BMPO[i]), x), hyperindsj)
#     cutinds_pos = Tuple(sort(vcat(indsposi, indsposj)))
    
#     push!(slice_inds_order, cutinds_pos)
#     sliced_factor = eachslice(array(BMPO[i]), dims=cutinds_pos)
#     push!(slices, sliced_factor)
# end
# slices[1]
# slice_inds_order[1]

## Slice up the main problem, We need to remember the order we cut BMPO2
hyperindsi = commoninds(inds(BMPO[factor]), keys(BMPO.bitsI))
hyperindsj = commoninds(inds(BMPO[factor]), keys(BMPO.bitsJ))
indsposi = map(x->findfirst(inds(BMPO[factor]), x), hyperindsi)
indsposj = map(x->findfirst(inds(BMPO[factor]), x), hyperindsj)
cutinds_pos = Tuple(sort(vcat(indsposi, indsposj)))
sliced_factor = eachslice(array(BMPO[2]), dims=cutinds_pos)
## construct LHS of problem

## This efficiently computes slices of the problems along the 


### The idea is that we need to have the available bits. We select a value for each
###  i2 | 0
###  i1 | 1
###  i0 | 0
###  j2 | 0
###  j1 | 0
###  j0 | 0
###  After picking this bit, we need to go into each array and pick the sub-array that corresponds

########################################
### This is the bitstring we will use for the rows and cols. This will combine rows and cols
### The order goes like i_l-1, ... i0 , j_l-1, ..., j0
### reverse(bitstring(UInt64(0)))
## This idea explicitly sticks on tensors to the end of the matricies to find the correct bits
zeroo = [1 0]
onee = [0 1]
projdict = Dict( 0 => zeroo,
1 => onee)
using ButterFlyMatrixMPO: map_bitstring_to_block_index
iblock_val = map_bitstring_to_block_index(3, 2)
jblock_val = map_bitstring_to_block_index(3, 1)

i_to_blockval = Dict(BMPO.hyperindsi[1] .=> [1,2,3])
j_to_blockval = Dict(BMPO.hyperindsj[end] .=> [1,2,3])

function get_projtensor(val_to_bit, bitmap, levels, ind)
    if dim(ind) == 1 
        return itensor([1,], ind)
    end

    bit_location = bitmap[ind]
    bitvals = map_bitstring_to_block_index(levels, val_to_bit)
    return itensor(projdict[bitvals[bit_location]-1], ind)
end

blocks = []
for factor in 1:4
    iind = factor == 1 ? 1 : factor - 1
    jind = factor
    block_projectorsi = map(x -> get_projtensor(1, BMPO.bitsI, BMPO.levels, x), BMPO.hyperindsi[iind])
    block_projectorsj = map(x -> get_projtensor(1, BMPO.bitsJ, BMPO.levels, x), BMPO.hyperindsj[jind])
    push!(blocks, contract([BMPO[factor], block_projectorsi..., block_projectorsj...]))
end

function get_block_factors(bit_value::Int, BMPO)
    
end
########################################
array(BMPO[1] * itensor(zero, i2) * itensor(one, i1))
array(BMPO[1])[:,:,1,2,1,1]
for x in 1:2 * BMPO.levels
    IndToBit = Dict()
    for (i, v) in zip(bitstring(UInt64(x-1))[end:-1:end-6], [commoninds(inds(BMPO[i]), BMPO.bitsI)...,  commoninds(inds(BMPO[i]), BMPO.bitsJ)[end:-1:1]...])
        IndToBit[v] = projdict[i]
    end

    p1 = [dim(x) != 1 ? itensor(IndToBit[x], x) : itensor([1,], x) for x in (BMPO.hyperindsi[1]..., BMPO.hyperindsj[1]...)]
end

p1 = [dim(x) != 1 ? itensor(IndToBit[x], x) : itensor([1,], x) for x in (BMPO.hyperindsi[1]..., BMPO.hyperindsj[1]...)]
contract([BMPO[1], p1...])

BMPO[2] * itensor(zero, i2) * itensor(zero, i1) * itensor(zero, j1)
