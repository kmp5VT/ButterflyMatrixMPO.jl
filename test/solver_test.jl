BMPO;

zeroo = [1 0]
onee = [0 1]
projdict = Dict( 0 => zeroo,
1 => onee)
function get_projtensor(val_to_bit, bitmap, levels, ind)
    if dim(ind) == 1 
        return itensor([1,], ind)
    end

    bit_location = findfirst(x-> x== ind, bitmap)
    bitvals = ButterFlyMatrixMPO.map_bitstring_to_block_index(levels, val_to_bit)
    return itensor(projdict[bitvals[bit_location]-1], ind)
end

factor = 1
#### This is the "easy mode" idea where we multiply by the onehot every
iind = factor == 1 ? 1 : factor - 1
jind = factor
block_projectorsi = map(x -> get_projtensor(1, BMPO.bitsI, BMPO.levels, x), BMPO.hyperindsi[iind])
block_projectorsj = map(x -> get_projtensor(1, BMPO.bitsJ, BMPO.levels, x), BMPO.hyperindsj[jind])
array(contract([BMPO[factor], block_projectorsi..., block_projectorsj...])) - BMPO.hyperind_sliced_factors[1][1]
## Pick a factor


### This will loop through all the values and corresponds to the tensor ordering
### defined in the  sliced_index_map variable.
factor = 1
elt = eltype(BMPO)
bits_on_factor = inds(BMPO[factor])[[BMPO.sliced_index_map[factor]...]]
tensor_block_indices = dim.(bits_on_factor)
bits_off_factor = noncommoninds(bits_on_factor, vcat(BMPO.iMPOinds[2:end], BMPO.jMPOinds[1:end-1]))
off_block_indices = Tuple(dim.(bits_off_factor))
i_nohyp = BMPO.iMPOinds[1]
j_nohyp = BMPO.jMPOinds[end]
MIT = itensor(M, inds(BMPO))

for i in 1:length(BMPO.hyperind_sliced_factors[factor])
    bits_for_solve = ButterFlyMatrixMPO.to_tensor_element(i, tensor_block_indices) .+ 1

    ## i is the tensor we are rewritting 
    MTtKRP = nothing
    gram_right = nothing
    gram_left = nothing
    for j in 1:prod(off_block_indices)
        bits_for_gram = ButterFlyMatrixMPO.to_tensor_element(j, off_block_indices) .+ 1

        label_to_bit = Dict(vcat(bits_off_factor..., bits_on_factor...) .=> vcat(bits_for_gram, bits_for_solve))
        iinds = map(x -> label_to_bit[x], BMPO.iMPOinds[2:end])
        jinds = map(x -> label_to_bit[x], BMPO.jMPOinds[end-1:-1:1])
        Target_Block = itensor(array(MIT)[:,iinds..., :, jinds...], i_nohyp, j_nohyp)
        
        RHS_KRP = 1
        if factor != 4
            RHS_KRP = contract([itensor(BMPO.hyperind_sliced_factors[x][ map_bit_vals_to_list_position(BMPO, x, label_to_bit)], BMPO.block_index_map[x])
            for x in factor+1:BMPO.levels + 1];)
            gr = RHS_KRP * dag(prime(RHS_KRP, tags=tags(BMPO.ranks[factor])))
            gram_right = isnothing(gram_right) ? gr : gram_right + gr
        end
        # for f1 in factor+1:BMPO.levels +1 
        #     ## finally we search our block-list for this block
        #     ## TODO make the bitlist to block-value a dict for fast lookup
        #     pos = map_bit_vals_to_list_position(BMPO, f1, label_to_bit)
        #     RHS_KRP *= itensor(BMPO.hyperind_sliced_factors[f1][pos], BMPO.block_index_map[f1])
        # end

        LHS_KRP = 1
        if factor != 1
            LHS_KRP = contract([itensor(BMPO.hyperind_sliced_factors[x][map_bit_vals_to_list_position(BMPO, x, label_to_bit)], BMPO.block_index_map[x]) for x in 1:factor-1])
            gr = LHS_KRP * dag(prime(LHS_KRP, tags=tags(BMPO.ranks[factor - 1])))
            gram_left = isnothing(gram_left) ? gr : gram_left + gr
            # LHS_KRP = contract()
        # for f1 in 1:factor-1
        #     ## Here we compute the product of all the factors to the right
        #     ## Given the bits_for_gram bits and ADD it to the RHS_KRP
        #     pos = map_bit_vals_to_list_position(BMPO, f1, label_to_bit)
        #     LHS_KRP *= itensor(BMPO.hyperind_sliced_factors[f1][pos], BMPO.block_index_map[f1])
        end
        MTtKRP = isnothing(MTtKRP) ? dag(LHS_KRP) * Target_Block * dag(RHS_KRP) : MTtKRP + dag(LHS_KRP) * Target_Block * dag(RHS_KRP)
    end
    @show  (array(gram_right) \ array(MTtKRP)')'
    return
end

function map_bit_vals_to_list_position(BMPO, factor, label_bit_map)
    ## Here we compute the product of all the factors to the right
    ## Given the bits_for_gram bits and ADD it to the RHS_KRP
    ## First we find the ordering of the blockwise indlist
    block_tensor_indlist = inds(BMPO[factor])[[BMPO.sliced_index_map[factor]...]]
    ## Then we find the value of each of the bits for this block
    block_ind_list = map(x -> label_bit_map[x], block_tensor_indlist)
    dim_list = dim.(block_tensor_indlist)
    return sum(map(x -> (block_ind_list[x] - 1) * prod(dim_list[1:x-1]), [1:length(block_ind_list)...])) + 1
    # , noncommoninds(inds(BMPO[factor]), block_tensor_indlist)
end


ComplexF64[0.2170965303673637 + 0.03915592242961101im 0.187174566241604 + 0.005703887957502922im 0.15843878432408484 + 0.07724374642465046im 1.2150367466560057 - 0.16328957298175795im; -0.07608693743824214 + 0.09640426533559791im -0.030736288222307763 + 0.14469586312675908im -0.19945622368008475 + 0.27556998411188116im 1.3755447675931167 - 0.2675578689597139im]'
