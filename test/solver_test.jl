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

for i in 1:length(BMPO.hyperind_sliced_factors[factor])
    bits_for_solve = to_tensor_element(i, tensor_block_indices) .+ 1
    @show bits_for_solve

    ## i is the tensor we are rewritting 
    MTtKRP = nothing
    Gram = nothing
    for j in 1:prod(off_block_indices)
        bits_for_gram = to_tensor_element(j, off_block_indices) .+ 1
        @show bits_for_gram

        label_to_bit = Dict(vcat(bits_off_factor..., bits_on_factor...) .=> vcat(bits_for_gram, bits_for_solve))
        iinds = map(x -> label_to_bit[x], BMPO.iMPOinds[2:end])
        jinds = map(x -> label_to_bit[x], BMPO.jMPOinds[end-1:-1:1])
        Target_Block = itensor(array(MIT)[:,iinds..., :, jinds...], i_nohyp, j_nohyp)
        
        RHS_KRP = 1
        for f1 in factor+1:BMPO.levels +1 
            ## finally we search our block-list for this block
            ## TODO make the bitlist to block-value a dict for fast lookup
            pos, is = map_bit_vals_to_list_position(BMPO, f1, label_to_bit)
            RHS_KRP *= itensor(BMPO.hyperind_sliced_factors[f1][pos], is)
            # @show filterinds(x-> x ∉ vcat(BMPO.iMPOinds[2:end], BMPO.jMPOinds[end-1:-1:1]), BMPO[f1])
        end

        Gram = isnothing(Gram) ? RHS_KRP * prime(RHS_KRP, tags=tags(BMPO.ranks[factor])) : Gram + RHS_KRP * prime(RHS_KRP, tags=tags(BMPO.ranks[factor]))
        RHS_KRP = 1
        for f1 in 1:factor-1
            ## Here we compute the product of all the factors to the right
            ## Given the bits_for_gram bits and ADD it to the RHS_KRP
        end
    end
    @show Gram
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
    @show block_ind_list
    return sum(map(x -> (block_ind_list[x] - 1) * prod(dim_list[1:x-1]), [1:length(block_ind_list)...])) + 1, noncommoninds(inds(BMPO[factor]), block_tensor_indlist)
end
