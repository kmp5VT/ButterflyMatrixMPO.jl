function least_squares_solve(BMPO, factor, M)
    bits_on_factor = inds(BMPO[factor])[[BMPO.sliced_index_map[factor]...]]
    bits_off_factor = noncommoninds(bits_on_factor, vcat(BMPO.iMPOinds[2:end], BMPO.jMPOinds[1:end-1]))
    

    tensor_block_indices = dim.(bits_on_factor)
    off_block_indices = Tuple(dim.(bits_off_factor))

    i_nohyp = BMPO.iMPOinds[1]
    j_nohyp = BMPO.jMPOinds[end]
    
    MIT = itensor(M, inds(BMPO))
    
    Sf = similar(BMPO[factor])
    Sfsliced = eachslice(array(Sf), dims=BMPO.sliced_index_map[factor])
    norm_fun = eachrow
    lam = Vector{eltype(BMPO)}()

    # @show bits_on_factor 
    # @show bits_off_factor

    m = 1
    for i in 1:length(BMPO.hyperind_sliced_factors[factor])
        bits_for_solve = ButterFlyMatrixMPO.to_tensor_element(i, tensor_block_indices) .+ 1

        ## i is the tensor we are rewritting 
        MTtKRP = nothing
        gram_right = nothing
        gram_left = nothing
        iindsPrev = nothing
        jindsPrev = nothing
        RHS_KRP = 1
        LHS_KRP = 1
        for j in 1:prod(off_block_indices)
            bits_for_gram = ButterFlyMatrixMPO.to_tensor_element(j, off_block_indices) .+ 1

            label_to_bit = Dict(vcat(bits_off_factor..., bits_on_factor...) .=> vcat(bits_for_gram, bits_for_solve))
            iinds = map(x -> label_to_bit[x], BMPO.iMPOinds[2:end])
            jinds = map(x -> label_to_bit[x], BMPO.jMPOinds[end-1:-1:1])
            ## TODO compute slices of the target tensor, then just find the position with function
            ## That way we don't have to recall array function and index search [].
            Target_Block = itensor(array(MIT)[:,iinds..., :, jinds...], i_nohyp, j_nohyp)
            
            if factor != length(BMPO) && jinds != jindsPrev
                RHS_KRP = contract([itensor(BMPO.hyperind_sliced_factors[x][ map_bit_vals_to_list_position(BMPO, x, label_to_bit)], BMPO.block_index_map[x])
                for x in factor+1:BMPO.levels + 1];)
                gr = RHS_KRP * dag(prime(RHS_KRP, tags=tags(BMPO.ranks[factor])))
                gram_right = isnothing(gram_right) ? gr : gram_right + gr
                jindsPrev = jinds
            end

            if factor != 1 && iinds != iindsPrev
                LHS_KRP = contract([itensor(BMPO.hyperind_sliced_factors[x][map_bit_vals_to_list_position(BMPO, x, label_to_bit)], BMPO.block_index_map[x]) for x in 1:factor-1])
                gr = LHS_KRP * dag(prime(LHS_KRP, tags=tags(BMPO.ranks[factor - 1])))
                gram_left = isnothing(gram_left) ? gr : gram_left + gr
                iindsPrev = iinds
            end

            MTtKRP = isnothing(MTtKRP) ? 
                    dag(LHS_KRP) * Target_Block * dag(RHS_KRP) : 
                    MTtKRP + dag(LHS_KRP) * Target_Block * dag(RHS_KRP)
        end
        label_to_bit = Dict(bits_on_factor .=> bits_for_solve)
        chol = false
        sol = solve_ls_problem(MTtKRP, gram_left, gram_right; chol)

        for j in norm_fun(sol)
            l = norm(j)
            j ./= l
            push!(lam, l)
            m += 1
        end 

        Sfsliced[map_bit_vals_to_list_position(BMPO, factor, label_to_bit)] .= sol
    end
    return Sf, lam
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

function solve_ls_problem(MTtKRP, ::Nothing, gram_right; chol = true)
        return (Symmetric(array(gram_right)) \ permutedims(array(MTtKRP), (2,1)))
end

function solve_ls_problem(MTtKRP, gram_left, ::Nothing; chol = true)
        return (Symmetric(array(gram_left)) \ array(MTtKRP))
end

function solve_ls_problem(MTtKRP, gram_left, gram_right; chol = true)
    return array(itensor(pinv(array(gram_left)), inds(gram_left)) * (MTtKRP * itensor(pinv(array(gram_right)), inds(gram_right))))
end
