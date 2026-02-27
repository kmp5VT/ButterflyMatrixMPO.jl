struct BFMatrixMPO
    ### These are fundemantal to the decomposition they hold the problem
    factors::Vector{ITensor}
    levels::Int

    ## The idea is that we need to know which inds are on the 
    ## factors and which shared inds live on each factor
    iMPOinds
    jMPOinds
    hyperindsi
    hyperindsj
    
    ## This is a dictionary that maps a bit array of number of number of levels long
    ## To the indices of the butterfly matrix in order of fastest moving to slowest moving
    bitsI
    bitsJ

    ## This holds the rank indices (i.e. the bond dimensions)
    ranks

    ## Since the hyperinds are fixed we can split all of the factors up immediately 
    hyperind_sliced_factors
    ## This tells the tensor ordering of the slices array.
    ## for example i2, i3, j1 => (2,2,2) tensor so slices (1,2,1) = (1-1) * 4 + (2 - 1) * 2 + 1 = 3
    sliced_index_map
    block_index_map

    lambda_pos
    lambda

    function BFMatrixMPO(factors, levels, iMPOinds, jMPOinds, hyperindsi, hyperindsj, ranks)
        bitsI = iMPOinds[2:end]
        bitsJ = jMPOinds[end-1:-1:1]

        hyperind_sliced_factors = Vector{AbstractSlices}()
        sliced_index_map = Vector{Tuple}()
        block_index_map = Vector{Vector{Index}}()
        for (i,j) in zip(factors, 1:length(factors))
            is = inds(i)
            hyi = commoninds(is, bitsI)
            hyj = commoninds(is, bitsJ)
            indsposi = map(x->findfirst(is, x), hyi)
            indsposj = map(x->findfirst(is, x), hyj)
            cutinds_pos = Tuple(sort(vcat(indsposi, indsposj)))
            push!(block_index_map, noncommoninds(is, is[[cutinds_pos...]]))
            
            push!(sliced_index_map, cutinds_pos)
            sliced_factor = eachslice(array(i), dims=cutinds_pos)
            push!(hyperind_sliced_factors, sliced_factor)
        end

        lambda = ones(eltype(factors[1]), 2^(levels+1))
        return new(factors, levels, 
        iMPOinds, jMPOinds, hyperindsi, hyperindsj, 
        bitsI, bitsJ, 
        ranks,
        hyperind_sliced_factors, 
        sliced_index_map, 
        block_index_map,
        [1,], 
        lambda
        )
    end
end

## You should be able to set the "quantics" of the buttefly matrix (it shouldn't have to be 2 only) but for now
## fix it to be 2, later leverage n as the fixed index set
function IndexSetFromLevels(n, levels)
    is = [Index(i == levels + 1 ? 1 : 2, "i$(levels+1-i)") for i in 1:levels+1]
    js = [Index(j == levels + 1 ? 1 : 2, "j$(levels+1-j)") for j in 1:levels+1]
    T = ITensor(is..., js...)
    jinds = Vector{Index}()
    iinds = Vector{Index}()
    hyperindsi = []
    hyperindsj = []
    for i in 1:levels+1
        push!(iinds, inds(T)[i])
        push!(jinds, inds(T)[2 * levels + 2 - (i-1)])
        hyperi = [inds(T)[(i == levels + 1 ? i : i+1):levels+1]...]
        hyperj = [inds(T)[(i == levels + 1 ? 2 * levels + 2 - (i - 2) : 2 * levels + 2 - (i -1)): 2 * levels + 2 ]...]
        push!(hyperindsi, hyperi)
        push!(hyperindsj, hyperj)
    end
    return iinds, jinds, hyperindsi, hyperindsj
end

function RandomButteflyMatrixMPO(M::AbstractArray, ranks=nothing)
    @assert ndims(M) == 2
    i,j = size(M)
    @assert i==j
    num_levels = Int(log2(i))
    elt = eltype(M)

    iinds, jinds, hyperindsi, hyperindsj = IndexSetFromLevels(i, num_levels)
    if isnothing(rank)
        ranks = Index.([2 for _ in 1:num_levels], ["r$(i)" for i in 1:num_levels])
    elseif ranks isa Number
        ranks = Index.([ranks for _ in 1:num_levels], ["r$(i)" for i in 1:num_levels])
    end
    factors = Vector{ITensor}(undef, num_levels + 1)
    for i in 1:num_levels + 1
        rks = i == 1 ? ranks[1] : i == num_levels + 1 ? ranks[num_levels] : (ranks[i-1], ranks[i],)
        f = random_itensor(elt, rks, unique([iinds[i], hyperindsi[i]...]), unique([jinds[i], hyperindsj[i]...]))
        factors[i] = f
    end
    BMPO = BFMatrixMPO(factors, num_levels, iinds, jinds, hyperindsi, hyperindsj, ranks)

    ## Normalize the rows before the mid point
    for sliced_factors in BMPO.hyperind_sliced_factors[1:num_levels÷2+1]
        for slice in sliced_factors
            normalize!.(eachrow(slice))
        end
    end
    ## Normalize the columns after the midpoint. This way normalization always
    ## points inwards.
    for sliced_factor in BMPO.hyperind_sliced_factors[num_levels÷ 2 + 2:end]
        for slice in sliced_factor
            normalize!.(eachcol(slice))
        end
    end

    return BMPO
end

factors(bmpo::BFMatrixMPO) = getproperty(bmpo, :factors)
ITensors.inds(bmpo::BFMatrixMPO) = [getproperty(bmpo, :iMPOinds)..., getproperty(bmpo, :jMPOinds)[end:-1:1]...]
ITensors.ind(bmpo::BFMatrixMPO, i::Int) = inds(bmpo)[i]
ITensors.itensor2inds(A::BFMatrixMPO)::Any = inds(A)
Base.getindex(cp::BFMatrixMPO, i) = cp.factors[i]
Base.eltype(bmpo::BFMatrixMPO) = eltype(getproperty(bmpo, :factors)[1].tensor)

Base.length(bmpo::BFMatrixMPO) = length(bmpo.factors)

function reconstruct_butterfly(BM::BFMatrixMPO)
    FuseMPO = ITensorCPD.had_contract(BM[1], BM[2], BM.hyperindsi[1]..., BM.hyperindsj[1]...)
    lampos = BM.lambda_pos[1]
    left = lampos > (BM.levels+1 ÷ 2)
    lambda = ITensor(BM.lambda, ind(BM[lampos], left ? 1 : 2), inds(BM[lampos])[[BM.sliced_index_map[lampos]...]]...)
    scaled = ITensorCPD.had_contract(BM[lampos], lambda, inds(lambda)...)
    for i in 3:length(BM)
        ten = (i == lampos ? scaled : BM[i])
        FuseMPO = ITensorCPD.had_contract(FuseMPO, ten, BM.hyperindsi[i-1]..., BM.hyperindsj[i-1]...)
    end
    return FuseMPO
end

function map_bitstring_to_block_index(levels, bitval; base=2)
    return digits(UInt64(bitval - 1); base, pad=levels) .+1
end

### This function tries to imagine the sliced array indices as tensor indices
### of a block sparse tensor. then we just need to use the ranges to find the 
### correct position.
function to_tensor_element(v::Int, range::Tuple)
    idx = zeros(Int, length(range))
    for r in 1:length(range) - 1
        ext = prod(range[r+1:end])
        m = (v-1) ÷ ext
        idx[r] = m
        v -= m * ext
    end
    idx[end] = (v-1)
    return idx
end