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

    function BFMatrixMPO(factors, levels, iMPOinds, jMPOinds, hyperindsi, hyperindsj, ranks)
        bitsI = Dict(iMPOinds[2:end] .=> [1:length(iMPOinds) - 1...])
        bitsJ = Dict(jMPOinds[end-1:-1:1] .=> [1:length(jMPOinds) - 1 ...])

        hyperind_sliced_factors = Vector{AbstractSlices}()
        sliced_index_map = Vector{Tuple}()
        for (i,j) in zip(factors, 1:length(factors))
            is = inds(i)
            hyi = commoninds(is, keys(bitsI))
            hyj = commoninds(is, keys(bitsJ))
            indsposi = map(x->findfirst(is, x), hyi)
            indsposj = map(x->findfirst(is, x), hyj)
            cutinds_pos = Tuple(sort(vcat(indsposi, indsposj)))
            
            push!(sliced_index_map, cutinds_pos)
            sliced_factor = eachslice(array(i), dims=cutinds_pos)
            push!(hyperind_sliced_factors, sliced_factor)
        end

        return new(factors, levels, 
        iMPOinds, jMPOinds, hyperindsi, hyperindsj, 
        bitsI, bitsJ, 
        ranks, 
        hyperind_sliced_factors, sliced_index_map)
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

function RandomButteflyMatrixMPO(M::AbstractArray)
    @assert ndims(M) == 2
    i,j = size(M)
    @assert i==j
    num_levels = Int(log2(i))
    elt = eltype(M)

    iinds, jinds, hyperindsi, hyperindsj = IndexSetFromLevels(i, num_levels)
    ranks = Index.([2 for i in 1:num_levels], ["r$(i)" for i in 1:num_levels])
    factors = Vector{ITensor}(undef, num_levels + 1)
    for i in 1:num_levels + 1
        rks = i == 1 ? ranks[1] : i == num_levels + 1 ? ranks[num_levels] : (ranks[i], ranks[i-1])
        f = random_itensor(elt, rks, unique([iinds[i], hyperindsi[i]...]), unique([jinds[i], hyperindsj[i]...]))

        ## Do we need to normalize, I am not sure.
        # indsposi = [findfirst(x -> x == i, inds(f)) for i in hyperindsi[i]]
        # indsposj = [findfirst(x -> x == j, inds(f)) for j in hyperindsj[i]]
        # slicef = eachslice(array(f), dims=(indsposi..., indsposj...))

        # f, lambda = ITensorCPD.row_norm(itensor(slicef, ))

        factors[i] = f
    end
    BMPO = ButterFlyMatrixMPO(factors, num_levels, iinds, jinds, hyperindsi, hyperindsj, ranks)
    return BMPO
end

factors(bmpo::BFMatrixMPO) = getproperty(bmpo, :factors)
ITensors.inds(bmpo::BFMatrixMPO) = [getproperty(bmpo, :iMPOinds)..., getproperty(bmpo, :jMPOinds)[end:-1:1]...]
ITensors.ind(bmpo::BFMatrixMPO, i::Int) = inds(bmpo)[i]
ITensors.itensor2inds(A::BFMatrixMPO)::Any = inds(A)
Base.getindex(cp::BFMatrixMPO, i) = cp.factors[i]

Base.length(bmpo::BFMatrixMPO) = length(bmpo.factors)

function reconstruct_butterfly(BM::BFMatrixMPO)
    FuseMPO = ITensorCPD.had_contract(BM[1], BM[2], BM.hyperindsi[1]..., BM.hyperindsj[1]...)
    for i in 3:length(BM)
        FuseMPO = ITensorCPD.had_contract(FuseMPO, BM[i], BM.hyperindsi[i-1]..., BM.hyperindsj[i-1]...)
    end
    return FuseMPO
end

function map_bitstring_to_block_index(levels, bitval; base=2)
    bs = digits(UInt64(bitval - 1); base, pad=levels) .+1
    return bs
end