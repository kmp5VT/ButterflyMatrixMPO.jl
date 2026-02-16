### Generic method to convert an NxN matrix into a quantized tensor
    begin
        l = 3
        i00 = Index(1, "i00")
        j00 = Index(1, "j00")
        T = itensor(M, (i2,i1,i0,i00,j2,j1,j0,j00))
        jinds = []
        iinds = []
        hyperindsi = []
        hyperindsj = []
        i = 2
        for i in 1:l+1
            jind = inds(T)[2 * l + 2 - (i-1)]
            iind = inds(T)[i]
            hyperi = [inds(T)[i+1:l+1]...]
            hyperj = [inds(T)[2 * l + 2 - (i -1): 2 * l + 2 ]...]
            push!(jinds, jind)
            push!(iinds, iind)
            push!(hyperindsi, hyperi)
            push!(hyperindsj, hyperj)
        end
    end

## Function to reconstruct the butterfly decomp, not sure if it works.
    function reconstruct_butterfly(S1MPO, S2MPO, S3MPO, S4MPO, hyperindsi, hyperindsj)
        S2S3MPO = ITensorCPD.had_contract(S2MPO, S3MPO,
        hyperindsi[2]..., hyperindsfj[2]...)
        S1S2S3MPO = ITensorCPD.had_contract(S1MPO, S2S3MPO,
        hyperindsi[1]..., hyperindsj[1]...)
        S1S2S3S4MPO = ITensorCPD.had_contract(S1S2S3MPO, S4MPO, 
        hyperindsi[3]..., hyperindsj[3]...)

        return S1S2S3S4MPO
    end

## Transform the TT-SVD based tensors into MPO ones for ALS optimization
    begin
        S1IT
        S1MPO = ITensor(array(S1IT), i2, r1, i1, i0, i00, j00)
        norm(reshape(array(S1T), (2,rank1, 2, 2)) - array(S1MPO))

        S2IT
        S2MPO = ITensor(array(S2IT), r1, i1, r2, i0, j0, i00, j00)
        norm(reshape(array(S2IT), (rank1,2,rank2, 2, 2)) - array(S2MPO))

        S3IT
        S3MPO = ITensor(array(S3IT), r2, r3, j1, i0, j0, i00, j00)
        norm(reshape(array(S3IT), (rank2,rank3, 2, 2,2)) - array(S3MPO))

        S4IT
        S4MPO = ITensor(array(S4IT), r3, j2,j1,j0,j00,i00)
        norm(reshape(array(S4IT), rank3, 2,2,2) - array(S4MPO))
    end