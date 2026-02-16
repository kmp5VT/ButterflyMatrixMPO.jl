using ITensors, ITensorCPD, LinearAlgebra, Random
using FFTW
rng = Xoshiro(123);
M = randn(rng, Float64, 8,8)
M = fft(M)

rank1 = 2
rank2 = 4
r2_eff = 4
rank3 = 2
### Get the solution via a TT SVD based method.
elt = eltype(M)
begin
    scalings = Array{elt}(undef, rank2, 2,2)
    leftT = Array{elt}(undef, 8, rank3,2,2) # ordered n x r3, j1, j0
    S4T = Array{elt}(undef, rank3,2, 2,2) ## Ordered r3 x j2, j1, j0
    ## m is ordered i2 i1 i0 j2 j1 j0
    MT = reshape(M, (8,2,2,2))
    for j0 in 1:2
        for j1 in 1:2
            u,s,v = svd(MT[:,:, j1, j0])
            leftT[:,:,j1,j0] .= u * diagm((s))[:,1:rank3]
            S4T[:,:,j1, j0] .=  (v')[1:rank3,:]
        end
    end
    # ## Santity check with tensor implementation. 
    # ## Here j10 is a hyperrank and can be delt with using ITensorCPD.
    n = 8
    j2, j1, j0 = 2,2,2
    r3 = rank3
    j10 = 4
    n,j2,j1,j0,r3,j10 = Index.((n,j2,j1,j0,r3,j10), ("n","j2","j1","j0","r3","j10"))
    leftIT = itensor(leftT, n,r3,j10)
    S4IT = itensor(S4T, r3,j2,j10)
    N1 = norm(M - reshape(array(ITensorCPD.had_contract(leftIT, S4IT, j10)), (8,8))) / norm(M)
    println("Error in M after removing S4: $(N1)")

    r3j1j0 = (rank3 * 2 * 2)
    centerT = Array{elt}(undef, rank1,2,2,r3j1j0) ## This is ordered r1 x i1 x i0 x (r3 x j1 x j0)
    S1T = Array{elt}(undef, 2,rank1, 2,2) ## Ordered i2 x r, i1, i0
    ## left is ordered i2 i1 i0 r3 j1 j0
    leftT = reshape(leftT, (2,2,2,r3j1j0))
    for i0 in 1:2
        for i1 in 1:2
            u,s,v = svd(leftT[:, i1, i0, :])
            S1T[:,:,i1, i0] = u[:,1:rank1]
            centerT[:,i1,i0,:] .= (diagm(s) * v')[1:rank1, :]
        end
    end

    m = r3j1j0
    i2, i1, i0 = 2,2,2
    r1 = rank1
    i10 = 4
    m,i2,i1,i0,r1,i10 = Index.((m,i2,i1,i0,r1,i10), ("m","i2","i1","i0","r1","i10"))
    S1IT = itensor(S1T, i2, r1, i10)
    centerIT = itensor(centerT, r1, i10, m)
    N2 = norm(reshape(leftT, (8,r3j1j0)) - reshape(array(ITensorCPD.had_contract(S1IT, centerIT, i10), (i2, i10, m)), (8,r3j1j0))) / norm(leftT)
    println("Error in remaining (left) tensor after removing S1: $(N2)")

    centerT = reshape(centerT, ((rank1 * 2),2,(rank3 * 2),2)) # This is ordered (r1, i1) x i0 x (r3, j1) x j0
    S2T = Array{elt}(undef, (rank1 * 2),rank2,2,2) # This is ordered (r1, i1) x r2 x i0 x j0
    fill!(S2T, 0.0)
    S3T = Array{elt}(undef, rank2,(rank3 * 2),2,2) # This is ordered r2 x (r3,j1) x i0 x j0
    fill!(S3T, 0.0)
    for i0 in 1:2
        for j0 in 1:2
            u,s,v, = svd(centerT[:,i0,:,j0])
            scalings[:,i0,j0] .= s[1:rank2]
            S2T[:,1:r2_eff,i0,j0] .= (u)[:,1:r2_eff]
            S3T[1:r2_eff,:,i0,j0] .= (v')[1:r2_eff, :]
        end
    end

    i0j0 = Index(4,"i0j0")
    r2 = Index(rank2, "r2")
    lambda = itensor(scalings, r2, i0j0)
    S2IT = itensor(S2T, r1, i1, r2, i0j0)
    S2ITl = ITensorCPD.had_contract(S2IT, lambda, r2, i0j0)
    S3IT = itensor(S3T, r2, r3, j1, i0j0)
    reconstructCenterT = reshape(array(itensor(array(ITensorCPD.had_contract(S2ITl, S3IT, i0j0)), r1, i1, r3,j1, i0, j0), r1, i1, i0, r3, j1, j0), size(centerT))
    N3 = norm(centerT - reconstructCenterT) / norm(centerT)
    println("Error in Center tensor after S2 and S3 factorization: $(N3)")
    ;

    centerIT = itensor(reconstructCenterT, r1, i10, m)
    reconstructLeftT = reshape(array(ITensorCPD.had_contract(S1IT, centerIT, i10), (i2, i10, m)), size(leftT))
    N4 = norm(leftT- reconstructLeftT) / norm(leftT)
    println("Error in Left tensor after S1, S2 and S3 reconstructions: $(N4)")
    leftIT = itensor(reconstructLeftT, n,r3,j10)
    N5 = norm(M - reshape(array(ITensorCPD.had_contract(leftIT, S4IT, j10)), (8,8))) / norm(M)
end