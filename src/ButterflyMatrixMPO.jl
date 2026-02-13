## The algorithm is:
## Look at the tensor we are updating, it should have 2 modes which are the in and out
## It will also have a left hand side problem and right hand side problem.
## The goal is to solve |M - A X| by leveraging the outer product structure of the the hyper-indices
## This will allow us to solve for sections of M at a time and it will allow us to build X by sections at a time.
## We can then determine how important each section of X is in the solution to the LS problem.

module ButterFlyMatrixMPO
    using ITensors, ITensorCPD
    include("BFMatrixMPO.jl")
end