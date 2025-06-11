# runtests.jl
using Test

include("testgibbsinput.jl")
include("testgibbsstate.jl")
include("testgibbsout.jl")
include("testCausalMixtures.jl")
include("testgibbsppd.jl")
# ... include other test files as needed