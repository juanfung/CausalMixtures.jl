## ---------------------------------------------------------------------------#
## Output

mutable struct GibbsOut
    out_data::Array{StateData}
    out_dp::Array{StateDP}
    out_theta::Array{StateTheta}
end

## generate GibbsOut given num iterations, M (or input.params?)
GibbsOut(M::Int64;
         out_data=Array{StateData}(undef, M),
         out_dp=Array{StateDP}(undef, M),
         out_theta=Array{StateTheta}(undef, M)) = GibbsOut(out_data, out_dp, out_theta)

## default constructor with no arguments
GibbsOut() = GibbsOut(0) 

## TODO: output as subtype of AbstractArray
## TODO: DO NOT SAVE state_sampler ...
##GibbsOut = Array{GibbsState, 1}

## --------------------------------------------------------------------------- #
export GibbsOut
