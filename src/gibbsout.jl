## ---------------------------------------------------------------------------#
## Output

mutable struct GibbsOut
    out_data::Array{StateData}
    out_dp::Array{StateDP}
    out_theta::Array{StateTheta}
end

GibbsOut(; out_data=Array(StateData,0), out_dp=Array(StateDP,0), out_theta=Array(StateTheta,0)) =
    GibbsOut(out_data, out_dp, out_theta)

## generate GibbsOut given num iterations, M (or input.params?)
GibbsOut(M::Int64) = GibbsOut(Array(StateData, M), Array(StateDP, M), Array(StateTheta, M))

## TODO: output as subtype of AbstractArray
## TODO: DO NOT SAVE state_sampler ...
##GibbsOut = Array{GibbsState, 1}

## --------------------------------------------------------------------------- #
export GibbsOut
