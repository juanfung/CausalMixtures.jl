using Test
using CausalMixtures

# Test the GibbsTuple type alias
@testset "GibbsTuple type alias" begin
    @test GibbsTuple isa Type
    @test GibbsTuple == Tuple{CausalMixtures.GibbsState, CausalMixtures.GibbsInput, CausalMixtures.GibbsOut}
end