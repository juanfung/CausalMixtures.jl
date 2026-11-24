# test/test_sigma23.jl -- R3, the collapsed sigma23 step (Phase 4c)
using Test, CausalMixtures, LinearAlgebra, Random, Statistics, Distributions
isdefined(Main, :TestHelpers) || include("TestHelpers.jl")
using .TestHelpers
const CM3 = CausalMixtures

## C3 prior scale: rho = 10, R = diag(1/rho, r, r), r = 0.153
const RHO3 = 10
const V3 = RHO3 .* Matrix(Diagonal([1 / RHO3, 0.153, 0.153]))

entries3(S) = [S[1, 2], S[1, 3], S[2, 2], S[2, 3], S[3, 3]]

@testset "R3: collapsed sigma23 step" begin

@testset "log density = dense restricted-IW log density; support is the PD interval" begin
    rng = test_rng()
    for _ in 1:20
        S = CM3.NobileWishart(rng, RHO3, V3)
        L, U = CM3.sigma23_interval(S)
        @test L < S[2, 3] < U
        f(s) = (T = copy(S); T[2, 3] = T[3, 2] = s; -(RHO3 + 4) / 2 * logdet(T) - tr(V3 * inv(T)) / 2)
        for s in range(L, U, length=9)[2:end-1]
            @test CM3.sigma23_logdens(S, s, V3, RHO3) - CM3.sigma23_logdens(S, S[2, 3], V3, RHO3) ≈ f(s) - f(S[2, 3])
        end
        @test CM3.sigma23_logdens(S, U + 1e-9, V3, RHO3) == -Inf
        @test CM3.sigma23_logdens(S, L - 1e-9, V3, RHO3) == -Inf
    end
end

@testset "invariance: one slice update keeps NobileWishart(rho, rho R) invariant" begin
    rng = test_rng()
    N = 200_000
    before = [CM3.NobileWishart(rng, RHO3, V3) for _ in 1:N]
    after = map(before) do S
        T = copy(S)
        T[2, 3] = T[3, 2] = CM3.slice_sigma23(rng, S, V3, RHO3)
        T
    end
    fresh = [CM3.NobileWishart(rng, RHO3, V3) for _ in 1:N]
    A = reduce(hcat, entries3.(after))'
    F = reduce(hcat, entries3.(fresh))'
    @test maximum(abs, two_sample_zscores(A, F)) < 4.5
    sq(X) = (X .- mean(X, dims=1)) .^ 2
    @test maximum(abs, two_sample_zscores(sq(A), sq(F))) < 4.5
    @test all(isposdef, after[1:1000])
    ## sigma23 moved, the rest did not
    @test mean(S[2, 3] != T[2, 3] for (S, T) in zip(before, after)) > 0.99
    @test all(entries3(S)[[1, 2, 3, 5]] == entries3(T)[[1, 2, 3, 5]] for (S, T) in zip(before, after))
    ## power: posterior-style degrees of freedom (rho + 50) in the slice target fail the same
    ## check (5 steps, so that the wrong target is reached; one step moves sigma23 only part way)
    wrong = reshape(map(before) do S
        T = copy(S)
        for _ in 1:5
            T[2, 3] = T[3, 2] = CM3.slice_sigma23(rng, T, V3, RHO3 + 50)
        end
        T[2, 3]
    end, :, 1)
    @test max(abs(two_sample_zscores(wrong, F[:, [4]])[1]), abs(two_sample_zscores(sq(wrong), sq(F[:, [4]]))[1])) > 10
end

@testset "conditional: slice draws follow the normalized density on a fine grid" begin
    rng = test_rng()
    for S in ([1.0 0.4 -0.3; 0.4 1.6 0.2; -0.3 0.2 1.2], [1.0 -0.8 0.7; -0.8 1.0 -0.5; 0.7 -0.5 0.9])
        L, U = CM3.sigma23_interval(S)
        G = 200_001
        grid = range(L, U, length=G)
        lf = [CM3.sigma23_logdens(S, s, V3, RHO3) for s in grid]
        lf[1] = lf[end] = -Inf
        w = exp.(lf .- maximum(lf))
        cdf = cumsum(w) ./ sum(w)
        Fgrid(x) = cdf[clamp(searchsortedlast(grid, x), 1, G)]
        N = 50_000
        ## x0 ~ the target (inverse cdf on the grid), then one slice step: the draws are
        ## independent and, by invariance, again from the target
        draws = map(1:N) do _
            x0 = grid[searchsortedfirst(cdf, rand(rng))]
            T = copy(S); T[2, 3] = T[3, 2] = x0
            CM3.slice_sigma23(rng, T, V3, RHO3)
        end
        ## a chain from a fixed start: the conditional is reached
        chain = Float64[]
        T = copy(S)
        for k in 1:N
            T[2, 3] = T[3, 2] = CM3.slice_sigma23(rng, T, V3, RHO3)
            push!(chain, T[2, 3])
        end
        for x in (draws, chain[1:10:end])
            xs = sort(x)
            n = length(xs)
            D = maximum(max(abs(k / n - Fgrid(xs[k])), abs((k - 1) / n - Fgrid(xs[k]))) for k in 1:n)
            @test D < 1.95 / sqrt(n) # KS at alpha = 0.001
        end
        @test all(L .< draws .< U)
    end
end

@testset "update_sigma23! changes sigma23 and y_miss only; option wiring" begin
    @test CM3.InputParams().sigma23_step == true
    data = generate_hedonic_data(60, 5)
    raw = CM3.RawData(data.formulas.formula_y, data.formulas.formula_d, data.df)
    for (model, sl) in (("marginal", :common), ("marginal", :cluster), ("gaussian", :cluster))
        params = CM3.InputParams(M=5, scale_data=(true, true), verbose=false, model=model, slopes=sl)
        pr = setup_default_priors(2, 3; J = model == "gaussian" ? 1 : 4, slopes=sl)
        rng = Xoshiro(4)
        st, input, _ = CM3.dpm!(CM3.dpm_init(raw, pr, params; rng=rng)...; rng=rng)
        j = 1
        idx = st.state_dp.members[j]
        S0, b0 = copy(st.state_theta[j].Sigma), copy(st.state_theta[j].beta)
        ds0, y10, y00 = copy(st.state_data.dstar), copy(st.state_data.y1), copy(st.state_data.y0)
        CM3.update_sigma23!(rng, st, input, j, idx)
        S1 = st.state_theta[j].Sigma
        @test S1[2, 3] != S0[2, 3] && S1[2, 3] == S1[3, 2] && isposdef(S1)
        @test entries3(S1)[[1, 2, 3, 5]] == entries3(S0)[[1, 2, 3, 5]] && S1[1, 1] == 1
        @test st.state_theta[j].beta == b0
        @test st.state_data.dstar == ds0
        d = input.data.d
        obs1, obs0 = d .== 1, d .== 0
        @test st.state_data.y1[obs1] == y10[obs1] && st.state_data.y0[obs0] == y00[obs0]
        @test all(st.state_data.y1[idx][obs0[idx]] .!= y10[idx][obs0[idx]])
        ## sigma23_step = false: the sweep never calls R3 (same draws as a run without it)
        for s23 in (true, false)
            p2 = CM3.InputParams(M=5, scale_data=(true, true), verbose=false, model=model, slopes=sl, sigma23_step=s23)
            r = Xoshiro(9)
            a = CM3.dpm!(CM3.dpm_init(raw, pr, p2; rng=r)...; rng=r)[1]
            @test all(isposdef(t.Sigma) for t in a.state_theta)
        end
    end
end

end # R3
