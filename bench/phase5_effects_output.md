# Phase 5 demonstration: raw output of bench/phase5_effects.jl

```
   3511.6 ms  ✓ CausalMixtures
hedonic | marginal_common | sample ATE 5.155, TT 6.671, TUT 3.565 | population ATE 5.001, TT 6.528, TUT 3.474
  pate: 5.251 [4.773, 5.699] (width 0.93), ESS 187, R-hat 1.047
  patt: 6.127 [5.380, 6.707] (width 1.33), ESS 105, R-hat 1.059
  patu: 4.331 [3.785, 4.962] (width 1.18), ESS 298, R-hat 1.020
  satt: 6.125 [5.410, 6.705] (width 1.29), ESS 110, R-hat 1.055
  satu: 4.339 [3.784, 4.961] (width 1.18), ESS 305, R-hat 1.015
  predictive gain at the means (rand_ppd + dpm_ate, chain 1): 5.205 [2.682, 7.921] (width 5.24)
   3213.0 ms  ✓ CausalMixtures
hedonic | marginal_cluster | sample ATE 5.155, TT 6.671, TUT 3.565 | population ATE 5.001, TT 6.528, TUT 3.474
  pate: 5.278 [4.794, 5.758] (width 0.96), ESS 262, R-hat 1.023
  patt: 6.118 [5.389, 6.745] (width 1.36), ESS 222, R-hat 1.025
  patu: 4.397 [3.851, 5.046] (width 1.20), ESS 292, R-hat 1.025
  satt: 6.117 [5.366, 6.735] (width 1.37), ESS 222, R-hat 1.024
  satu: 4.402 [3.868, 5.055] (width 1.19), ESS 302, R-hat 1.022
  predictive gain at the means (rand_ppd + dpm_ate, chain 1): 5.274 [2.749, 7.893] (width 5.14)
   3319.9 ms  ✓ CausalMixtures
hedonic | gaussian | sample ATE 5.155, TT 6.671, TUT 3.565 | population ATE 5.001, TT 6.528, TUT 3.474
  pate: 5.261 [4.845, 5.697] (width 0.85), ESS 178, R-hat 1.027
  patt: 6.149 [5.499, 6.759] (width 1.26), ESS 154, R-hat 1.043
  patu: 4.328 [3.801, 4.901] (width 1.10), ESS 205, R-hat 1.044
  satt: 6.148 [5.479, 6.757] (width 1.28), ESS 157, R-hat 1.040
  satu: 4.330 [3.823, 4.898] (width 1.08), ESS 205, R-hat 1.041
  predictive gain at the means (rand_ppd + dpm_ate, chain 1): 5.280 [2.592, 8.085] (width 5.49)
   3447.7 ms  ✓ CausalMixtures
hedonic | blocked | sample ATE 5.155, TT 6.671, TUT 3.565 | population ATE 5.001, TT 6.528, TUT 3.474
  pate: 5.201 [4.722, 5.663] (width 0.94), ESS 55, R-hat 1.122
  patt: 6.090 [5.425, 6.675] (width 1.25), ESS 127, R-hat 1.060
  patu: 4.269 [3.682, 4.934] (width 1.25), ESS 77, R-hat 1.107
  satt: 6.159 [5.523, 6.721] (width 1.20), ESS 240, R-hat 1.027
  satu: 4.328 [3.753, 4.969] (width 1.22), ESS 211, R-hat 1.053
  predictive gain at the means (rand_ppd + dpm_ate, chain 1): 5.350 [2.761, 7.995] (width 5.23)
   3571.4 ms  ✓ CausalMixtures
twocomp | marginal_common | sample ATE 4.920, TT 6.653, TUT 3.383 | population ATE 4.997, TT 6.533, TUT 3.470
  pate: 5.437 [4.597, 6.264] (width 1.67), ESS 173, R-hat 1.048
  patt: 7.069 [5.794, 8.092] (width 2.30), ESS 76, R-hat 1.107
  patu: 3.985 [2.858, 5.209] (width 2.35), ESS 170, R-hat 1.029
  satt: 7.072 [5.753, 8.064] (width 2.31), ESS 85, R-hat 1.101
  satu: 3.994 [2.855, 5.198] (width 2.34), ESS 171, R-hat 1.026
  predictive gain at the means (rand_ppd + dpm_ate, chain 1): 5.555 [1.060, 9.970] (width 8.91)
   3378.9 ms  ✓ CausalMixtures
twocomp | marginal_cluster | sample ATE 4.920, TT 6.653, TUT 3.383 | population ATE 4.997, TT 6.533, TUT 3.470
  pate: 5.501 [4.610, 6.367] (width 1.76), ESS 53, R-hat 1.110
  patt: 7.161 [5.953, 8.068] (width 2.11), ESS 132, R-hat 1.047
  patu: 4.021 [2.843, 5.346] (width 2.50), ESS 53, R-hat 1.107
  satt: 7.166 [5.982, 8.102] (width 2.12), ESS 137, R-hat 1.043
  satu: 4.033 [2.895, 5.365] (width 2.47), ESS 52, R-hat 1.109
  predictive gain at the means (rand_ppd + dpm_ate, chain 1): 5.456 [0.820, 10.021] (width 9.20)
   3265.2 ms  ✓ CausalMixtures
twocomp | gaussian | sample ATE 4.920, TT 6.653, TUT 3.383 | population ATE 4.997, TT 6.533, TUT 3.470
  pate: 5.503 [4.607, 6.483] (width 1.88), ESS 35, R-hat 1.149
  patt: 7.173 [5.932, 8.081] (width 2.15), ESS 166, R-hat 1.069
  patu: 4.014 [2.809, 5.411] (width 2.60), ESS 39, R-hat 1.134
  satt: 7.175 [5.916, 8.088] (width 2.17), ESS 167, R-hat 1.061
  satu: 4.021 [2.826, 5.399] (width 2.57), ESS 42, R-hat 1.125
  predictive gain at the means (rand_ppd + dpm_ate, chain 1): 5.415 [0.903, 10.160] (width 9.26)
   3065.6 ms  ✓ CausalMixtures
twocomp | blocked | sample ATE 4.920, TT 6.653, TUT 3.383 | population ATE 4.997, TT 6.533, TUT 3.470
  pate: 5.401 [4.463, 6.275] (width 1.81), ESS 52, R-hat 1.113
  patt: 6.837 [4.567, 8.023] (width 3.46), ESS 19, R-hat 1.340
  patu: 4.116 [2.991, 5.389] (width 2.40), ESS 90, R-hat 1.073
  satt: 6.902 [4.633, 8.072] (width 3.44), ESS 20, R-hat 1.322
  satu: 4.156 [3.033, 5.450] (width 2.42), ESS 93, R-hat 1.071
  predictive gain at the means (rand_ppd + dpm_ate, chain 1): 5.511 [0.972, 10.001] (width 9.03)
```
