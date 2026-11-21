gaussian          chain 1: ATE 5.309 (sd 0.291)  k 1.00  alpha 1.00  0.5 s
gaussian          chain 2: ATE 5.292 (sd 0.194)  k 1.00  alpha 1.00  0.3 s
gaussian          chain 3: ATE 5.229 (sd 0.220)  k 1.00  alpha 1.00  0.3 s
gaussian          chain 4: ATE 5.311 (sd 0.216)  k 1.00  alpha 1.00  0.3 s
gaussian          chain 5: ATE 5.299 (sd 0.201)  k 1.00  alpha 1.00  0.3 s
gaussian          chain 6: ATE 5.231 (sd 0.214)  k 1.00  alpha 1.00  0.3 s
gaussian          chain 7: ATE 5.212 (sd 0.188)  k 1.00  alpha 1.00  0.3 s
gaussian          chain 8: ATE 5.217 (sd 0.208)  k 1.00  alpha 1.00  0.3 s
marginal_k1       chain 1: ATE 5.231 (sd 0.211)  k 1.00  alpha 0.00  7.4 s
marginal_k1       chain 2: ATE 5.273 (sd 0.224)  k 1.00  alpha 0.00  7.6 s
marginal_k1       chain 3: ATE 5.358 (sd 0.207)  k 1.00  alpha 0.00  7.5 s
marginal_k1       chain 4: ATE 5.210 (sd 0.243)  k 1.00  alpha 0.00  7.5 s
marginal_k1       chain 5: ATE 5.241 (sd 0.224)  k 1.00  alpha 0.00  7.8 s
marginal_k1       chain 6: ATE 5.224 (sd 0.218)  k 1.00  alpha 0.00  7.6 s
marginal_k1       chain 7: ATE 5.233 (sd 0.226)  k 1.00  alpha 0.00  7.9 s
marginal_k1       chain 8: ATE 5.186 (sd 0.220)  k 1.00  alpha 0.00  7.8 s
marginal_cluster  chain 1: ATE 5.238 (sd 0.215)  k 1.09  alpha 0.26  8.3 s
marginal_cluster  chain 2: ATE 5.283 (sd 0.206)  k 1.14  alpha 0.26  8.5 s
marginal_cluster  chain 3: ATE 5.290 (sd 0.201)  k 1.07  alpha 0.25  8.4 s
marginal_cluster  chain 4: ATE 5.237 (sd 0.232)  k 1.10  alpha 0.25  8.5 s
marginal_cluster  chain 5: ATE 5.219 (sd 0.207)  k 1.10  alpha 0.25  8.5 s
marginal_cluster  chain 6: ATE 5.151 (sd 0.294)  k 1.32  alpha 0.28  10.2 s
marginal_cluster  chain 7: ATE 5.153 (sd 0.299)  k 1.17  alpha 0.27  8.9 s
marginal_cluster  chain 8: ATE 5.261 (sd 0.242)  k 1.16  alpha 0.26  8.9 s
marginal_common   chain 1: ATE 4.869 (sd 0.685)  k 1.42  alpha 0.28  6.8 s
marginal_common   chain 2: ATE 5.232 (sd 0.234)  k 1.12  alpha 0.25  6.5 s
marginal_common   chain 3: ATE 5.238 (sd 0.244)  k 1.12  alpha 0.26  6.6 s
marginal_common   chain 4: ATE 5.309 (sd 0.229)  k 1.11  alpha 0.26  6.8 s
marginal_common   chain 5: ATE 5.298 (sd 0.208)  k 1.08  alpha 0.25  6.5 s
marginal_common   chain 6: ATE 5.210 (sd 0.206)  k 1.11  alpha 0.25  6.7 s
marginal_common   chain 7: ATE 5.278 (sd 0.241)  k 1.19  alpha 0.26  6.8 s
marginal_common   chain 8: ATE 5.206 (sd 0.218)  k 1.11  alpha 0.25  7.0 s
blocked           chain 1: ATE 5.265 (sd 0.265)  k 1.11  alpha 0.29  1.3 s
blocked           chain 2: ATE 5.215 (sd 0.220)  k 1.90  alpha 0.45  1.3 s
blocked           chain 3: ATE 5.276 (sd 0.198)  k 1.55  alpha 0.41  1.3 s
blocked           chain 4: ATE 5.209 (sd 0.224)  k 2.92  alpha 0.81  1.6 s
blocked           chain 5: ATE 5.211 (sd 0.230)  k 4.67  alpha 1.56  1.4 s
blocked           chain 6: ATE 5.279 (sd 0.244)  k 3.59  alpha 1.11  1.4 s
blocked           chain 7: ATE 5.240 (sd 0.227)  k 1.94  alpha 0.56  1.4 s
blocked           chain 8: ATE 4.825 (sd 0.710)  k 3.52  alpha 1.12  1.6 s

## Per configuration (8 chains, M = 3000, burn-in 1000)

| Configuration | ATE mean ± se | ATE sd (mean over chains) | k | alpha | R-hat ATE | R-hat k | R-hat alpha | bulk ESS ATE | ms / iter |
|---|---|---|---|---|---|---|---|---|---|
| gaussian | 5.263 ± 0.015 | 0.216 | 1.00 | 1.00 | 1.033 | — | — | 177 | 0.10 |
| marginal_k1 | 5.244 ± 0.018 | 0.222 | 1.00 | 0.00 | 1.056 | — | — | 156 | 2.55 |
| marginal_cluster | 5.229 ± 0.019 | 0.237 | 1.15 | 0.26 | 1.035 | 1.068 | 1.005 | 197 | 2.92 |
| marginal_common | 5.205 ± 0.050 | 0.283 | 1.16 | 0.26 | 1.144 | 1.095 | 1.006 | 41 | 2.23 |
| blocked | 5.190 ± 0.053 | 0.290 | 2.65 | 0.79 | 1.122 | 1.479 | 1.475 | 46 | 0.47 |

## (b) alpha fixed tiny (k = 1): marginal_k1 vs gaussian

| Statistic | marginal_k1 | gaussian | z |
|---|---|---|---|
| ATE mean | 5.2444 ± 0.0185 | 5.2625 ± 0.0155 | -0.75 |
| ATE sd | 0.2217 ± 0.0039 | 0.2163 ± 0.0113 | +0.45 |
| s22 | 0.0536 ± 0.0004 | 0.0533 ± 0.0003 | +0.67 |
| s33 | 0.0660 ± 0.0004 | 0.0663 ± 0.0003 | -0.62 |
| s12 | 0.0652 ± 0.0036 | 0.0617 ± 0.0042 | +0.62 |
| s13 | -0.0978 ± 0.0033 | -0.1009 ± 0.0028 | +0.72 |
| s23 | -0.0066 ± 0.0004 | -0.0061 ± 0.0005 | -0.84 |

## (c) Design A (:cluster) vs blocked: marginal_cluster vs blocked

| Statistic | marginal_cluster | blocked | z |
|---|---|---|---|
| ATE mean | 5.2290 ± 0.0189 | 5.1900 ± 0.0531 | +0.69 |
| ATE sd | 0.2371 ± 0.0139 | 0.2896 ± 0.0604 | -0.85 |
| s22 | 0.0547 ± 0.0010 | 0.0597 ± 0.0043 | -1.15 |
| s33 | 0.0660 ± 0.0003 | 0.0685 ± 0.0006 | -3.51 |
| s12 | 0.0636 ± 0.0032 | 0.0697 ± 0.0048 | -1.06 |
| s13 | -0.0946 ± 0.0033 | -0.1006 ± 0.0038 | +1.20 |
| s23 | -0.0064 ± 0.0004 | -0.0062 ± 0.0006 | -0.21 |

## (d) :common vs :cluster: marginal_common vs marginal_cluster

| Statistic | marginal_common | marginal_cluster | z |
|---|---|---|---|
| ATE mean | 5.2050 ± 0.0500 | 5.2290 ± 0.0189 | -0.45 |
| ATE sd | 0.2833 ± 0.0576 | 0.2371 ± 0.0139 | +0.78 |
| s22 | 0.0539 ± 0.0003 | 0.0547 ± 0.0010 | -0.71 |
| s33 | 0.0698 ± 0.0033 | 0.0660 ± 0.0003 | +1.12 |
| s12 | 0.0620 ± 0.0036 | 0.0636 ± 0.0032 | -0.32 |
| s13 | -0.0920 ± 0.0084 | -0.0946 ± 0.0033 | +0.29 |
| s23 | -0.0062 ± 0.0006 | -0.0064 ± 0.0004 | +0.18 |
