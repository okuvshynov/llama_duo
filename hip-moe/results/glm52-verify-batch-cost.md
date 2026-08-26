# GLM-5.2 UD-Q3_K_XL verify-batch cost study, 2026-08-26
# llama-bench, one load, -ngl 99 -ncmoe 56 -ts 59/7/7/7 -t 16, -p 1..17 -d 0,512 -r 4.
# Motivation: is there a kernel boundary that makes a draft of 8 (verify batch 9) special?
# Derived ms/batch (N / pp-N t/s): marginal token ~120-150 ms within an octave; crossing
# a multiple of 8 adds a ~275-285 ms step (batch 9 and batch 17, identical at d=0 and
# d=512 -> depth-independent, i.e. matmul column tiling of width 8, not attention).
# tg128 = 4.05 t/s cross-checks the server's no-spec baseline (4.09).

| model                          |       size |     params | backend    | ngl |  n_cpu_moe | ts           |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | ---------: | ------------ | --------------: | -------------------: |
| glm-dsa 744B.A40B Q3_K - Medium | 319.40 GiB |   753.86 B | ROCm       |  99 |         56 | 59.00/7.00/7.00/7.00 |             pp1 |          3.82 ± 0.57 |
| glm-dsa 744B.A40B Q3_K - Medium | 319.40 GiB |   753.86 B | ROCm       |  99 |         56 | 59.00/7.00/7.00/7.00 |             pp2 |          4.84 ± 0.38 |
| glm-dsa 744B.A40B Q3_K - Medium | 319.40 GiB |   753.86 B | ROCm       |  99 |         56 | 59.00/7.00/7.00/7.00 |             pp3 |          5.47 ± 0.33 |
| glm-dsa 744B.A40B Q3_K - Medium | 319.40 GiB |   753.86 B | ROCm       |  99 |         56 | 59.00/7.00/7.00/7.00 |             pp4 |          5.75 ± 0.23 |
| glm-dsa 744B.A40B Q3_K - Medium | 319.40 GiB |   753.86 B | ROCm       |  99 |         56 | 59.00/7.00/7.00/7.00 |             pp5 |          5.80 ± 0.20 |
| glm-dsa 744B.A40B Q3_K - Medium | 319.40 GiB |   753.86 B | ROCm       |  99 |         56 | 59.00/7.00/7.00/7.00 |             pp6 |          5.93 ± 0.18 |
| glm-dsa 744B.A40B Q3_K - Medium | 319.40 GiB |   753.86 B | ROCm       |  99 |         56 | 59.00/7.00/7.00/7.00 |             pp7 |          6.14 ± 0.19 |
| glm-dsa 744B.A40B Q3_K - Medium | 319.40 GiB |   753.86 B | ROCm       |  99 |         56 | 59.00/7.00/7.00/7.00 |             pp8 |          6.21 ± 0.18 |
| glm-dsa 744B.A40B Q3_K - Medium | 319.40 GiB |   753.86 B | ROCm       |  99 |         56 | 59.00/7.00/7.00/7.00 |             pp9 |          5.76 ± 0.18 |
| glm-dsa 744B.A40B Q3_K - Medium | 319.40 GiB |   753.86 B | ROCm       |  99 |         56 | 59.00/7.00/7.00/7.00 |            pp10 |          5.96 ± 0.12 |
| glm-dsa 744B.A40B Q3_K - Medium | 319.40 GiB |   753.86 B | ROCm       |  99 |         56 | 59.00/7.00/7.00/7.00 |            pp11 |          6.05 ± 0.09 |
| glm-dsa 744B.A40B Q3_K - Medium | 319.40 GiB |   753.86 B | ROCm       |  99 |         56 | 59.00/7.00/7.00/7.00 |            pp12 |          6.23 ± 0.16 |
| glm-dsa 744B.A40B Q3_K - Medium | 319.40 GiB |   753.86 B | ROCm       |  99 |         56 | 59.00/7.00/7.00/7.00 |            pp13 |          6.29 ± 0.17 |
| glm-dsa 744B.A40B Q3_K - Medium | 319.40 GiB |   753.86 B | ROCm       |  99 |         56 | 59.00/7.00/7.00/7.00 |            pp14 |          6.42 ± 0.12 |
| glm-dsa 744B.A40B Q3_K - Medium | 319.40 GiB |   753.86 B | ROCm       |  99 |         56 | 59.00/7.00/7.00/7.00 |            pp15 |          6.50 ± 0.15 |
| glm-dsa 744B.A40B Q3_K - Medium | 319.40 GiB |   753.86 B | ROCm       |  99 |         56 | 59.00/7.00/7.00/7.00 |            pp16 |          6.61 ± 0.16 |
| glm-dsa 744B.A40B Q3_K - Medium | 319.40 GiB |   753.86 B | ROCm       |  99 |         56 | 59.00/7.00/7.00/7.00 |            pp17 |          6.28 ± 0.08 |
| glm-dsa 744B.A40B Q3_K - Medium | 319.40 GiB |   753.86 B | ROCm       |  99 |         56 | 59.00/7.00/7.00/7.00 |           tg128 |          4.05 ± 0.04 |
| glm-dsa 744B.A40B Q3_K - Medium | 319.40 GiB |   753.86 B | ROCm       |  99 |         56 | 59.00/7.00/7.00/7.00 |      pp1 @ d512 |          3.81 ± 0.39 |
| glm-dsa 744B.A40B Q3_K - Medium | 319.40 GiB |   753.86 B | ROCm       |  99 |         56 | 59.00/7.00/7.00/7.00 |      pp2 @ d512 |          4.95 ± 0.33 |
| glm-dsa 744B.A40B Q3_K - Medium | 319.40 GiB |   753.86 B | ROCm       |  99 |         56 | 59.00/7.00/7.00/7.00 |      pp3 @ d512 |          5.43 ± 0.33 |
| glm-dsa 744B.A40B Q3_K - Medium | 319.40 GiB |   753.86 B | ROCm       |  99 |         56 | 59.00/7.00/7.00/7.00 |      pp4 @ d512 |          5.73 ± 0.26 |
| glm-dsa 744B.A40B Q3_K - Medium | 319.40 GiB |   753.86 B | ROCm       |  99 |         56 | 59.00/7.00/7.00/7.00 |      pp5 @ d512 |          5.92 ± 0.20 |
| glm-dsa 744B.A40B Q3_K - Medium | 319.40 GiB |   753.86 B | ROCm       |  99 |         56 | 59.00/7.00/7.00/7.00 |      pp6 @ d512 |          6.04 ± 0.22 |
| glm-dsa 744B.A40B Q3_K - Medium | 319.40 GiB |   753.86 B | ROCm       |  99 |         56 | 59.00/7.00/7.00/7.00 |      pp7 @ d512 |          6.24 ± 0.22 |
| glm-dsa 744B.A40B Q3_K - Medium | 319.40 GiB |   753.86 B | ROCm       |  99 |         56 | 59.00/7.00/7.00/7.00 |      pp8 @ d512 |          6.28 ± 0.16 |
| glm-dsa 744B.A40B Q3_K - Medium | 319.40 GiB |   753.86 B | ROCm       |  99 |         56 | 59.00/7.00/7.00/7.00 |      pp9 @ d512 |          5.92 ± 0.19 |
| glm-dsa 744B.A40B Q3_K - Medium | 319.40 GiB |   753.86 B | ROCm       |  99 |         56 | 59.00/7.00/7.00/7.00 |     pp10 @ d512 |          6.13 ± 0.12 |
| glm-dsa 744B.A40B Q3_K - Medium | 319.40 GiB |   753.86 B | ROCm       |  99 |         56 | 59.00/7.00/7.00/7.00 |     pp11 @ d512 |          6.21 ± 0.15 |
| glm-dsa 744B.A40B Q3_K - Medium | 319.40 GiB |   753.86 B | ROCm       |  99 |         56 | 59.00/7.00/7.00/7.00 |     pp12 @ d512 |          6.34 ± 0.09 |
| glm-dsa 744B.A40B Q3_K - Medium | 319.40 GiB |   753.86 B | ROCm       |  99 |         56 | 59.00/7.00/7.00/7.00 |     pp13 @ d512 |          6.48 ± 0.12 |
| glm-dsa 744B.A40B Q3_K - Medium | 319.40 GiB |   753.86 B | ROCm       |  99 |         56 | 59.00/7.00/7.00/7.00 |     pp14 @ d512 |          6.57 ± 0.10 |
| glm-dsa 744B.A40B Q3_K - Medium | 319.40 GiB |   753.86 B | ROCm       |  99 |         56 | 59.00/7.00/7.00/7.00 |     pp15 @ d512 |          6.65 ± 0.12 |
| glm-dsa 744B.A40B Q3_K - Medium | 319.40 GiB |   753.86 B | ROCm       |  99 |         56 | 59.00/7.00/7.00/7.00 |     pp16 @ d512 |          6.72 ± 0.11 |
| glm-dsa 744B.A40B Q3_K - Medium | 319.40 GiB |   753.86 B | ROCm       |  99 |         56 | 59.00/7.00/7.00/7.00 |     pp17 @ d512 |          6.45 ± 0.16 |
| glm-dsa 744B.A40B Q3_K - Medium | 319.40 GiB |   753.86 B | ROCm       |  99 |         56 | 59.00/7.00/7.00/7.00 |    tg128 @ d512 |          4.07 ± 0.01 |

build: 60eeeb608 (10472)
