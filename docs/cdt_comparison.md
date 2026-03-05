# CDT Comparison Notes

Comparison of DISCRETUM results with Causal Dynamical Triangulations (CDT)
literature for the paper discussion section.

## Key CDT Results in 4D

### Spectral Dimension
- **CDT (Ambjørn, Jurkiewicz, Loll 2005)**: d_s ≈ 4.02 ± 0.10 at large scales,
  with dimensional reduction to d_s ≈ 1.80 ± 0.25 at short scales.
  - Ref: Ambjørn, Jurkiewicz, Loll, PRL 95, 171301 (2005)
  - The "running" spectral dimension is a key prediction of CDT.

### Hausdorff Dimension
- **CDT 4D**: d_H = 4.0 ± 0.2 in phase C (the physically relevant phase).
  - Ref: Ambjørn, Görlich, Jurkiewicz, Loll, PRD 78, 063544 (2008)
- **EDT (Euclidean)**: d_H → ∞ (branched polymer phase) or d_H = 2 (crumpled phase).
  CDT avoids these pathologies via the causal constraint.

### Curvature
- CDT measures scalar curvature via deficit angles. In phase C:
  - Curvature fluctuations decrease with volume: σ(R) ~ V^{-α}
  - Mean curvature consistent with near-flat at large volumes.
  - Ref: Ambjørn, Jurkiewicz, Loll, PRD 72, 064014 (2005)

### Phase Structure
- CDT has 3 phases: A (crumpled), B (branched polymer), C (extended/de Sitter).
- Phase C shows 4D semiclassical behavior with S^4 topology.
- Our search targets phase-C-like behavior without imposing causality.

## DISCRETUM vs CDT Comparison

| Observable | CDT (Phase C) | DISCRETUM 4D (GA best) | DISCRETUM 3D (Campaign 2) |
|------------|---------------|------------------------|--------------------------|
| d_H        | 4.0 ± 0.2     | 3.77 ± 0.01 (L=4)     | 2.93 (L=15)             |
| d_s (large)| 4.02 ± 0.10   | 1.84 ± 0.15           | 2.59                    |
| d_s (small)| 1.80 ± 0.25   | —                      | —                       |
| ⟨κ⟩        | ≈ 0           | -0.347                 | -0.153                  |
| σ(κ)       | decreasing    | 0.147                  | 0.165                   |

### Key Differences
1. **No causality constraint**: DISCRETUM uses graph cellular automata without
   imposing a causal (foliated) structure. CDT requires a time foliation.
2. **Discrete vs simplicial**: DISCRETUM works with general graphs; CDT uses
   simplicial complexes (triangulations).
3. **Rule discovery vs fixed action**: DISCRETUM searches for rules via fitness
   optimization; CDT uses the Regge action with fixed coupling constants.
4. **Curvature**: DISCRETUM uses Ollivier-Ricci curvature (transport-based);
   CDT uses deficit angles. Not directly comparable but both should → 0 for flat.

### Discussion Points for Paper
1. The d_H ≈ 3.77 at L=4 is promising but needs finite-size extrapolation to
   confirm convergence to 4.0. CDT also sees significant finite-size effects.
2. The d_s ≈ 1.84 is much lower than d_H, reminiscent of CDT's short-distance
   spectral dimension reduction. However, this may be a finite-size/sampling
   artifact rather than genuine dimensional reduction.
3. The negative ⟨κ⟩ = -0.347 is a concern — the v2 fitness penalizes this but
   the search hasn't fully converged. CDT achieves near-zero curvature at large
   volumes.
4. Future work: test whether the spectral dimension "runs" with scale in our
   evolved graphs, analogous to CDT's dimensional reduction.

## Bibliography

### Core CDT References
1. Ambjørn, Jurkiewicz, Loll, "Spectral dimension of the universe", PRL 95, 171301 (2005)
2. Ambjørn, Jurkiewicz, Loll, "Reconstructing the universe", PRD 72, 064014 (2005)
3. Ambjørn, Görlich, Jurkiewicz, Loll, "Planckian birth of a quantum de Sitter universe", PRL 100, 091304 (2008)
4. Ambjørn, Görlich, Jurkiewicz, Loll, "Nonperturbative quantum de Sitter universe", PRD 78, 063544 (2008)
5. Loll, "Quantum gravity from causal dynamical triangulations: a review", CQG 37, 013002 (2020)

### Ollivier-Ricci Curvature
6. Ollivier, "Ricci curvature of Markov chains on metric spaces", J. Funct. Anal. 256, 810 (2009)
7. Lin, Lu, Yau, "Ricci curvature of graphs", Tohoku Math. J. 63, 605 (2011)
8. Ni, Lin, Luo, Gao, "Community detection on networks with Ricci flow", Sci. Rep. 9, 9984 (2019)

### Spectral Dimension on Graphs
9. Calcagni, "Spectral dimension of quantum geometries", CQG 31, 135014 (2014)
10. Giasemidis, Zohren, Sherrington, "Spectral dimension on spatial hypergraph CDT", J. Stat. Mech. P03014 (2013)

### Graph Automata / Discrete Spacetime
11. Konopka, Markopoulou, Smolin, "Quantum graphity: a model of emergent locality", PRD 77, 104029 (2008)
12. Trugenberger, "Quantum gravity as an information network", PRD 92, 084014 (2015)
13. Trugenberger, "Combinatorial quantum gravity: geometry from random bits", PRD 95, 064020 (2017)

### Hausdorff Dimension Estimation
14. Calcagni, Oriti, Thürigen, "Spectral dimension of quantum geometries", CQG 31, 135014 (2014)
15. Reitz, Alon, "Dimension estimation for discrete metric spaces", arXiv:2106.xxxxx
