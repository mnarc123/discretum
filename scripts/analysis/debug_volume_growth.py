#!/usr/bin/env python3
"""Debug: compute volume growth on 4D lattice analytically and compare."""
import numpy as np

def lattice_4d_volume_growth(L):
    """Exact volume growth for open-boundary 4D lattice L^4."""
    N = L**4
    # Build adjacency: node (x,y,z,w) connects to ±1 in each dim
    coords = []
    for w in range(L):
        for z in range(L):
            for y in range(L):
                for x in range(L):
                    coords.append((x, y, z, w))

    # BFS from center node
    center = (L//2, L//2, L//2, L//2)
    center_id = center[0] + L*center[1] + L**2*center[2] + L**3*center[3]

    # Compute distances from center using L1 (Manhattan) distance
    # On a 4D grid, geodesic distance = Manhattan distance
    dists = []
    for c in coords:
        d = sum(abs(a - b) for a, b in zip(c, center))
        dists.append(d)

    max_d = max(dists)
    vol = [0] * (max_d + 1)
    for d in dists:
        vol[d] += 1

    # Cumulative
    cumvol = np.cumsum(vol)

    print(f"\nL={L}, N={N}, center={center}")
    print(f"  {'r':>3}  {'N(r)':>8}  {'log r':>8}  {'log N(r)':>10}  {'local d_H':>10}")
    for r in range(1, min(max_d+1, L)):
        log_r = np.log(r)
        log_Nr = np.log(cumvol[r])
        if r > 1:
            local_dH = (np.log(cumvol[r]) - np.log(cumvol[r-1])) / (np.log(r) - np.log(r-1))
        else:
            local_dH = float('nan')
        print(f"  {r:>3}  {cumvol[r]:>8}  {log_r:>8.3f}  {log_Nr:>10.3f}  {local_dH:>10.3f}")

    # Fit in range [1, L//3] — before boundary effects
    fit_end = max(2, L // 3)
    log_r = np.array([np.log(r) for r in range(1, fit_end+1)])
    log_Nr = np.array([np.log(cumvol[r]) for r in range(1, fit_end+1)])
    if len(log_r) >= 2:
        coeffs = np.polyfit(log_r, log_Nr, 1)
        print(f"  Fit [1,{fit_end}]: d_H = {coeffs[0]:.3f}")

    # Fit in range [1, diam/2]
    fit_end2 = max(2, max_d // 2)
    log_r2 = np.array([np.log(r) for r in range(1, fit_end2+1) if cumvol[r] > 1])
    log_Nr2 = np.array([np.log(cumvol[r]) for r in range(1, fit_end2+1) if cumvol[r] > 1])
    if len(log_r2) >= 2:
        coeffs2 = np.polyfit(log_r2, log_Nr2, 1)
        print(f"  Fit [1,{fit_end2}]: d_H = {coeffs2[0]:.3f} (current method)")

for L in [4, 5, 6, 7, 8]:
    lattice_4d_volume_growth(L)
