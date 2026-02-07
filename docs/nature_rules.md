# Nature-derived junction rules

This MVP implements **the consequences** of the Nature minimal-surface framework as a fast CAD primitive library:

## 1) Sprouting vs branching transition (ρ≈0.6)

When the thin link is sufficiently thin relative to thick links:

- The thick path stays straight.
- The thin branch sprouts orthogonally.

When the thin link is thicker, steering emerges and the junction morphs toward a more symmetric configuration.

Implementation: `junctions.RHO_THRESHOLD = 0.6`

## 2) Angle steering via Ω ↔ θ mapping

The paper reports a 3D steering angle Ω related to branching angle θ through:

Ω = 4π sin²((π − θ)/4)

This MVP uses that mapping to:

- Set Ω=0 below ρ_th (sprout regime)
- Increase Ω linearly above ρ_th up to Ω(θ=120°) as ρ→1
- Invert Ω→θ to get a *target thick-link angle*

Implementation: `junctions.omega_from_theta`, `junctions.theta_from_omega`, `junctions.thick_angle_from_rho`

## 3) Trifurcation emergence (χ≈0.83)

When links are “thick enough” relative to separation, the paper observes a transition where
two bifurcations collapse into a single trifurcation around χ≈0.83.

This MVP uses a local proxy:

χ_local = circumference / link_length

If χ_local ≥ 0.83, we merge two nearby bifurcations.

Implementation: `junctions.should_merge_to_trifurcation` and a merge pass in `generator._merge_trifurcations_in_tree`.
