# Kerr-Newman 3D Visualization

An interactive 3D visualizer for the **Kerr-Newman** metric (rotating and charged black hole) in Boyer-Lindquist coordinates. No geodesic included.

### Features
- Exact computation of Kerr-Newman metric quantities in geometric units (\(G = c = 1\))
- Real-time interactive 3D plot with:
  - Coordinate grid lines (r, θ, φ)
  - Inner and outer horizons
  - Ergosphere surface
  - Frame-dragging (ZAMO angular velocity) arrows
  - Ring singularity
- Adjustable parameters: Mass \(M\), Spin \(a\), Charge \(Q\), domain size, line width, and transparency

# Important Notes

Metric quantities (g_{\mu\nu}, α, ω, horizons, etc.) are calculated exactly.
Visual amplification is applied to coordinate lines for better clarity and aesthetics (not a pure isometric embedding).

This tool is designed for educational purposes and intuitive exploration of rotating charged black holes. 

#update 1: 2D Schwarzschild black hole + white hole simulation.

### Requirements
```bash
pip install numpy matplotlib
