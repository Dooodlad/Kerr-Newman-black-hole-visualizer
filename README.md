# Kerr-Newman 3D Visualization

An interactive 2D and 3D visualizer for the **Kerr-Newman** metric (rotating and charged black hole) in Boyer-Lindquist coordinates. No geodesic included. 

### Features
- Exact computation of Kerr-Newman metric quantities NOT in geometric units (\(G is G, c is c, and not = 1\))
- Real-time interactive 3D plot with:
  - Coordinate grid lines (x, y, z)
  - Inner and outer horizons
  - Ergosphere surface
  - Frame-dragging (ZAMO angular velocity) arrows
  - Ring singularity
- Adjustable parameters: Mass \(M\), Spin \(a\), Charge \(Q\), domain size, line width, and transparency

# Important Notes

Metric quantities (g_{\mu\nu}, α, ω, horizons, etc.) are calculated exactly.
Visual amplification is applied to coordinate lines for better clarity and aesthetics (not a pure isometric embedding).
Grid calculated using the 1/alpha variable. 

This tool is designed for educational purposes and intuitive exploration of rotating charged black holes. 

#update 1: 2D Schwarzschild black hole + white hole simulation.
#update 2: 3D Schwarzschild black hole + white hole simulation. 

### Requirements
```bash
pip install numpy matplotlib
