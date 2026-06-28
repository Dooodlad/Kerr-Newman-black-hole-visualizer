"""
KERR-NEWMAN EXACT BOYER-LINDQUIST METRIC VISUALIZER
====================================================

This version removes the old toy-model radial deformation.
It computes the Kerr-Newman metric quantities in geometric units G = c = 1:

    Sigma = r^2 + a^2 cos^2(theta)
    Delta = r^2 - 2 M r + a^2 + Q^2
    A     = (r^2 + a^2)^2 - a^2 Delta sin^2(theta)

Metric components in Boyer-Lindquist coordinates (-,+,+,+):

    g_tt     = -(Delta - a^2 sin^2(theta)) / Sigma
             = -(1 - (2 M r - Q^2) / Sigma)
    g_tphi   = -a sin^2(theta) (2 M r - Q^2) / Sigma
    g_rr     = Sigma / Delta
    g_thetatheta = Sigma
    g_phiphi = A sin^2(theta) / Sigma

3+1 quantities outside the horizon / where Delta > 0:

    alpha^2 = Delta Sigma / A
    omega   = -g_tphi / g_phiphi = a (2 M r - Q^2) / A

Important limitation:
The plot is an exact Boyer-Lindquist coordinate visualization of surfaces,
coordinate lines, horizons, ergosphere, and frame-dragging field values.
It is NOT an isometric embedding of the full 4D Lorentzian spacetime into 3D
Euclidean space, because that is not generally possible.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, CheckButtons

EPS = 1e-12


# =====================================================================
# EXACT KERR-NEWMAN FUNCTIONS, GEOMETRIC UNITS G = c = 1
# =====================================================================

def kn_quantities(r, theta, M, a, Q):
    """Return exact Kerr-Newman metric quantities in Boyer-Lindquist coordinates."""
    r = np.asarray(r, dtype=float)
    theta = np.asarray(theta, dtype=float)

    sin_t = np.sin(theta)
    cos_t = np.cos(theta)
    sin2 = sin_t**2

    Sigma = r**2 + a**2 * cos_t**2
    Delta = r**2 - 2.0 * M * r + a**2 + Q**2
    A = (r**2 + a**2)**2 - a**2 * Delta * sin2

    Sigma_safe = np.where(np.abs(Sigma) < EPS, np.nan, Sigma)
    Delta_safe = np.where(np.abs(Delta) < EPS, np.nan, Delta)
    A_safe = np.where(np.abs(A) < EPS, np.nan, A)

    g_tt = -(Delta - a**2 * sin2) / Sigma_safe
    g_tphi = -a * sin2 * (2.0 * M * r - Q**2) / Sigma_safe
    g_rr = Sigma_safe / Delta_safe
    g_thetatheta = Sigma
    g_phiphi = A * sin2 / Sigma_safe

    # Exact ADM lapse in Boyer-Lindquist split. Real outside regions where Delta > 0.
    alpha_sq = Delta * Sigma_safe / A_safe
    alpha = np.where(alpha_sq >= 0, np.sqrt(alpha_sq), np.nan)
    inv_alpha = np.where(alpha_sq > 0, 1.0 / np.sqrt(alpha_sq), np.nan)

    # Exact ZAMO angular velocity / frame dragging.
    omega = a * (2.0 * M * r - Q**2) / A_safe

    return {
        "Sigma": Sigma,
        "Delta": Delta,
        "A": A,
        "g_tt": g_tt,
        "g_tphi": g_tphi,
        "g_rr": g_rr,
        "g_thetatheta": g_thetatheta,
        "g_phiphi": g_phiphi,
        "alpha": alpha,
        "inv_alpha": inv_alpha,
        "omega": omega,
    }


def horizons(M, a, Q):
    """Return r_plus, r_minus, discriminant, status."""
    disc = M**2 - a**2 - Q**2
    tol = 1e-12

    if disc > tol:
        root = np.sqrt(disc)
        return M + root, M - root, disc, "Regular Kerr-Newman black hole"
    if abs(disc) <= tol:
        return M, M, disc, "Extremal Kerr-Newman black hole"
    return None, None, disc, "Over-extremal: no horizon / naked singularity"


def ergosurface_outer(theta, M, a, Q):
    """Outer stationary-limit surface r_E+(theta)."""
    val = M**2 - a**2 * np.cos(theta)**2 - Q**2
    return np.where(val >= 0, M + np.sqrt(np.maximum(val, 0.0)), np.nan)


def bl_to_cartesian(r, theta, phi, a):
    """
    Auxiliary Boyer-Lindquist -> Cartesian-like map:
        x = sqrt(r^2 + a^2) sin(theta) cos(phi)
        y = sqrt(r^2 + a^2) sin(theta) sin(phi)
        z = r cos(theta)

    This is the standard oblate-spheroidal coordinate picture used for display.
    It is not an isometric embedding of the full spacetime metric.
    """
    Rxy = np.sqrt(np.maximum(r**2 + a**2, 0.0)) * np.sin(theta)
    x = Rxy * np.cos(phi)
    y = Rxy * np.sin(phi)
    z = r * np.cos(theta)
    return x, y, z


def visual_space_pull(x, y, z, r, theta, phi, M, a, Q,
                      radial_strength=1.15, twist_strength=1.35):
    """
    Visual-only amplification of Kerr-Newman spatial distortion.

    The metric quantities remain exact. This function only bends the displayed
    coordinate lines so the 3D picture shows stronger radial stretching and
    frame-dragging twist. Horizons are not transformed by this function, so r+
    and r- stay as simple reference rings.
    """
    q = kn_quantities(r, theta, M, a, Q)

    Sigma = q["Sigma"]
    Delta = q["Delta"]
    A = q["A"]

    # Use |g_rr|^(1/2) = sqrt(Sigma/|Delta|) as a spatial stretching indicator.
    # This remains finite numerically except extremely close to Delta=0.
    stretch = np.sqrt(np.abs(Sigma) / (np.abs(Delta) + EPS))
    pull = 1.0 / (1.0 + radial_strength * np.log1p(stretch))

    # Keep the central singular ring from swallowing the whole drawing.
    pull = np.where(np.isfinite(pull), pull, 1.0)

    # Exact omega, but visually scaled into a display twist.
    omega = a * (2.0 * M * r - Q**2) / (A + EPS)
    twist = twist_strength * omega * np.exp(-0.18 * np.maximum(r, 0.0))
    twist = np.where(np.isfinite(twist), twist, 0.0)

    cosw = np.cos(twist)
    sinw = np.sin(twist)

    x2 = pull * (x * cosw - y * sinw)
    y2 = pull * (x * sinw + y * cosw)
    z2 = pull * z
    return x2, y2, z2


# =====================================================================
# DRAWING HELPERS
# =====================================================================

def draw_constant_r_rings(ax, r0, a, color, linewidth, alpha, label, linestyle="-"):
    """Draw 3 principal intersections of a constant-r Boyer-Lindquist surface."""
    if r0 is None or not np.isfinite(r0) or r0 < 0:
        return

    t = np.linspace(0.0, 2.0 * np.pi, 360)

    # Equatorial ring: theta = pi/2
    x, y, z = bl_to_cartesian(r0, np.pi / 2.0, t, a)
    ax.plot(x, y, z, color=color, linewidth=linewidth, alpha=alpha, linestyle=linestyle, label=label)

    # Meridian in XZ: phi = 0
    theta = t
    x, y, z = bl_to_cartesian(r0, theta, 0.0, a)
    ax.plot(x, y, z, color=color, linewidth=linewidth, alpha=alpha, linestyle=linestyle)

    # Meridian in YZ: phi = pi/2
    x, y, z = bl_to_cartesian(r0, theta, np.pi / 2.0, a)
    ax.plot(x, y, z, color=color, linewidth=linewidth, alpha=alpha, linestyle=linestyle)


def draw_ring_singularity(ax, a, linewidth=1.1, alpha=0.55):
    """Kerr-Newman ring singularity: r=0, theta=pi/2, radius |a|. Point if a=0."""
    if abs(a) < 1e-8:
        ax.scatter([0], [0], [0], color="black", s=70, alpha=alpha,
                   edgecolor="gold", linewidth=0.8, label="Singularity: r=0")
        return

    phi = np.linspace(0.0, 2.0 * np.pi, 360)
    x = abs(a) * np.cos(phi)
    y = abs(a) * np.sin(phi)
    z = np.zeros_like(phi)
    ax.plot(x, y, z, color="black", linewidth=linewidth, alpha=alpha,
            label="Ring singularity: r=0, theta=pi/2")


def plot_bl_coordinate_grid(ax, M, a, Q, r_max, n_r=9, n_theta=13, n_phi=18,
                            show_r=True, show_theta=True, show_phi=True,
                            line_width=0.75, transparency=0.55,
                            radial_strength=1.15, twist_strength=1.35):
    """Plot exact BL coordinate lines in the auxiliary oblate-spheroidal display."""
    r_min = max(0.05, 0.015 * r_max)
    r_vals = np.linspace(r_min, r_max, n_r)
    theta_vals = np.linspace(0.12, np.pi - 0.12, n_theta)
    phi_vals = np.linspace(0.0, 2.0 * np.pi, n_phi, endpoint=False)

    black = "black"

    # r-coordinate lines: theta, phi fixed; r varies.
    if show_r:
        r_line = np.linspace(r_min, r_max, 240)
        for th in theta_vals:
            for ph in phi_vals[::2]:
                x, y, z = bl_to_cartesian(r_line, th, ph, a)
                x, y, z = visual_space_pull(
                    x, y, z, r_line, th, ph, M, a, Q,
                    radial_strength=radial_strength,
                    twist_strength=twist_strength,
                )
                ax.plot(x, y, z, color=black, linewidth=line_width, alpha=transparency)

    # theta-coordinate lines: r, phi fixed; theta varies.
    if show_theta:
        th_line = np.linspace(0.02, np.pi - 0.02, 240)
        for rr in r_vals:
            for ph in phi_vals[::2]:
                x, y, z = bl_to_cartesian(rr, th_line, ph, a)
                x, y, z = visual_space_pull(
                    x, y, z, rr, th_line, ph, M, a, Q,
                    radial_strength=radial_strength,
                    twist_strength=twist_strength,
                )
                ax.plot(x, y, z, color=black, linewidth=line_width * 0.9,
                        alpha=transparency * 0.85, linestyle="--")

    # phi-coordinate lines: r, theta fixed; phi varies.
    if show_phi:
        ph_line = np.linspace(0.0, 2.0 * np.pi, 240)
        for rr in r_vals:
            for th in theta_vals[1:-1:2]:
                x, y, z = bl_to_cartesian(rr, th, ph_line, a)
                x, y, z = visual_space_pull(
                    x, y, z, rr, th, ph_line, M, a, Q,
                    radial_strength=radial_strength,
                    twist_strength=twist_strength,
                )
                ax.plot(x, y, z, color=black, linewidth=line_width * 0.9,
                        alpha=transparency * 0.75, linestyle=":")


def plot_ergosurface(ax, M, a, Q, r_max, linewidth=0.65, alpha=0.32):
    """Plot outer ergosurface as exact r_E+(theta), if present in the domain."""
    if abs(a) < 1e-10:
        return

    theta = np.linspace(0.0, np.pi, 72)
    phi = np.linspace(0.0, 2.0 * np.pi, 72)
    TH, PH = np.meshgrid(theta, phi, indexing="ij")
    RE = ergosurface_outer(TH, M, a, Q)

    if np.all(~np.isfinite(RE)):
        return

    mask = np.isfinite(RE) & (RE > 0) & (RE <= r_max * 1.05)
    RE_plot = np.where(mask, RE, np.nan)

    X, Y, Z = bl_to_cartesian(RE_plot, TH, PH, a)
    ax.plot_wireframe(X, Y, Z, rstride=5, cstride=5,
                      color="purple", linewidth=linewidth, alpha=alpha,
                      label="Outer ergosurface: g_tt = 0")


def plot_frame_dragging_arrows(ax, M, a, Q, r_max, r_plus, alpha=0.55):
    """Plot normalized exact ZAMO frame-dragging angular velocity omega."""
    if abs(a) < 1e-10:
        return

    start_r = max(0.35, (r_plus * 1.08 if r_plus is not None else 0.45))
    if start_r >= r_max * 0.9:
        return

    r_vals = np.linspace(start_r, r_max * 0.82, 4)
    phi_vals = np.linspace(0.0, 2.0 * np.pi, 12, endpoint=False)
    theta = np.pi / 2.0

    points = []
    vectors = []
    magnitudes = []

    for rr in r_vals:
        for ph in phi_vals:
            x, y, z = bl_to_cartesian(rr, theta, ph, a)
            q = kn_quantities(rr, theta, M, a, Q)
            omega = float(q["omega"])
            if not np.isfinite(omega):
                continue
            tangent = np.array([-np.sin(ph), np.cos(ph), 0.0])
            points.append([x, y, z])
            vectors.append(tangent * omega)
            magnitudes.append(abs(omega))

    if not points:
        return

    points = np.array(points)
    vectors = np.array(vectors)
    magnitudes = np.array(magnitudes)
    max_mag = np.nanmax(magnitudes)
    if not np.isfinite(max_mag) or max_mag < EPS:
        return

    # Normalize vector length for display only; omega values are exact, arrows are scaled.
    display_vectors = vectors / max_mag * (0.06 * r_max)
    ax.quiver(points[:, 0], points[:, 1], points[:, 2],
              display_vectors[:, 0], display_vectors[:, 1], display_vectors[:, 2],
              color="darkgreen", linewidth=0.8, alpha=alpha, arrow_length_ratio=0.35,
              label="Frame dragging omega = -g_tphi/g_phiphi")


def finite_str(x, digits=4):
    if x is None or not np.isfinite(x):
        return "nan"
    return f"{x:.{digits}g}"


# =====================================================================
# MAIN PLOT
# =====================================================================

def plot_kerr_newman_exact(ax, M, a, Q, r_max, line_width=0.75, transparency=0.55,
                           show_r=True, show_theta=True, show_phi=True,
                           show_horizon=True, show_inner=True,
                           show_ergo=True, show_drag=True):
    ax.clear()

    r_plus, r_minus, disc, status = horizons(M, a, Q)

    plot_bl_coordinate_grid(
        ax, M, a, Q, r_max,
        n_r=9, n_theta=13, n_phi=18,
        show_r=show_r, show_theta=show_theta, show_phi=show_phi,
        line_width=line_width, transparency=transparency,
        radial_strength=1.15, twist_strength=1.35,
    )

    if show_horizon and r_plus is not None and 0.0 <= r_plus <= r_max:
        draw_constant_r_rings(ax, r_plus, a, color="red", linewidth=1.15, alpha=0.45,
                              label=f"Outer horizon r+ = {r_plus:.4f}", linestyle="-")

    if show_inner and r_minus is not None and r_minus > 0.0 and r_minus <= r_max:
        draw_constant_r_rings(ax, r_minus, a, color="orange", linewidth=0.9, alpha=0.38,
                              label=f"Inner horizon r- = {r_minus:.4f}", linestyle="--")

    if show_ergo:
        plot_ergosurface(ax, M, a, Q, r_max)

    if show_drag:
        plot_frame_dragging_arrows(ax, M, a, Q, r_max, r_plus)

    draw_ring_singularity(ax, a)

    # Coordinate axes, auxiliary display only.
    axis_scale = r_max * 0.78
    ax.plot([-axis_scale, axis_scale], [0, 0], [0, 0], color="black", linewidth=1.2,
            alpha=0.45, linestyle="-", label="X display axis")
    ax.plot([0, 0], [-axis_scale, axis_scale], [0, 0], color="black", linewidth=1.2,
            alpha=0.45, linestyle="--", label="Y display axis")
    ax.plot([0, 0], [0, 0], [-axis_scale, axis_scale], color="black", linewidth=1.2,
            alpha=0.45, linestyle=":", label="Z spin axis")

    ax.set_xlabel("X auxiliary BL display")
    ax.set_ylabel("Y auxiliary BL display")
    ax.set_zlabel("Z spin axis")

    lim = r_max
    # BL constant-r surfaces have equatorial display radius sqrt(r^2+a^2).
    lim = max(lim, np.sqrt(r_max**2 + a**2))
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_zlim(-r_max, r_max)
    ax.set_box_aspect([1, 1, max(0.55, r_max / lim)])

    ax.grid(True, alpha=0.16, linestyle="--", linewidth=0.45)
    ax.xaxis.pane.set_facecolor((0.96, 0.96, 0.96, 0.35))
    ax.yaxis.pane.set_facecolor((0.96, 0.96, 0.96, 0.35))
    ax.zaxis.pane.set_facecolor((0.96, 0.96, 0.96, 0.35))

    title = "Kerr-Newman metric quantities with amplified spatial grid pull\n"
    title += f"M = {M:.3f}, a = {a:.3f}, Q = {Q:.3f} | {status}"
    ax.set_title(title, fontsize=11, fontweight="bold", pad=16)

    # Probe metric at equator outside r+ if possible.
    if r_plus is not None:
        r_probe = min(r_max * 0.82, max(r_plus + 0.25, r_max * 0.45))
    else:
        r_probe = r_max * 0.55
    theta_probe = np.pi / 2.0
    q = kn_quantities(r_probe, theta_probe, M, a, Q)

    info = []
    info.append("Exact KN metric, G=c=1")
    info.append(f"Delta = r^2 - 2Mr + a^2 + Q^2")
    info.append(f"disc = M^2-a^2-Q^2 = {disc:.5g}")
    info.append(f"r+ = {finite_str(r_plus)}")
    info.append(f"r- = {finite_str(r_minus)}")
    if abs(a) > 1e-10:
        info.append("Ergosurface: rE+(theta)=M+sqrt(M^2-a^2 cos^2theta-Q^2)")
    else:
        info.append("a=0: no distinct ergosphere")
    info.append("")
    info.append(f"Metric probe: r={r_probe:.3g}, theta=pi/2")
    info.append(f"g_tt     = {finite_str(float(q['g_tt']))}")
    info.append(f"g_tphi   = {finite_str(float(q['g_tphi']))}")
    info.append(f"g_rr     = {finite_str(float(q['g_rr']))}")
    info.append(f"g_phiphi = {finite_str(float(q['g_phiphi']))}")
    info.append(f"alpha    = {finite_str(float(q['alpha']))}")
    info.append(f"omega    = {finite_str(float(q['omega']))}")
    info.append("")
    info.append("Grid: visually amplified KN pull/twist")
    info.append("Horizon rings: unwarped r+ and r- references")
    info.append("Note: 3D view is not an isometric embedding.")

    fig = ax.figure
    fig.text(0.755, 0.86, "\n".join(info), fontsize=8.5, family="monospace",
             va="top", ha="left",
             bbox=dict(boxstyle="round", facecolor="white", edgecolor="gray", alpha=0.92, pad=0.75))

    ax.legend(loc="upper left", bbox_to_anchor=(1.03, 0.60), fontsize=7.6,
              framealpha=0.92, borderpad=0.6)


# =====================================================================
# INTERACTIVE PROGRAM
# =====================================================================

def update_plot(val=None):
    M = slider_m.val
    a = slider_a.val
    Q = slider_Q.val
    r_max = slider_r.val
    line_width = slider_lw.val
    transparency = slider_alpha.val

    states = check_boxes.get_status()
    show_r = states[0]
    show_theta = states[1]
    show_phi = states[2]
    show_horizon = states[3]
    show_inner = states[4]
    show_ergo = states[5]
    show_drag = states[6]

    plot_kerr_newman_exact(
        ax, M, a, Q, r_max,
        line_width=line_width,
        transparency=transparency,
        show_r=show_r,
        show_theta=show_theta,
        show_phi=show_phi,
        show_horizon=show_horizon,
        show_inner=show_inner,
        show_ergo=show_ergo,
        show_drag=show_drag,
    )
    fig.canvas.draw_idle()


if __name__ == "__main__":
    fig = plt.figure(figsize=(18.5, 11.0))
    ax = fig.add_subplot(111, projection="3d")
    plt.subplots_adjust(bottom=0.34, left=0.05, right=0.73, top=0.91)

    # Physics sliders
    ax_m = plt.axes([0.12, 0.27, 0.28, 0.022], facecolor="white")
    ax_a = plt.axes([0.12, 0.23, 0.28, 0.022], facecolor="white")
    ax_Q = plt.axes([0.12, 0.19, 0.28, 0.022], facecolor="white")
    ax_r = plt.axes([0.12, 0.15, 0.28, 0.022], facecolor="white")

    slider_m = Slider(ax_m, "Mass M", 0.3, 20, valinit=1.0, valstep=0.05,
                      color="black", track_color="lightgray")
    slider_a = Slider(ax_a, "Spin a", 0.0, 20, valinit=0.6, valstep=0.05,
                      color="black", track_color="lightgray")
    slider_Q = Slider(ax_Q, "Charge Q", 0.0, 20, valinit=0.0, valstep=0.05,
                      color="black", track_color="lightgray")
    slider_r = Slider(ax_r, "Domain r_max", 2.0, 25.0, valinit=8.0, valstep=0.5,
                      color="black", track_color="lightgray")

    # Visual sliders
    ax_lw = plt.axes([0.52, 0.27, 0.20, 0.022], facecolor="white")
    ax_alpha = plt.axes([0.52, 0.23, 0.20, 0.022], facecolor="white")

    slider_lw = Slider(ax_lw, "Line Width", 0.25, 2.2, valinit=0.75, valstep=0.05,
                       color="black", track_color="lightgray")
    slider_alpha = Slider(ax_alpha, "Grid Alpha", 0.15, 1.0, valinit=0.55, valstep=0.05,
                          color="black", track_color="lightgray")

    for s in (slider_m, slider_a, slider_Q, slider_r, slider_lw, slider_alpha):
        s.on_changed(update_plot)

    # Checkboxes
    ax_check = plt.axes([0.12, 0.035, 0.31, 0.095])
    ax_check.axis("off")
    check_boxes = CheckButtons(
        ax_check,
        ("r-coordinate lines",
         "theta-coordinate lines",
         "phi-coordinate lines",
         "Outer horizon r+",
         "Inner horizon r-",
         "Outer ergosurface",
         "Frame dragging omega"),
        (True, True, True, True, True, True, True),
    )

    try:
        check_boxes.set_frame_props({"linewidth": 1.1, "edgecolor": "black"})
        check_boxes.set_check_props({"color": "black"})
    except Exception:
        pass

    check_boxes.on_clicked(lambda label: update_plot())

    update_plot()
    plt.show()
