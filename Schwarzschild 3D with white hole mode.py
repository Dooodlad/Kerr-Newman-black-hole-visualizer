import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, CheckButtons
from matplotlib.patches import FancyArrowPatch
from matplotlib.lines import Line2D
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 (side-effect import for 3D)

# =====================================================
# CONSTANTS
# =====================================================

G = 6.67430e-11
c = 299792458.0
M_SUN = 1.9891e30

# =====================================================
# INITIAL PARAMETERS
# =====================================================

Mx = 0.0
My = 0.0
Mz = 0.0  # center z

INITIAL_MASS = 1.4
INITIAL_RADIUS = 12.0

# =====================================================
# DISPLAY SETTINGS
# =====================================================

X_MIN = -100.0
X_MAX = 100.0
Y_MIN = -100.0
Y_MAX = 100.0
Z_MIN = -100.0
Z_MAX = 100.0

NUM_LINES = 151
NUM_POINTS = 1100

# 3D full-grid resolution (kept modest for interactivity)
NUM_LINES_3D = 25    # number of discrete positions per axis where lines pass through
NUM_POINTS_3D = 200  # points per drawn grid-line

BASE_LINE_WIDTH = 0.35
BASE_CENTER_WIDTH = 1.0
BASE_ALPHA = 0.38
BASE_COLOR = "0.15"

SEGMENT_LINE_WIDTH = 1.6
SEGMENT_CENTER_WIDTH = 2.4
SEGMENT_ALPHA = 0.92

# Colors for each asymptotic region
REGION1_COLOR = "#cc6a00"  # outside H1 -> H1 infinity (orange)
REGION2_COLOR = "#9b1d60"  # H1 -> S infinity (purple)
REGION3_COLOR = "#006b8f"  # S -> H2 infinity (blue)
REGION4_COLOR = "#008c5a"  # H2 -> outside (green)

# Arrow colors matching regions
ARROW1_COLOR = "#cc6a00"
ARROW2_COLOR = "#9b1d60"
ARROW3_COLOR = "#006b8f"
ARROW4_COLOR = "#008c5a"

COLOR_INV_ALPHA_MIN = 2.0
ALPHA2_EPS = 1.0e-14

# 3D lower-res axes for full-grid drawing
x_axis_3d = np.linspace(X_MIN, X_MAX, NUM_POINTS_3D)
y_axis_3d = np.linspace(Y_MIN, Y_MAX, NUM_POINTS_3D)
z_axis_3d = np.linspace(Z_MIN, Z_MAX, NUM_POINTS_3D)
x_slices_3d = np.linspace(X_MIN, X_MAX, NUM_LINES_3D)
y_slices_3d = np.linspace(Y_MIN, Y_MAX, NUM_LINES_3D)
z_slices_3d = np.linspace(Z_MIN, Z_MAX, NUM_LINES_3D)

# =====================================================
# PHYSICS: same as original
# =====================================================

def mass_kg(m_solar):
    return float(m_solar) * M_SUN


def schwarzschild_radius_km(m_solar):
    return (2.0 * G * mass_kg(m_solar) / c**2) / 1000.0


def alpha2(u_km, m_solar):
    u_km = np.asarray(u_km, dtype=float)
    u_m = u_km * 1000.0
    m = mass_kg(m_solar)

    out = np.full_like(u_km, np.nan, dtype=float)
    valid = u_m > 0.0
    out[valid] = 1.0 - (2.0 * G * m) / (u_m[valid] * c**2)
    return out


def inv_alpha_complex(u_km, m_solar):
    a2 = alpha2(u_km, m_solar)
    z = np.full(a2.shape, np.nan + 0j, dtype=complex)

    regular = np.isfinite(a2) & (np.abs(a2) >= ALPHA2_EPS)
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        alpha = np.sqrt(a2[regular].astype(complex))
        z[regular] = 1.0 / alpha
    return z


def projected_inv_alpha(u_km, m_solar):
    a2 = alpha2(u_km, m_solar)
    z = inv_alpha_complex(u_km, m_solar)

    out = np.full(a2.shape, np.nan, dtype=float)
    real_branch = np.isfinite(a2) & (a2 > ALPHA2_EPS)
    imag_branch = np.isfinite(a2) & (a2 < -ALPHA2_EPS)

    out[real_branch] = z[real_branch].real
    out[imag_branch] = z[imag_branch].imag
    return out


def gap_sizes(rs):
    return 0.0, 0.0


# folded-branch labels
BRANCH_OUTSIDE_H1 = "outside_H1"
BRANCH_H1_TO_S = "H1_to_S"
BRANCH_S_TO_H2 = "S_to_H2"
BRANCH_OUTSIDE_H2 = "outside_H2"


def folded_radius_branch(rho, R, mass, branch):
    rho = np.asarray(rho, dtype=float)
    R = float(R)
    rs = schwarzschild_radius_km(mass)
    mapped = np.full_like(rho, np.nan, dtype=float)
    if rs <= 0.0 or R >= rs:
        return mapped
    if branch == BRANCH_OUTSIDE_H1:
        mask = rho >= rs
        if np.any(mask):
            rh = rho[mask]
            u = rs + R * ((rh - rs) / rs)
            branch_distance = rh
            factor = projected_inv_alpha(u, mass)
            mapped[mask] = +branch_distance * factor
        return mapped
    if branch == BRANCH_H1_TO_S:
        mask = rho <= rs
        if np.any(mask):
            rh = rho[mask]
            branch_distance = rs - rh
            u = rs + R * (rh / rs)
            factor = projected_inv_alpha(u, mass)
            mapped[mask] = -branch_distance * factor
        return mapped
    if branch == BRANCH_S_TO_H2:
        mask = rho <= rs
        if np.any(mask):
            rh = rho[mask]
            branch_distance = rh
            u = rs + R * ((rs - rh) / rs)
            factor = projected_inv_alpha(u, mass)
            mapped[mask] = +branch_distance * factor
        return mapped
    if branch == BRANCH_OUTSIDE_H2:
        mask = rho >= rs
        if np.any(mask):
            rh = rho[mask]
            branch_distance = rh - rs
            u = rs + R * ((rh - rs) / rs)
            factor = projected_inv_alpha(u, mass)
            mapped[mask] = -branch_distance * factor
        return mapped
    return mapped


def ordinary_radius(rho, R, mass):
    rho = np.asarray(rho, dtype=float)
    R = float(R)
    rs = schwarzschild_radius_km(mass)
    if rs <= 0.0:
        return rho
    u = np.maximum(rho, R)
    factor = projected_inv_alpha(u, mass)
    return rho * factor


def distort_points_3d(x0, y0, z0, R, mass, branch=None):
    dx = x0 - Mx
    dy = y0 - My
    dz = z0 - Mz
    rho = np.sqrt(dx**2 + dy**2 + dz**2)
    rs = schwarzschild_radius_km(mass)
    if R < rs and branch is not None:
        mapped_radius = folded_radius_branch(rho, R, mass, branch)
    else:
        mapped_radius = ordinary_radius(rho, R, mass)
    with np.errstate(divide="ignore", invalid="ignore"):
        radial_scale = np.where(rho > 0.0, mapped_radius / rho, np.nan)
    xd = Mx + dx * radial_scale
    yd = My + dy * radial_scale
    zd = Mz + dz * radial_scale
    return xd, yd, zd


def branch_u_value(rho, R, mass, branch):
    rho = np.asarray(rho, dtype=float)
    R = float(R)
    rs = schwarzschild_radius_km(mass)
    u = np.full_like(rho, np.nan, dtype=float)
    if rs <= 0.0 or R >= rs:
        return u
    if branch == BRANCH_OUTSIDE_H1:
        mask = rho >= rs
        u[mask] = rs + R * ((rho[mask] - rs) / rs)
        return u
    if branch == BRANCH_H1_TO_S:
        mask = rho <= rs
        u[mask] = rs + R * (rho[mask] / rs)
        return u
    if branch == BRANCH_S_TO_H2:
        mask = rho <= rs
        u[mask] = rs + R * ((rs - rho[mask]) / rs)
        return u
    if branch == BRANCH_OUTSIDE_H2:
        mask = rho >= rs
        u[mask] = rs + R * ((rho[mask] - rs) / rs)
        return u
    return u


def get_asymptotic_mask(rho, R, mass, branch):
    rho = np.asarray(rho, dtype=float)
    u = branch_u_value(rho, R, mass, branch)
    inva = np.abs(projected_inv_alpha(u, mass))
    return np.isfinite(inva) & (inva >= COLOR_INV_ALPHA_MIN)


def get_region_mask(rho, R, mass, region):
    rho = np.asarray(rho, dtype=float)
    rs = schwarzschild_radius_km(mass)
    if R >= rs:
        return np.full_like(rho, False, dtype=bool)
    if region == 1:
        return rho >= rs
    elif region == 2:
        return rho <= rs
    elif region == 3:
        return rho <= rs
    elif region == 4:
        return rho >= rs
    else:
        return np.full_like(rho, False, dtype=bool)


# =====================================================
# FIGURE LAYOUT & CONTROLS
# =====================================================

fig = plt.figure(figsize=(15.6, 10.0))
ax3d = fig.add_axes([0.06, 0.20, 0.62, 0.75], projection='3d')

ax_info = plt.axes([0.71, 0.47, 0.26, 0.48])
ax_checks = plt.axes([0.72, 0.19, 0.23, 0.24])
ax_region_checks = plt.axes([0.72, 0.08, 0.23, 0.10])

ax_mass = plt.axes([0.16, 0.105, 0.50, 0.03])
ax_radius = plt.axes([0.16, 0.055, 0.50, 0.03])

mass_slider = Slider(ax=ax_mass, label="Mass (Solar Masses)", valmin=0.1, valmax=20.0, valinit=INITIAL_MASS)
radius_slider = Slider(ax=ax_radius, label="Physical radius R (km)", valmin=0.01, valmax=120.0, valinit=INITIAL_RADIUS)

checks = CheckButtons(
    ax_checks,
    ["show R", "show H", "base lines", "show arrows", "show white hole"],
    [True, True, True, True, True],
)

region_checks = CheckButtons(
    ax_region_checks,
    ["Region 1 (outside→H)", "Region 2 (H→S)", "Region 3 (S→H)", "Region 4 (H→outside)"],
    [True, True, True, True],
)


def get_state():
    s = checks.get_status()
    return {
        "show_R": bool(s[0]),
        "show_H": bool(s[1]),
        "base_lines": bool(s[2]),
        "show_arrows": bool(s[3]),
        "show_white_hole": bool(s[4]),
    }


def get_region_state():
    s = region_checks.get_status()
    return {
        "region1": bool(s[0]),
        "region2": bool(s[1]),
        "region3": bool(s[2]),
        "region4": bool(s[3]),
    }


# =====================================================
# DRAW HELPERS
# =====================================================

def _plot_line_from_distorted_3d(xd, yd, zd, color, linewidth, alpha, zorder, linestyle="-"):
    # keep NaNs to allow segment breaks; ax3d.plot accepts NaNs
    good = np.isfinite(xd) & np.isfinite(yd) & np.isfinite(zd)
    if np.count_nonzero(good) < 2:
        return
    ax3d.plot(xd, yd, zd, color=color, linewidth=linewidth, alpha=alpha, linestyle=linestyle, zorder=zorder)


# =====================================================
# 3D: FULL GRID DRAWING
# =====================================================

def draw_field_3d_fullgrid(R, mass):
    ax3d.clear()
    ax3d.set_facecolor('white')
    rs = schwarzschild_radius_km(mass)
    ax3d.set_title("3D full-grid: distorted grid lines (entire volume)")
    ax3d.set_xlabel("x")
    ax3d.set_ylabel("y")
    ax3d.set_zlabel("z")
    region_state = get_region_state()
    show_white_hole = get_state()["show_white_hole"]
    show_base_lines = get_state()["base_lines"]

    region_list = [
        (1, BRANCH_OUTSIDE_H1, REGION1_COLOR),
        (2, BRANCH_H1_TO_S, BRANCH_H1_TO_S and REGION2_COLOR),
        (3, BRANCH_S_TO_H2, REGION3_COLOR),
        (4, BRANCH_OUTSIDE_H2, REGION4_COLOR),
    ]

    # Draw lines parallel to X (vary x), at every (y_slice, z_slice)
    for y0 in y_slices_3d:
        for z0 in z_slices_3d:
            # For white hole mode, only show colored segments in upper half (z>=0)
            # But base grid lines should still show everywhere
            x0 = x_axis_3d
            y_line = np.full_like(x0, y0)
            z_line = np.full_like(x0, z0)
            rho = np.sqrt((x0 - Mx)**2 + (y_line - My)**2 + (z_line - Mz)**2)
            
            # Determine visibility for colored segments (white hole mode)
            visible_colored = (z0 >= 0) if show_white_hole else np.full_like(rho, True, dtype=bool)

            any_colored = np.zeros_like(rho, dtype=bool)
            for region_num, branch, color in [
                (1, BRANCH_OUTSIDE_H1, REGION1_COLOR),
                (2, BRANCH_H1_TO_S, REGION2_COLOR),
                (3, BRANCH_S_TO_H2, REGION3_COLOR),
                (4, BRANCH_OUTSIDE_H2, REGION4_COLOR),
            ]:
                if not region_state[f"region{region_num}"]:
                    continue
                asymp_mask = get_asymptotic_mask(rho, R, mass, branch)
                region_mask = get_region_mask(rho, R, mass, region_num)
                combined = asymp_mask & region_mask & visible_colored
                if not np.any(combined):
                    continue
                any_colored |= combined
                xd, yd, zd = distort_points_3d(x0, y_line, z_line, R, mass, branch=branch)
                xd_masked = np.where(combined, xd, np.nan)
                yd_masked = np.where(combined, yd, np.nan)
                zd_masked = np.where(combined, zd, np.nan)
                _plot_line_from_distorted_3d(xd_masked, yd_masked, zd_masked, color, SEGMENT_LINE_WIDTH, SEGMENT_ALPHA, 3)

            # Neutral segments (base grid lines) - show everywhere regardless of white hole mode
            if show_base_lines:
                neutral_mask = (~any_colored) & np.isfinite(rho)
                if np.any(neutral_mask):
                    xd, yd, zd = distort_points_3d(x0, y_line, z_line, R, mass, branch=None)
                    xd_masked = np.where(neutral_mask, xd, np.nan)
                    yd_masked = np.where(neutral_mask, yd, np.nan)
                    zd_masked = np.where(neutral_mask, zd, np.nan)
                    _plot_line_from_distorted_3d(xd_masked, yd_masked, zd_masked, BASE_COLOR, BASE_LINE_WIDTH, 0.6, 1)

    # Draw lines parallel to Y (vary y), at every (x_slice, z_slice)
    for x0 in x_slices_3d:
        for z0 in z_slices_3d:
            visible_colored = (z0 >= 0) if show_white_hole else True
            y0 = y_axis_3d
            x_line = np.full_like(y0, x0)
            z_line = np.full_like(y0, z0)
            rho = np.sqrt((x_line - Mx)**2 + (y0 - My)**2 + (z_line - Mz)**2)

            any_colored = np.zeros_like(rho, dtype=bool)
            for region_num, branch, color in [
                (1, BRANCH_OUTSIDE_H1, REGION1_COLOR),
                (2, BRANCH_H1_TO_S, REGION2_COLOR),
                (3, BRANCH_S_TO_H2, REGION3_COLOR),
                (4, BRANCH_OUTSIDE_H2, REGION4_COLOR),
            ]:
                if not region_state[f"region{region_num}"]:
                    continue
                asymp_mask = get_asymptotic_mask(rho, R, mass, branch)
                region_mask = get_region_mask(rho, R, mass, region_num)
                combined = asymp_mask & region_mask & visible_colored
                if not np.any(combined):
                    continue
                any_colored |= combined
                xd, yd, zd = distort_points_3d(x_line, y0, z_line, R, mass, branch=branch)
                xd_masked = np.where(combined, xd, np.nan)
                yd_masked = np.where(combined, yd, np.nan)
                zd_masked = np.where(combined, zd, np.nan)
                _plot_line_from_distorted_3d(xd_masked, yd_masked, zd_masked, color, SEGMENT_LINE_WIDTH, SEGMENT_ALPHA, 3)

            # Neutral segments (base grid lines) - show everywhere
            if show_base_lines:
                neutral_mask = (~any_colored) & np.isfinite(rho)
                if np.any(neutral_mask):
                    xd, yd, zd = distort_points_3d(x_line, y0, z_line, R, mass, branch=None)
                    xd_masked = np.where(neutral_mask, xd, np.nan)
                    yd_masked = np.where(neutral_mask, yd, np.nan)
                    zd_masked = np.where(neutral_mask, zd, np.nan)
                    _plot_line_from_distorted_3d(xd_masked, yd_masked, zd_masked, BASE_COLOR, BASE_LINE_WIDTH, 0.6, 1)

    # Draw lines parallel to Z (vary z), at every (x_slice, y_slice)
    for x0 in x_slices_3d:
        for y0 in y_slices_3d:
            z0 = z_axis_3d
            x_line = np.full_like(z0, x0)
            y_line = np.full_like(z0, y0)
            rho = np.sqrt((x_line - Mx)**2 + (y_line - My)**2 + (z0 - Mz)**2)
            # For base grid lines, show all z values
            # For colored segments, only show z>=0 in white hole mode
            visible_colored = (z0 >= 0) if show_white_hole else np.full_like(z0, True, dtype=bool)

            any_colored = np.zeros_like(rho, dtype=bool)
            for region_num, branch, color in [
                (1, BRANCH_OUTSIDE_H1, REGION1_COLOR),
                (2, BRANCH_H1_TO_S, REGION2_COLOR),
                (3, BRANCH_S_TO_H2, REGION3_COLOR),
                (4, BRANCH_OUTSIDE_H2, REGION4_COLOR),
            ]:
                if not region_state[f"region{region_num}"]:
                    continue
                asymp_mask = get_asymptotic_mask(rho, R, mass, branch)
                region_mask = get_region_mask(rho, R, mass, region_num)
                combined = asymp_mask & region_mask & visible_colored
                if not np.any(combined):
                    continue
                any_colored |= combined
                xd, yd, zd = distort_points_3d(x_line, y_line, z0, R, mass, branch=branch)
                plot_mask = combined
                xd_masked = np.where(plot_mask, xd, np.nan)
                yd_masked = np.where(plot_mask, yd, np.nan)
                zd_masked = np.where(plot_mask, zd, np.nan)
                _plot_line_from_distorted_3d(xd_masked, yd_masked, zd_masked, color, SEGMENT_LINE_WIDTH, SEGMENT_ALPHA, 3)

            # Neutral segments (base grid lines) - show everywhere
            if show_base_lines:
                neutral_mask = (~any_colored) & np.isfinite(rho)
                if np.any(neutral_mask):
                    xd, yd, zd = distort_points_3d(x_line, y_line, z0, R, mass, branch=None)
                    xd_masked = np.where(neutral_mask, xd, np.nan)
                    yd_masked = np.where(neutral_mask, yd, np.nan)
                    zd_masked = np.where(neutral_mask, zd, np.nan)
                    _plot_line_from_distorted_3d(xd_masked, yd_masked, zd_masked, BASE_COLOR, BASE_LINE_WIDTH, 0.6, 1)

    # Draw R and horizon wireframe spheres (low-res)
    if get_state()["show_R"]:
        u = np.linspace(0, 2*np.pi, 36)
        v = np.linspace(0, np.pi, 18)
        xR = Mx + R * np.outer(np.cos(u), np.sin(v))
        yR = My + R * np.outer(np.sin(u), np.sin(v))
        zR = Mz + R * np.outer(np.ones_like(u), np.cos(v))
        ax3d.plot_wireframe(xR, yR, zR, color='blue', linewidth=0.6, alpha=0.9)

    if get_state()["show_H"]:
        u = np.linspace(0, 2*np.pi, 36)
        v = np.linspace(0, np.pi, 18)
        rs = schwarzschild_radius_km(mass)
        xH = Mx + rs * np.outer(np.cos(u), np.sin(v))
        yH = My + rs * np.outer(np.sin(u), np.sin(v))
        zH = Mz + rs * np.outer(np.ones_like(u), np.cos(v))
        ax3d.plot_wireframe(xH, yH, zH, color='red', linewidth=0.6, alpha=0.6, linestyles='dashed')

    ax3d.set_xlim(X_MIN, X_MAX)
    ax3d.set_ylim(Y_MIN, Y_MAX)
    ax3d.set_zlim(Z_MIN, Z_MAX)
    ax3d.view_init(elev=30, azim=-60)
    ax3d.grid(True)


def draw_info_panel(mass, R, rs):
    ax_info.clear()
    ax_info.axis("off")
    a2_R = float(alpha2(np.array([R]), mass)[0])
    inv_R = complex(inv_alpha_complex(np.array([R]), mass)[0])
    proj_R = float(projected_inv_alpha(np.array([R]), mass)[0])

    if R < rs:
        mode = "folded sequence: R < H"
        branch_note = (
            "Asymptotic evolution:\n"
            "Region 1 (orange): outside → H (∞ at H)\n"
            "Region 2 (purple): H → S (∞ at S)\n"
            "Region 3 (blue): S → H (∞ at H)\n"
            "Region 4 (green): H → outside → ∞\n\n"
            "Arrows show direction of evolution"
        )
    else:
        mode = "ordinary mode: R >= H"
        branch_note = "inside R uses u = R; outside uses u = rho."

    info = (
        f"Mass = {mass:.2f} M_sun\n"
        f"R = {R:.4f} km\n"
        f"H = rs = 2GM/c^2 = {rs:.4f} km\n"
        f"R/rs = {R/rs:.5f}\n"
        f"mode = {mode}\n\n"
        f"alpha^2(R) = {a2_R:.6g}\n"
        f"1/alpha(R) = {inv_R.real:.6g} {inv_R.imag:+.6g}i\n"
        f"projection(R) = {proj_R:.6g}\n\n"
        "Only field equation:\n"
        "alpha^2 = 1 - 2GM/(u c^2)\n"
        "alpha = sqrt(alpha^2 + 0j)\n"
        "1/alpha = 1/alpha\n\n"
        "Colored asymptotic lines show\n"
        "regions where 1/alpha approaches infinity\n\n"
        f"{branch_note}"
    )

    ax_info.text(0.02, 0.98, info, va="top", ha="left", fontsize=8.45, family="monospace",
                 bbox=dict(facecolor="white", alpha=0.96, edgecolor="0.7"))

    handles = [
        Line2D([0], [0], color=BASE_COLOR, lw=1.2, label="neutral grid lines"),
        Line2D([0], [0], color="blue", lw=2.0, label="R marker"),
        Line2D([0], [0], color="red", lw=2.0, ls="--", label="H = rs"),
        Line2D([0], [0], color=REGION1_COLOR, lw=2.0, label="Region 1: outside→H"),
        Line2D([0], [0], color=REGION2_COLOR, lw=2.0, label="Region 2: H→S"),
        Line2D([0], [0], color=REGION3_COLOR, lw=2.0, label="Region 3: S→H"),
        Line2D([0], [0], color=REGION4_COLOR, lw=2.0, label="Region 4: H→outside"),
    ]
    ax_info.legend(handles=handles, loc="lower left", fontsize=7.5, frameon=True)


# =====================================================
# DRAW FIELD (2D <-> 3D dispatcher)
# =====================================================

def draw_field():
    mass = float(mass_slider.val)
    R = float(radius_slider.val)
    rs = schwarzschild_radius_km(mass)

    draw_field_3d_fullgrid(R, mass)
    draw_info_panel(mass, R, rs)
    fig.canvas.draw_idle()


def update_slider(val):
    draw_field()


mass_slider.on_changed(update_slider)
radius_slider.on_changed(update_slider)
checks.on_clicked(lambda label: draw_field())
region_checks.on_clicked(lambda label: draw_field())

draw_field()
plt.show()