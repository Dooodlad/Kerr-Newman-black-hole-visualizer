import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, CheckButtons
from matplotlib.patches import Circle, FancyArrowPatch
from matplotlib.lines import Line2D
from matplotlib.patches import ArrowStyle

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

INITIAL_MASS = 1.4
INITIAL_RADIUS = 12.0

# =====================================================
# DISPLAY SETTINGS
# =====================================================

X_MIN = -100.0
X_MAX = 100.0
Y_MIN = -100.0
Y_MAX = 100.0

NUM_LINES = 151
NUM_POINTS = 1100

BASE_LINE_WIDTH = 0.50
BASE_CENTER_WIDTH = 1.35
BASE_ALPHA = 0.38
BASE_COLOR = "0.15"

SEGMENT_LINE_WIDTH = 1.8
SEGMENT_CENTER_WIDTH = 2.8
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

HORIZON_GAP_FRAC = 0.0
CENTER_GAP_FRAC = 0.0
MIN_GAP_KM = 1.0e-4

# Colored segments are selected only by the value of 1/alpha(u_branch)
COLOR_INV_ALPHA_MIN = 2.0  # Lower threshold to show more of the asymptotic approach

ALPHA2_EPS = 1.0e-14

x_axis = np.linspace(X_MIN, X_MAX, NUM_POINTS)
y_axis = np.linspace(Y_MIN, Y_MAX, NUM_POINTS)
x_slices = np.linspace(X_MIN, X_MAX, NUM_LINES)
y_slices = np.linspace(Y_MIN, Y_MAX, NUM_LINES)

# =====================================================
# THE ONLY FIELD EQUATION
# =====================================================

def mass_kg(m_solar):
    return float(m_solar) * M_SUN


def schwarzschild_radius_km(m_solar):
    return (2.0 * G * mass_kg(m_solar) / c**2) / 1000.0


def alpha2(u_km, m_solar):
    """
    The only field equation used by this file:

        alpha^2 = 1 - 2GM/(u c^2)

    u is the branch-local radial variable fed into the equation, in km.
    """
    u_km = np.asarray(u_km, dtype=float)
    u_m = u_km * 1000.0
    m = mass_kg(m_solar)

    out = np.full_like(u_km, np.nan, dtype=float)
    valid = u_m > 0.0
    out[valid] = 1.0 - (2.0 * G * m) / (u_m[valid] * c**2)
    return out


def inv_alpha_complex(u_km, m_solar):
    """
        alpha = sqrt(alpha^2 + 0j)
        1/alpha = 1 / alpha

    No finite fake infinity is inserted.
    """
    a2 = alpha2(u_km, m_solar)
    z = np.full(a2.shape, np.nan + 0j, dtype=complex)

    regular = np.isfinite(a2) & (np.abs(a2) >= ALPHA2_EPS)

    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        alpha = np.sqrt(a2[regular].astype(complex))
        z[regular] = 1.0 / alpha

    return z


def projected_inv_alpha(u_km, m_solar):
    """
    Project complex 1/alpha into the 2D drawing plane.

    alpha^2 >= 0: use Re(1/alpha)
    alpha^2 <  0: use Im(1/alpha), naturally negative.
    """
    a2 = alpha2(u_km, m_solar)
    z = inv_alpha_complex(u_km, m_solar)

    out = np.full(a2.shape, np.nan, dtype=float)
    real_branch = np.isfinite(a2) & (a2 > ALPHA2_EPS)
    imag_branch = np.isfinite(a2) & (a2 < -ALPHA2_EPS)

    out[real_branch] = z[real_branch].real
    out[imag_branch] = z[imag_branch].imag
    return out


def gap_sizes(rs):
    # No avoidance gaps at H or S.
    h_gap = 0.0
    s_gap = 0.0
    return h_gap, s_gap


# =====================================================
# FOLDED SEQUENCE
# =====================================================

BRANCH_OUTSIDE_H1 = "outside_H1"
BRANCH_H1_TO_S = "H1_to_S"
BRANCH_S_TO_H2 = "S_to_H2"
BRANCH_OUTSIDE_H2 = "outside_H2"


def folded_radius_branch(rho, R, mass, branch):
    """
    Four-branch folded sequence.

        outside H1 -> infinity
        touch H1 -> reset 0

        H1 -> S -> infinity
        touch S -> reset 0

        S -> H2 -> infinity
        touch H2 -> reset 0

        H2 -> outside

    Every branch uses the same alpha equation.  Branch differences are only:
        mask, branch_distance, u, sign.
    """
    rho = np.asarray(rho, dtype=float)
    R = float(R)
    rs = schwarzschild_radius_km(mass)

    mapped = np.full_like(rho, np.nan, dtype=float)

    if rs <= 0.0 or R >= rs:
        return mapped

    h_gap, s_gap = gap_sizes(rs)

    if branch == BRANCH_OUTSIDE_H1:
        mask = rho >= rs
        if np.any(mask):
            rh = rho[mask]
            # R enters only through u for 1/alpha.
            # H1 is the asymptote: rh -> rs+ gives u -> rs.
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
            # R enters only through u for 1/alpha.
            # H2 is the asymptote: rh -> rs+ gives u -> rs.
            u = rs + R * ((rh - rs) / rs)
            factor = projected_inv_alpha(u, mass)
            mapped[mask] = -branch_distance * factor
        return mapped

    return mapped


def ordinary_radius(rho, R, mass):
    """
    Ordinary-object mode when R >= rs.
    Inside the physical body marker, use u = R; outside use u = rho.
    """
    rho = np.asarray(rho, dtype=float)
    R = float(R)
    rs = schwarzschild_radius_km(mass)

    if rs <= 0.0:
        return rho

    u = np.maximum(rho, R)
    factor = projected_inv_alpha(u, mass)
    return rho * factor


def distort_points(x0, y0, R, mass, branch=None):
    dx = x0 - Mx
    dy = y0 - My
    rho = np.sqrt(dx**2 + dy**2)

    rs = schwarzschild_radius_km(mass)

    if R < rs and branch is not None:
        mapped_radius = folded_radius_branch(rho, R, mass, branch)
    else:
        mapped_radius = ordinary_radius(rho, R, mass)

    with np.errstate(divide="ignore", invalid="ignore"):
        radial_scale = np.where(rho > 0.0, mapped_radius / rho, np.nan)

    xd = Mx + dx * radial_scale
    yd = My + dy * radial_scale
    return xd, yd


def branch_u_value(rho, R, mass, branch):
    """
    Return u_branch for the branch.

    This is used both for geometry and for coloring.  No separate spatial
    selection around H or S is used.
    """
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
    """
    Get mask for asymptotic region where 1/alpha is large.
    """
    rho = np.asarray(rho, dtype=float)
    u = branch_u_value(rho, R, mass, branch)
    inva = np.abs(projected_inv_alpha(u, mass))
    return np.isfinite(inva) & (inva >= COLOR_INV_ALPHA_MIN)


def get_region_mask(rho, R, mass, region):
    """
    Create radial mask for each region.
    Region 1: outside H (rho >= rs) - approaching H from outside
    Region 2: inside H to S (rho <= rs) - approaching S from H side
    Region 3: S to H (rho <= rs) - approaching H from S side
    Region 4: H to outside (rho >= rs) - going from H to infinity
    """
    rho = np.asarray(rho, dtype=float)
    rs = schwarzschild_radius_km(mass)
    
    if R >= rs:
        return np.full_like(rho, False, dtype=bool)
    
    if region == 1:  # outside H1 -> approaching H1
        return rho >= rs
    elif region == 2:  # H1 -> approaching S
        return rho <= rs
    elif region == 3:  # S -> approaching H2
        return rho <= rs
    elif region == 4:  # H2 -> going outside
        return rho >= rs
    else:
        return np.full_like(rho, False, dtype=bool)


# =====================================================
# FIGURE LAYOUT
# =====================================================

fig = plt.figure(figsize=(15.6, 10.0))
ax = plt.axes([0.06, 0.20, 0.62, 0.75])
ax_info = plt.axes([0.71, 0.47, 0.26, 0.48])
ax_checks = plt.axes([0.72, 0.19, 0.23, 0.24])
ax_region_checks = plt.axes([0.72, 0.08, 0.23, 0.10])

ax_mass = plt.axes([0.16, 0.105, 0.50, 0.03])
ax_radius = plt.axes([0.16, 0.055, 0.50, 0.03])

mass_slider = Slider(
    ax=ax_mass,
    label="Mass (Solar Masses)",
    valmin=0.1,
    valmax=20.0,
    valinit=INITIAL_MASS,
)

radius_slider = Slider(
    ax=ax_radius,
    label="Physical radius R (km)",
    valmin=0.01,
    valmax=120.0,
    valinit=INITIAL_RADIUS,
)

checks = CheckButtons(
    ax_checks,
    ["show R", "show H", "base lines", "show arrows"],
    [True, True, True, True],
)

# Checkboxes for each region
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
# DRAWING
# =====================================================

def _plot_line_from_distorted(xd, yd, color, linewidth, alpha, zorder, linestyle="-"):
    good = np.isfinite(xd) & np.isfinite(yd)
    if np.count_nonzero(good) < 2:
        return
    xd = np.where(good, xd, np.nan)
    yd = np.where(good, yd, np.nan)
    ax.plot(
        xd,
        yd,
        color=color,
        linewidth=linewidth,
        alpha=alpha,
        linestyle=linestyle,
        zorder=zorder,
        solid_capstyle="round",
    )


def _plot_original_grid_line(x0, y0, R, mass, branch, color, linewidth, alpha, zorder, mask=None):
    """
    Draw a line from the original grid after the selected branch distortion.

    If mask is provided, only that segment is drawn.  The segment is still taken
    from the same original grid line, then distorted by the same branch map.
    """
    xd, yd = distort_points(x0, y0, R, mass, branch=branch)

    if mask is not None:
        xd = np.where(mask, xd, np.nan)
        yd = np.where(mask, yd, np.nan)

    _plot_line_from_distorted(xd, yd, color, linewidth, alpha, zorder)


def draw_ordinary_grid(R, mass):
    for x_slice in x_slices:
        x0 = np.full_like(y_axis, x_slice)
        y0 = y_axis
        is_center = np.isclose(x_slice, 0.0, atol=1e-8)
        _plot_original_grid_line(
            x0, y0, R, mass, branch=None,
            color=BASE_COLOR,
            linewidth=BASE_CENTER_WIDTH if is_center else BASE_LINE_WIDTH,
            alpha=1.0 if is_center else 0.72,
            zorder=2,
        )

    for y_slice in y_slices:
        x0 = x_axis
        y0 = np.full_like(x_axis, y_slice)
        is_center = np.isclose(y_slice, 0.0, atol=1e-8)
        _plot_original_grid_line(
            x0, y0, R, mass, branch=None,
            color=BASE_COLOR,
            linewidth=BASE_CENTER_WIDTH if is_center else BASE_LINE_WIDTH,
            alpha=1.0 if is_center else 0.72,
            zorder=2,
        )


def draw_folded_base_lines(R, mass):
    """
    Draw the full folded sequence as neutral lines only.
    These are the lines that later receive colored segments.
    """
    branches = [
        BRANCH_OUTSIDE_H1,
        BRANCH_H1_TO_S,
        BRANCH_S_TO_H2,
        BRANCH_OUTSIDE_H2,
    ]

    for branch in branches:
        for x_slice in x_slices:
            x0 = np.full_like(y_axis, x_slice)
            y0 = y_axis
            is_center = np.isclose(x_slice, 0.0, atol=1e-8)
            _plot_original_grid_line(
                x0, y0, R, mass, branch=branch,
                color=BASE_COLOR,
                linewidth=BASE_CENTER_WIDTH if is_center else BASE_LINE_WIDTH,
                alpha=BASE_ALPHA if not is_center else 0.70,
                zorder=1,
            )

        for y_slice in y_slices:
            x0 = x_axis
            y0 = np.full_like(x_axis, y_slice)
            is_center = np.isclose(y_slice, 0.0, atol=1e-8)
            _plot_original_grid_line(
                x0, y0, R, mass, branch=branch,
                color=BASE_COLOR,
                linewidth=BASE_CENTER_WIDTH if is_center else BASE_LINE_WIDTH,
                alpha=BASE_ALPHA if not is_center else 0.70,
                zorder=1,
            )


def draw_asymptotic_lines(R, mass):
    """
    Draw colored asymptotic lines for each region.
    These lines show the 1/alpha behavior approaching infinity at boundaries.
    """
    rs = schwarzschild_radius_km(mass)
    if R >= rs:
        return
    
    region_state = get_region_state()
    
    # Define branches and their corresponding regions with evolution direction
    region_config = [
        (1, BRANCH_OUTSIDE_H1, REGION1_COLOR, "Region 1", "→ H"),
        (2, BRANCH_H1_TO_S, REGION2_COLOR, "Region 2", "→ S"),
        (3, BRANCH_S_TO_H2, REGION3_COLOR, "Region 3", "→ H"),
        (4, BRANCH_OUTSIDE_H2, REGION4_COLOR, "Region 4", "→ ∞"),
    ]
    
    for region_num, branch, color, name, direction in region_config:
        if not region_state[f"region{region_num}"]:
            continue
            
        # For each grid line, draw the asymptotic segment
        for x_slice in x_slices:
            x0 = np.full_like(y_axis, x_slice)
            y0 = y_axis
            is_center = np.isclose(x_slice, 0.0, atol=1e-8)
            
            # Get the distorted points
            xd, yd = distort_points(x0, y0, R, mass, branch=branch)
            
            # Create mask for this region based on radial distance
            rho = np.sqrt((x0 - Mx)**2 + (y0 - My)**2)
            
            # Get asymptotic mask (where 1/alpha is large)
            asymp_mask = get_asymptotic_mask(rho, R, mass, branch)
            
            # Get region mask
            region_mask = get_region_mask(rho, R, mass, region_num)
            
            # Combine masks
            combined_mask = region_mask & asymp_mask
            
            # Apply mask
            xd_masked = np.where(combined_mask, xd, np.nan)
            yd_masked = np.where(combined_mask, yd, np.nan)
            
            # Only draw if we have enough points
            good = np.isfinite(xd_masked) & np.isfinite(yd_masked)
            if np.count_nonzero(good) < 2:
                continue
                
            _plot_line_from_distorted(
                xd_masked, yd_masked,
                color=color,
                linewidth=SEGMENT_LINE_WIDTH if not is_center else SEGMENT_CENTER_WIDTH,
                alpha=SEGMENT_ALPHA if not is_center else 1.0,
                zorder=3,
                linestyle='-'
            )
        
        # Do the same for y-slices
        for y_slice in y_slices:
            x0 = x_axis
            y0 = np.full_like(x_axis, y_slice)
            is_center = np.isclose(y_slice, 0.0, atol=1e-8)
            
            xd, yd = distort_points(x0, y0, R, mass, branch=branch)
            
            rho = np.sqrt((x0 - Mx)**2 + (y0 - My)**2)
            
            asymp_mask = get_asymptotic_mask(rho, R, mass, branch)
            region_mask = get_region_mask(rho, R, mass, region_num)
            combined_mask = region_mask & asymp_mask
            
            xd_masked = np.where(combined_mask, xd, np.nan)
            yd_masked = np.where(combined_mask, yd, np.nan)
            
            good = np.isfinite(xd_masked) & np.isfinite(yd_masked)
            if np.count_nonzero(good) < 2:
                continue
                
            _plot_line_from_distorted(
                xd_masked, yd_masked,
                color=color,
                linewidth=SEGMENT_LINE_WIDTH if not is_center else SEGMENT_CENTER_WIDTH,
                alpha=SEGMENT_ALPHA if not is_center else 1.0,
                zorder=3,
                linestyle='-'
            )


def draw_evolution_arrows(R, mass):
    """
    Draw arrows showing the evolution direction for each region.
    Arrows point in the radial direction of the asymptotic approach.
    """
    rs = schwarzschild_radius_km(mass)
    if R >= rs:
        return
    
    if not get_state()["show_arrows"]:
        return
    
    region_state = get_region_state()
    
    # Define arrow positions for each region (at different angles)
    angles = np.linspace(0, 2*np.pi, 8, endpoint=False)
    arrow_radius = rs * 1.2  # Position arrows near the horizon
    
    # Region 1: outside → H (arrows point inward)
    if region_state["region1"]:
        for angle in angles:
            # Start from outside, point toward H
            start_r = rs * 1.8
            end_r = rs * 1.1
            x_start = Mx + start_r * np.cos(angle)
            y_start = My + start_r * np.sin(angle)
            x_end = Mx + end_r * np.cos(angle)
            y_end = My + end_r * np.sin(angle)
            
            arrow = FancyArrowPatch(
                (x_start, y_start), (x_end, y_end),
                arrowstyle='->,head_width=4,head_length=6',
                color=ARROW1_COLOR,
                linewidth=1.5,
                alpha=0.7,
                zorder=10
            )
            ax.add_patch(arrow)
    
    # Region 2: H → S (arrows point inward toward S)
    if region_state["region2"]:
        s_radius = rs * 0.5  # S is at some fraction of rs
        for angle in angles:
            start_r = rs * 0.9
            end_r = s_radius * 1.1
            x_start = Mx + start_r * np.cos(angle)
            y_start = My + start_r * np.sin(angle)
            x_end = Mx + end_r * np.cos(angle)
            y_end = My + end_r * np.sin(angle)
            
            arrow = FancyArrowPatch(
                (x_start, y_start), (x_end, y_end),
                arrowstyle='->,head_width=4,head_length=6',
                color=ARROW2_COLOR,
                linewidth=1.5,
                alpha=0.7,
                zorder=10
            )
            ax.add_patch(arrow)
    
    # Region 3: S → H (arrows point outward toward H)
    if region_state["region3"]:
        s_radius = rs * 0.5
        for angle in angles:
            start_r = s_radius * 1.1
            end_r = rs * 0.9
            x_start = Mx + start_r * np.cos(angle)
            y_start = My + start_r * np.sin(angle)
            x_end = Mx + end_r * np.cos(angle)
            y_end = My + end_r * np.sin(angle)
            
            arrow = FancyArrowPatch(
                (x_start, y_start), (x_end, y_end),
                arrowstyle='->,head_width=4,head_length=6',
                color=ARROW3_COLOR,
                linewidth=1.5,
                alpha=0.7,
                zorder=10
            )
            ax.add_patch(arrow)
    
    # Region 4: H → outside (arrows point outward)
    if region_state["region4"]:
        for angle in angles:
            start_r = rs * 1.1
            end_r = rs * 1.8
            x_start = Mx + start_r * np.cos(angle)
            y_start = My + start_r * np.sin(angle)
            x_end = Mx + end_r * np.cos(angle)
            y_end = My + end_r * np.sin(angle)
            
            arrow = FancyArrowPatch(
                (x_start, y_start), (x_end, y_end),
                arrowstyle='->,head_width=4,head_length=6',
                color=ARROW4_COLOR,
                linewidth=1.5,
                alpha=0.7,
                zorder=10
            )
            ax.add_patch(arrow)


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

    ax_info.text(
        0.02,
        0.98,
        info,
        va="top",
        ha="left",
        fontsize=8.45,
        family="monospace",
        bbox=dict(facecolor="white", alpha=0.96, edgecolor="0.7"),
    )

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


def draw_field():
    ax.clear()

    mass = float(mass_slider.val)
    R = float(radius_slider.val)
    rs = schwarzschild_radius_km(mass)
    state = get_state()

    if R < rs:
        if state["base_lines"]:
            draw_folded_base_lines(R, mass)
        # Draw the asymptotic lines on top
        draw_asymptotic_lines(R, mass)
        # Draw evolution arrows
        draw_evolution_arrows(R, mass)
    else:
        draw_ordinary_grid(R, mass)

    if state["show_R"]:
        ax.add_patch(Circle((Mx, My), R, fill=False, linewidth=2.0, color="blue", zorder=8))

    if state["show_H"]:
        ax.add_patch(Circle((Mx, My), rs, fill=False, linestyle="--", linewidth=2.0, color="red", zorder=8))

    ax.scatter([Mx], [My], color="black", s=42, zorder=9)

    ax.set_title("Asymptotic 1/alpha lines with evolution arrows")
    ax.set_xlabel("x coordinate")
    ax.set_ylabel("y coordinate")
    ax.set_xlim(X_MIN, X_MAX)
    ax.set_ylim(Y_MIN, Y_MAX)
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.14)

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