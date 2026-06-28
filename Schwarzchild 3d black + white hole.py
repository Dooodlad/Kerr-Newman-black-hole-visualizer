import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, CheckButtons
from matplotlib.lines import Line2D
from mpl_toolkits.mplot3d import Axes3D

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
Mz = 0.0

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

NUM_LINES = 21
NUM_POINTS = 300

BASE_LINE_WIDTH = 0.50
BASE_CENTER_WIDTH = 1.35
BASE_ALPHA = 0.38
BASE_COLOR = "0.15"

SEGMENT_LINE_WIDTH = 1.8
SEGMENT_CENTER_WIDTH = 2.8
SEGMENT_ALPHA = 0.92

# Colors
REGION1_COLOR = "#cc6a00"
REGION2_COLOR = "#9b1d60"
REGION3_COLOR = "#006b8f"
REGION4_COLOR = "#008c5a"

COLOR_INV_ALPHA_MIN = 2.0
ALPHA2_EPS = 1.0e-14

x_axis = np.linspace(X_MIN, X_MAX, NUM_POINTS)
y_axis = np.linspace(Y_MIN, Y_MAX, NUM_POINTS)
z_axis = np.linspace(Z_MIN, Z_MAX, NUM_POINTS)

x_slices = np.linspace(X_MIN, X_MAX, NUM_LINES)
y_slices = np.linspace(Y_MIN, Y_MAX, NUM_LINES)
z_slices = np.linspace(Z_MIN, Z_MAX, NUM_LINES)

# Tách riêng Z dương và Z âm
z_positive_slices = z_slices[z_slices >= 0]
z_negative_slices = z_slices[z_slices <= 0]

# =====================================================
# FIELD EQUATION
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

# =====================================================
# FOLDED SEQUENCE
# =====================================================

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

def distort_points_3d(x0, y0, z0, R, mass, branch=None, is_white_hole=False):
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
    
    if is_white_hole:
        zd = Mz - (dz * radial_scale)
    
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
# FIGURE LAYOUT
# =====================================================

fig = plt.figure(figsize=(18, 10))
ax3d = fig.add_axes([0.06, 0.15, 0.65, 0.80], projection='3d')
ax_info = plt.axes([0.73, 0.55, 0.25, 0.40])
ax_checks = plt.axes([0.73, 0.25, 0.25, 0.25])
ax_region_checks = plt.axes([0.73, 0.08, 0.25, 0.14])

ax_mass = plt.axes([0.16, 0.05, 0.50, 0.025])
ax_radius = plt.axes([0.16, 0.015, 0.50, 0.025])

mass_slider = Slider(ax_mass, "Mass (Solar Masses)", 0.1, 20.0, valinit=INITIAL_MASS)
radius_slider = Slider(ax_radius, "Physical radius R (km)", 0.01, 120.0, valinit=INITIAL_RADIUS)

checks = CheckButtons(
    ax_checks,
    ["show R (sphere)", "show H (sphere)", "base lines", "show arrows", "show white hole"],
    [True, True, True, True, True]
)

region_checks = CheckButtons(
    ax_region_checks,
    ["Region 1 (outside→H)", "Region 2 (H→S)", "Region 3 (S→H)", "Region 4 (H→outside)"],
    [True, True, True, True]
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
# DRAWING 3D
# =====================================================

def _plot_line_3d(xd, yd, zd, color, linewidth, alpha, zorder, linestyle="-"):
    if xd is None or len(xd) == 0:
        return
    good = np.isfinite(xd) & np.isfinite(yd) & np.isfinite(zd)
    if np.count_nonzero(good) < 2:
        return
    xd = np.where(good, xd, np.nan)
    yd = np.where(good, yd, np.nan)
    zd = np.where(good, zd, np.nan)
    ax3d.plot(xd, yd, zd, color=color, linewidth=linewidth, 
              alpha=alpha, linestyle=linestyle, zorder=zorder)

def _plot_grid_line_3d(x0, y0, z0, R, mass, branch, color, linewidth, alpha, zorder, 
                       mask=None, is_white_hole=False):
    xd, yd, zd = distort_points_3d(x0, y0, z0, R, mass, branch=branch, 
                                   is_white_hole=is_white_hole)
    if mask is not None:
        xd = np.where(mask, xd, np.nan)
        yd = np.where(mask, yd, np.nan)
        zd = np.where(mask, zd, np.nan)
    _plot_line_3d(xd, yd, zd, color, linewidth, alpha, zorder)

def draw_ordinary_grid_3d(R, mass):
    for z_slice in z_slices:
        for x_slice in x_slices:
            x0 = np.full_like(y_axis, x_slice)
            y0 = y_axis
            z0 = np.full_like(y_axis, z_slice)
            is_center = np.isclose(x_slice, 0.0, atol=1e-8) and np.isclose(z_slice, 0.0, atol=1e-8)
            _plot_grid_line_3d(x0, y0, z0, R, mass, None, BASE_COLOR,
                              BASE_CENTER_WIDTH if is_center else BASE_LINE_WIDTH,
                              1.0 if is_center else 0.72, 2)
        
        for y_slice in y_slices:
            x0 = x_axis
            y0 = np.full_like(x_axis, y_slice)
            z0 = np.full_like(x_axis, z_slice)
            is_center = np.isclose(y_slice, 0.0, atol=1e-8) and np.isclose(z_slice, 0.0, atol=1e-8)
            _plot_grid_line_3d(x0, y0, z0, R, mass, None, BASE_COLOR,
                              BASE_CENTER_WIDTH if is_center else BASE_LINE_WIDTH,
                              1.0 if is_center else 0.72, 2)

    for x_slice in x_slices:
        for y_slice in y_slices:
            x0 = np.full_like(z_axis, x_slice)
            y0 = np.full_like(z_axis, y_slice)
            z0 = z_axis
            is_center = np.isclose(x_slice, 0.0, atol=1e-8) and np.isclose(y_slice, 0.0, atol=1e-8)
            _plot_grid_line_3d(x0, y0, z0, R, mass, None, BASE_COLOR,
                              BASE_CENTER_WIDTH if is_center else BASE_LINE_WIDTH,
                              1.0 if is_center else 0.72, 2)

def draw_folded_base_lines_3d(R, mass):
    branches = [BRANCH_OUTSIDE_H1, BRANCH_H1_TO_S, BRANCH_S_TO_H2, BRANCH_OUTSIDE_H2]
    
    for branch in branches:
        for z_slice in z_slices:
            for x_slice in x_slices:
                x0 = np.full_like(y_axis, x_slice)
                y0 = y_axis
                z0 = np.full_like(y_axis, z_slice)
                is_center = np.isclose(x_slice, 0.0, atol=1e-8) and np.isclose(z_slice, 0.0, atol=1e-8)
                _plot_grid_line_3d(x0, y0, z0, R, mass, branch, BASE_COLOR,
                                  BASE_CENTER_WIDTH if is_center else BASE_LINE_WIDTH,
                                  BASE_ALPHA if not is_center else 0.70, 1)
            
            for y_slice in y_slices:
                x0 = x_axis
                y0 = np.full_like(x_axis, y_slice)
                z0 = np.full_like(x_axis, z_slice)
                is_center = np.isclose(y_slice, 0.0, atol=1e-8) and np.isclose(z_slice, 0.0, atol=1e-8)
                _plot_grid_line_3d(x0, y0, z0, R, mass, branch, BASE_COLOR,
                                  BASE_CENTER_WIDTH if is_center else BASE_LINE_WIDTH,
                                  BASE_ALPHA if not is_center else 0.70, 1)

        for x_slice in x_slices:
            for y_slice in y_slices:
                x0 = np.full_like(z_axis, x_slice)
                y0 = np.full_like(z_axis, y_slice)
                z0 = z_axis
                is_center = np.isclose(x_slice, 0.0, atol=1e-8) and np.isclose(y_slice, 0.0, atol=1e-8)
                _plot_grid_line_3d(x0, y0, z0, R, mass, branch, BASE_COLOR,
                                  BASE_CENTER_WIDTH if is_center else BASE_LINE_WIDTH,
                                  BASE_ALPHA if not is_center else 0.70, 1)

def draw_black_hole_lines(R, mass):
    """Vẽ lỗ đen (nửa trên Z>0)"""
    rs = schwarzschild_radius_km(mass)
    if R >= rs:
        return
    
    region_state = get_region_state()
    
    black_hole_branches = [(1, BRANCH_OUTSIDE_H1, REGION1_COLOR),
                           (2, BRANCH_H1_TO_S, REGION2_COLOR)]
    
    for region_num, branch, color in black_hole_branches:
        if not region_state[f"region{region_num}"]:
            continue
        
        # Dùng z_positive_slices đã được định nghĩa sẵn
        for z_slice in z_positive_slices:
            for x_slice in x_slices:
                x0 = np.full_like(y_axis, x_slice)
                y0 = y_axis
                z0 = np.full_like(y_axis, z_slice)
                is_center = np.isclose(x_slice, 0.0, atol=1e-8)
                
                xd, yd, zd = distort_points_3d(x0, y0, z0, R, mass, branch=branch, is_white_hole=False)
                
                rho = np.sqrt((x0 - Mx)**2 + (y0 - My)**2 + (z0 - Mz)**2)
                asymp_mask = get_asymptotic_mask(rho, R, mass, branch)
                region_mask = get_region_mask(rho, R, mass, region_num)
                combined_mask = region_mask & asymp_mask & (zd >= Mz)
                
                xd_masked = np.where(combined_mask, xd, np.nan)
                yd_masked = np.where(combined_mask, yd, np.nan)
                zd_masked = np.where(combined_mask, zd, np.nan)
                
                good = np.isfinite(xd_masked) & np.isfinite(yd_masked) & np.isfinite(zd_masked)
                if np.count_nonzero(good) < 2:
                    continue
                
                _plot_line_3d(xd_masked, yd_masked, zd_masked, color,
                             SEGMENT_LINE_WIDTH if not is_center else SEGMENT_CENTER_WIDTH,
                             SEGMENT_ALPHA, 3, '-')
            
            for y_slice in y_slices:
                x0 = x_axis
                y0 = np.full_like(x_axis, y_slice)
                z0 = np.full_like(x_axis, z_slice)
                is_center = np.isclose(y_slice, 0.0, atol=1e-8)
                
                xd, yd, zd = distort_points_3d(x0, y0, z0, R, mass, branch=branch, is_white_hole=False)
                
                rho = np.sqrt((x0 - Mx)**2 + (y0 - My)**2 + (z0 - Mz)**2)
                asymp_mask = get_asymptotic_mask(rho, R, mass, branch)
                region_mask = get_region_mask(rho, R, mass, region_num)
                combined_mask = region_mask & asymp_mask & (zd >= Mz)
                
                xd_masked = np.where(combined_mask, xd, np.nan)
                yd_masked = np.where(combined_mask, yd, np.nan)
                zd_masked = np.where(combined_mask, zd, np.nan)
                
                good = np.isfinite(xd_masked) & np.isfinite(yd_masked) & np.isfinite(zd_masked)
                if np.count_nonzero(good) < 2:
                    continue
                
                _plot_line_3d(xd_masked, yd_masked, zd_masked, color,
                             SEGMENT_LINE_WIDTH if not is_center else SEGMENT_CENTER_WIDTH,
                             SEGMENT_ALPHA, 3, '-')

        for x_slice in x_slices:
            for y_slice in y_slices:
                x0 = np.full_like(z_axis, x_slice)
                y0 = np.full_like(z_axis, y_slice)
                z0 = z_axis
                is_center = np.isclose(x_slice, 0.0, atol=1e-8) and np.isclose(y_slice, 0.0, atol=1e-8)
                
                xd, yd, zd = distort_points_3d(x0, y0, z0, R, mass, branch=branch, is_white_hole=False)
                
                rho = np.sqrt((x0 - Mx)**2 + (y0 - My)**2 + (z0 - Mz)**2)
                asymp_mask = get_asymptotic_mask(rho, R, mass, branch)
                region_mask = get_region_mask(rho, R, mass, region_num)
                combined_mask = region_mask & asymp_mask & (zd >= Mz)
                
                xd_masked = np.where(combined_mask, xd, np.nan)
                yd_masked = np.where(combined_mask, yd, np.nan)
                zd_masked = np.where(combined_mask, zd, np.nan)
                
                good = np.isfinite(xd_masked) & np.isfinite(yd_masked) & np.isfinite(zd_masked)
                if np.count_nonzero(good) < 2:
                    continue
                
                _plot_line_3d(xd_masked, yd_masked, zd_masked, color,
                             SEGMENT_LINE_WIDTH if not is_center else SEGMENT_CENTER_WIDTH,
                             SEGMENT_ALPHA, 3, '-')

def draw_white_hole_lines(R, mass):
    """Vẽ lỗ trắng (nửa dưới Z<0)"""
    rs = schwarzschild_radius_km(mass)
    if R >= rs:
        return
    
    state = get_state()
    if not state["show_white_hole"]:
        return
    
    region_state = get_region_state()
    
    white_hole_branches = [(3, BRANCH_S_TO_H2, REGION3_COLOR),
                           (4, BRANCH_OUTSIDE_H2, REGION4_COLOR)]
    
    for region_num, branch, color in white_hole_branches:
        if not region_state[f"region{region_num}"]:
            continue
        
        # Dùng z_negative_slices đã được định nghĩa sẵn
        for z_slice in z_negative_slices:
            for x_slice in x_slices:
                x0 = np.full_like(y_axis, x_slice)
                y0 = y_axis
                z0 = np.full_like(y_axis, z_slice)
                is_center = np.isclose(x_slice, 0.0, atol=1e-8)
                
                xd, yd, zd = distort_points_3d(x0, y0, z0, R, mass, branch=branch, is_white_hole=True)
                
                rho = np.sqrt((x0 - Mx)**2 + (y0 - My)**2 + (z0 - Mz)**2)
                asymp_mask = get_asymptotic_mask(rho, R, mass, branch)
                region_mask = get_region_mask(rho, R, mass, region_num)
                combined_mask = region_mask & asymp_mask & (zd <= Mz)
                
                xd_masked = np.where(combined_mask, xd, np.nan)
                yd_masked = np.where(combined_mask, yd, np.nan)
                zd_masked = np.where(combined_mask, zd, np.nan)
                
                good = np.isfinite(xd_masked) & np.isfinite(yd_masked) & np.isfinite(zd_masked)
                if np.count_nonzero(good) < 2:
                    continue
                
                _plot_line_3d(xd_masked, yd_masked, zd_masked, color,
                             SEGMENT_LINE_WIDTH if not is_center else SEGMENT_CENTER_WIDTH,
                             SEGMENT_ALPHA, 3, '-')
            
            for y_slice in y_slices:
                x0 = x_axis
                y0 = np.full_like(x_axis, y_slice)
                z0 = np.full_like(x_axis, z_slice)
                is_center = np.isclose(y_slice, 0.0, atol=1e-8)
                
                xd, yd, zd = distort_points_3d(x0, y0, z0, R, mass, branch=branch, is_white_hole=True)
                
                rho = np.sqrt((x0 - Mx)**2 + (y0 - My)**2 + (z0 - Mz)**2)
                asymp_mask = get_asymptotic_mask(rho, R, mass, branch)
                region_mask = get_region_mask(rho, R, mass, region_num)
                combined_mask = region_mask & asymp_mask & (zd <= Mz)
                
                xd_masked = np.where(combined_mask, xd, np.nan)
                yd_masked = np.where(combined_mask, yd, np.nan)
                zd_masked = np.where(combined_mask, zd, np.nan)
                
                good = np.isfinite(xd_masked) & np.isfinite(yd_masked) & np.isfinite(zd_masked)
                if np.count_nonzero(good) < 2:
                    continue
                
                _plot_line_3d(xd_masked, yd_masked, zd_masked, color,
                             SEGMENT_LINE_WIDTH if not is_center else SEGMENT_CENTER_WIDTH,
                             SEGMENT_ALPHA, 3, '-')

        for x_slice in x_slices:
            for y_slice in y_slices:
                x0 = np.full_like(z_axis, x_slice)
                y0 = np.full_like(z_axis, y_slice)
                z0 = z_axis
                is_center = np.isclose(x_slice, 0.0, atol=1e-8) and np.isclose(y_slice, 0.0, atol=1e-8)
                
                xd, yd, zd = distort_points_3d(x0, y0, z0, R, mass, branch=branch, is_white_hole=True)
                
                rho = np.sqrt((x0 - Mx)**2 + (y0 - My)**2 + (z0 - Mz)**2)
                asymp_mask = get_asymptotic_mask(rho, R, mass, branch)
                region_mask = get_region_mask(rho, R, mass, region_num)
                combined_mask = region_mask & asymp_mask & (zd <= Mz)
                
                xd_masked = np.where(combined_mask, xd, np.nan)
                yd_masked = np.where(combined_mask, yd, np.nan)
                zd_masked = np.where(combined_mask, zd, np.nan)
                
                good = np.isfinite(xd_masked) & np.isfinite(yd_masked) & np.isfinite(zd_masked)
                if np.count_nonzero(good) < 2:
                    continue
                
                _plot_line_3d(xd_masked, yd_masked, zd_masked, color,
                             SEGMENT_LINE_WIDTH if not is_center else SEGMENT_CENTER_WIDTH,
                             SEGMENT_ALPHA, 3, '-')

def draw_evolution_arrows_3d(R, mass):
    if not get_state()["show_arrows"]:
        return
    
    rs = schwarzschild_radius_km(mass)
    if R >= rs:
        return
    
    region_state = get_region_state()
    state = get_state()
    
    # Black hole arrows (upper half)
    if region_state["region1"]:
        for theta in np.linspace(0.2, np.pi/2-0.2, 3):
            for phi in np.linspace(0, 2*np.pi, 6):
                start_r = rs * 1.8
                end_r = rs * 1.1
                x_start = Mx + start_r * np.sin(theta) * np.cos(phi)
                y_start = My + start_r * np.sin(theta) * np.sin(phi)
                z_start = Mz + start_r * np.cos(theta)
                x_end = Mx + end_r * np.sin(theta) * np.cos(phi)
                y_end = My + end_r * np.sin(theta) * np.sin(phi)
                z_end = Mz + end_r * np.cos(theta)
                ax3d.plot([x_start, x_end], [y_start, y_end], [z_start, z_end],
                         color=REGION1_COLOR, linewidth=1.5, alpha=0.5, zorder=10)
    
    if region_state["region2"]:
        for theta in np.linspace(0.2, np.pi/2-0.2, 3):
            for phi in np.linspace(0, 2*np.pi, 6):
                start_r = rs * 0.9
                end_r = rs * 0.4
                x_start = Mx + start_r * np.sin(theta) * np.cos(phi)
                y_start = My + start_r * np.sin(theta) * np.sin(phi)
                z_start = Mz + start_r * np.cos(theta)
                x_end = Mx + end_r * np.sin(theta) * np.cos(phi)
                y_end = My + end_r * np.sin(theta) * np.sin(phi)
                z_end = Mz + end_r * np.cos(theta)
                ax3d.plot([x_start, x_end], [y_start, y_end], [z_start, z_end],
                         color=REGION2_COLOR, linewidth=1.5, alpha=0.5, zorder=10)
    
    # White hole arrows (lower half)
    if state["show_white_hole"]:
        if region_state["region3"]:
            for theta in np.linspace(np.pi/2+0.2, np.pi-0.2, 3):
                for phi in np.linspace(0, 2*np.pi, 6):
                    start_r = rs * 0.4
                    end_r = rs * 0.9
                    x_start = Mx + start_r * np.sin(theta) * np.cos(phi)
                    y_start = My + start_r * np.sin(theta) * np.sin(phi)
                    z_start = Mz + start_r * np.cos(theta)
                    x_end = Mx + end_r * np.sin(theta) * np.cos(phi)
                    y_end = My + end_r * np.sin(theta) * np.sin(phi)
                    z_end = Mz + end_r * np.cos(theta)
                    ax3d.plot([x_start, x_end], [y_start, y_end], [z_start, z_end],
                             color=REGION3_COLOR, linewidth=1.5, alpha=0.5, zorder=10)
        
        if region_state["region4"]:
            for theta in np.linspace(np.pi/2+0.2, np.pi-0.2, 3):
                for phi in np.linspace(0, 2*np.pi, 6):
                    start_r = rs * 1.1
                    end_r = rs * 1.8
                    x_start = Mx + start_r * np.sin(theta) * np.cos(phi)
                    y_start = My + start_r * np.sin(theta) * np.sin(phi)
                    z_start = Mz + start_r * np.cos(theta)
                    x_end = Mx + end_r * np.sin(theta) * np.cos(phi)
                    y_end = My + end_r * np.sin(theta) * np.sin(phi)
                    z_end = Mz + end_r * np.cos(theta)
                    ax3d.plot([x_start, x_end], [y_start, y_end], [z_start, z_end],
                             color=REGION4_COLOR, linewidth=1.5, alpha=0.5, zorder=10)

def draw_info_panel_3d(mass, R, rs):
    ax_info.clear()
    ax_info.axis("off")
    
    a2_R = float(alpha2(np.array([R]), mass)[0])
    inv_R = complex(inv_alpha_complex(np.array([R]), mass)[0])
    proj_R = float(projected_inv_alpha(np.array([R]), mass)[0])
    
    info = (
        f"3D BLACK HOLE + WHITE HOLE\n"
        f"{'='*35}\n"
        f"Mass = {mass:.2f} M_sun\n"
        f"R = {R:.4f} km\n"
        f"H = rs = {rs:.4f} km\n"
        f"R/rs = {R/rs:.5f}\n\n"
        f"alpha^2(R) = {a2_R:.6g}\n"
        f"1/alpha(R) = {inv_R.real:.6g} {inv_R.imag:+.6g}i\n"
        f"projection(R) = {proj_R:.6g}\n\n"
        "UPPER HALF (Z>0): BLACK HOLE\n"
        "  Region 1 (orange): outside → H\n"
        "  Region 2 (purple): H → S\n"
        "  Vectors point INWARD\n\n"
        "LOWER HALF (Z<0): WHITE HOLE\n"
        "  Region 3 (blue): S → H\n"
        "  Region 4 (green): H → outside\n"
        "  Vectors point OUTWARD\n"
        "  Grid is EXPANDED by 1/alpha"
    )
    
    ax_info.text(0.02, 0.98, info, va="top", ha="left", fontsize=8.0,
                family="monospace", bbox=dict(facecolor="white", alpha=0.96, edgecolor="0.7"))
    
    handles = [
        Line2D([0], [0], color=BASE_COLOR, lw=1.2, label="grid lines"),
        Line2D([0], [0], color="blue", lw=2.0, label="R sphere"),
        Line2D([0], [0], color="red", lw=2.0, ls="--", label="H = rs"),
        Line2D([0], [0], color=REGION1_COLOR, lw=2.0, label="Region 1"),
        Line2D([0], [0], color=REGION2_COLOR, lw=2.0, label="Region 2"),
        Line2D([0], [0], color=REGION3_COLOR, lw=2.0, label="Region 3 (white)"),
        Line2D([0], [0], color=REGION4_COLOR, lw=2.0, label="Region 4 (white)"),
    ]
    ax_info.legend(handles=handles, loc="lower left", fontsize=7.0, frameon=True)

def draw_field_3d():
    ax3d.clear()
    
    mass = float(mass_slider.val)
    R = float(radius_slider.val)
    rs = schwarzschild_radius_km(mass)
    state = get_state()
    
    if R < rs:
        if state["base_lines"]:
            draw_folded_base_lines_3d(R, mass)
        draw_black_hole_lines(R, mass)
        if state["show_white_hole"]:
            draw_white_hole_lines(R, mass)
        draw_evolution_arrows_3d(R, mass)
    else:
        draw_ordinary_grid_3d(R, mass)
    
    # Draw disk at z=0
    if R < rs:
        theta = np.linspace(0, 2*np.pi, 30)
        r_disk = np.linspace(0, rs, 15)
        T, R_disk = np.meshgrid(theta, r_disk)
        x_disk = R_disk * np.cos(T)
        y_disk = R_disk * np.sin(T)
        z_disk = np.zeros_like(x_disk)
        ax3d.plot_surface(x_disk, y_disk, z_disk, color='gray', alpha=0.1, zorder=5)
    
    # R sphere
    if state["show_R"]:
        u = np.linspace(0, 2*np.pi, 15)
        v = np.linspace(0, np.pi, 15)
        x_sphere = R * np.outer(np.cos(u), np.sin(v)) + Mx
        y_sphere = R * np.outer(np.sin(u), np.sin(v)) + My
        z_sphere = R * np.outer(np.ones(np.size(u)), np.cos(v)) + Mz
        ax3d.plot_surface(x_sphere, y_sphere, z_sphere, color='blue', alpha=0.10, zorder=8)
    
    # H sphere
    if state["show_H"]:
        u = np.linspace(0, 2*np.pi, 15)
        v = np.linspace(0, np.pi, 15)
        x_sphere = rs * np.outer(np.cos(u), np.sin(v)) + Mx
        y_sphere = rs * np.outer(np.sin(u), np.sin(v)) + My
        z_sphere = rs * np.outer(np.ones(np.size(u)), np.cos(v)) + Mz
        ax3d.plot_surface(x_sphere, y_sphere, z_sphere, color='red', alpha=0.08, zorder=7)
        ax3d.plot_wireframe(x_sphere, y_sphere, z_sphere, color='red', alpha=0.2, linewidth=0.5)
    
    ax3d.scatter([Mx], [My], [Mz], color="black", s=50, zorder=9)
    
    ax3d.set_xlabel("X")
    ax3d.set_ylabel("Y")
    ax3d.set_zlabel("Z")
    ax3d.set_xlim(X_MIN, X_MAX)
    ax3d.set_ylim(Y_MIN, Y_MAX)
    ax3d.set_zlim(Z_MIN, Z_MAX)
    ax3d.set_title(f"3D Black + White Hole: M={mass:.2f} M_sun, R={R:.2f} km")
    ax3d.set_box_aspect([1, 1, 1])
    
    draw_info_panel_3d(mass, R, rs)
    fig.canvas.draw_idle()

# =====================================================
# EVENT HANDLERS
# =====================================================

mass_slider.on_changed(lambda val: draw_field_3d())
radius_slider.on_changed(lambda val: draw_field_3d())
checks.on_clicked(lambda label: draw_field_3d())
region_checks.on_clicked(lambda label: draw_field_3d())

draw_field_3d()
plt.show()