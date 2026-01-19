""" Some Python utilities for general use
"""
import os, sys, re
import traceback

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from tqdm import trange, tqdm

from scipy.optimize import curve_fit, least_squares
from scipy.special import erfc

import adios2

def Try(func, verbose=True):
    try:
        func()
        if verbose: print(f'{func.__name__:<15} -> Succeed')
    except Exception as exc:
        print(f'{func.__name__:<15} -> Failed')
        print(f"Error: {exc}")
        print("Traceback details:")
        traceback.print_exc()

#--- plotting helpers
MPL_COLORS = matplotlib.rcParams['axes.prop_cycle'].by_key()['color']

def get_mpl_color(n, idx, cycle=None):
    if cycle is None:
        cycle = matplotlib.rcParams['axes.prop_cycle'].by_key()['color']
        
    if idx >= 0:
        return cycle[idx%len(cycle)] 
    else:
        return cycle[(n+idx)%len(cycle)]

#--- basic XGC IO helpers
def gen_arr(start, interval, number):
    return np.arange(start, start + interval*number, interval)

def get_existing_steps(fdir='./', header='3d'):
    existing_dirs = []
    for (dirpath, dirnames, filenames) in os.walk(fdir):
        existing_dirs.extend(dirnames)
        break

    existing = sorted([x for x in dirnames if f'xgc.{header}' in x], key=lambda x: int(''.join(filter(str.isdigit, x))))

    existing_steps = []
    regex = re.compile(r'\d+')
    for _indx, _dir in enumerate(existing):
        try:
            existing_steps.append(int(regex.findall(_dir)[-1]))
        except:
            pass
    
    return np.array(existing_steps)

def closest_existing_step(step_in, fdir='./', header='3d'):
    existing_steps = get_existing_steps(fdir, header)
    step_out = existing_steps[np.argmin(abs(existing_steps-step_in))]
    return step_out

# select N steps around 'step_base' in 'existing_steps'
def select_step_window(existing_steps, step_base, N):
    istep_base = np.argmin(abs(existing_steps-step_base))
    
    half = N//2 # divide and round down
    start = max(0, istep_base-half)
    end = start+N-1

    # if 'end' goes past the total, shift window backward
    if end > len(existing_steps)-1:
        end = len(existing_steps)-1
        start = max(0, end-N+1)

    return existing_steps[start:end+1]

def convert_time_to_step(xr, tm, target_time): # target_time in [ms]
    step_tm = xr.od.step[tm]
    time_tm = xr.od.time[tm]

    _target_time = time_tm[np.argmin(abs(time_tm-(np.array(target_time)/1e3)))]*1e3 # [ms]
    _itm = np.argmin(abs(time_tm-_target_time/1e3))
    _istep = tm[_itm]    
    return _istep

def convert_step_to_time(xr, tm, target_step): # []
    step_tm = xr.od.step[tm]
    time_tm = xr.od.time[tm]

    _target_step = step_tm[np.argmin(abs(step_tm-np.array(target_step)))]
    _itm = np.argmin(abs(step_tm-_target_step))
    _itime = tm[_itm]
    return _itime

def get_adios2_var(filestr, varstr):
    f=adios2.FileReader(filestr)
    #f.__next__()
    var=f.read(varstr)
    f.close()
    return var

def get_adios2_var_step(filestr, varstr, step):
    with adios2.FileReader(filestr) as f:
        vars=f.available_variables()
        shape_str = vars[varstr].get("Shape")
    
        if shape_str == "": # scalar
            count = []
            start = []
        else:
            count = [int(i) for i in shape_str.split(',') if i.strip()]
            start = [0] * len(count)
    
        data = f.read(varstr, start=start, count=count, step_selection=[step, 1])
        
    return data

def get_adios2_var_allstep(filestr, varstr):
    with adios2.FileReader(filestr) as f:
        vars=f.available_variables()
        stc=vars[varstr].get("AvailableStepsCount")
        ct=vars[varstr].get("Shape")
        stc=int(stc)

        if ct!='':
            c=[int(i) for i in ct.split(',')]  #
            if len(c)==1 :
                return np.reshape(f.read(varstr, start=[0],    count=c, step_selection=[0,stc]), [stc, c[0]])
            elif len(c)==2 :
                return np.reshape(f.read(varstr, start=[0,0],  count=c, step_selection=[0,stc]), [stc, c[0], c[1]])
            elif ( len(c)==3 ):
                return np.reshape(f.read(varstr, start=[0,0,0],count=c, step_selection=[0,stc]), [stc, c[0], c[1], c[2]])
        else:
            return f.read(varstr, step_selection=[0,stc])

def get_3d_array(dir_run, step, varstr, header='3d', op_name="0th plane", verbose=True):
    operations = {"0th plane": lambda v: v[0,:],
                  "mean"     : lambda v: np.mean(v, axis=0),
                  "w/o n=0"  : lambda v: v[0,:]-np.mean(v, axis=0) }

    var = get_adios2_var(f"{dir_run}/xgc.{header}.{step:05}.bp", varstr)

    op_name_used = "no op."
    if var.ndim==2:
        # some arrays are [inode, iphi]. Force it to be [iphi, inode]
        if var.shape[1] < var.shape[0]:
            var = var.T

        op_name_used = op_name
        # op_name_used = "mean"
        # op_name_used = "w/o n=0"
    
        var = operations[op_name_used](var) # One of 1) "0th plane", 2) "mean", and 3) "w/o n=0"
        
        if verbose: print(f"reading f3d file: {varstr}, operation [{op_name_used}]")
    else:
        if verbose: print(f"reading f3d file: {varstr}, operation [{op_name_used}] (`no op.` is forced as `var.ndim != 2`)")
    
    return var, op_name_used

def get_f3d_components(dir_run, step, varstr, op_name="0th plane", verbose=False):
    #--- Read xgc.f3d file for suffixes
    sufxs = ["_n0_f0", "_n0_df", "_turb_f0", "_turb_df"] # suffixes
    descs = [r"$\bar{f}_{A}$", r"$\bar{f}_{NA}$", r"$\tilde{f}_{A}$", r"$\tilde{f}_{NA}$"] # descriptions for each suffixes
    colors = ["tab:blue", "tab:orange", "tab:green", "tab:red"]
    
    # Container
    var = {}
    total = None  # placeholder for accumulating the total
    
    # Load and annotate
    for _sufx, _desc, _c in zip(sufxs, descs, colors):
        try:
            _data, _op_name = get_3d_array(dir_run, step, varstr+_sufx, header="f3d", op_name=op_name, verbose=verbose)
            
            # simple sanity checks
            if _data.ndim != 1:       raise ValueError("The target values from `xgc.f3d` are expected to be toroidal averaged quantity")                
            if np.isnan(_data).any(): raise ValueError("NaN detected")
            
            var[_sufx[1:]] = { # strip leading underscore for cleaner keys
                "data": _data,
                "desc": _desc,
                "c": _c
            }

            # Accumulate total
            total = _data if total is None else total + _data
            
        except FileNotFoundError:
            print(f"Suffix '{_sufx}' not found for {varstr}")

    # Add total if any component was found
    if total is not None:
        var["total"] = {
            "data": total,
            "desc": "Total",
            "c": "k"
        }
    
    return var

def get_OMP_index(xr, isurf, LFS=True):
    inodes = xr.mesh.surf_idx[isurf, 0:xr.mesh.surf_len[isurf]]-1
    if LFS:
        msk_lfs = (xr.mesh.r[inodes] > xr.eq_axis_r)
    else:
        msk_lfs = (xr.mesh.r[inodes] < xr.eq_axis_r)
    
    idx = np.argmin(abs(xr.mesh.z[inodes[msk_lfs]]-xr.eq_axis_z))
    idx_omp = inodes[msk_lfs][idx]
    inode_idx_omp = np.argmin(abs(inodes-idx_omp))
    
    return inode_idx_omp

def get_OMP_index_inode(xr, inodes, LFS=True):
    if LFS:
        msk_lfs = (xr.mesh.r[inodes] > xr.eq_axis_r)
    else:
        msk_lfs = (xr.mesh.r[inodes] < xr.eq_axis_r)
    
    idx = np.argmin(abs(xr.mesh.z[inodes[msk_lfs]]-xr.eq_axis_z))
    idx_omp = inodes[msk_lfs][idx]
    inode_idx_omp = np.argmin(abs(inodes-idx_omp))
    
    return inode_idx_omp


#--- Velocity contour plots
def contour_vgrid_f0(ax, iphi, inode, f0_f, vpara=np.linspace(-4,4,29), vperp=np.linspace(0,4,33), title="", use_log=False, draw_contour=False, **kwargs):
    if f0_f.ndim ==4:
        _f = f0_f[iphi,:,inode,:] # [iphi, iperp, inode, ipara]
    else:
        _f = f0_f[:,inode,:]
        print(f"'iphi' {iphi} is given, but it seems that toroidally averaged f0_f is given. So ignoring 'iphi'")

    if use_log:
        title = title+" (log)"
        cntr = ax.contourf(vpara, vperp, np.log(_f[:,:]), **kwargs)
    else:
        cntr = ax.contourf(vpara, vperp, _f[:,:], **kwargs)
        
    plt.colorbar(cntr)
    cntr.colorbar.formatter.set_powerlimits((0,0))
    ax.set_xlabel('vpara')
    ax.set_ylabel('vperp')
    ax.set_title(title)
    
    if draw_contour:
        levels=[1e-2,1e-1,1e0,1e1,1e2]
        cs = ax.contour(vpara, vperp, _f[:,:], levels=levels, colors='white', linewidths=1.5)
        ax.clabel(cs, fmt="%.2f", colors='white', fontsize=10)

def contour_vgrid_f0_only_node(ax, f0_f_target, vpara=np.linspace(-4,4,29), vperp=np.linspace(0,4,33), title="", use_log=False, draw_contour=False, **kwargs):
    _f = f0_f_target # [iperp, ipara]

    if use_log:
        title = title+" (log)"
        cntr = ax.contourf(vpara, vperp, np.log(_f[:,:]), **kwargs)
    else:
        cntr = ax.contourf(vpara, vperp, _f[:,:], **kwargs)
        
    plt.colorbar(cntr)
    cntr.colorbar.formatter.set_powerlimits((0,0))
    ax.set_xlabel('vpara')
    ax.set_ylabel('vperp')
    ax.set_title(title)
    
    if draw_contour:
        levels=[1e-2,1e-1,1e0,1e1,1e2]
        cs = ax.contour(vpara, vperp, _f[:,:], levels=levels, colors='white', linewidths=1.5)
        ax.clabel(cs, fmt="%.2f", colors='white', fontsize=10)

#--- f0 integral
UNIT_CHARGE = 1.6022e-19;  # Charge of an electron (C)
EV_2_J = UNIT_CHARGE;        # Conversion rate ev to J
J_2_EV = 1.0/EV_2_J;         # Conversion rate J to ev
PROTON_MASS = 1.6720e-27;  # proton mass (MKS)

# e_f & i_f
isp_e = 0; mass_au_e = 5.454E-4;
isp_i = 1; mass_au_i = 2E0;

def v_vol_fac(iv: int, nv: int) -> float:
    return 0.5 if iv == 0 or iv == nv - 1 else 1.0

def get_cs(dir_run, x): # m/s
    Ti0_ev = get_adios2_var(f"{dir_run}/xgc.f0.mesh.bp", "f0_fg_T_ev")[isp_i,:]
    Te0_ev = get_adios2_var(f"{dir_run}/xgc.f0.mesh.bp", "f0_fg_T_ev")[isp_e,:]

    cs = np.sqrt((Ti0_ev+Te0_ev)*EV_2_J/(mass_au_i*PROTON_MASS))
    
    return cs

def get_q_par(dir_run, x, f0_f, mass_au, isp, vperp, vpara, sheath_pot_map=None):
    fg_temp_ev = get_adios2_var(f"{dir_run}/xgc.f0.mesh.bp", "f0_fg_T_ev")[isp,:]
    en_th = fg_temp_ev*EV_2_J
    vth = np.sqrt(en_th/(mass_au*PROTON_MASS))

    vol_vonly = get_adios2_var(f"{dir_run}/xgc.f0.mesh.bp", "f0_grid_vol_vonly")[isp,:]

    nvr = np.shape(f0_f)[0]
    nvz = np.shape(f0_f)[2]

    q_par_tot = np.zeros_like(x.mesh.r)
    q_par_to_target = np.zeros_like(x.mesh.r)
    for ivr in tqdm(range(nvr)):
        smu = vperp[ivr]
        mu = smu*smu
        if ivr==0: smu = vperp[1]/3.0
    
        for ivz in range(nvz):
            vp = vpara[ivz]
            en = 0.5 * (mu+vp*vp)
            
            vp_mks = vp*vth
            en_mks = mass_au*PROTON_MASS * en * vth**2

            vol = vol_vonly * v_vol_fac(ivr,nvr) * v_vol_fac(ivz,nvz)

            # calculate target moment
            _moment = en_mks * vp_mks

            q_par_tot += f0_f[ivr,:,ivz] * _moment * vol
        
            if vp < 0: # for DIII-D turb-QH mode geometry, outer target
                if sheath_pot_map is not None:
                    if isp != 0: print("Warning: sheath_pot filtering on with isp!=0. Be careful to not use this in the analysis.")

                    en_para = 0.5 * vp*vp
                    en_para_mks = mass_au*PROTON_MASS * en_para * vth**2

                    msk_sheath = en_para_mks > sheath_pot_map*EV_2_J
                
                    q_par_to_target[msk_sheath] += f0_f[ivr,msk_sheath,ivz] * _moment[msk_sheath] * vol[msk_sheath]
                else:
                    q_par_to_target += f0_f[ivr,:,ivz] * _moment * vol

    return q_par_tot, q_par_to_target

#--- Drawing line
def find_mesh_indice(r,z,r0,z0):
    dist2 = (r - r0)**2 + (z - z0)**2  # squared distance
    return np.argmin(dist2)

def gen_line_pt(pt1, pt2, x):
    slope = (pt2[1]-pt1[1])/(pt2[0]-pt1[0])
    C = pt1[1] - (pt2[1]-pt1[1])/(pt2[0]-pt1[0])*pt1[0]
    y = slope * x + C
    return x, y

def gen_line_theta(pt1, theta, x):
    slope = np.tan(theta)
    C = pt1[1] - slope*pt1[0]
    y = slope * x + C
    return x, y

def gen_line_theta_dist(pt1, theta, dist=1.0, dist_backward=0.0, npt=1000):
    # x = np.linspace(pt1[0],pt1[0]+np.cos(theta)*dist,1000)
    # slope = np.tan(theta)
    # C = pt1[1] - slope*pt1[0]
    # y = slope * x + C
    # return x, y

    epsilon = 1e-8  # tolerance for floating-point comparison
    if np.isclose(np.cos(theta), 0.0, atol=epsilon):
        # Vertical line
        x = np.full(npt, pt1[0])  # constant x
        if np.sin(theta) > 0:
            y = np.linspace(pt1[1], pt1[1] + dist, npt)
        else:
            y = np.linspace(pt1[1], pt1[1] - dist, npt)
    else:
        x = np.linspace(pt1[0] - np.cos(theta)*dist_backward, pt1[0] + np.cos(theta)*dist, npt)
        slope = np.tan(theta)
        C = pt1[1] - slope * pt1[0]
        y = slope * x + C
    return x, y

def pt_to_theta(pt1, pt2): # rad
    dx = pt2[0] - pt1[0]
    dy = pt2[1] - pt1[1]
    theta = np.atan2(dy, dx)
    return theta

def deg2rad(deg):
    return deg * np.pi / 180

def rad2deg(rad):
    return rad * 180 / np.pi

def normalize_to_pi(angle): # maps angle (radian) onto (-pi, pi]
    return np.arctan2(np.sin(angle), np.cos(angle))

def get_dist(pt1,pt2):
    return np.sqrt((pt1[0]-pt2[0])**2 + (pt1[1]-pt2[1])**2)

#--- Mesh relevant points (for SOL profiles)
def insert_level_point_on_line(
    f,                         # callable: f(r_array, z_array) -> array (vectorized; wrap if needed)
    r0, z0, theta, dist_fwd, dist_bwd,
    r_line, z_line,            # existing points along the same line (ascending in s)
    level=1.0, M=2001, atol=1e-10
):
    c, s = np.cos(theta), np.sin(theta)

    # 1) dense param along the line
    S = np.linspace(-dist_bwd, dist_fwd, M)
    R = r0 + S*c
    Z = z0 + S*s

    # 2) evaluate and look for crossing of f-level
    vals = f(R, Z)
    g = vals - level

    # exact hit on the grid?
    hit = np.where(np.isclose(g, 0.0, atol=atol))[0]
    if hit.size:
        Sstar = S[hit[0]]
    else:
        # sign-change brackets
        sign = np.sign(g)
        sc_idx = np.where(sign[:-1] * sign[1:] < 0)[0]
        if sc_idx.size == 0:
            # no crossing; nothing to insert
            return r_line, z_line, None

        # pick the crossing closest to s=0 (change policy if you prefer first/last)
        candidates = []
        for i in sc_idx:
            # 3) linear interpolation for the root inside [S[i], S[i+1]]
            S0, S1 = S[i], S[i+1]
            g0, g1 = g[i], g[i+1]
            Sx = S0 - g0 * (S1 - S0) / (g1 - g0)
            candidates.append(Sx)
        Sstar = min(candidates, key=lambda t: abs(t))

    # compute (r*, z*)
    r_star = r0 + Sstar*c
    z_star = z0 + Sstar*s

    # 4) insert into your existing (r_line, z_line) in order of s
    # recover s for current points
    S_line = (r_line - r0)*c + (z_line - z0)*s
    # if already present (within tolerance), skip
    if np.any(np.isclose(S_line, Sstar, atol=1e-12)):
        return r_line, z_line, (r_star, z_star)

    k = np.searchsorted(S_line, Sstar)
    r_ins = np.insert(r_line, k, r_star)
    z_ins = np.insert(z_line, k, z_star)
    return r_ins, z_ins, (r_star, z_star)

import matplotlib.tri as mtri
from shapely.geometry import LineString
from shapely.ops import unary_union

#--- helpes
def _seg_seg_intersection(p0, p1, a, b, eps=1e-12):
    """Return t in [0,1] along p0->p1 where the segment intersects a->b; None if no hit."""
    p0 = np.asarray(p0); p1 = np.asarray(p1); a = np.asarray(a); b = np.asarray(b)
    r = p1 - p0; s = b - a
    den = r[0]*s[1] - r[1]*s[0]
    if abs(den) < eps:
        return None
    ap  = a - p0
    t   = (ap[0]*s[1] - ap[1]*s[0]) / den
    u   = (ap[0]*r[1] - ap[1]*r[0]) / den
    if -eps <= t <= 1+eps and -eps <= u <= 1+eps:
        return float(np.clip(t, 0.0, 1.0))
    return None

def find_polyline_intersections(x1, y1, x2, y2):
    """
    Finds all geometric intersection points between two polylines.
    
    Args:
        x1, y1: Coordinates of the first polyline (arrays).
        x2, y2: Coordinates of the second polyline (arrays).
        
    Returns:
        intersections: List of (x, y) tuples representing intersection points.
    """
    x1 = np.asarray(x1); y1 = np.asarray(y1)
    x2 = np.asarray(x2); y2 = np.asarray(y2)
    
    intersections = []
    
    # Iterate over all segments in Line 1
    for i in range(len(x1) - 1):
        p0 = np.array([x1[i], y1[i]])
        p1 = np.array([x1[i+1], y1[i+1]])
        
        # Iterate over all segments in Line 2
        for j in range(len(x2) - 1):
            a = np.array([x2[j], y2[j]])
            b = np.array([x2[j+1], y2[j+1]])
            
            t = _seg_seg_intersection(p0, p1, a, b)
            
            if t is not None:
                # Calculate the physical intersection point
                # Point = p0 + t * (p1 - p0)
                int_pt = p0 + t * (p1 - p0)
                
                # Deduplication:
                # If an intersection occurs exactly at a vertex (e.g., t=1 of segment i 
                # and t=0 of segment i+1), we might find it twice.
                # We check if this point is effectively the same as the last one found.
                if intersections:
                    if np.linalg.norm(int_pt - intersections[-1]) < 1e-9:
                        continue
                
                intersections.append(tuple(int_pt))
                
    return intersections

def line_edge_hits(tri, p0, p1):
    """
    Compute the EXACT intersection points of the line with mesh edges,
    and return interpolated values there (plus endpoints).
    Useful to build piecewise integrations without bias from uneven cell sizes.
    """
    # collect t where the line hits any edge
    edges = tri.edges
    ts = [0.0, 1.0]
    for i, j in edges:
        t = _seg_seg_intersection(p0, p1, (tri.x[i], tri.y[i]), (tri.x[j], tri.y[j]))
        if t is not None:
            ts.append(t)
            
    ts = np.unique(np.clip(ts, 0.0, 1.0))
    xs = p0[0] + (p1[0] - p0[0]) * ts
    ys = p0[1] + (p1[1] - p0[1]) * ts
    s  = np.hypot(*(np.array(p1) - np.array(p0))) * ts

    return s, xs, ys

def line_contour_hits_tri(tri, z, p0, p1, level):
    """
    Intersections of the line segment p0->p1 with the tricontour at `level`.
    Returns s (arclength), x, y sorted along the segment.
    """
    # 1) build isoline(s); use allsegs (portable across mpl versions)
    # cs = plt.tricontour(tri, z, levels=[level])
    # segs = cs.allsegs[0]                    # list of (N,2) arrays (x,y)
    # plt.close(cs.figure)
    tmp_fig, tmp_ax = plt.subplots()
    cs = tmp_ax.tricontour(tri, z, levels=[level])
    segs = cs.allsegs[0] if cs.allsegs else []
    plt.close(tmp_fig)  # closes only the temp fig

    if not segs:
        return np.empty(0), np.empty(0), np.empty(0)

    # 2) union of polylines
    contour = unary_union([LineString(seg) for seg in segs if len(seg) >= 2])

    # 3) intersect with the line segment
    cut = LineString([p0, p1])
    inter = cut.intersection(contour)

    pts = []
    if inter.is_empty:
        pass
    elif inter.geom_type == "Point":
        pts = [inter]
    elif inter.geom_type == "MultiPoint":
        pts = list(inter.geoms)
    elif inter.geom_type in ("LineString", "MultiLineString"):
        # overlapping case: take segment endpoints as "intersections"
        geoms = getattr(inter, "geoms", [inter])
        for g in geoms:
            a, b = g.boundary.geoms
            pts += [a, b]

    if not pts:
        return np.empty(0), np.empty(0), np.empty(0)

    # 4) param along p0->p1
    L  = np.hypot(p1[0]-p0[0], p1[1]-p0[1])
    dx = p1[0]-p0[0]; dy = p1[1]-p0[1]
    t  = ((np.array([p.x for p in pts]) - p0[0])*dx +
          (np.array([p.y for p in pts]) - p0[1])*dy) / (L**2)
    t  = np.clip(t, 0, 1)
    order = np.argsort(t)
    t = t[order]
    x = p0[0] + t*dx
    y = p0[1] + t*dy
    s = t*L
    return s, x, y

def unique_xy_exact(x, y):
    xy = np.column_stack([x, y])                            # shape (N,2)
    _, idx = np.unique(xy, axis=0, return_index=True)       # first indices kept
    keep = np.sort(idx)                                     # preserve input order
    return x[keep], y[keep], keep

def unique_xy_tol(x, y, tol=1e-9):
    xy = np.column_stack([x, y])
    q  = np.round(xy / tol).astype(np.int64)                # quantized keys
    _, idx = np.unique(q, axis=0, return_index=True)
    keep = np.sort(idx)
    return x[keep], y[keep], keep

#--- main API
# !!! Setting specific for a run !!! TODO: clear this AD-HOC 
#isurf_sol = [92,93,94,96,98,100]       # for old DIII-D run
isurf_sol = [103,104,105,107,109,111,113,115,117,119,121]  # for new DIII-D run

def sample_line_grid_relevant(x, r_line, z_line, psin_min=1.0):
    # start main logic
    p0 = [r_line[0], z_line[0]]
    p1 = [r_line[-1], z_line[-1]]

    x_mesh_relevant = np.array([], dtype=float)
    y_mesh_relevant = np.array([], dtype=float)
 
    if psin_min < 1.0:
        isep = np.argmin(abs(x.mesh.psi_surf - x.psix))
        for i in range(np.argmin(abs(x.mesh.psi_surf/x.psix - psin_min)), isep):
            _, xtmp, ytmp = line_contour_hits_tri(x.mesh.triobj, x.mesh.psi/x.psix, p0, p1, level=x.mesh.psi_surf[i]/x.psix); 
            x_mesh_relevant = np.append(x_mesh_relevant, xtmp); 
            y_mesh_relevant = np.append(y_mesh_relevant, ytmp)
   
    # append flux surfaces in SOL
    for i in range(len(isurf_sol)):
        _, xtmp, ytmp = line_contour_hits_tri(x.mesh.triobj, x.mesh.psi/x.psix, p0, p1, level=x.mesh.psi_surf[isurf_sol[i]]/x.psix); 
        x_mesh_relevant = np.append(x_mesh_relevant, xtmp); 
        y_mesh_relevant = np.append(y_mesh_relevant, ytmp)

    # remove redundant points
    x_mesh_relevant, y_mesh_relevant, _ = unique_xy_tol(x_mesh_relevant, y_mesh_relevant, tol=1e-4)

    return x_mesh_relevant, y_mesh_relevant

def sample_line_grid_relevant_incl_unstruc(x, r_line, z_line, psin_min=1.0):
    # !!! Setting specific for a run !!!
    isurf_sep         = np.argmin(abs(x.mesh.psi_surf - x.psix))
    isurf_unstruc_min = isurf_sep - 1
    isurf_unstruc_max = isurf_sep + 1

    # start main logic
    p0 = [r_line[0], z_line[0]]
    p1 = [r_line[-1], z_line[-1]]

    x_mesh_relevant = np.array([], dtype=float)
    y_mesh_relevant = np.array([], dtype=float)

    # 1) add inner surfaces
    if psin_min < x.mesh.psi_surf[isurf_unstruc_min]/x.psix:
        for i in range(np.argmin(abs(x.mesh.psi_surf/x.psix - psin_min)), isurf_unstruc_min):
            _, xtmp, ytmp = line_contour_hits_tri(x.mesh.triobj, x.mesh.psi/x.psix, p0, p1, level=x.mesh.psi_surf[i]/x.psix); 
            x_mesh_relevant = np.append(x_mesh_relevant, xtmp); 
            y_mesh_relevant = np.append(y_mesh_relevant, ytmp)
    
    # 2) add unstruc part
    _, x_unstruc_min, y_unstruc_min = line_contour_hits_tri(x.mesh.triobj, x.mesh.psi/x.psix, p0, p1, level=x.mesh.psi_surf[isurf_unstruc_min]/x.psix);

    if len(x_unstruc_min) == 0: # at least to start from the separatrix
        _, x_unstruc_min, y_unstruc_min = line_contour_hits_tri(x.mesh.triobj, x.mesh.psi/x.psix, p0, p1, level=1.0)
        
        # use the given point when possible !!! (the logic seems not so appropriate) !!!
        if x_unstruc_min > p0[0]:
            x_unstruc_min[0] = p0[0]
            y_unstruc_min[0] = p0[1]
        
    if len(x_unstruc_min) == 0: # still zero?
        _, x_unstruc_min, y_unstruc_min = line_contour_hits_tri(x.mesh.triobj, x.mesh.psi/x.psix, p0, p1, level=1.0+1e-4)
    _, x_unstruc_max, y_unstruc_max = line_contour_hits_tri(x.mesh.triobj, x.mesh.psi/x.psix, p0, p1, level=x.mesh.psi_surf[isurf_unstruc_max]/x.psix);

    # keep edge-crossing points in unstructured mesh region
    _, xtmp, ytmp = line_edge_hits(x.mesh.triobj, (x_unstruc_min[0], y_unstruc_min[0]), (x_unstruc_max[0], y_unstruc_max[0]))
    x_mesh_relevant = np.append(x_mesh_relevant, xtmp); 
    y_mesh_relevant = np.append(y_mesh_relevant, ytmp)

    # 3) append flux surfaces in SOL
    for i in range(len(isurf_sol)):
        _, xtmp, ytmp = line_contour_hits_tri(x.mesh.triobj, x.mesh.psi/x.psix, p0, p1, level=x.mesh.psi_surf[isurf_sol[i]]/x.psix); 
        x_mesh_relevant = np.append(x_mesh_relevant, xtmp); 
        y_mesh_relevant = np.append(y_mesh_relevant, ytmp)

    # remove redundant points
    x_mesh_relevant, y_mesh_relevant, _ = unique_xy_tol(x_mesh_relevant, y_mesh_relevant, tol=1e-4)

    return x_mesh_relevant, y_mesh_relevant

#--- Eich fit
def eich(xdata,q0,s,lq,dsep):
  return 0.5*q0*np.exp((0.5*s/lq)**2-(xdata-dsep)/lq)*erfc(0.5*s/lq-(xdata-dsep)/s)

def eich_fit1(ydata,rmidsepmm,pmask=None):
  q0init=np.max(ydata)  
  sinit=0.1 # 1mm
  lqinit=3 # 3mm
  dsepinit=0.1 # 0.1 mm

  p0=np.array([q0init, sinit, lqinit, dsepinit])
  if(pmask==None):
      popt,pconv = curve_fit(eich,rmidsepmm,ydata,p0=p0)
  else:
      popt,pconv = curve_fit(eich,rmidsepmm[pmask],ydata[pmask],p0=p0)

  return popt, pconv

#--- Fitting suite
# models
def eich(x, q0, s, lq, dsep):
    z = x - dsep
    a = 0.5 * s / lq
    return 0.5 * q0 * np.exp(a*a - z/lq) * erfc(a - z/s)

def eich_bg(x, q0, s, lq, qbg, dsep):
    z = x - dsep
    a = 0.5 * s / lq
    return 0.5 * q0 * np.exp(a*a - z/lq) * erfc(a - z/s) + qbg

def eich_dsep0(x, q0, s, lq):
    z = x
    a = 0.5 * s / lq
    return 0.5 * q0 * np.exp(a*a - z/lq) * erfc(a - z/s)

def single_exp(x, qn, ln, dsep):
    z = x - dsep
    return qn*np.exp(-z/ln)

def single_exp_dsep0(x, qn, ln):
    z = x
    return qn*np.exp(-z/ln)

def single_exp_dsep0_bg(x, qn, ln, qbg):
    z = x
    return qn*np.exp(-z/ln) + qbg

def double_exp(x, qn, ln, qf, lf, dsep):
    z = x - dsep
    return qn*np.exp(-z/ln) + qf*np.exp(-z/lf)

def double_exp_dsep0(x, qn, ln, qf, lf):
    z = x
    return qn*np.exp(-z/ln) + qf*np.exp(-z/lf)

def double_exp_bg(x, qn, ln, qf, lf, qbg, dsep):
    z = x - dsep
    return qn*np.exp(-z/ln) + qf*np.exp(-z/lf) + qbg

MODELS = {
    "eich"                : (eich,                ["q0","s","lq", "dsep"]),
    "eich_bg"             : (eich_bg,             ["q0","s","lq", "qbg", "dsep"]),
    "eich_dsep0"          : (eich_dsep0,          ["q0","s","lq"]),
    "single_exp"          : (single_exp,          ["qn","ln","dsep"]),
    "single_exp_dsep0"    : (single_exp_dsep0,    ["qn","ln"]),
    "single_exp_dsep0_bg" : (single_exp_dsep0_bg, ["qn","ln","qbg"]),
    "double_exp"          : (double_exp,          ["qn","ln","qf","lf","dsep"]),
    "double_exp_dsep0"    : (double_exp_dsep0,    ["qn","ln","qf","lf"]),
    "double_exp_bg"       : (double_exp_bg,       ["qn","ln","qf","lf","qbg","dsep"]),
}

def _default_p0(model, x, y):
    xm = float(x[np.nanargmax(y)])
    ymax = float(np.nanmax(y))
    span = max(float(np.ptp(x)), 1.0)
    match model:
        case "eich":
            return [ymax, 0.1, 3.0, xm]  # s≈0.1, lq≈3, dsep at peak
        case "eich_bg":
            return [ymax, 0.1, 3.0, 0.0, xm]
        case "eich_dsep0":
            return [ymax, 0.1, 3.0]
        case "single_exp":
            return [0.6*ymax, span/20, xm]
        case "single_exp_dsep0":
            return [0.6*ymax, span/20]
        case "single_exp_dsep0_bg":
            return [0.6*ymax, span/20, 0.0]
        case "double_exp":
            return [0.6*ymax, span/20, 0.4*ymax, span/4, xm]
        case "double_exp_dsep0":
            return [0.6*ymax, span/20, 0.4*ymax, span/4]
        case "double_exp_bg":
            return [0.6*ymax, span/20, 0.4*ymax, span/4, 0.0, xm]
        case _:
            print(f"Warning: {model} is not implemented. Assume Eich-like parameters.")
            return [ymax, 0.1, 3.0, 0.0, xm]

#----------------
def format_popt(model, popt, sep=", ", prefix=""):
    try:
        _, names = MODELS[model]
    except KeyError as e:
        raise ValueError(f"Unknown model '{model}'. Available: {list(MODELS)}") from e

    names = list(names)
    popt = np.asarray(popt).ravel()
    if len(popt) != len(names):
        print(f"Warning: {model} expects {len(names)} params {names} but got {len(popt)}")

    def fmt(name, val):
        return f"{name}: {val:.3f}"

    body = sep.join(fmt(n, v) for n, v in zip(names, popt))
    return f"{prefix}{body}"

def print_popt_line(model: str, popt, prefix=""):
    print(format_popt(model, popt, sep=", ", prefix=prefix))

def reconstruc_1d(x, model, popt):
    y = np.zeros_like(x)
    match model:
        case "eich":
            y = eich(x, *popt)
        case "eich_bg":
            y = eich_bg(x, *popt)
        case "eich_dsep0":
            y = eich_dsep0(x, *popt)
        case "single_exp":
            y = single_exp(x, *popt)
        case "single_exp_dsep0":
            y = single_exp_dsep0(x, *popt)
        case "single_exp_dsep0_bg":
            y = single_exp_dsep0_bg(x, *popt)
        case "double_exp":
            y = double_exp(x, *popt)
        case "double_exp_dsep0":
            y = double_exp_dsep0(x, *popt)
        case "double_exp_bg":
            y = double_exp_bg(x, *popt)
        case _:
            print(f"Warning: {model} is not implemented. Skip reconstruc_1d, giving zero y.")
            
    return y

def _local_spacing_weights(x, alpha=1.0, min_dx=1e-9):
    """
    Compute per-point weights ~ (local Δx)^alpha.
    alpha=1 gives integral weighting; alpha in (0,1) partially downweights dense clusters.
    """
    x = np.asarray(x).ravel()
    order = np.argsort(x)
    xs = x[order]

    dx = np.empty_like(xs)
    if xs.size == 1:
        dx[...] = 1.0
    else:
        dx[1:-1] = 0.5 * (xs[2:] - xs[:-2])        # half-interval around xi
        dx[0]    = xs[1] - xs[0]
        dx[-1]   = xs[-1] - xs[-2]
        dx = np.clip(dx, min_dx, None)

    w = dx**alpha
    # normalize so mean weight ~ 1 (keeps sigma magnitudes sane)
    w *= (xs.size / w.sum())
    # un-sort back to original order
    inv = np.empty_like(order)
    inv[order] = np.arange(order.size)
    return w[inv]

def fit_1d(x, y, model="eich", p0=None, bounds=None, sigma=None):
    """Minimal, extensible 1D fit. Returns (popt, pcov, yfit, param_names, R2)."""
    f, names = MODELS[model]
    x = np.asarray(x).ravel(); y = np.asarray(y).ravel() # flattens to 1d
    m = np.isfinite(x) & np.isfinite(y)
    x, y = x[m], y[m]

    if p0 is None: p0 = _default_p0(model, x, y)
    if bounds is None:
        lo = np.full(len(p0), 0.0); hi = np.full(len(p0), np.inf)
        lo[-1], hi[-1] = -np.inf, np.inf # dsep
        
        bounds = (lo, hi)

    try:
        popt, pcov = curve_fit(
            f, x, y, p0=p0, bounds=bounds, sigma=sigma,
            absolute_sigma=(sigma is not None), maxfev=20000
        )
    except Exception:
        res = lambda th: f(x, *th) - y
        r = least_squares(res, p0, bounds=bounds, max_nfev=20000)
        popt, pcov = r.x, None

    yfit = f(x, *popt)
    ss_res = np.sum((y - yfit)**2)
    ss_tot = np.sum((y - np.mean(y))**2)
    r2 = 1.0 - ss_res/ss_tot if ss_tot > 0 else np.nan

    return popt, pcov, yfit, names, r2

def fit_1d_weight(x, y, model="eich", p0=None, bounds=None, sigma=None,
                  weight_by_spacing=True, alpha=1.0,
                  robust=None, f_scale=1.0):
    """
    Minimal, extensible 1D fit with optional spacing-based weighting and robust loss.
    Returns (popt, pcov, yfit, param_names, R2).

    weight_by_spacing: if True and sigma is None, use sigma_i = 1/sqrt(w_i) with w_i ~ (Δx_i)^alpha
    alpha: 1.0 = full integral weighting; 0.0 = unweighted; e.g. 0.5–0.8 partially downweights dense regions
    robust: None | 'soft_l1' | 'huber' | 'cauchy' (passed to least_squares)
    """
    f, names = MODELS[model]
    x = np.asarray(x).ravel(); y = np.asarray(y).ravel()
    m = np.isfinite(x) & np.isfinite(y)
    x, y = x[m], y[m]

    if p0 is None: p0 = _default_p0(model, x, y)
    if bounds is None:
        lo = np.full(len(p0), 0.0); hi = np.full(len(p0), np.inf)
        lo[-1], hi[-1] = -np.inf, np.inf # dsep
        
        bounds = (lo, hi)

    # If user didn't pass sigma, optionally build one from local spacing
    if sigma is None and weight_by_spacing:
        w = _local_spacing_weights(x, alpha=alpha)         # larger in sparse regions
        sigma = 1.0 / np.sqrt(w)                           # curve_fit minimizes sum(((f-y)/σ)^2)
        abs_sigma = True
    else:
        abs_sigma = (sigma is not None)

    try:
        if robust is None:
            popt, pcov = curve_fit(
                f, x, y, p0=p0, bounds=bounds, sigma=sigma,
                absolute_sigma=abs_sigma, maxfev=20000
            )
        else:
            # Use least_squares with robust loss; apply weights by scaling residuals
            if sigma is None:
                scale = 1.0
            else:
                scale = 1.0 / sigma  # since curve_fit would divide by sigma, we multiply residual by 1/sigma
            def res(th):
                return (f(x, *th) - y) * scale
            r = least_squares(res, p0, bounds=bounds, loss=robust, f_scale=f_scale, max_nfev=20000)
            popt, pcov = r.x, None
    except Exception:
        # Fallback if curve_fit struggles
        if sigma is None:
            scale = 1.0
        else:
            scale = 1.0 / sigma
        def res(th): return (f(x, *th) - y) * scale
        r = least_squares(res, p0, bounds=bounds, max_nfev=20000)
        popt, pcov = r.x, None

    yfit = f(x, *popt)
    ss_res = np.sum((y - yfit)**2)
    ss_tot = np.sum((y - np.mean(y))**2)
    r2 = 1.0 - ss_res/ss_tot if ss_tot > 0 else np.nan
    return popt, pcov, yfit, names, r2

# --- old API ---
def eich_fit1(ydata, rmidsepmm):
    popt, pcov, _, _, R2 = fit_1d(rmidsepmm, ydata, model="eich")
    return popt, pcov, R2

