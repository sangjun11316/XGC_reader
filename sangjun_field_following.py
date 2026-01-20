""" Python functions to make fieldline following diagnostics from XGC
"""
import os, sys, re
import traceback

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.tri import Triangulation, LinearTriInterpolator

from scipy.optimize import curve_fit
from scipy.special import erfc

import adios2
import pyvista as pv

import sangjun_utils as sj

#--- Fieldline Following
def normalize_index(i: int, n: int) -> int:
    # """Map i in [-n, n-1] to a canonical index in [0, n-1]."""
    # if n <= 0:
    #     raise ValueError("n must be > 0")
    # if not (-n <= i < n):
    #     raise IndexError(f"index {i} out of range for length {n}")
    return i%n    

def get_length_cylindrical(p0, p1):
    r0, z0, phi0 = p0
    r1, z1, phi1 = p1
    
    d2 = r0**2 + r1**2 - 2*r0*r1*np.cos(phi1-phi0) + (z1-z0)**2
    return np.sqrt(d2)

def signed_angle_AOB_2D(A, O, B, deg=False):
    """
    Signed angle ∠AOB in 2D: CCW positive, range (-π, π].
    """
    a = np.asarray(A, float) - np.asarray(O, float)
    b = np.asarray(B, float) - np.asarray(O, float)
    if a.shape[-1] != 2 or b.shape[-1] != 2:
        raise ValueError("signed_angle_AOB_2D requires 2D points.")
    cross_z = a[0]*b[1] - a[1]*b[0]
    dot = a[0]*b[0] + a[1]*b[1]
    ang = np.arctan2(cross_z, dot)
    return np.degrees(ang) if deg else ang

def follow_field_to_next_plane(r, z, iphi, bfield_interpolator, nplanes, wedge_n, bt_sign, 
                               bt_backward=False, ff_step=4, ff_order=4):
    """

    """
    # derived quantities
    wedge_angle = 2*np.pi / wedge_n
    delta_phi = wedge_angle / nplanes
    inv_delta_phi = 1.0 / delta_phi

    # Trace fieldline backward
    bt_back = 1
    if bt_backward: bt_back = -1

    # Compute total phi step to reach mid-plane and sub-step size
    dphi_total = delta_phi
    dphi = dphi_total / ff_step * bt_sign*bt_back  # increment per sub-step
    
    # Initialize current positions for integration
    r_curr = r
    z_curr = z
    phi_curr = iphi*delta_phi  # assume iphi is at midplane
    
    # Helper function to get derivatives dr/dphi, dz/dphi for all particles at once
    def get_derivs(r, z, phi):
        # Interpolate B field at given (r, z, phi)
        Br   = bfield_interpolator[0](r, z)
        Bz   = bfield_interpolator[1](r, z)
        Bphi = bfield_interpolator[2](r, z)
    
        # Compute derivatives: dr/dphi and dz/dphi
        # Avoid division by zero in Bphi (e.g., on axis) by adding a tiny epsilon if needed
        # (assuming Bphi is never zero in our domain for normal field lines)
        deriv_r = (Br / Bphi) * r
        deriv_z = (Bz / Bphi) * r
        return deriv_r, deriv_z

    s_arc = 0.0
    
    # First-order (Euler) integration
    if ff_order == 1:
        for _ in range(ff_step):
            P0 = (r_curr, z_curr, phi_curr)
            
            k1r, k1z = get_derivs(r_curr, z_curr, phi_curr)
            # Update all particles with the Euler step
            r_curr += k1r * dphi
            z_curr += k1z * dphi
            phi_curr += dphi

            P1 = (r_curr, z_curr, phi_curr)
            
            s_arc += get_length_cylindrical(P0,P1)

    # Second-order (Midpoint/RK2) integration
    elif ff_order == 2:
        for _ in range(ff_step):
            P0 = (r_curr, z_curr, phi_curr)
            
            # Compute k1 at the beginning of the interval
            k1r, k1z = get_derivs(r_curr, z_curr, phi_curr)
            # Estimate midpoint values
            r_mid = r_curr + 0.5 * k1r * dphi
            z_mid = z_curr + 0.5 * k1z * dphi
            phi_mid_step = phi_curr + 0.5 * dphi
            # Compute k2 at the midpoint
            k2r, k2z = get_derivs(r_mid, z_mid, phi_mid_step)
            # Update all particles using k2 (midpoint derivative)
            r_curr += k2r * dphi
            z_curr += k2z * dphi
            phi_curr += dphi

            P1 = (r_curr, z_curr, phi_curr)

            s_arc += get_length_cylindrical(P0,P1)

    # Fourth-order (RK4) integration (existing implementation)
    elif ff_order == 4:
        for _ in range(ff_step):
            P0 = (r_curr, z_curr, phi_curr)
            
            # Step 1 (first midpoint)
            k1r, k1z = get_derivs(r_curr, z_curr, phi_curr)
            r_mid1 = r_curr + 0.5 * k1r * dphi
            z_mid1 = z_curr + 0.5 * k1z * dphi
            phi_mid1 = phi_curr + 0.5 * dphi

            # Step 2 (second midpoint)
            k2r, k2z = get_derivs(r_mid1, z_mid1, phi_mid1)
            r_mid2 = r_curr + 0.5 * k2r * dphi
            z_mid2 = z_curr + 0.5 * k2z * dphi

            # Step 3 (full step)
            k3r, k3z = get_derivs(r_mid2, z_mid2, phi_mid1)
            r_end = r_curr + k3r * dphi
            z_end = z_curr + k3z * dphi
            phi_end = phi_curr + dphi

            # Step 4 (combine increments)
            k4r, k4z = get_derivs(r_end, z_end, phi_end)
            r_curr += (k1r + 2*k2r + 2*k3r + k4r) * (dphi / 6.0)
            z_curr += (k1z + 2*k2z + 2*k3z + k4z) * (dphi / 6.0)
            phi_curr += dphi

            P1 = (r_curr, z_curr, phi_curr)

            s_arc += get_length_cylindrical(P0,P1)
            
    else:
        raise ValueError("Unsupported ff_order, use 1, 2, or 4.")
            
    return r_curr, z_curr, normalize_index(iphi+bt_sign*bt_back, nplanes), s_arc

def trace_fieldline_values(r, z, iphi, value_interpolator, bfield_interpolator, xr, nplanes, wedge_n, bt_sign, bt_backward=False, 
                           ff_step=4, ff_order=4, toroidal_turns=1, follow_until_exit=False, max_exit_turns=10, 
                           verbose=True):
    # derived quantities
    wedge_angle = 2*np.pi / wedge_n
    delta_phi = wedge_angle / nplanes
    inv_delta_phi = 1.0 / delta_phi

    # direction of phi
    bt_back = 1
    if bt_backward: bt_back = -1
    
    if follow_until_exit:
        if verbose: print(f"Trace until exit: max_turns {max_exit_turns}")
        step_max = int(np.ceil(nplanes*wedge_n*max_exit_turns))
    else:
        if verbose: print(f"Trace fieldline: max_turns {toroidal_turns}")
        step_max = nplanes*wedge_n*toroidal_turns
    
    # placeholders; ff for 'fieldline following'
    r_ff = []
    z_ff = []
    phi_ff = []
    iphi_ff = []
    value_ff = []
    s_ff = []
    tri_ff = []

    trifinder = xr.mesh.triobj.get_trifinder()
    
    r_curr = r
    z_curr = z
    phi_curr = iphi*delta_phi # should I 0.5 here?
    iphi_curr = iphi

    s_arc = 0.0
    
    # default: track for one wedge_angle
    step = 0
    while True:
        if step >= step_max:
            break 
        
        r_ff.append(r_curr)
        z_ff.append(z_curr)
        phi_ff.append(phi_curr)
        iphi_ff.append(iphi_curr)
        value_ff.append(value_interpolator[iphi_curr](r_curr, z_curr))
        s_ff.append(s_arc)
        
        _tri = trifinder(r_curr, z_curr)
        tri_ff.append(_tri)

        # Anyway stop if we left the (r, z) triangulation domain. 
        if _tri == -1:
            # replace the value_ff to the nearest one (to fill the most reasonable value)
            idx_closest_node = sj.find_mesh_indice(xr.mesh.r, xr.mesh.z, r_curr, z_curr)
            
            r_ff[-1] = xr.mesh.r[idx_closest_node]
            z_ff[-1] = xr.mesh.z[idx_closest_node]
            value_ff[-1] = value_interpolator[iphi_curr](xr.mesh.r[idx_closest_node], xr.mesh.z[idx_closest_node])
            break

        # follow field line to next plane
        _r, _z, _iphi, _s = follow_field_to_next_plane(r_curr, z_curr, iphi_curr, bfield_interpolator, nplanes, wedge_n, bt_sign, bt_backward=bt_backward, 
                                                       ff_step=ff_step, ff_order=ff_order)
        
        r_curr = _r
        z_curr = _z
        phi_curr += delta_phi*bt_sign*bt_back
        iphi_curr = _iphi
        s_arc += _s

        step += 1
    
    r_ff = np.asarray(r_ff, dtype=float)
    z_ff = np.asarray(z_ff, dtype=float)
    phi_ff = np.asarray(phi_ff, dtype=float)
    iphi_ff = np.asarray(iphi_ff, dtype=int)
    s_ff = np.asarray(s_ff, dtype=float)
    tri_ff = np.asarray(tri_ff, dtype=int)
    
    return r_ff, z_ff, phi_ff, iphi_ff, value_ff, s_ff, tri_ff

def trace_fieldline_values_from_OMP(isurf_ff, iphi, value_interpolator, bfield_interpolator, xr, nplanes, wedge_n, bt_sign, bt_backward=False, 
                                    ff_step=4, ff_order=4, toroidal_turns=1, follow_until_exit=False, max_exit_turns=10,
                                    verbose=True):
    if verbose: print(f"tracing OMP on isurf {isurf_ff}")
    
    inodes_ff = xr.mesh.surf_idx[isurf_ff,0:xr.mesh.surf_len[isurf_ff]]-1

    msk_lfs = (xr.mesh.r[inodes_ff] > xr.eq_axis_r)
    idx_ff = np.argmin(abs(xr.mesh.z[inodes_ff[msk_lfs]]-xr.eq_axis_z))

    r_start = xr.mesh.r[inodes_ff[msk_lfs][idx_ff]]
    z_start = xr.mesh.z[inodes_ff[msk_lfs][idx_ff]]
    iphi_start = iphi

    r_ff, z_ff, phi_ff, iphi_ff, value_ff, s_ff, tri_ff = trace_fieldline_values(r_start, z_start, iphi_start, value_interpolator, bfield_interpolator, xr, nplanes, wedge_n, bt_sign, bt_backward=bt_backward, 
                                                                                 ff_step=ff_step, ff_order=ff_order, toroidal_turns=toroidal_turns, follow_until_exit=follow_until_exit, max_exit_turns=max_exit_turns, 
                                                                                 verbose=False)
    
    return r_ff, z_ff, phi_ff, iphi_ff, value_ff, s_ff, tri_ff

#--- Save into .vtp file
def _cyl_to_xyz(r, z, phi):
    x = r*np.cos(phi)
    y = r*np.sin(phi)
    return x, y, z

def save_ff_vtp(r_ff, z_ff, phi_ff, s_ff, filename, point_scalars=None):
    print(f"saving {filename}")
    
    # Convert it into XYZ coordinate
    x_ff, y_ff, z_ff = _cyl_to_xyz(r_ff, z_ff, phi_ff)

    # into points
    #  * to match VTK's XYZ coordinate, y z are swapped with reversed y
    points_xyz = np.column_stack((x_ff, z_ff, -y_ff))
    
    # into poly
    poly = pv.lines_from_points(points_xyz)
    poly.point_data["arc_length"] = s_ff.astype(np.float32)
    if point_scalars:
        for name, arr in point_scalars.items():
            arr = np.asarray(arr)
            poly.point_data[name] = arr.astype(np.float32)

    # write .vtp
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    poly.save(filename)
