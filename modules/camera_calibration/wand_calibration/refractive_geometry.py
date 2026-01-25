import numpy as np
from dataclasses import dataclass
from typing import Optional, Tuple

@dataclass
class Ray:
    o: np.ndarray      # (3,) world coord
    d: np.ndarray      # (3,) normalized, world coord
    valid: bool
    reason: str = ""
    cam_id: int = -1
    window_id: int = -1
    frame_id: int = -1
    endpoint: str = "" # "A" or "B"
    uv: Tuple[float, float] = (np.nan, np.nan)

def normalize(v: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(v)
    if norm < 1e-12:
        return v
    return v / norm

def _extract_line3d(line):
    """
    Robustly extract pt and unit_vector from a C++ Line3D object.
    Supports multiple binding styles.
    """
    # Try .pt and .unit_vector (found in pyMatrix.cpp)
    if hasattr(line, "pt") and hasattr(line, "unit_vector"):
        o = np.array([line.pt[0], line.pt[1], line.pt[2]])
        d = np.array([line.unit_vector[0], line.unit_vector[1], line.unit_vector[2]])
        return o, d
    
    # Try .p0 and .p1 (alternative style)
    if hasattr(line, "p0") and hasattr(line, "p1"):
        o = np.array([line.p0[0], line.p0[1], line.p0[2]])
        p1 = np.array([line.p1[0], line.p1[1], line.p1[2]])
        return o, normalize(p1 - o)
    
    # Try getter methods
    try:
        pt = line.getPoint()
        vec = line.getVector()
        return np.array([pt[0], pt[1], pt[2]]), np.array([vec[0], vec[1], vec[2]])
    except AttributeError:
        pass

    # If all fail, provide diagnostic info
    attrs = dir(line)
    raise AttributeError(f"Could not extract geometry from Line3D object. Available attributes: {attrs}")

# Removed _pixel_to_undist_normalized as pinplateLine expects pixels when cam_mtx is set

_PROBE_LOGGED = False
_RAY_ORIGIN_AUDIT_COUNT = 0
_RAY_ORIGIN_AUDIT_PRINTED = False

def build_pinplate_ray_cpp(cam, uv, *, cam_id=-1, window_id=-1, frame_id=-1, endpoint="") -> Ray:
    """
    Build object-side refracted ray using C++ lineOfSight.
    
    Input 'uv' should be PIXEL coordinates.
    The C++ Camera.lineOfSight() handles undistortion and normalization internally.
    """
    global _PROBE_LOGGED, _RAY_ORIGIN_AUDIT_COUNT, _RAY_ORIGIN_AUDIT_PRINTED
    import pyopenlpt as lpt
    
    u, v = float(uv[0]), float(uv[1])
    
    try:
        # Use lineOfSight which accepts pixel coordinates directly
        # It calls undistort() internally, then passes to pinplateLine()
        line = cam.lineOfSight(lpt.Pt2D(u, v))
        o, d = _extract_line3d(line)
        
        if np.all(np.isfinite(o)) and np.all(np.isfinite(d)):
            if not _PROBE_LOGGED:
                # print(f"[Refractive] lineOfSight: Input ({u:.1f},{v:.1f}) pixel coords")
                _PROBE_LOGGED = True
            
            # Sanity check on direction z (should be >> 0 for forward facing)
            d_norm = normalize(d)
            if abs(d_norm[2]) < 0.1:
                # Diagnostics for the "collapsed ray" issue
                return Ray(o=o, d=d_norm, valid=False, reason=f"collapsed_ray_z={d_norm[2]:.4f}", 
                           cam_id=cam_id, window_id=window_id, frame_id=frame_id, endpoint=endpoint, uv=(u, v))
                
            return Ray(o=o, d=d_norm, valid=True, cam_id=cam_id, window_id=window_id, frame_id=frame_id, endpoint=endpoint, uv=(u, v))
            
    except Exception as e:
        reason = f"C++ lineOfSight failed: {str(e)}"
        if "total internal reflection" in reason.lower():
             reason = "total_internal_reflection"
        return Ray(o=np.zeros(3), d=np.array([0, 0, 1.0]), valid=False, reason=reason, 
                   cam_id=cam_id, window_id=window_id, frame_id=frame_id, endpoint=endpoint, uv=(u, v))
    
    # Fallback
    return Ray(o=np.zeros(3), d=np.array([0, 0, 1.0]), valid=False, reason="extraction_failed", 
               cam_id=cam_id, window_id=window_id, frame_id=frame_id, endpoint=endpoint, uv=(u, v))

def triangulate_point(rays_list: list) -> Tuple[np.ndarray, float, bool, str]:
    """
    Intersection of N rays in 3D using least squares.
    Formula: Σ (I - di*di.T) * X = Σ (I - di*di.T) * oi
    """
    valid_rays = [r for r in rays_list if r.valid]
    if len(valid_rays) < 2:
        return np.zeros(3), 0.0, False, "insufficient_valid_rays"
    
    A = np.zeros((3, 3))
    b = np.zeros(3)
    I = np.eye(3)
    
    for ray in valid_rays:
        d = ray.d.reshape(3, 1)
        # (I - d*d.T)
        proj_perp = I - d @ d.T
        A += proj_perp
        b += proj_perp @ ray.o
    
    try:
        # Check conditioning
        cond = np.linalg.cond(A)
        # Using solve for better accuracy if conditioned well
        if cond < 1e12:
            X = np.linalg.solve(A, b)
            return X, cond, True, ""
        else:
            # Ill-conditioned: Fallback to lstsq but mark as success=False for wand triangulation
            X, _, rank, _ = np.linalg.lstsq(A, b, rcond=None)
            # For wand calib, we usually want 3 rays and full rank
            return X, cond, False, f"ill_conditioned: cond={cond:.2e}, rank={rank}"
    except (np.linalg.LinAlgError, ValueError) as e:
        return np.zeros(3), np.inf, False, f"linalg_error: {e}"

def point_to_ray_dist(X: np.ndarray, o: np.ndarray, d: np.ndarray) -> float:
    """
    Distance from point X to ray (o, d) treated as a HALF-LINE.
    If projection is backward (t < 0), clamp to ray origin.
    
    This enforces forward-only ray geometry to prevent non-physical
    solutions where objects appear on the camera-side of refractive planes.
    """
    d = d / (np.linalg.norm(d) + 1e-12)  # safety normalize
    v = X - o
    t = float(np.dot(v, d))

    if t >= 0.0:
        # Forward projection: use perpendicular distance
        v_perp = v - t * d
        return float(np.linalg.norm(v_perp))
    else:
        # Backward projection: clamp to origin, return distance to origin
        return float(np.linalg.norm(v))


def point_to_ray_dist_vec(X: np.ndarray, O: np.ndarray, D: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """
    Vectorized INFINITE LINE distance from point X to rays (O, D).
    
    Ray distance residual measures only geometric deviation from the ray line
    and is invariant to ray origin translation. Physical feasibility 
    (point ordering / plane side) is enforced separately via a soft side penalty.
    
    Args:
        X: (3,) single 3D point
        O: (N, 3) ray origins
        D: (N, 3) ray directions (will be normalized)
        eps: small value for numerical stability
    
    Returns:
        (N,) array of perpendicular distances from X to each ray (infinite line)
    """
    # Normalize directions
    Dn = D / (np.linalg.norm(D, axis=1, keepdims=True) + eps)
    
    # Vector from each origin to point X
    V = X - O  # (N, 3)
    
    # Projection parameter t for each ray (can be negative, that's OK)
    t = np.sum(V * Dn, axis=1)  # (N,)
    
    # Perpendicular component (always non-negative distance)
    perp = V - t[:, None] * Dn  # (N, 3)
    
    # Distance = perpendicular distance (infinite line, no clamp)
    return np.linalg.norm(perp, axis=1)  # (N,)

def closest_distance_rays(ray1: Ray, ray2: Ray) -> float:
    """
    Compute minimal distance between two 3D lines.
    L1: P1 + s1*d1
    L2: P2 + s2*d2
    """
    p1, d1 = ray1.o, ray1.d
    p2, d2 = ray2.o, ray2.d
    
    w0 = p1 - p2
    a = np.dot(d1, d1)
    b = np.dot(d1, d2)
    c = np.dot(d2, d2)
    d = np.dot(d1, w0)
    e = np.dot(d2, w0)
    
    denom = a * c - b * b
    if abs(denom) < 1e-12:
        # Parallel lines
        return np.linalg.norm(np.cross(w0, d1)) / np.sqrt(a)
    
    s = (b * e - c * d) / denom
    t = (a * e - b * d) / denom
    
    closest1 = p1 + s * d1
    closest2 = p2 + t * d2
    
    return np.linalg.norm(closest1 - closest2)


def compute_tangent_basis(n: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Construct an orthonormal basis (t1, t2) tangent to unit normal n.
    n must be normalized.
    """
    # Pick a helper vector not parallel to n
    if abs(n[0]) < 0.9:
        a = np.array([1.0, 0.0, 0.0])
    else:
        a = np.array([0.0, 1.0, 0.0])
        
    t1 = np.cross(n, a)
    t1 /= np.linalg.norm(t1)
    t2 = np.cross(n, t1)
    
    return t1, t2

def update_normal_tangent(n_current: np.ndarray, alpha: float, beta: float) -> np.ndarray:
    """
    Update normal using tangent space parameters (alpha, beta).
    Returns normalized new normal.
    """
    t1, t2 = compute_tangent_basis(n_current)
    n_new = n_current + alpha * t1 + beta * t2
    return normalize(n_new)


def rodrigues_to_R(rvec: np.ndarray) -> np.ndarray:
    """
    Convert Rodrigues vector (3,) to rotation matrix (3,3).
    Uses cv2.Rodrigues if available, otherwise manual implementation.
    """
    rvec = np.asarray(rvec).flatten()
    theta = np.linalg.norm(rvec)
    if theta < 1e-12:
        return np.eye(3)
    k = rvec / theta
    K = np.array([
        [0, -k[2], k[1]],
        [k[2], 0, -k[0]],
        [-k[1], k[0], 0]
    ])
    R = np.eye(3) + np.sin(theta) * K + (1 - np.cos(theta)) * (K @ K)
    return R


def camera_center(R: np.ndarray, t: np.ndarray) -> np.ndarray:
    """
    Compute camera center in world coordinates.
    C = -R^T @ t
    """
    return -R.T @ np.asarray(t).flatten()


def angle_between_vectors(v1: np.ndarray, v2: np.ndarray) -> float:
    """
    Compute angle between two vectors in DEGREES.
    """
    v1 = np.asarray(v1).flatten()
    v2 = np.asarray(v2).flatten()
    n1 = np.linalg.norm(v1)
    n2 = np.linalg.norm(v2)
    if n1 < 1e-12 or n2 < 1e-12:
        return 0.0
    cos_theta = np.clip(np.dot(v1, v2) / (n1 * n2), -1.0, 1.0)
    return np.degrees(np.arccos(cos_theta))


def optical_axis_world(R: np.ndarray) -> np.ndarray:
    """
    Get camera optical axis (z-axis) in world coordinates.
    z_world = R^T @ [0, 0, 1]
    """
    return R.T @ np.array([0.0, 0.0, 1.0])


def compute_plane_intersection_line(n0: np.ndarray, pt0: np.ndarray,
                                     n1: np.ndarray, pt1: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute intersection line of two planes.
    
    Returns:
        line_dir: Unit direction vector of intersection line
        line_pt: A point on the intersection line
    """
    n0 = n0 / (np.linalg.norm(n0) + 1e-12)
    n1 = n1 / (np.linalg.norm(n1) + 1e-12)
    
    # Direction is cross product of normals
    line_dir = np.cross(n0, n1)
    norm_dir = np.linalg.norm(line_dir)
    
    if norm_dir < 1e-9:
        # Planes are parallel, no intersection line
        return np.array([0.0, 1.0, 0.0]), np.zeros(3)
    
    line_dir = line_dir / norm_dir
    
    # Find a point on the intersection line
    # Solve: dot(n0, p - pt0) = 0, dot(n1, p - pt1) = 0
    # Use least squares with constraint that p lies on both planes
    d0 = np.dot(n0, pt0)
    d1 = np.dot(n1, pt1)
    
    A = np.vstack([n0, n1, line_dir])
    b = np.array([d0, d1, 0.0])
    
    try:
        line_pt = np.linalg.lstsq(A, b, rcond=None)[0]
    except:
        line_pt = (pt0 + pt1) / 2
    
    return line_dir, line_pt


def build_rotation_align_y_to_dir(target_dir: np.ndarray) -> np.ndarray:
    """
    Build 3x3 rotation matrix that aligns world Y-axis [0,1,0] to target_dir.
    
    Uses Rodrigues formula for rotation about axis perpendicular to both vectors.
    """
    target_dir = target_dir / (np.linalg.norm(target_dir) + 1e-12)
    y_axis = np.array([0.0, 1.0, 0.0])
    
    # If already aligned (or anti-aligned)
    cos_theta = np.dot(y_axis, target_dir)
    if abs(cos_theta) > 0.9999:
        if cos_theta > 0:
            return np.eye(3)
        else:
            # 180 degree rotation about X axis
            return np.diag([1.0, -1.0, -1.0])
    
    # Rotation axis (perpendicular to both)
    axis = np.cross(y_axis, target_dir)
    axis = axis / np.linalg.norm(axis)
    
    # Rotation angle
    theta = np.arccos(np.clip(cos_theta, -1.0, 1.0))
    
    # Rodrigues formula: R = I + sin(θ)K + (1-cos(θ))K²
    K = np.array([
        [0, -axis[2], axis[1]],
        [axis[2], 0, -axis[0]],
        [-axis[1], axis[0], 0]
    ])
    R = np.eye(3) + np.sin(theta) * K + (1 - np.cos(theta)) * (K @ K)
    
    return R


def apply_coordinate_rotation(
    R_world: np.ndarray,
    cam_params: dict,
    window_planes: dict,
    points_3d: Optional[list] = None,
    t_shift: Optional[np.ndarray] = None
) -> Tuple[dict, dict, Optional[list]]:
    """
    Apply world coordinate rotation AND translation to all calibration data.
    
    Transforms all coordinates: P_new = R_world @ (P_old + t_shift)
    
    Args:
        R_world: 3x3 rotation matrix
        cam_params: Dict of cam_id -> [rvec(3), tvec(3), ...]
        window_planes: Dict of window_id -> {'plane_pt': [...], 'plane_n': [...]}
        points_3d: Optional list of 3D points
        t_shift: Optional (3,) vector to shift world origin BEFORE rotation.
        
    Returns:
        (new_cam_params, new_window_planes, new_points_3d)
    """
    import cv2
    
    if t_shift is None:
        t_shift = np.zeros(3)
    else:
        t_shift = np.array(t_shift).flatten()
    
    new_cam_params = {}
    new_window_planes = {}
    
    # Transform camera extrinsics
    for cid, params in cam_params.items():
        params = np.array(params).flatten()
        rvec_old = params[0:3]
        tvec_old = params[3:6]
        
        R_old = cv2.Rodrigues(rvec_old.reshape(3, 1))[0]
        
        # C_world = -R_old.T @ t_old (camera center in world)
        # C_new = R_world @ (C_world + t_shift)
        
        # Rotations just compose: R_new = R_old @ R_world.T
        R_new = R_old @ R_world.T
        
        # Camera center transforms
        C_old = -R_old.T @ tvec_old
        C_new = R_world @ (C_old + t_shift)
        
        # t_new = -R_new @ C_new
        t_new = -R_new @ C_new
        
        rvec_new = cv2.Rodrigues(R_new)[0].flatten()
        
        new_params = params.copy()
        new_params[0:3] = rvec_new
        new_params[3:6] = t_new
        new_cam_params[cid] = new_params
    
    # Transform window planes
    for wid, pl in window_planes.items():
        pt_old = np.array(pl['plane_pt'])
        n_old = np.array(pl['plane_n'])
        
        # Plane point shifts and rotates
        pt_new = R_world @ (pt_old + t_shift)
        # Normal vector only rotates
        n_new = R_world @ n_old
        
        new_window_planes[wid] = {
            **pl,
            'plane_pt': pt_new.tolist(),
            'plane_n': n_new.tolist()
        }
    
    # Transform 3D points
    new_points_3d = None
    if points_3d is not None and len(points_3d) > 0:
        pts = np.array(points_3d).reshape(-1, 3)
        # Shift then Rotate
        pts_shifted = pts + t_shift
        pts_new = (R_world @ pts_shifted.T).T
        new_points_3d = pts_new.tolist()
    
    return new_cam_params, new_window_planes, new_points_3d


def align_world_y_to_plane_intersection(
    window_planes: dict,
    cam_params: dict,
    points_3d: Optional[list] = None
) -> Tuple[dict, dict, Optional[list], np.ndarray]:
    """
    Align world Y-axis to the intersection line of two window planes.
    
    Returns:
        (new_cam_params, new_window_planes, new_points_3d, R_world)
    """
    wids = list(window_planes.keys())
    
    if len(wids) < 2:
        # Only one plane, no intersection line
        print("[Coordinate Alignment] Only one window plane, skipping Y-axis alignment.")
        return cam_params, window_planes, points_3d, np.eye(3)
    
    # Use first two planes
    pl0 = window_planes[wids[0]]
    pl1 = window_planes[wids[1]]
    
    n0 = np.array(pl0['plane_n'])
    pt0 = np.array(pl0['plane_pt'])
    n1 = np.array(pl1['plane_n'])
    pt1 = np.array(pl1['plane_pt'])
    
    # Compute intersection line
    line_dir, line_pt = compute_plane_intersection_line(n0, pt0, n1, pt1)
    
    # Sign stabilization: ensure line_dir points towards +Y hemisphere
    line_dir = line_dir / (np.linalg.norm(line_dir) + 1e-12)
    if np.dot(line_dir, np.array([0.0, 1.0, 0.0])) < 0:
        line_dir = -line_dir
    
    print(f"[Coordinate Alignment] Intersection line direction: [{line_dir[0]:.4f}, {line_dir[1]:.4f}, {line_dir[2]:.4f}]")
    
    print(f"[Coordinate Alignment] Intersection line direction: [{line_dir[0]:.4f}, {line_dir[1]:.4f}, {line_dir[2]:.4f}]")
    
    # [USER REQUEST] Strict Alignment:
    # 1. Y-axis = Intersection Line (lies on Plane 0)
    # 2. X-axis = Parallel to Plane 0 (orthogonal to n0)
    # 3. Z-axis = Normal of Plane 0 (n0)
    
    # New Basis Vectors (in Old Frame)
    y_new = line_dir
    z_new = n0 / (np.linalg.norm(n0) + 1e-12)
    
    # Check if Z points roughly "Forward" (usually Z > 0). If n0 is pointing back, flip it?
    # But usually window normal points Out or In. 
    # Let's verify orthogonality. Y is on Plane 0, so Y.dot(n0) should be 0.
    # Force strict orthogonality for numerical stability
    z_new = z_new - np.dot(z_new, y_new) * y_new
    z_new = z_new / (np.linalg.norm(z_new) + 1e-12)
    
    # Construct X = Y cross Z (Right-handed)
    x_new = np.cross(y_new, z_new)
    x_new = x_new / (np.linalg.norm(x_new) + 1e-12)
    
    # Construct Rotation Matrix R_world (Old -> New)
    # Rows are the new basis vectors expressed in old frame
    R_world = np.vstack([x_new, y_new, z_new])
    
    # Verification: R_world @ line_dir should be [0, 1, 0]
    line_dir_new = R_world @ line_dir
    # Verification: R_world @ n0 should be [0, 0, 1] (or -1)
    n0_new = R_world @ n0
    
    print(f"[ALIGN CHECK] Y-Axis (Intersection): {line_dir_new}")
    print(f"[ALIGN CHECK] Z-Axis (Plane 0 Normal): {n0_new}")
    
    # [USER REQUEST] Translation: Center World Origin at 3D Cloud Centroid
    t_shift = np.zeros(3)
    if points_3d is not None and len(points_3d) > 0:
        pts = np.array(points_3d).reshape(-1, 3)
        centroid = np.mean(pts, axis=0)
        t_shift = -centroid
        print(f"[Coordinate Alignment] Centering World at Cloud Centroid: {centroid}")
    else:
        print("[Coordinate Alignment] No 3D points provided, skipping Centering (Translation = 0).")
    
    # Apply transformation
    new_cam_params, new_window_planes, new_points_3d = apply_coordinate_rotation(
        R_world, cam_params, window_planes, points_3d, t_shift=t_shift
    )
    
    print(f"[Coordinate Alignment] Applied rotation and translation to align coordinates.")
    
    return new_cam_params, new_window_planes, new_points_3d, R_world, t_shift


def update_cpp_camera_state(cam_obj, 
                            extrinsics: Optional[dict] = None,
                            intrinsics: Optional[dict] = None,
                            plane_geom: Optional[dict] = None,
                            media_props: Optional[dict] = None) -> None:
    """
    Unified function to update C++ Camera parameters (PinPlate model).
    
    Handles:
    1. Intrinsics (K matrix, distortion)
    2. Extrinsics (R, T, and their inverses)
    3. Plane Geometry (Coordinate Shift: Closest -> Farthest)
    4. Refractive Media Properties
    5. Internal State Refresh (updatePt3dClosest)
    
    Args:
        cam_obj: lpt.Camera instance
        extrinsics: Dict with keys 'rvec' (3,) and 'tvec' (3,)
        intrinsics: Dict with optional keys 'f', 'cx', 'cy', 'dist'
        plane_geom: Dict with keys 'pt', 'n' (point and normal at Closest Interface)
        media_props: Dict with keys 'thickness' (and optional 'n1','n2','n3' or 'n_air'...)
    """
    try:
        import pyopenlpt as lpt
    except ImportError:
        return # Cannot update if lpt not available
        
    # Python-side binding usually exposes a copy of the struct
    pp = cam_obj._pinplate_param
    
    # --- 1. Intrinsics ---
    if intrinsics:
        # Create fresh K matrix
        K = lpt.MatrixDouble(3, 3, 0.0)
        
        # Use existing as default
        current_K = pp.cam_mtx
        fx = intrinsics.get('f', current_K[0, 0])
        fy = intrinsics.get('f', current_K[1, 1]) # Assume fx=fy if 'f' passed
        cx = intrinsics.get('cx', current_K[0, 2])
        cy = intrinsics.get('cy', current_K[1, 2])
        
        K[0, 0] = float(fx)
        K[1, 1] = float(fy)
        K[0, 2] = float(cx)
        K[1, 2] = float(cy)
        K[2, 2] = 1.0
        pp.cam_mtx = K
        
        # Distortion
        if 'dist' in intrinsics and intrinsics['dist'] is not None:
             d = intrinsics['dist']
             pp.dist_coeff = [float(x) for x in d]
             pp.n_dist_coeff = len(d)
             pp.is_distorted = any(abs(x) > 1e-9 for x in d)

    # --- 2. Extrinsics ---
    if extrinsics:
        # Support partial updates by reading current state if needed
        # Note: We need both R and T to recompute inverses correctly.
        
        has_r = 'rvec' in extrinsics
        has_t = 'tvec' in extrinsics
        
        if not has_r and not has_t:
            pass # No extrinsics to update
        else:
            # Get current values
            current_R_lpt = pp.r_mtx
            current_t_lpt = pp.t_vec
            
            # Resolve R
            if has_r:
                rvec = extrinsics['rvec']
                R_np = rodrigues_to_R(rvec)
            else:
                # Reconstruct numpy R from MatrixDouble
                R_np = np.zeros((3, 3))
                for i in range(3):
                    for j in range(3):
                        R_np[i, j] = current_R_lpt[i, j]
            
            # Resolve T
            if has_t:
                tvec = extrinsics['tvec']
                t_np = np.asarray(tvec).flatten()
            else:
                 t_np = np.array([current_t_lpt[0], current_t_lpt[1], current_t_lpt[2]])
            
            # Update Forward
            R_lpt = lpt.MatrixDouble(3, 3, 0.0)
            for i in range(3):
                for j in range(3):
                     R_lpt[i, j] = float(R_np[i, j])
            
            pp.r_mtx = R_lpt
            pp.t_vec = lpt.Pt3D(float(t_np[0]), float(t_np[1]), float(t_np[2]))
            
            # Update Inverse (Always recompute based on final R, T)
            R_inv_np = R_np.T
            t_inv_np = -R_inv_np @ t_np
            
            R_inv_lpt = lpt.MatrixDouble(3, 3, 0.0)
            for i in range(3):
                for j in range(3):
                     R_inv_lpt[i, j] = float(R_inv_np[i, j])
                     
            pp.r_mtx_inv = R_inv_lpt
            pp.t_vec_inv = lpt.Pt3D(float(t_inv_np[0]), float(t_inv_np[1]), float(t_inv_np[2]))

    # --- 3. Media & Thickness ---
    current_thick = 0.0
    if pp.w_array and len(pp.w_array) > 0:
        current_thick = pp.w_array[0]
        
    if media_props:
        # Update refractive indices if provided
        n1 = media_props.get('n1', media_props.get('n_air'))
        n2 = media_props.get('n2', media_props.get('n_window', media_props.get('n_glass')))
        n3 = media_props.get('n3', media_props.get('n_object', media_props.get('n_medium')))
        
        if n1 is not None and n2 is not None and n3 is not None:
             # Convention: [n_obj, n_win, n_air] (Farthest -> Nearest)
             pp.refract_array = [float(n3), float(n2), float(n1)]
        
        if 'thickness' in media_props:
            current_thick = float(media_props['thickness'])
            pp.w_array = [current_thick]

    # --- 4. Plane Geometry (Coordinate Shift) ---
    if plane_geom:
        pt = np.asarray(plane_geom['pt']).flatten()
        plane_normal = np.asarray(plane_geom['n']).flatten()
        
        # [CRITICAL] Coordinate Alignment: Shift to Farthest Interface
        # C++ model wants point on Farthest interface
        # Python tracks Closest interface (or interface point)
        p_farthest = pt + plane_normal * current_thick
        
        pl_struct = pp.plane
        pl_struct.pt = lpt.Pt3D(float(p_farthest[0]), float(p_farthest[1]), float(p_farthest[2]))
        pl_struct.norm_vector = lpt.Pt3D(float(plane_normal[0]), float(plane_normal[1]), float(plane_normal[2]))
        
        pp.plane = pl_struct
        
    # --- 5. Commit and Refresh ---
    cam_obj._pinplate_param = pp
    cam_obj.updatePt3dClosest()


