import numpy as np

def is_in_court(pt):
    x,y,z = pt
    if -1 <= x <= 10 and -2 <= y <= 20:
        return True
    return False

def vec_vec_angle(v1, v2):
    return np.arccos(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)))

def vec_tilt_angle(vec):
    angle = vec_vec_angle(vec, [0, 0, 1])
    angle = np.pi / 2 - angle
    return angle

def vec_tilt_angle_deg(vec):
    return np.degrees(vec_tilt_angle(vec))

def vec_yaxis_angle(vec):
    vec = np.array(vec)
    vec[2] = 0
    angle = vec_vec_angle(vec, [0, 1, 0])
    if angle > np.pi / 2:
        angle = np.pi - angle
    return angle

def vec_yaxis_angle_deg(vec):
    return np.degrees(vec_yaxis_angle(vec))

def fit_trajectory(frame_pos3d: dict, order=2):
    t = np.asarray(list(map(int, frame_pos3d.keys())), dtype=float)
    pos3d_list = np.asarray(list(frame_pos3d.values()), dtype=float)
    pos3d_list_T = pos3d_list.T
    fits = [np.polyfit(t, pos3d_list_T[i], order) for i in range(3)]
    return fits
    
def get_trajectory_land_pos(fits):
    roots = np.roots(fits[2])
    reals = np.real(roots[np.imag(roots)==0])
    if len(reals) == 0:
        return None
    land_t = max(reals)
    pred_land = np.asarray([np.polyval(fits[i], land_t) for i in range(3)])
    return pred_land