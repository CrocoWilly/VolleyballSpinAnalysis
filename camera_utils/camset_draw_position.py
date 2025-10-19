import argparse
from .calibration import CameraFileCalibrator, CameraSetFileCalibrator
from .camera import Camera, CameraSet
import cv2 as cv
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser(description="Camera Set")
    parser.add_argument("camset_dir", type=str, help="path to the camera set directory")
    args = parser.parse_args()
    camset_dir = Path(args.camset_dir)
    if not camset_dir.exists():
        print(f"Camera set directory {camset_dir} does not exist")
        exit()

    calibrator = CameraSetFileCalibrator(camset_dir).load()
    camset = calibrator.get_camset()
    cameras = camset.cameras

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    view_arrow_length = 5
    color_map = plt.cm.get_cmap("rainbow")
    min_x, max_x = None, None
    min_y, max_y = None, None
    min_z, max_z = None, None
    cam_vecs = []

    for cid, camera in enumerate(cameras):
        camera: Camera
        color = color_map(cid / (len(cameras) + 3))
        T = camera.translation
        R = camera.rotation
        cam_world_pos = R.T @ (-T.reshape(-1))
        cam_view_pos = R.T @ (np.array([0, 0, 1])-T.reshape(-1))
        cam_view_dir = cam_view_pos - cam_world_pos
        ax.quiver(cam_world_pos[0], cam_world_pos[1], cam_world_pos[2], cam_view_dir[0], cam_view_dir[1], cam_view_dir[2], 
              length=view_arrow_length, normalize=True, color=color, arrow_length_ratio=0.5)
        ax.scatter(cam_world_pos[0], cam_world_pos[1], cam_world_pos[2], c=color, marker='o', s=10)
        # ax.scatter(cam_view_pos[0], cam_view_pos[1], cam_view_pos[2], c=color, marker='x', s=10)
        print((cam_world_pos[0], cam_world_pos[1], cam_world_pos[2]))
        ax.text(float(cam_world_pos[0] + 1), float(cam_world_pos[1] + 1), float(cam_world_pos[2]), camera.name, color=color)
        if min_x is None or cam_world_pos[0] < min_x:
            min_x = cam_world_pos[0]
        if max_x is None or cam_world_pos[0] > max_x:
            max_x = cam_world_pos[0]
        if min_y is None or cam_world_pos[1] < min_y:
            min_y = cam_world_pos[1]
        if max_y is None or cam_world_pos[1] > max_y:
            max_y = cam_world_pos[1]
        if min_z is None or cam_world_pos[2] < min_z:
            min_z = cam_world_pos[2]
        if max_z is None or cam_world_pos[2] > max_z:
            max_z = cam_world_pos[2]
        cam_vecs.append((camera, cam_world_pos, cam_view_dir, color))

    import itertools
    for cam1_tup, cam2_tup in itertools.combinations(cam_vecs, 2):
        cam1, cam1_pos, cam1_dir, cam1_color = cam1_tup
        cam2, cam2_pos, cam2_dir, cam2_color = cam2_tup
        # print(f"Camera {cam1.name} position: {cam1_pos} view direction: {cam1_dir}")
        # print(f"Camera {cam2.name} position: {cam2_pos} view direction: {cam2_dir}")

        # cam1_vec = cam1_dir - cam1_pos
        # cam2_vec = cam2_dir - cam2_pos
        cam1_vec = cam1_dir.copy()
        cam2_vec = cam2_dir.copy()
        angle = np.arccos(np.dot(cam1_vec, cam2_vec) / (np.linalg.norm(cam1_vec) * np.linalg.norm(cam2_vec)))
        print(f"Angle1 between {cam1.name} and {cam2.name}: {angle * 180 / np.pi:.2f} degrees")
        abs_val1 = abs(angle * 180 / np.pi - 90)
        # print(f"Cam1 vec: {cam1_vec} Cam2 vec: {cam2_vec}")
        cam1_vec[2] = 0
        cam2_vec[2] = 0
        # print(f"Cam1 vec: {cam1_vec} Cam2 vec: {cam2_vec}")
        angle = np.arccos(np.dot(cam1_vec, cam2_vec) / (np.linalg.norm(cam1_vec) * np.linalg.norm(cam2_vec)))
        print(f"Angle2 between {cam1.name} and {cam2.name}: {angle * 180 / np.pi:.2f} degrees")
        abs_val2 = abs(angle * 180 / np.pi - 90)

        print(f"Abs diff: {abs_val1:.2f} vs {abs_val2:.2f}")

        center_player_pos = (4.5, 9, 0)
        epipolar_plane = np.cross(cam1_pos - center_player_pos, cam2_pos - center_player_pos)
        if epipolar_plane[2] < 0:
            epipolar_plane = -epipolar_plane
        epipolar_plane /= np.linalg.norm(epipolar_plane)
        angle_epipolar = np.arccos(np.dot(epipolar_plane, np.array([0, 0, 1])))
        
        print(f"Angle between epipolar plane and z-axis: {angle_epipolar * 180 / np.pi:.2f} degrees")

    points = np.array([
        [0, 0, 0], [9, 0, 0], [0, 6, 0], [9, 6, 0], [0, 9, 0],
        [9, 9, 0], [0, 12, 0], [9, 12, 0], [0, 18, 0], [9, 18, 0]
    ])
    courtedge = [2, 0, 1, 3, 2, 4, 5, 3, 5, 7, 6, 4, 6, 8, 9, 7]
    curves = points[courtedge]

    netpoints = np.array([
        [0, 9, 0], [0, 9, 1.24], [0, 9, 2.24], [9, 9, 0], [9, 9, 1.24], [9, 9, 2.24]])
    netedge = [0, 1, 2, 5, 4, 1, 4, 3]
    netcurves = netpoints[netedge]

    court = points.T
    courtX, courtY, courtZ = court
    # plot 3D court reference points

    ax.scatter(courtX, courtY, courtZ, c='black', marker='o', s=1)
    ax.plot(curves[:, 0], curves[:, 1], c='k',
            linewidth=1, alpha=1.0)  # plot 3D court edges
    ax.plot(netcurves[:, 0], netcurves[:, 1], netcurves[:, 2],
            c='k', linewidth=1, alpha=0.5)  # plot 3D net edges
    ax.view_init(elev=90, azim=0)
    ax.set_xlim(min_x - 10, max_x + 10)
    ax.set_ylim(min_y - 10, max_y + 10)
    ax.set_zlim(min_z - 10, max_z + 10)
    fig.tight_layout()
    ax.set_aspect('equal', adjustable='box')
    fig.legend()
    fig.savefig(camset_dir / "camset_position.png", dpi=400)
    print("Saved", camset_dir / "camset_position.png")
    try:
        plt.show()
    except:
        pass


if __name__ == "__main__":
    main()