#!/usr/bin/env python3
"""
Project semi-transparent colored masks from multiple images onto a 3D point cloud.
"""
import os
import argparse
import numpy as np
import open3d as o3d

def parse_cameras(path):
    cams = {}
    with open(path) as f:
        for line in f:
            if line.startswith('#') or not line.strip():
                continue
            parts = line.strip().split()
            cam_id = int(parts[0])
            model = parts[1]
            width, height = map(int, parts[2:4])
            fx, fy, cx, cy = map(float, parts[4:8])
            cams[cam_id] = {
                'model': model,
                'width': width,
                'height': height,
                'fx': fx,
                'fy': fy,
                'cx': cx,
                'cy': cy
            }
    return cams

def parse_images(path):
    imgs = []
    with open(path) as f:
        for line in f:
            if line.startswith('#') or not line.strip():
                continue
            parts = line.strip().split()
            if len(parts) < 10:
                continue
            img = {
                'id': int(parts[0]),
                'q': tuple(map(float, parts[1:5])),
                't': tuple(map(float, parts[5:8])),
                'cam_id': int(parts[8]),
                'name': parts[9]
            }
            imgs.append(img)
    imgs.sort(key=lambda x: x['id'])
    return imgs

def quaternion_to_rotation_matrix(q):
    qw, qx, qy, qz = q
    # normalize quaternion
    n = np.linalg.norm([qw, qx, qy, qz])
    qw, qx, qy, qz = qw/n, qx/n, qy/n, qz/n
    # build rotation matrix
    return np.array([
        [1 - 2*(qy**2 + qz**2),     2*(qx*qy - qz*qw),     2*(qx*qz + qy*qw)],
        [    2*(qx*qy + qz*qw), 1 - 2*(qx**2 + qz**2),     2*(qy*qz - qx*qw)],
        [    2*(qx*qz - qy*qw),     2*(qy*qz + qx*qw), 1 - 2*(qx**2 + qy**2)]
    ])

def main():
    parser = argparse.ArgumentParser(
        description="Project colored masks onto 3D point cloud"
    )
    parser.add_argument(
        "input_dir",
        help="Folder containing images.txt, cameras.txt, points3D.ply, and imgs_6/"
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.5,
        help="Mask transparency (0=transparent,1=opaque)"
    )
    args = parser.parse_args()

    # paths
    images_txt = os.path.join(args.input_dir, 'images.txt')
    cameras_txt = os.path.join(args.input_dir, 'cameras.txt')
    ply_path     = os.path.join(args.input_dir, 'points3D.ply')

    # load data
    print("Loading cameras...")
    cameras = parse_cameras(cameras_txt)
    print("Loading image poses...")
    images  = parse_images(images_txt)
    print("Reading point cloud...")
    pcd     = o3d.io.read_point_cloud(ply_path)
    points  = np.asarray(pcd.points)
    colors  = np.asarray(pcd.colors)

    # define mask colors (RGB, normalized)
    mask_colors = [
        np.array([1.0, 0.0, 0.0]),
        np.array([0.0, 1.0, 0.0]),
        np.array([0.0, 0.0, 1.0]),
        np.array([1.0, 1.0, 0.0]),
        np.array([1.0, 0.0, 1.0]),
        np.array([0.0, 1.0, 1.0])
    ]
    alpha = args.alpha

    # apply masks sequentially
    for idx, img in enumerate(images):
        cam = cameras.get(img['cam_id'], None)
        if cam is None:
            print(f"Warning: Camera ID {img['cam_id']} not found. Skipping.")
            continue
        # extrinsic
        R_mat = quaternion_to_rotation_matrix(img['q'])
        t_vec = np.array(img['t']).reshape(3, 1)
        # intrinsics
        fx, fy, cx, cy = cam['fx'], cam['fy'], cam['cx'], cam['cy']
        w, h = cam['width'], cam['height']

        # project points into this camera
        X_cam = (R_mat @ points.T) + t_vec           # shape 3 x N
        z     = X_cam[2, :]
        valid = z > 0
        x     = X_cam[0, valid] / z[valid]
        y     = X_cam[1, valid] / z[valid]
        u     = fx * x + cx
        v     = fy * y + cy
        inside = (u >= 0) & (u < w) & (v >= 0) & (v < h)
        idxs   = np.where(valid)[0][inside]

        # blend mask color
        mask_col = mask_colors[idx % len(mask_colors)]
        colors[idxs] = alpha * mask_col + (1 - alpha) * colors[idxs]
        print(f"{img['name']}: tinted {len(idxs)} points.")

    # update and save
    pcd.colors = o3d.utility.Vector3dVector(colors)
    out_ply = os.path.join(args.input_dir, 'colored_points.ply')
    o3d.io.write_point_cloud(out_ply, pcd)
    print(f"Saved colored point cloud to: {out_ply}")

    # visualize
    o3d.visualization.draw_geometries([pcd])

if __name__ == '__main__':
    main()
