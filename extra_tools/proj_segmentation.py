#!/usr/bin/env python3
import os
import sys
import argparse
import numpy as np
from PIL import Image
import open3d as o3d

def find_depth_file(depth_dir, base_name):
    candidates = [f"{base_name}_depth.npy", f"{base_name}.npy"]
    for fname in candidates:
        path = os.path.join(depth_dir, fname)
        if os.path.isfile(path):
            return path
    raise FileNotFoundError(f"No depth .npy file found for '{base_name}' in {depth_dir}")

def load_camera_intrinsics(cameras_txt, cam_id):
    """
    Parses cameras.txt (COLMAP format) to extract intrinsics and image resolution.
    Returns: fx, fy, cx, cy, image_width, image_height
    """
    with open(cameras_txt, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            parts = line.split()
            cid = int(parts[0])
            if cid == cam_id:
                # COLMAP cameras.txt format: ID, MODEL, WIDTH, HEIGHT, FX, FY, CX, CY
                img_width = int(parts[2])
                img_height = int(parts[3])
                fx = float(parts[4])
                fy = float(parts[5])
                cx = float(parts[6])
                cy = float(parts[7])
                return fx, fy, cx, cy, img_width, img_height
    raise RuntimeError(f"Camera ID {cam_id} not found in {cameras_txt}")

def backproject(depth, fx, fy, cx, cy):
    h, w = depth.shape
    u = np.arange(w)
    v = np.arange(h)
    uu, vv = np.meshgrid(u, v)
    Z = depth
    X = (uu - cx) * Z / fx
    Y = (vv - cy) * Z / fy
    pts = np.stack((X, Y, Z), axis=-1)
    return pts.reshape(-1, 3)

# Quaternion to rotation matrix
def quaternion_to_rotation_matrix(qw, qx, qy, qz):
    return np.array([
        [1-2*(qy*qy+qz*qz),   2*(qx*qy- qz*qw), 2*(qx*qz+qy*qw)],
        [2*(qx*qy+ qz*qw), 1-2*(qx*qx+qz*qz),   2*(qy*qz- qx*qw)],
        [2*(qx*qz- qy*qw),   2*(qy*qz+ qx*qw), 1-2*(qx*qx+qy*qy)]
    ])

def main(main_dir):
    images_txt = os.path.join(main_dir, 'images.txt')
    cameras_txt = os.path.join(main_dir, 'cameras.txt')
    depth_dir = os.path.join(main_dir, 'depth_maps')
    rgb_dir = os.path.join(main_dir, 'imgs_6')
    mask_dir   = os.path.join(rgb_dir, 'masks')

    entries = []
    with open(images_txt, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            entries.append(line.split())

    if not entries:
        print("No image entries found in images.txt", file=sys.stderr)
        sys.exit(1)

    pcds = []
    proj_pcds = []
    for parts in entries:
        qw, qx, qy, qz = map(float, parts[1:5])
        tx, ty, tz = map(float, parts[5:8])
        cam_id = int(parts[8])
        filename = parts[9]
        base = os.path.splitext(filename)[0]

        try:
            fx, fy, cx, cy, cam_w, cam_h = load_camera_intrinsics(cameras_txt, cam_id)
        except Exception as e:
            print(e, file=sys.stderr)
            continue

        print(f"\n--- Image: {filename} ---")
        print(f"Camera Resolution (from cameras.txt): width={cam_w}, height={cam_h}")
        print(f"Camera Intrinsics: fx={fx}, fy={fy}, cx={cx}, cy={cy}")
        print(f"Camera Extrinsics: qw={qw}, qx={qx}, qy={qy}, qz={qz}, tx={tx}, ty={ty}, tz={tz}")

        try:
            depth = np.load(find_depth_file(depth_dir, base))
            dh, dw = depth.shape
            ratio_w = dw / cam_w
            ratio_h = dh / cam_h
            print(f"Depth map size: width={dw}, height={dh}")
            print(f"Depth/Camera resolution ratio: width_ratio={ratio_w:.3f}, height_ratio={ratio_h:.3f}")
        except Exception as e:
            print(e, file=sys.stderr)
            continue

        fx_s, fy_s = fx * ratio_w, fy * ratio_h
        cx_s, cy_s = cx * ratio_w, cy * ratio_h
        print(f"Scaled Intrinsics for Depth: fx'={fx_s:.3f}, fy'={fy_s:.3f}, cx'={cx_s:.3f}, cy'={cy_s:.3f}")

        rgb_path = os.path.join(rgb_dir, filename)
        if not os.path.isfile(rgb_path):
            print(f"RGB image not found: {rgb_path}", file=sys.stderr)
            continue
        rgb_img = Image.open(rgb_path)
        iw, ih = rgb_img.size
        rgb = np.array(rgb_img)
        
        # Load mask
        mask_path = os.path.join(mask_dir, base + '_mask.png')
        if os.path.isfile(mask_path):
            mask_img = Image.open(mask_path).convert('L')
            mask = (np.array(mask_img) > 128)   # boolean array: True where mask is white
            print(f"Loaded mask: {mask_path}")
        else:
            # fallback: keep all pixels
            mask = np.ones_like(depth, dtype=bool)
            print(f"No mask found for {base}, defaulting to all-True mask")


        pts_flat   = backproject(depth, fx_s, fy_s, cx_s, cy_s)
        cols_flat = rgb.reshape(-1, 3) / 255.0
        # flatten the mask and apply:
        mask_flat = mask.reshape(-1)
        pts  = pts_flat[mask_flat]
        cols = cols_flat[mask_flat]
        step = 10
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pts[::step])
        pcd.colors = o3d.utility.Vector3dVector(cols[::step])

        # Transform to world coordinates
        R = quaternion_to_rotation_matrix(qw, qx, qy, qz)
        R_inv = R.T
        t = np.array([tx, ty, tz])
        C = -R_inv.dot(t)
        pcd.rotate(R_inv, center=(0,0,0))
        pcd.translate(C)

        pcds.append(pcd)
        proj_pcds.append(pcd)

    # merge and save just the projected (masked) point clouds
    if proj_pcds:
        merged = o3d.geometry.PointCloud()
        for cloud in proj_pcds:
            merged += cloud
        save_path = os.path.join(main_dir, 'points3D_seg.ply')
        o3d.io.write_point_cloud(save_path, merged)
        print(f"Saved merged masked point cloud to {save_path}")

    # ask once whether to include the extra PLY clouds
    resp = input("Include extra PLY files in the final view? [y/N]: ").strip().lower()
    include_ply = resp in ('y', 'yes')
    
    if include_ply:
    # Load PLYs
        for root, _, files in os.walk(main_dir):
            for fname in files:
                if fname.lower().endswith('.ply'):
                    path = os.path.join(root, fname)
                    try:
                        extra = o3d.io.read_point_cloud(path)
                        if extra.has_colors():
                            col = np.asarray(extra.colors)
                            if col.max() > 1.0:
                                extra.colors = o3d.utility.Vector3dVector(col/255.0)
                        pcds.append(extra)
                        print(f"Loaded extra PLY: {path}")
                    except Exception as e:
                        print(f"Could not load {path}: {e}", file=sys.stderr)
    else:
        print("Skipping extra PLY files.")


    if not pcds:
        print("No point clouds to display.", file=sys.stderr)
        sys.exit(1)

    o3d.visualization.draw_geometries(pcds,
                                      window_name="All Point Clouds",
                                      width=1024, height=768)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Visualize all depth-based and PLY point clouds in global frame"
    )
    parser.add_argument('main_dir', help="Directory with images.txt, cameras.txt, depth_maps/, imgs_6/, and optional .ply files")
    args = parser.parse_args()
    main(args.main_dir)