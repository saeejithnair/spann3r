import rerun as rr
import numpy as np
import os.path as osp
import argparse
import cv2
from glob import glob
import re

def get_args_parser():
    parser = argparse.ArgumentParser('Spann3R Point Cloud Viewer')
    parser.add_argument('--demo_name', type=str, default='kitchen', help='Name of demo to visualize')
    parser.add_argument('--demos_folder', type=str, 
                       default='/pub0/smnair/robotics/scene_rep/graph-robotics/third-party/spann3r/output/demo',
                       help='Path to demos folder')
    parser.add_argument('--use_original_images', action='store_true', 
                       help='Use original images from examples/kitchen instead of resized ones')
    return parser

def main(args):
    output_folder = osp.join(args.demos_folder, args.demo_name)
    params_path = osp.join(output_folder, f"{args.demo_name}.npy")
    params = np.load(params_path, allow_pickle=True).item()

    rr.init(f"spann3r_{args.demo_name}", spawn=True)

    if args.use_original_images:
        # Find all images in examples/kitchen
        img_paths = glob(f"examples/{args.demo_name}/frame_*.jpg")
        
        for img_path in sorted(img_paths):
            # Extract frame number from filename
            frame_num = int(re.search(r'frame_(\d+)', img_path).group(1)) - 1
            
            img = cv2.imread(img_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            h, w = img.shape[:2]
            
            # Create pixel coordinates grid
            y, x = np.meshgrid(np.arange(h), np.arange(w), indexing='ij')
            pixels = np.stack([x.flatten(), y.flatten()], axis=-1)
            
            # Get camera parameters
            K = params['intrinsic']
            pose = params['poses_all'][frame_num]
            
            # Convert pixels to 3D points
            points = pixels_to_3d(pixels, K, pose)
            colors = img.reshape(-1, 3)
            
            rr.log(
                f"img_{frame_num}",
                rr.Points3D(points, colors=colors, radii=0.001)
            )
    else:
        # Original visualization code
        for i in range(len(params['images_all'])):
            points = params['pts_all'][i].reshape(-1, 3)
            colors = (params['images_all'][i].reshape(-1, 3) * 255).astype(np.uint8)
            
            rr.log(
                f"img_{i}",
                rr.Points3D(points, colors=colors, radii=0.001)
            )


def pixels_to_3d(pixels, K, pose):
    """Convert pixel coordinates to 3D points using camera parameters"""
    # Normalize pixel coordinates
    fx, fy = K[0,0], K[1,1]
    cx, cy = K[0,2], K[1,2]
    x = (pixels[:,0] - cx) / fx
    y = (pixels[:,1] - cy) / fy
    
    # Create rays in camera space
    rays = np.stack([x, y, np.ones_like(x)], axis=-1)
    
    # Transform rays to world space using pose
    R = pose[:3, :3]
    t = pose[:3, 3]
    rays_world = (R @ rays.T).T
    
    # TODO: You'll need to determine the depth for each ray
    # For now, setting a fixed depth
    depth = 5.0
    points = rays_world * depth[:, None] + t
    
    return points

if __name__ == '__main__':
    parser = get_args_parser()
    args = parser.parse_args()
    main(args)