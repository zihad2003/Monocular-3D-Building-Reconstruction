import argparse
import cv2
import torch
from pathlib import Path

from src.models.multitask_net import MultiTaskBuildingNet
from src.reconstruction.predict import predict_mask_and_height
from src.reconstruction.extrude import mask_and_height_to_3d_mesh

def run_inference(image_path, checkpoint_path, output_path):
    device = torch.device("cpu") # Using CPU as requested for Windows
    
    print(f"Loading image: {image_path}")
    image_bgr = cv2.imread(image_path)
    if image_bgr is None:
        raise ValueError(f"Could not read {image_path}")
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    
    print(f"Loading model from {checkpoint_path}...")
    model = MultiTaskBuildingNet(encoder_name="resnet18", encoder_weights=None)
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state"] if "model_state" in ckpt else ckpt)
    model.to(device)
    model.eval()
    
    print("Predicting mask and height...")
    mask_prob, mask_bin, height_m = predict_mask_and_height(model, image_rgb, device)
    
    print("Extruding to 3D OBJ...")
    out_obj, n_bldgs = mask_and_height_to_3d_mesh(
        image_rgb=image_rgb,
        mask_bin=mask_bin,
        height_m=height_m,
        output_path=output_path,
        min_area_px=10, 
        separate_touching=False
    )
    print(f"Success! Generated {n_bldgs} building(s). Output saved to: {out_obj}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=str, required=True, help="Path to input image")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/best_model.pt", help="Path to model checkpoint")
    parser.add_argument("--output", type=str, default="outputs/building.obj", help="Path to output .obj file")
    args = parser.parse_args()
    
    run_inference(args.image, args.checkpoint, args.output)
