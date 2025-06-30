import os
import torch
import matplotlib.pyplot as plt
from torchvision.transforms.functional import to_pil_image
import torchvision.transforms.functional as TF
import numpy as np

import random
from tqdm import tqdm
from models.sr_vanilla_model import SRModel
from models.srflow_model import SRFlowGenerator
from datasets.div2k_dataset import get_dataloaders 
from utils.metrics import get_psnr_ssim_metrics

from config import (
    DEVICE, UPSCALE_FACTOR, CHANNELS,
    SRFLOW_NF, SRFLOW_NB, RESIZE_HEIGHT, RESIZE_WIDTH 
)

GLOBAL_SEED = 42

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
# ---------- Utility Functions ----------

def extract_zoom_patches(image, regions, target_size=None):
    """
    Extracts zoomed-in patches from an image.
    Args:
        image (torch.Tensor): The input image tensor (C, H, W).
        regions (list): A list of tuples, where each tuple is (x, y, size)
                        defining the top-left corner and size of the square patch.
        target_size (tuple, optional): (H, W) to resize the image to before extracting patches.
                                       Useful for aligning different resolution images.
    Returns:
        list: A list of image patch tensors.
    """
    patches = []
    # Ensure image is 2D (H, W) or 3D (C, H, W)
    if image.dim() == 4: # If it's a batch, take the first image
        image = image[0]
    
    if target_size and (image.shape[-2], image.shape[-1]) != target_size:
        # Only resize if target_size is provided and differs from current size
        image = TF.resize(image, target_size, antialias=True) # Use antialias for better quality

    _, H, W = image.shape # Get height and width after potential resize

    for x, y, size in regions:
        # Adjust coordinates to ensure they are within image bounds
        x_start = max(0, x)
        y_start = max(0, y)
        x_end = min(W, x + size)
        y_end = min(H, y + size)
        
        patch = image[:, y_start:y_end, x_start:x_end]
        
        # Pad if the extracted patch is smaller than the requested size (e.g., at borders)
        if patch.shape[1] < size or patch.shape[2] < size:
            pad_h_start = 0
            pad_h_end = size - patch.shape[1]
            pad_w_start = 0
            pad_w_end = size - patch.shape[2]
            patch = torch.nn.functional.pad(patch, (pad_w_start, pad_w_end, pad_h_start, pad_h_end), 'constant', 0)

        patches.append(patch)
    return patches


def plot_comparison_with_zooms(lr_img, hr_img, sr1_img, sr2_img,
                                psnr1, psnr2, ssim1, ssim2,
                                regions, title, save_path=None):
    """
    Plots a comparison of LR, HR, SR1, and SR2 images with zoomed-in patches.
    Args:
        lr_img (torch.Tensor): Low-resolution input image.
        hr_img (torch.Tensor): High-resolution ground truth image.
        sr1_img (torch.Tensor): Super-resolved image from Model 1 (SRModel).
        sr2_img (torch.Tensor): Super-resolved image from Model 2 (SRFlowGenerator).
        psnr1 (float): PSNR score for SR1 vs HR.
        psnr2 (float): PSNR score for SR2 vs HR.
        ssim1 (float): SSIM score for SR1 vs HR.
        ssim2 (float): SSIM score for SR2 vs HR.
        regions (list): List of (x, y, size) for zoom patches (relative to SR output size).
        title (str): Title for the main plot.
        save_path (str, optional): Path to save the plot. Defaults to None.
    """
    num_zoom_cols = len(regions)
    fig, axes = plt.subplots(4, num_zoom_cols + 1, figsize=(4 * (num_zoom_cols + 1), 16))
    
    # Get the target full resolution for all images to align them (e.g., 1024x1024)
    target_display_h, target_display_w = sr1_img.shape[-2], sr1_img.shape[-1]
    
    images = [lr_img, sr1_img, sr2_img, hr_img]
    row_titles = ['Low-Res Input (Upscaled)', 'SRModel Output', 'SRFlow Output', 'Ground Truth']

    for row, (img_tensor, label) in enumerate(zip(images, row_titles)):
        # Determine the target size for this specific image for displaying and patch extraction
        current_target_size = (target_display_h, target_display_w)
        
        # Resize main image for display
        display_img = TF.resize(img_tensor.cpu().clamp(0,1), current_target_size, antialias=True)
        
        axes[row][0].imshow(to_pil_image(display_img))
        axes[row][0].set_title(label, fontsize=14)
        axes[row][0].axis('off')

        # Extract patches, resizing the source image to target_display_h/w first
        patches = extract_zoom_patches(img_tensor, regions, target_size=current_target_size)
        for col, patch in enumerate(patches):
            axes[row][col + 1].imshow(to_pil_image(patch.cpu().clamp(0, 1)))
            axes[row][col + 1].set_title(f'Zoom {col+1}', fontsize=12)
            axes[row][col + 1].axis('off')

    psnr_str = f'PSNR – SRModel: {psnr1:.2f}, SRFlow: {psnr2:.2f}'
    ssim_str = f'SSIM – SRModel: {ssim1:.4f}, SRFlow: {ssim2:.4f}'
    plt.suptitle(f"{title}\n{psnr_str} | {ssim_str}", fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
    plt.show()

# ---------- Load Models and Data ----------

def load_model(model_type, seed=42):
    """
    Loads a pre-trained model based on its type and seed.
    Args:
        model_type (str): Type of the model ('SRModel' or 'SRFlowGenerator').
        seed (int): Random seed used for training the model.
    Returns:
        torch.nn.Module: The loaded model, or None if weights are not found.
    """
    if model_type == 'SRModel':
        model = SRModel(upscale_factor=UPSCALE_FACTOR, channels=CHANNELS)
    elif model_type == 'SRFlowGenerator':
        model = SRFlowGenerator(
            in_nc=CHANNELS, out_nc=CHANNELS,
            nf=SRFLOW_NF, nb=SRFLOW_NB,
            upscale_factor=UPSCALE_FACTOR
        )
    else:
        raise ValueError("Invalid model_type. Choose 'SRModel' or 'SRFlowGenerator'.")
    
    weight_path = f'./{model_type}_seed_{seed}_weights.pth'
    if not os.path.exists(weight_path):
        print(f"Warning: Weights not found at {weight_path}. Skipping model load for {model_type} seed {seed}.")
        return None
    model.load_state_dict(torch.load(weight_path, map_location=DEVICE))
    return model.to(DEVICE).eval()

# ---------- Main Comparison and Evaluation Functions ----------

def compare_models_qualitative(num_images=5):
    """
    Performs a qualitative comparison of SRModel and SRFlowGenerator,
    generating plots with zoomed-in patches and categorizing examples.
    Args:
        num_images (int): Number of *individual images* from the test set to process for qualitative analysis.
    Returns:
        list: A list of dictionaries, each containing data for one comparison sample.
    """
    # Get all dataloaders, but we'll manually sample from the test_dataset
    _, _, test_loader = get_dataloaders()
    test_dataset = test_loader.dataset # Get the dataset directly

    model_sr = load_model('SRModel')
    model_flow = load_model('SRFlowGenerator')

    if model_sr is None or model_flow is None:
        print("Cannot run qualitative comparison: one or both models could not be loaded.")
        return []

    # Initialize torchmetrics for individual image PSNR/SSIM calculation
    psnr_metric_individual = get_psnr_ssim_metrics(data_range=1.0)[0].to(DEVICE)
    ssim_metric_individual = get_psnr_ssim_metrics(data_range=1.0)[1].to(DEVICE)

    results = []
    
    # Randomly select indices for the desired number of images
    # Ensure num_images doesn't exceed dataset size
    num_images = min(num_images, len(test_dataset))
    selected_indices = np.random.choice(len(test_dataset), num_images, replace=False)

    print(f"Collecting {num_images} samples for qualitative comparison...")
    with torch.no_grad():
        for i, idx in enumerate(tqdm(selected_indices, desc="Processing selected images")):
            lr_img, hr_img = test_dataset[idx] # Get single image
            lr_img = lr_img.unsqueeze(0).to(DEVICE) # Add batch dimension
            hr_img = hr_img.unsqueeze(0).to(DEVICE) # Add batch dimension

            sr1_img = model_sr(lr_img)
            sr2_img = model_flow(lr_img)

            # Calculate PSNR/SSIM for each image pair individually
            psnr1 = psnr_metric_individual(sr1_img, hr_img).item()
            ssim1 = ssim_metric_individual(sr1_img, hr_img).item()
            psnr2 = psnr_metric_individual(sr2_img, hr_img).item()
            ssim2 = ssim_metric_individual(sr2_img, hr_img).item()

            results.append({
                "index": idx, # Store original index for reference
                "lr": lr_img.squeeze(0).cpu().clamp(0, 1),
                "hr": hr_img.squeeze(0).cpu().clamp(0, 1),
                "sr1": sr1_img.squeeze(0).cpu().clamp(0, 1),
                "sr2": sr2_img.squeeze(0).cpu().clamp(0, 1),
                "psnr1": psnr1,
                "psnr2": psnr2,
                "ssim1": ssim1,
                "ssim2": ssim2
            })
    return results

def classify_and_visualize(results):
    """
    Categorizes the comparison results and generates plots for specific cases.
    Args:
        results (list): List of comparison results from compare_models_qualitative.
    """
    if not results:
        print("No results to classify and visualize.")
        return

    both_good = []
    both_bad = []
    model1_better = []
    model2_better = []

    GOOD_PSNR_THRESHOLD = 30.0
    BAD_PSNR_THRESHOLD = 25.0
    PSNR_DIFFERENCE_THRESHOLD = 2.0 # Significant difference in PSNR

    for item in results:
        psnr1 = item['psnr1']
        psnr2 = item['psnr2']
        delta = psnr1 - psnr2

        # Classification based primarily on PSNR
        if psnr1 >= GOOD_PSNR_THRESHOLD and psnr2 >= GOOD_PSNR_THRESHOLD:
            both_good.append(item)
        elif psnr1 <= BAD_PSNR_THRESHOLD and psnr2 <= BAD_PSNR_THRESHOLD:
            both_bad.append(item)
        elif delta >= PSNR_DIFFERENCE_THRESHOLD: # Model 1 PSNR significantly higher than Model 2
            model1_better.append(item)
        elif delta <= -PSNR_DIFFERENCE_THRESHOLD: # Model 2 PSNR significantly higher than Model 1
            model2_better.append(item)
    
    # Sort for best/worst representatives based on PSNR for selection
    both_good.sort(key=lambda x: (x['psnr1'] + x['psnr2']) / 2, reverse=True) # Still average for "both good"
    both_bad.sort(key=lambda x: (x['psnr1'] + x['psnr2']) / 2) # Still average for "both bad"
    model1_better.sort(key=lambda x: x['psnr1'] - x['psnr2'], reverse=True) # Sort by PSNR difference
    model2_better.sort(key=lambda x: x['psnr2'] - x['psnr1'], reverse=True) # Sort by PSNR difference


    cases = {
        "Both Models Good": both_good[0] if both_good else None,
        "Both Models Poor": both_bad[0] if both_bad else None,
        "SRModel Significantly Better": model1_better[0] if model1_better else None,
        "SRFlow Significantly Better": model2_better[0] if model2_better else None,
    }

    # Define the target HR output resolution (e.g., 1024x1024 if LR=256 and UPSCALE_FACTOR=4)
    target_output_h = RESIZE_HEIGHT * UPSCALE_FACTOR
    target_output_w = RESIZE_WIDTH * UPSCALE_FACTOR

    # Define zoom regions relative to the target output resolution (e.g., 1024x1024)
    # Using larger, more distinct patches.
    zoom_patch_size = 256 # A larger patch size for better detail visibility

    regions_for_plot = [
        (int(target_output_w * 0.1), int(target_output_h * 0.1), zoom_patch_size), # Top-leftish
        (int(target_output_w * 0.4), int(target_output_h * 0.4), zoom_patch_size), # Mid-centerish
        (int(target_output_w * 0.7), int(target_output_h * 0.7), zoom_patch_size), # Bottom-rightish
    ]
    # Ensure coordinates + size don't exceed target_output_w/h.
    # The extract_zoom_patches function has padding, but it's good to aim for within bounds.

    for label, case in cases.items():
        if case:
            print(f"\n--- Visualizing: {label} ---")
            plot_comparison_with_zooms(
                case['lr'], case['hr'], case['sr1'], case['sr2'],
                case['psnr1'], case['psnr2'],
                case['ssim1'], case['ssim2'],
                regions_for_plot,
                title=label,
                save_path=f'./comparison_plots/{label.replace(" ", "_").replace(":", "")}.png'
            )
        else:
            print(f"No qualifying example found for: {label}")

if __name__ == '__main__':
    print("Starting comprehensive model comparison and evaluation...")
    set_seed(GLOBAL_SEED)
    print("\n--- Running qualitative comparison (generating example plots) ---")
    qualitative_results = compare_models_qualitative(num_images=500)
    classify_and_visualize(qualitative_results)
    print("\nComparison and evaluation complete.")