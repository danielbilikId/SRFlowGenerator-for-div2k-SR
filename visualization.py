#Osher sidi – 318420239
#Daniel Bilik – 213196207

import torch
import matplotlib.pyplot as plt
import os
import argparse
from torchvision import transforms

from models.sr_vanilla_model import SRModel
from models.srflow_model import SRFlowGenerator

from datasets.div2k_dataset import get_dataloaders
from config import (
    DEVICE, UPSCALE_FACTOR, CHANNELS,
    SRFLOW_NF, SRFLOW_NB,
    RESIZE_HEIGHT, RESIZE_WIDTH
)

def visualize_generated_images(model_type, ablate, num_samples_to_show=4, seed=42):
    """
    Loads a trained model, generates HR images from LR test samples,
    and visualizes/saves the LR, Generated HR, and Ground Truth HR images.
    Now includes a comparison with bicubic interpolation.

    Args:
        model_type (str): The type of model to load ('SRModel', 'SRFlowGenerator', 'RealNVP_SR').
        num_samples_to_show (int): Number of sample images to display.
    """
    _, _, test_loader = get_dataloaders()

    if (ablate==True):  model_weights_path = f'./{model_type}_ablated_seed_{seed}_weights.pth'
    else: model_weights_path = f'./{model_type}_seed_{seed}_weights.pth'

    if model_type == 'SRModel':
        model = SRModel(upscale_factor=UPSCALE_FACTOR, channels=CHANNELS).to(DEVICE)
    elif model_type == 'SRFlowGenerator':
        model = SRFlowGenerator(
            in_nc=CHANNELS,
            out_nc=CHANNELS,
            nf=SRFLOW_NF,
            nb=SRFLOW_NB,
            upscale_factor=UPSCALE_FACTOR
        ).to(DEVICE)
    else:
        raise ValueError(f"Unknown model type: {model_type}. Choose 'SRModel', 'SRFlowGenerator', or 'RealNVP_SR'.")

    print(f"Loading weights for model: {model_type} from {model_weights_path}")
    if not os.path.exists(model_weights_path):
        print(f"Error: Model weights not found at {model_weights_path}. Please train the model first.")
        return 

    model.load_state_dict(torch.load(model_weights_path, map_location=DEVICE))
    model.eval()

    print(f"\nDisplaying {num_samples_to_show} sample outputs for {model_type}...")
    
    sample_lr_images = []
    sample_hr_images = []
    count = 0
    with torch.no_grad():
        for lr, hr in test_loader:
            sample_lr_images.append(lr)
            sample_hr_images.append(hr)
            count += lr.size(0)
            if count >= num_samples_to_show:
                break
    
    sample_lr_images = torch.cat(sample_lr_images)[:num_samples_to_show]
    sample_hr_images = torch.cat(sample_hr_images)[:num_samples_to_show]

    sample_lr_images = sample_lr_images.to(DEVICE)
    sample_hr_images = sample_hr_images.to(DEVICE)

    with torch.no_grad():
        sample_output = model(sample_lr_images)

    # --- Generate Bicubic Upsampled Images ---
    bicubic_upsample_transform = transforms.Compose([
        transforms.ToPILImage(), # Convert tensor to PIL Image
        transforms.Resize((RESIZE_HEIGHT*4, RESIZE_WIDTH*4), interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.ToTensor() # Convert back to tensor
    ])
    bicubic_hr_images = torch.stack([bicubic_upsample_transform(img.cpu()) for img in sample_lr_images]).to(DEVICE)
    # --- End Bicubic Generation ---

    viz_output_dir = f'./visualizations/{model_type}_vs_bicubic'
    os.makedirs(viz_output_dir, exist_ok=True)

    for i in range(num_samples_to_show):
        lr_display = sample_lr_images[i].cpu().permute(1, 2, 0).numpy()
        bicubic_display = bicubic_hr_images[i].cpu().permute(1, 2, 0).numpy()
        output_display = sample_output[i].cpu().clamp(0, 1).permute(1, 2, 0).numpy()
        hr_display = sample_hr_images[i].cpu().permute(1, 2, 0).numpy()

        hr_h, hr_w, _ = hr_display.shape
        dpi_val = 100 
        fig_width = (lr_display.shape[1] + bicubic_display.shape[1] + output_display.shape[1] + hr_display.shape[1]) / dpi_val
        fig_height = max(lr_display.shape[0], bicubic_display.shape[0], output_display.shape[0], hr_display.shape[0]) / dpi_val

        plt.figure(figsize=(fig_width, fig_height), dpi=dpi_val)
        plt.suptitle(f'{model_type} Sample {i+1}: LR | Bicubic HR | Generated HR | Original HR', fontsize=16)

        plt.subplot(1, 4, 1)
        plt.imshow(lr_display)
        plt.title(f"LR Input\n({lr_display.shape[0]}x{lr_display.shape[1]})")
        plt.axis('off')

        plt.subplot(1, 4, 2)
        plt.imshow(bicubic_display)
        plt.title(f"Bicubic HR\n({bicubic_display.shape[0]}x{bicubic_display.shape[1]})")
        plt.axis('off')

        plt.subplot(1, 4, 3)
        plt.imshow(output_display)
        plt.title(f"Generated HR\n({output_display.shape[0]}x{output_display.shape[1]})")
        plt.axis('off')

        plt.subplot(1, 4, 4)
        plt.imshow(hr_display)
        plt.title(f"Original HR\n({hr_display.shape[0]}x{hr_display.shape[1]})")
        plt.axis('off')
        
        plt.tight_layout(rect=[0, 0.03, 1, 0.95]) 
        
        save_path_individual = os.path.join(viz_output_dir, f'{model_type}_sample_{i+1}_vs_bicubic.png')
        plt.savefig(save_path_individual)
        print(f"Individual visualization saved to {save_path_individual}")
        plt.show()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Visualize generated images from trained Super-Resolution models.")
    parser.add_argument('--model', type=str, default='SRModel',
                        choices=['SRModel', 'SRFlowGenerator'],
                        help="Specify the model to visualize: 'SRModel', 'SRFlowGenerator'. (default: SRModel)")
    parser.add_argument('--ablate', type= bool, default=False, choices=[True,False])
    parser.add_argument('--num_samples', type=int, default=4,
                        help="Number of sample images to display. (default: 4)")
    
    args = parser.parse_args()
    visualize_generated_images(args.model,args.ablate, args.num_samples)
