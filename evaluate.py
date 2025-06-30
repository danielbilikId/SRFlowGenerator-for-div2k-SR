import torch
import numpy as np
import os
import matplotlib.pyplot as plt
from tqdm import tqdm
from torchvision.utils import save_image
import argparse
import shutil

from pytorch_fid.fid_score import calculate_fid_given_paths

from config import (
    DEVICE, UPSCALE_FACTOR, CHANNELS, BATCH_SIZE,
    SRFLOW_NF, SRFLOW_NB,
)
from models.sr_vanilla_model import SRModel
from models.srflow_model import SRFlowGenerator
from datasets.div2k_dataset import get_dataloaders
from utils.metrics import get_psnr_ssim_metrics

def evaluate(model_type, seeds=[42, 123, 789]):
    _, _, test_loader = get_dataloaders()

    psnr_scores, ssim_scores, fid_scores = [], [], []

    for seed in seeds:
        model_weights_path = f'./{model_type}_seed_{seed}_weights.pth'

        # Model selection
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
            raise ValueError(f"Unknown model type: {model_type}. Choose 'SRModel' or 'SRFlowGenerator'.")

        if not os.path.exists(model_weights_path):
            print(f"Skipping seed {seed}: weights not found at {model_weights_path}")
            continue

        print(f"\n--- Evaluating {model_type} (Seed {seed}) ---")
        model.load_state_dict(torch.load(model_weights_path, map_location=DEVICE))
        model.eval()

        psnr_metric, ssim_metric = get_psnr_ssim_metrics(data_range=1.0)

        output_dir_real = f'real_test_images_{model_type}_seed{seed}'
        output_dir_generated = f'generated_test_images_{model_type}_seed{seed}'
        os.makedirs(output_dir_real, exist_ok=True)
        os.makedirs(output_dir_generated, exist_ok=True)

        with torch.no_grad():
            for i, (lr_images, hr_images) in enumerate(tqdm(test_loader, desc=f"Evaluating Seed {seed}")):
                lr_images = lr_images.to(DEVICE)
                hr_images = hr_images.to(DEVICE)

                outputs = model(lr_images)
                psnr_metric.update(outputs, hr_images)
                ssim_metric.update(outputs, hr_images)

                for j in range(outputs.size(0)):
                    save_image(outputs[j].cpu().clamp(0, 1), os.path.join(output_dir_generated, f'img_{i*BATCH_SIZE + j:04d}.png'))
                    save_image(hr_images[j].cpu().clamp(0, 1), os.path.join(output_dir_real, f'img_{i*BATCH_SIZE + j:04d}.png'))

        avg_psnr = psnr_metric.compute().item()
        avg_ssim = ssim_metric.compute().item()
        psnr_scores.append(avg_psnr)
        ssim_scores.append(avg_ssim)

        print(f"PSNR: {avg_psnr:.4f}, SSIM: {avg_ssim:.4f}")

        fid_value = calculate_fid_given_paths([output_dir_real, output_dir_generated], BATCH_SIZE, DEVICE, dims=2048)
        fid_scores.append(fid_value)

        print(f"FID Score: {fid_value:.4f}")

        shutil.rmtree(output_dir_real)
        shutil.rmtree(output_dir_generated)

        # Optional: visualize results for first seed
        if seed == seeds[0]:
            model.eval()
            with torch.no_grad():
                sample_lr, sample_hr = next(iter(test_loader))
                sample_lr = sample_lr.to(DEVICE)
                sample_output = model(sample_lr)

                num_samples = min(4, sample_lr.size(0))
                plt.figure(figsize=(15, 5 * num_samples))
                for i in range(num_samples):
                    plt.subplot(num_samples, 3, i * 3 + 1)
                    plt.imshow(sample_lr[i].cpu().permute(1, 2, 0).numpy())
                    plt.title(f"LR Input {i+1}")
                    plt.axis('off')

                    plt.subplot(num_samples, 3, i * 3 + 2)
                    plt.imshow(sample_output[i].cpu().permute(1, 2, 0).numpy())
                    plt.title(f"Generated HR {i+1}")
                    plt.axis('off')

                    plt.subplot(num_samples, 3, i * 3 + 3)
                    plt.imshow(sample_hr[i].cpu().permute(1, 2, 0).numpy())
                    plt.title(f"Original HR {i+1}")
                    plt.axis('off')
                plt.tight_layout()
                plt.show()
            import torchvision.transforms.functional as TF
            vis_dir = f'visualizations_{model_type}_seed{seed}'
            os.makedirs(vis_dir, exist_ok=True)

            print(f"\nCollecting individual PSNR scores for qualitative examples...")

            from torchmetrics.functional import peak_signal_noise_ratio as compute_psnr

            model.eval()
            psnr_per_image = []
            samples = []

            with torch.no_grad():
                for lr_images, hr_images in tqdm(test_loader, desc="Collecting Samples"):
                    lr_images = lr_images.to(DEVICE)
                    hr_images = hr_images.to(DEVICE)
                    sr_images = model(lr_images)

                    for i in range(lr_images.size(0)):
                        lr = lr_images[i].unsqueeze(0)
                        hr = hr_images[i].unsqueeze(0)
                        sr = sr_images[i].unsqueeze(0)

                        psnr_val = compute_psnr(sr, hr, data_range=1.0).item()
                        psnr_per_image.append(psnr_val)
                        samples.append((lr.cpu(), sr.cpu(), hr.cpu(), psnr_val))

            # Sort samples by PSNR
            sorted_samples = sorted(samples, key=lambda x: x[3], reverse=True)
            top_n = sorted_samples[:4]   # Best 4
            worst_n = sorted_samples[-4:]  # Worst 4

            def save_sample_set(sample_set, title_prefix, save_path_prefix):
                num = len(sample_set)
                plt.figure(figsize=(10, 4.5 * num))
                for i, (lr, sr, hr, score) in enumerate(sample_set):
                    # Convert to images
                    img_lr = TF.to_pil_image(lr.squeeze(0).clamp(0, 1))
                    img_sr = TF.to_pil_image(sr.squeeze(0).clamp(0, 1))
                    img_hr = TF.to_pil_image(hr.squeeze(0).clamp(0, 1))

                    # Save individual images
                    img_lr.save(os.path.join(vis_dir, f'{save_path_prefix}_sample_{i+1}_LR.png'))
                    img_sr.save(os.path.join(vis_dir, f'{save_path_prefix}_sample_{i+1}_SR.png'))
                    img_hr.save(os.path.join(vis_dir, f'{save_path_prefix}_sample_{i+1}_GT.png'))

                    # Plot
                    plt.subplot(num, 3, i * 3 + 1)
                    plt.imshow(img_lr)
                    plt.title(f"{title_prefix} #{i+1} - LR")
                    plt.axis("off")

                    plt.subplot(num, 3, i * 3 + 2)
                    plt.imshow(img_sr)
                    plt.title(f"{title_prefix} #{i+1} - SR (PSNR={score:.2f})")
                    plt.axis("off")

                    plt.subplot(num, 3, i * 3 + 3)
                    plt.imshow(img_hr)
                    plt.title(f"{title_prefix} #{i+1} - Ground Truth")
                    plt.axis("off")
                plt.tight_layout()
                plt.savefig(os.path.join(vis_dir, f'{save_path_prefix}_qualitative_grid.png'))
                plt.show()

            # Save and plot both top and bottom performing samples
            save_sample_set(top_n, "Best", f"{model_type}_seed{seed}_best")
            save_sample_set(worst_n, "Worst", f"{model_type}_seed{seed}_worst")

    # Final summary
    print(f"\n=== {model_type} Final Test Results Over Seeds {seeds} ===")
    print(f"PSNR: Mean = {np.mean(psnr_scores):.4f}, Std = {np.std(psnr_scores):.4f}")
    print(f"SSIM: Mean = {np.mean(ssim_scores):.4f}, Std = {np.std(ssim_scores):.4f}")
    print(f"FID : Mean = {np.mean(fid_scores):.4f}, Std = {np.std(fid_scores):.4f}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Evaluate Super-Resolution models over multiple seeds.")
    parser.add_argument('--model', type=str, default='SRModel',
                        choices=['SRModel', 'SRFlowGenerator'],
                        help="Specify the model to evaluate.")
    args = parser.parse_args()
    evaluate(args.model)
