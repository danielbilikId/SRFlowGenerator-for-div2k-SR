#Osher sidi – 318420239
#Daniel Bilik – 213196207

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt
import os
import argparse 
import numpy as np 
import random

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"Random seed set to {seed}")

from config import (
    EPOCHS, LEARNING_RATE, DEVICE, 
    UPSCALE_FACTOR, CHANNELS,
    SRFLOW_NF, SRFLOW_NB,
)
from models.sr_vanilla_model import SRModel
from models.srflow_model import SRFlowGenerator

from datasets.div2k_dataset import get_dataloaders
from utils.metrics import overall_loss_func, MeanGradientError

def train(model_type, load_weights,ablate, random_seed=None):
    if random_seed is not None:
        set_seed(random_seed)

    train_loader, val_loader, _ = get_dataloaders()

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
    if (ablate == True):
            print(f"ablating is on")
            model_weights_path = f'./{model_type}_ablated_seed_{random_seed}_weights.pth'
            history_path = f'./{model_type}_ablated_seed_{random_seed}_training_history.npy'
            plot_path = f'./{model_type}_ablated_seed_{random_seed}_training_history.png'
    else: 
        model_weights_path = f'./{model_type}_seed_{random_seed}_weights.pth'
        history_path = f'./{model_type}_seed_{random_seed}_training_history.npy'
        plot_path = f'./{model_type}_seed_{random_seed}_training_history.png'
        
    print(f"Using model: {model_type}")
    print(model)

    if load_weights:
        if os.path.exists(model_weights_path):
            print(f"Loading pre-trained weights from {model_weights_path}")
            model.load_state_dict(torch.load(model_weights_path, map_location=DEVICE))
            print("Weights loaded successfully.")
        else:
            print(f"Warning: --load_weights was specified, but no weights found at {model_weights_path}. Training from scratch for this seed.")
    else:
        print(f"Training {model_type} from scratch for seed {random_seed}.")


    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    mge_metric = MeanGradientError().to(DEVICE)

    history = {'loss': [], 'val_loss': []}

    print(f"Training on {DEVICE}")
    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0

        for batch_idx, (lr_images, hr_images) in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS} (Train)")):
            lr_images = lr_images.to(DEVICE)
            hr_images = hr_images.to(DEVICE)

            optimizer.zero_grad()

            outputs = model(lr_images)
            if (ablate==True):
                mge_weight = 0.2
                mse_weight = 1.0
                mge_loss = mge_metric(outputs, hr_images)
                mse_loss = overall_loss_func(outputs, hr_images)
                loss = mge_weight * mge_loss + mse_weight * mse_loss
            else: loss = overall_loss_func(outputs, hr_images)

            loss.backward() 
            optimizer.step() 
            running_loss += loss.item() * lr_images.size(0)

        epoch_loss = running_loss / len(train_loader.dataset)
        
        history['loss'].append(epoch_loss)

        model.eval()
        val_running_loss = 0.0

        with torch.no_grad(): 
            for lr_images, hr_images in tqdm(val_loader, desc=f"Epoch {epoch+1}/{EPOCHS} (Val)"):
                lr_images = lr_images.to(DEVICE)
                hr_images = hr_images.to(DEVICE)

                outputs = model(lr_images)
                if (ablate==True):
                    mge_weight = 0.2
                    mse_weight = 1.0
                    mge_loss = mge_metric(outputs, hr_images)
                    mse_loss = overall_loss_func(outputs, hr_images)
                    val_loss = mge_weight * mge_loss + mse_weight * mse_loss
                else: val_loss = overall_loss_func(outputs, hr_images)
                
                val_running_loss += val_loss.item() * lr_images.size(0)

        val_epoch_loss = val_running_loss / len(val_loader.dataset)

        history['val_loss'].append(val_epoch_loss)

        print(f"Epoch {epoch+1} - Train Loss: {epoch_loss:.4f} | Val Loss: {val_epoch_loss:.4f}")

    torch.save(model.state_dict(), model_weights_path)
    print(f"Model weights saved to {model_weights_path}")

    np.save(history_path, history)
    print(f"Training history saved to {history_path}")

    plt.figure(figsize=(12, 5))
    plt.plot(history['loss'], label='train_loss', marker='o')
    plt.plot(history['val_loss'], label='val_loss', marker='o')
    plt.title(f'{model_type} Loss History (Seed: {random_seed if random_seed is not None else "Default"})') 
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.tight_layout()
    plt.savefig(plot_path) 
    plt.show() 

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train Super-Resolution models.")
    parser.add_argument('--model', type=str, default='SRModel',
                        choices=['SRModel', 'SRFlowGenerator'], 
                        help="Specify the model to train: 'SRModel', 'SRFlowGenerator'. (default: SRModel)")
    parser.add_argument('--ablate', type= bool, default=False, choices=[True,False])

    parser.add_argument('--load_weights', action='store_true', 
                        help="Load pre-trained weights for the specified model if available.")
    args = parser.parse_args()
    random_seeds = [42,123,789]
    
    for seed in random_seeds:
        print(f"\n--- Starting training for {args.model} with seed: {seed} ---")

        train(args.model, args.load_weights,ablate= args.ablate, random_seed=seed)
        print(f"--- Finished training for {args.model} with seed: {seed} ---\n")
