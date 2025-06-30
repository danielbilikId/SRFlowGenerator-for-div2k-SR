# train.py

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt
import os
import argparse 
import numpy as np 

from config import (
    EPOCHS, BATCH_SIZE, LEARNING_RATE, DEVICE, 
    UPSCALE_FACTOR, CHANNELS,
    SRFLOW_NF, SRFLOW_NB, 
)
from models.sr_vanilla_model import SRModel
from models.srflow_model import SRFlowGenerator
from datasets.div2k_dataset import get_dataloaders
from utils.metrics import overall_loss_func, MeanGradientError

def train(model_type, load_weights): # Accept load_weights as an argument
    # 1. Setup DataLoaders
    train_loader, val_loader, _ = get_dataloaders()

    # 2. Initialize Model based on model_type argument
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

    # Define model weights path dynamically
    model_weights_path = f'./{model_type}_weights.pth'
    
    print(f"Using model: {model_type}")
    print(model) # Print model architecture for verification

    # Load pre-trained weights if requested
    if load_weights:
        if os.path.exists(model_weights_path):
            print(f"Loading pre-trained weights from {model_weights_path}")
            model.load_state_dict(torch.load(model_weights_path, map_location=DEVICE))
            print("Weights loaded successfully.")
        else:
            print(f"Warning: --load_weights was specified, but no weights found at {model_weights_path}. Training from scratch.")

    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    mae_loss_func = nn.L1Loss()
    mge_metric = MeanGradientError().to(DEVICE)

    # 3. Training Loop
    history = {'loss': [], 'val_loss': [], 'mae': [], 'val_mae': [], 'mean_gradient_error': [], 'val_mean_gradient_error': []}

    print(f"Training on {DEVICE}")
    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0
        running_mae = 0.0
        running_mge = 0.0

        for batch_idx, (lr_images, hr_images) in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS} (Train)")):
            lr_images = lr_images.to(DEVICE)
            hr_images = hr_images.to(DEVICE)

            optimizer.zero_grad() # Zero the gradients

            outputs = model(lr_images)
            
            loss = overall_loss_func(outputs, hr_images)
            mae = mae_loss_func(outputs, hr_images)
            mge = mge_metric(outputs, hr_images)

            loss.backward() # Backpropagation
            optimizer.step() # Update weights

            running_loss += loss.item() * lr_images.size(0)
            running_mae += mae.item() * lr_images.size(0)
            running_mge += mge.item() * lr_images.size(0)

        epoch_loss = running_loss / len(train_loader.dataset)
        epoch_mae = running_mae / len(train_loader.dataset)
        epoch_mge = running_mge / len(train_loader.dataset)
        
        history['loss'].append(epoch_loss)
        history['mae'].append(epoch_mae)
        history['mean_gradient_error'].append(epoch_mge)

        # Validation
        model.eval() # Set model to evaluation mode
        val_running_loss = 0.0
        val_running_mae = 0.0
        val_running_mge = 0.0

        with torch.no_grad(): # Disable gradient calculations for validation
            for lr_images, hr_images in tqdm(val_loader, desc=f"Epoch {epoch+1}/{EPOCHS} (Val)"):
                lr_images = lr_images.to(DEVICE)
                hr_images = hr_images.to(DEVICE)

                outputs = model(lr_images)
                
                val_loss = overall_loss_func(outputs, hr_images)
                val_mae = mae_loss_func(outputs, hr_images)
                val_mge = mge_metric(outputs, hr_images)

                val_running_loss += val_loss.item() * lr_images.size(0)
                val_running_mae += val_mae.item() * lr_images.size(0)
                val_running_mge += val_mge.item() * lr_images.size(0)

        val_epoch_loss = val_running_loss / len(val_loader.dataset)
        val_epoch_mae = val_running_mae / len(val_loader.dataset)
        val_epoch_mge = val_running_mge / len(val_loader.dataset)
        
        history['val_loss'].append(val_epoch_loss)
        history['val_mae'].append(val_epoch_mae)
        history['val_mean_gradient_error'].append(val_epoch_mge)

        print(f"Epoch {epoch+1} - Train Loss: {epoch_loss:.4f}, Train MAE: {epoch_mae:.4f}, Train MGE: {epoch_mge:.4f} | Val Loss: {val_epoch_loss:.4f}, Val MAE: {val_epoch_mae:.4f}, Val MGE: {val_epoch_mge:.4f}")

    # Save model weights dynamically based on MODEL_TYPE
    torch.save(model.state_dict(), model_weights_path)
    print(f"Model weights saved to {model_weights_path}")

    # Save the history dictionary for future plotting
    history_path = f'./{model_type}_training_history.npy'
    np.save(history_path, history)
    print(f"Training history saved to {history_path}")

    # Plotting training history (still display during execution)
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(history['loss'], label='train_loss', marker='o')
    plt.plot(history['val_loss'], label='val_loss', marker='o')
    plt.title('Loss History')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history['mae'], label='train_mae', marker='o')
    plt.plot(history['val_mae'], label='val_mae', marker='o')
    plt.title('MAE History')
    plt.xlabel('Epoch')
    plt.ylabel('MAE')
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'{model_type}_training_history.png') 
    plt.show() 

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train Super-Resolution models.")
    parser.add_argument('--model', type=str, default='SRModel',
                        choices=['SRModel', 'SRFlowGenerator', 'RealNVP_SR'], 
                        help="Specify the model to train: 'SRModel', 'SRFlowGenerator', or 'RealNVP_SR'. (default: SRModel)")
    parser.add_argument('--load_weights', action='store_true', 
                        help="Load pre-trained weights for the specified model if available.")
    
    args = parser.parse_args()
    train(args.model, args.load_weights)
