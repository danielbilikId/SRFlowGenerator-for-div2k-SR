#Osher sidi –318420239
#Daniel Bilik – 213196207

import numpy as np
import matplotlib.pyplot as plt
import os

ablated = True
model_type = 'SRFlowGenerator'  # or 'SRModel'
seeds = [42, 123, 789]
histories = []

for seed in seeds:
    if ablated: path = f'./{model_type}_ablated_seed_{seed}_training_history.npy'
    else: path = f'./{model_type}_seed_{seed}_training_history.npy'
    if not os.path.exists(path):
        print(f"Warning: Missing file at {path}")
        continue
    try:
        history = np.load(path, allow_pickle=True).item()
        histories.append(history)
    except Exception as e:
        print(f"Error loading {path}: {e}")

if not histories:
    print("Error: No valid history files found. Cannot proceed.")
    exit()

num_epochs = len(histories[0]['loss'])
metrics = ['loss', 'val_loss']

avg_history = {metric: np.zeros(num_epochs) for metric in metrics}

for metric in metrics:
    for hist in histories:
        avg_history[metric] += np.array(hist[metric])
    avg_history[metric] /= len(histories)

plt.figure(figsize=(6, 5))

plt.plot(avg_history['loss'], label='Avg Train Loss', marker='o')
plt.plot(avg_history['val_loss'], label='Avg Val Loss', marker='o')
if ablated: plt.title(f'{model_type} Averaged Loss History (Seeds {seeds}) - Ablated study') 
else: plt.title(f'{model_type} Averaged Loss History (Seeds {seeds})')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()
