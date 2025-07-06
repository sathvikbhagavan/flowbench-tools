# Import necessary modules
from codes.data.dataset import LidDrivenDataset
from codes.models.FNO import TensorizedFNO
from codes.utils.trainer import TrainFNO
from codes.utils.device import device
import torch
import numpy as np
from einops import rearrange

print("Packages imported...")

# Create an instance of the LidDrivenDataset
res = 128
dataset_name = 'combined'
LidDriven_dataset = LidDrivenDataset(
   file_path_x_train=f'/work/cvlab/students/bhagavan/SemesterProject/LDC_NS_2D/{res}x{res}/processed/{dataset_name}_lid_driven_cavity_X_train.npy',
   file_path_x_valid=f'/work/cvlab/students/bhagavan/SemesterProject/LDC_NS_2D/{res}x{res}/processed/{dataset_name}_lid_driven_cavity_X_test.npy',
   file_path_y_train=f'/work/cvlab/students/bhagavan/SemesterProject/LDC_NS_2D/{res}x{res}/processed/{dataset_name}_lid_driven_cavity_Y_train.npy',
   file_path_y_valid=f'/work/cvlab/students/bhagavan/SemesterProject/LDC_NS_2D/{res}x{res}/processed/{dataset_name}_lid_driven_cavity_Y_test.npy'
)

# Create data loaders for training and validation
train_loader, test_loader = LidDriven_dataset.create_dataloader(batch_size=100, shuffle=False)

# Create an instance of the TensorizedFNO model
model = TensorizedFNO(n_modes=(256, 256), in_channels=2, out_channels=3, hidden_channels=64, 
                      projection_channels=128, n_layers=3).to(device)
save_folder = '/work/cvlab/students/bhagavan/SemesterProject/flowbench-tools/Training/wandb/run-20250613_145353-msclahtz/files/'
model.load_checkpoint(save_name='1000', save_folder=f'{save_folder}/checkpoints')
model.eval()

# Containers for all batches
all_train_inputs = []
all_train_outputs = []
all_train_targets = []

all_test_inputs = []
all_test_outputs = []
all_test_targets = []

# Disable gradient calculation
with torch.no_grad():
    # Process training data
    for batch in train_loader:
        inputs, targets = batch[0].to(device), batch[1].to(device)
        outputs = model(inputs)
        all_train_outputs.append(outputs.cpu())
        all_train_targets.append(targets.cpu())

    # Process validation data
    for batch in test_loader:
        inputs, targets = batch[0].to(device), batch[1].to(device)
        outputs = model(inputs)
        all_test_outputs.append(outputs.cpu())
        all_test_targets.append(targets.cpu())

# Concatenate all batches
train_outputs = torch.cat(all_train_outputs, dim=0).numpy()
train_outputs = rearrange(train_outputs, 'b c h w -> b h w c')

train_targets = torch.cat(all_train_targets, dim=0).numpy()
train_targets = rearrange(train_targets, 'b c h w -> b h w c')

test_outputs = torch.cat(all_test_outputs, dim=0).numpy()
test_outputs = rearrange(test_outputs, 'b c h w -> b h w c')

test_targets = torch.cat(all_test_targets, dim=0).numpy()
test_targets = rearrange(test_targets, 'b c h w -> b h w c')

def denormalize_from_minus_one_one(normalized_arr, min_val, max_val):
    # Rearrange min and max to match (b, h, w, c) format
    min_val = rearrange(min_val, '1 c 1 1 -> 1 1 1 c')
    max_val = rearrange(max_val, '1 c 1 1 -> 1 1 1 c')
    return (normalized_arr + 1) * (max_val[:, :, :, :3] - min_val[:, :, :, :3]) / 2 + min_val[:, :, :, :3]

min_val = np.load(f"/work/cvlab/students/bhagavan/SemesterProject/LDC_NS_2D/{res}x{res}/processed/{dataset_name}_lid_driven_cavity_Y_train_min_stats.npy")
max_val = np.load(f"/work/cvlab/students/bhagavan/SemesterProject/LDC_NS_2D/{res}x{res}/processed/{dataset_name}_lid_driven_cavity_Y_train_max_stats.npy")

train_outputs = denormalize_from_minus_one_one(train_outputs, min_val, max_val)
train_targets = denormalize_from_minus_one_one(train_targets, min_val, max_val)

np.save(f"{save_folder}/{res}_{dataset_name}_preds_train.npy", train_outputs)
np.save(f"{save_folder}/{res}_{dataset_name}_gt_train.npy", train_targets)

test_outputs = denormalize_from_minus_one_one(test_outputs, min_val, max_val)
test_targets = denormalize_from_minus_one_one(test_targets, min_val, max_val)

np.save(f"{save_folder}/{res}_{dataset_name}_preds_test.npy", test_outputs)
np.save(f"{save_folder}/{res}_{dataset_name}_gt_test.npy", test_targets)

print('Saving done...')
