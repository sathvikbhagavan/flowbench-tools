# Import necessary modules
from codes.data.dataset import LidDrivenDataset
from codes.models.CNO import CompressedCNO
from codes.utils.trainer import TrainCNO
from codes.utils.device import device
import torch
import wandb

print("Packages imported...")

# Initialize Weights & Biases
wandb.init(project="CNO", config={
    "learning_rate": 5e-4,
    "epochs": 1500,
    "batch_size": 128,
    "model": "CNO",
    "optimizer": "Adam + LR Scheduler",
    "dataset": "harmonics",
    "loss_fn": "L1",
    "in_dim": 2,
    "out_dim": 3,
    "N_layers": 5,
    "in_size": 128,
    "out_size": 128,
    "weight_decay": 1e-5
})
config = wandb.config

# Create an instance of the LidDrivenDataset
res = 128
dataset_name = 'harmonics'
LidDriven_dataset = LidDrivenDataset(
   file_path_x_train=f'/work/cvlab/students/bhagavan/SemesterProject/LDC_NS_2D/{res}x{res}/processed/{dataset_name}_lid_driven_cavity_X_train.npy',
   file_path_x_valid=f'/work/cvlab/students/bhagavan/SemesterProject/LDC_NS_2D/{res}x{res}/processed/{dataset_name}_lid_driven_cavity_X_test.npy',
   file_path_y_train=f'/work/cvlab/students/bhagavan/SemesterProject/LDC_NS_2D/{res}x{res}/processed/{dataset_name}_lid_driven_cavity_Y_train.npy',
   file_path_y_valid=f'/work/cvlab/students/bhagavan/SemesterProject/LDC_NS_2D/{res}x{res}/processed/{dataset_name}_lid_driven_cavity_Y_test.npy'
)

# Create data loaders
train_loader, val_loader = LidDriven_dataset.create_dataloader(batch_size=wandb.config.batch_size, shuffle=True)

# Create an instance of the CompressedCNO model
model = CompressedCNO(in_dim=config.in_dim, out_dim=config.out_dim, N_layers=config.N_layers, in_size=config.in_size, out_size=config.out_size)
print(f'The number of parameters in CNO are: {sum(p.numel() for p in model.parameters() if p.requires_grad)}')

# Optionally log model
# wandb.watch(model, log="all", log_freq=100)

# Set hyperparameters
learning_rate = wandb.config.learning_rate
num_epochs = wandb.config.epochs
weight_decay = wandb.config.weight_decay

# Define loss and optimizer
criterion = torch.nn.L1Loss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
scheduler = torch.optim.lr_scheduler.LinearLR(
    optimizer,
    start_factor=1.0,
    end_factor=0.1,
    total_iters=config.epochs
)

# Create an instance of the TrainCNO class
CNO_trainer = TrainCNO(model=model, optimizer=optimizer, scheduler=scheduler, loss_fn=criterion,
                      train_loader=train_loader, val_loader=val_loader,
                      log_dir=wandb.run.dir,
                      epochs=num_epochs, device=device)

# Start training
CNO_trainer.train()
