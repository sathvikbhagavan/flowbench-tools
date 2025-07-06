# Import necessary modules
from codes.data.dataset import LidDrivenDataset
from codes.models.FNO import TensorizedFNO
from codes.utils.trainer import TrainFNO
from codes.utils.device import device
import torch
import wandb

wandb.init(project="FNO", config={
    "learning_rate": 0.001,
    "epochs": 1000,
    "batch_size": 128,
    "model": "FNO",
    "optimizer": "Adam",
    "loss_fn": "MSELoss",
    "in_dim": 2,
    "out_dim": 3,
    "n_layers": 3,
    "in_size": 128,
    "out_size": 128,
    "n_modes": (256, 256),
    "hidden_channels": 64,
    "projection_channels": 128,
    "weight_decay": 1e-5
})
config = wandb.config

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
train_loader, val_loader = LidDriven_dataset.create_dataloader(batch_size=wandb.config.batch_size, shuffle=True)

# Create an instance of the TensorizedFNO model
model = TensorizedFNO(n_modes=config.n_modes, in_channels=config.in_dim, out_channels=config.out_dim, hidden_channels=config.hidden_channels, 
                      projection_channels=config.projection_channels, n_layers=config.n_layers)
print(f'The number of parameters in FNO are: {sum(p.numel() for p in model.parameters() if p.requires_grad)}')

# Set the learning rate and number of epochs
learning_rate = wandb.config.learning_rate
num_epochs = wandb.config.epochs
weight_decay = wandb.config.weight_decay

# Define loss function and optimizer
criterion = torch.nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

# Create an instance of the TrainFNO class
FNO_trainer = TrainFNO(model=model, optimizer=optimizer, loss_fn=criterion,
                       train_loader=train_loader, val_loader=val_loader, log_dir=wandb.run.dir,
                       epochs=num_epochs, device=device)

FNO_trainer.train()