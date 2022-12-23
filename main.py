import os
import sys

import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

import data
from model import ChessModel


# Experiment parameters:
RUN_CONFIGURATION = {
    "learning_rate": 0.0004,
    "dataset_cap": 100000,
    "epochs": 1000,
    "latent_size": 256,
}

# Logging:
wandb = None
try:
    import wandb
    wandb.init("assembly_ai_hackathon_2022", config=RUN_CONFIGURATION)
except ImportError:
    print("Weights and Biases not found in packages.")


def train():
    learning_rate = RUN_CONFIGURATION["learning_rate"]
    latent_size = RUN_CONFIGURATION["latent_size"]
    data_cap = RUN_CONFIGURATION["dataset_cap"]
    num_epochs = RUN_CONFIGURATION["epochs"]

    device_string = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device_string)
    model = ChessModel(latent_size).to(torch.float32).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=learning_rate)
    reconstruction_loss_fn = nn.CrossEntropyLoss().to(torch.float32).to(device)
    popularity_loss_fn = nn.L1Loss().to(torch.float32).to(device)
    evaluation_loss_fn = nn.L1Loss().to(torch.float32).to(device)
    data_loader = DataLoader(data.LichessPuzzleDataset(cap_data=data_cap), batch_size=64, num_workers=1)  # 1 to avoid threading madness.
    save_every_nth_epoch = 50
    upload_logs_every_nth_epoch = 1

    for epoch in range(num_epochs):
        model.train()
        total_reconstruction_loss = 0.0
        total_popularity_loss = 0.0
        total_evaluation_loss = 0.0
        total_batch_loss = 0.0
        num_batches = 0
        for batch_idx, (board_vec, popularity, evaluation) in tqdm(enumerate(data_loader)):
            board_vec = board_vec.to(torch.float32).to(device)  # [batch_size x 903]
            popularity = popularity.to(torch.float32).to(device).unsqueeze(1)  # enforce [batch_size, 1]
            evaluation = evaluation.to(torch.float32).to(device).unsqueeze(1)

            _embedding, predicted_popularity, predicted_evaluation, predicted_board_vec = model(board_vec)

            reconstruction_loss = reconstruction_loss_fn(predicted_board_vec, board_vec)
            popularity_loss = popularity_loss_fn(predicted_popularity, popularity)
            evaluation_loss = evaluation_loss_fn(predicted_evaluation, evaluation)
            #total_loss = reconstruction_loss + popularity_loss + evaluation_loss
            total_loss = popularity_loss

            opt.zero_grad()
            total_loss.backward()
            opt.step()

            total_reconstruction_loss += reconstruction_loss.cpu().item()
            total_popularity_loss += popularity_loss.cpu().item()
            total_evaluation_loss += evaluation_loss.cpu().item()
            total_batch_loss += total_loss.cpu().item()
            num_batches += 1
        print(f"Average reconstruction loss: {total_reconstruction_loss/num_batches}")
        print(f"Average popularity loss: {total_popularity_loss/num_batches}")
        print(f"Average evaluation loss: {total_evaluation_loss/num_batches}")
        print(f"Average batch loss: {total_batch_loss/num_batches}")

        if save_every_nth_epoch > 0 and (epoch % save_every_nth_epoch) == 0:
            torch.save(model, f"checkpoints/epoch_{epoch}.pth")

        if wandb:
            wandb.log(
                # For now, just log popularity.
                {"popularity_loss": total_popularity_loss},
                commit=(epoch+1) % upload_logs_every_nth_epoch == 0
            )

    torch.save(model, "checkpoints/final.pth")
    if wandb:
        wandb.finish()


def infer(fen):
    pass


def test():
    pass


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python {sys.argv[0]} --train|infer")
    elif sys.argv[1] == "--train":
        train()
    elif sys.argv[2] == "--infer":
        infer(sys.argv[3])
