import os
import sys

import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

import data
from model import ChessModel

def train():
    device_string = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device_string)
    model = ChessModel(128).to(torch.float32).to(device)
    opt = torch.optim.Adam(model.parameters())
    reconstruction_loss_fn = nn.CrossEntropyLoss().to(torch.float32).to(device)
    popularity_loss_fn = nn.L1Loss().to(torch.float32).to(device)
    evaluation_loss_fn = nn.L1Loss().to(torch.float32).to(device)
    data_loader = DataLoader(data.LichessPuzzleDataset(cap_data=65536), batch_size=64, num_workers=1)  # 1 to avoid threading madness.
    num_epochs = 10

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
            total_loss = reconstruction_loss + popularity_loss + evaluation_loss

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

        torch.save(model, f"checkpoints/epoch_{epoch}.pth")


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
