import torch
from torch import nn

from data import BOARD_VECTOR_SIZE


class ChessModel(nn.Module):
    def __init__(self, embedding_dims):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(BOARD_VECTOR_SIZE, 512),
            nn.SiLU(),
            nn.Linear(512, 1024),
            nn.SiLU(),
            nn.Linear(1024, 1024),
            nn.SiLU(),
            nn.Linear(1024, embedding_dims),
            nn.SiLU(),
        )

        self.popularity_head = nn.Sequential(
            nn.Linear(embedding_dims, 512),
            nn.SiLU(),
            nn.Linear(512, 1),
            nn.Tanh(),
        )

        # Since it will take too long for this to evaluate before the jam is over, just noop it.
        #self.evaluation_head = nn.Sequential(
        #    nn.Linear(embedding_dims, 512),
        #    nn.SiLU(),
        #    nn.Linear(512, 1),
        #    nn.Tanh(),
        #)
        self.evaluation_head = nn.Sequential(
            nn.Linear(embedding_dims, 1),
        )

        self.reconstruction_head = nn.Sequential(
            nn.Linear(embedding_dims, 512),
            nn.SiLU(),
            nn.Linear(512, BOARD_VECTOR_SIZE),
            nn.Sigmoid(),
        )

    def forward(self, x):
        """Return the embedding, popularity, evaluation, and reconstruction."""
        # Outputs have three heads: one for the board reconstruction, one for the popularity, and one for the eval.
        embedding = self.encoder(x)
        popularity = self.popularity_head(embedding)
        evaluation = self.evaluation_head(embedding)
        reconstruction = self.reconstruction_head(embedding)
        return embedding, popularity, evaluation, reconstruction
