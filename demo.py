import gradio as gr
import torch

from board import BitBoard
from data import bitboard_to_tensor

model = torch.load("checkpoints/model.pth")

def evaluate_fen(fen):
    board = BitBoard.from_fen(fen)
    arr = bitboard_to_tensor(board)
    _embedding, predicted_popularity, _predicted_evaluation, _predicted_board_vec = model(arr)
    return f"Predicted popularity: {predicted_popularity}"

demo = gr.Interface(fn=evaluate_fen, inputs="text", outputs="text")

demo.launch()