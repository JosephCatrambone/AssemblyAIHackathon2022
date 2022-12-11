import gradio as gr
import torch

from board import BitBoard
from data import bitboard_to_tensor
from model import ChessModel

mdl = torch.load("checkpoints/model.pth", map_location='cpu')

def evaluate_fen(fen):
    board = BitBoard.from_fen(fen)
    arr = bitboard_to_tensor(board).to(torch.float32)
    _embedding, predicted_popularity, _predicted_evaluation, _predicted_board_vec = mdl(arr)
    return f"Estimated popularity: {predicted_popularity.cpu().item()}"

demo = gr.Interface(fn=evaluate_fen, inputs="text", outputs="text")

demo.launch()