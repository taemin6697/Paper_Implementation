import torch

from Transformer_Train_code import device
from dataset import load_dataset
from main import SRC_PAD_IDX, TRG_PAD_IDX
from model import Transformer
from utils import show_bleu

train_dataset, valid_dataset, test_dataset,SRC,TRG = load_dataset()

model = Transformer(
    SRC_PAD_IDX,
    TRG_PAD_IDX,
    INPUT_DIM=len(SRC.vocab),
    OUTPUT_DIM = len(TRG.vocab),
    HIDDEN_DIM = 512,
    ENC_LAYERS = 3,
    DEC_LAYERS = 3,
    ENC_HEADS = 8,
    DEC_HEADS = 8,
    ENC_PF_DIM = 512,
    DEC_PF_DIM = 512,
    ENC_DROPOUT = 0.1,
    DEC_DROPOUT = 0.1,
    device=device
)

model.load_state_dict(torch.load("./save_pt/transformer_german_to_english.pt"))
show_bleu(test_dataset, SRC, TRG, model, device)