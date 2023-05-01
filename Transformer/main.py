import time
import math
import random

import torch
import torch.optim as optim
from torch import nn
from torchtext.data import BucketIterator

from Transformer_Train_code import train, evaluate
from dataset import load_dataset
from model import Transformer
from utils import count_parameters, initialize_weights, epoch_time

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

train_dataset, valid_dataset, test_dataset,SRC,TRG = load_dataset()
train_iterator, valid_iterator, test_iterator = BucketIterator.splits(
    (train_dataset, valid_dataset, test_dataset),
    batch_size=128,shuffle=True,
    device=device)

SRC_PAD_IDX = SRC.vocab.stoi[SRC.pad_token]
TRG_PAD_IDX = TRG.vocab.stoi[TRG.pad_token]

model = Transformer(
    SRC_PAD_IDX,
    TRG_PAD_IDX,
    INPUT_DIM=len(SRC.vocab),
    OUTPUT_DIM = len(TRG.vocab),
    HIDDEN_DIM = 512,
    ENC_LAYERS = 4,
    DEC_LAYERS = 4,
    ENC_HEADS = 8,
    DEC_HEADS = 8,
    ENC_PF_DIM = 512,
    DEC_PF_DIM = 512,
    ENC_DROPOUT = 0.1,
    DEC_DROPOUT = 0.1,
    device=device
)

print(f'The model has {count_parameters(model):,} trainable parameter')
model.apply(initialize_weights)
print(model)
N_EPOCHS = 200
CLIP = 1
best_valid_loss = float('inf')

LEARNING_RATE = 0.0005
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
criterion = nn.CrossEntropyLoss(ignore_index = TRG_PAD_IDX)

if __name__ == "__main__":
    for epoch in range(N_EPOCHS):
        start_time = time.time() # 시작 시간 기록

        train_loss = train(model, train_iterator, optimizer, criterion, CLIP,SRC,TRG,test_dataset)
        valid_loss = evaluate(model, valid_iterator, criterion)

        end_time = time.time() # 종료 시간 기록
        epoch_mins, epoch_secs = epoch_time(start_time, end_time)

        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            torch.save(model.state_dict(), './save_pt/transformer_german_to_english.pt')

        print(f'Epoch: {epoch + 1:02} | Time: {epoch_mins}m {epoch_secs}s')
        print(f'\tTrain Loss: {train_loss:.3f} | Train PPL: {math.exp(train_loss):.3f}')
        print(f'\tValidation Loss: {valid_loss:.3f} | Validation PPL: {math.exp(valid_loss):.3f}')