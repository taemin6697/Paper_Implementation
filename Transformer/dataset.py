from torchtext.data import Field, BucketIterator
from torchtext.datasets import Multi30k
import spacy

from utils import tokenize_de, tokenize_en

def load_dataset():
    spacy_en = spacy.load('en_core_web_sm')  # 영어 토큰화(tokenization)
    spacy_de = spacy.load('de_core_news_sm')
    SRC = Field(tokenize=tokenize_de, init_token="<sos>", eos_token="<eos>", lower=True, batch_first=True)
    TRG = Field(tokenize=tokenize_en, init_token="<sos>", eos_token="<eos>", lower=True, batch_first=True)
    train_dataset, valid_dataset, test_dataset = Multi30k.splits(exts=(".de", ".en"), fields=(SRC, TRG))
    SRC.build_vocab(train_dataset, min_freq=2)
    TRG.build_vocab(train_dataset, min_freq=2)
    return train_dataset, valid_dataset, test_dataset,SRC,TRG
