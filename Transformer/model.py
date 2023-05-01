from Transformer_Attention_block import Transformer_model, Encoder, Decoder


def Transformer(SRC_PAD_IDX,TRG_PAD_IDX,INPUT_DIM,OUTPUT_DIM,HIDDEN_DIM,ENC_LAYERS,DEC_LAYERS,ENC_HEADS,DEC_HEADS,ENC_PF_DIM,DEC_PF_DIM,ENC_DROPOUT,DEC_DROPOUT,device):
    enc = Encoder(INPUT_DIM, HIDDEN_DIM, ENC_LAYERS, ENC_HEADS, ENC_PF_DIM, ENC_DROPOUT, device)
    dec = Decoder(OUTPUT_DIM, HIDDEN_DIM, DEC_LAYERS, DEC_HEADS, DEC_PF_DIM, DEC_DROPOUT, device)

    return Transformer_model(enc, dec, SRC_PAD_IDX, TRG_PAD_IDX, device).to(device)