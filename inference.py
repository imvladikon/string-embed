import numpy as np
import torch
import networks
from string import ascii_letters, punctuation, digits


alphabet = ascii_letters + punctuation + digits + " "

def to_ord(c, all_chars, alphabet):
    if not (c in all_chars):
        alphabet += c
        all_chars[c] = all_chars["counter"]
        all_chars["counter"] = all_chars["counter"] + 1
    return all_chars[c]

def encode_token(t, alphabet, max_length=90):
    M = max_length
    C = len(alphabet)
    all_chars = {c: idx for idx, c in enumerate(alphabet)}
    all_chars["counter"] = len(all_chars)
    x = [[to_ord(c, all_chars, alphabet) for c in t]]
    index = 0
    encode = np.zeros((C, M), dtype=np.float32)
    encode[np.array(x[index]), np.arange(len(x[index]))] = 1.0
    return torch.from_numpy(encode)


def main():
    t = encode_token("test", alphabet).unsqueeze(0)
    emb = networks.TwoLayerCNN(C=len(alphabet), M=90, embedding=128, channel=8, mtc_input=1)
    print(emb(t))
    # emb = torch.load("model.torch", map_location=torch.device('cpu')).embedding_net


if __name__ == '__main__':
    main()