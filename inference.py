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

def pytorch_cos_sim(a, b):
    if not isinstance(a, torch.Tensor):
        a = torch.tensor(a)

    if not isinstance(b, torch.Tensor):
        b = torch.tensor(b)

    if len(a.shape) == 1:
        a = a.unsqueeze(0)

    if len(b.shape) == 1:
        b = b.unsqueeze(0)

    a_norm = torch.nn.functional.normalize(a, p=2, dim=1)
    b_norm = torch.nn.functional.normalize(b, p=2, dim=1)
    return torch.mm(a_norm, b_norm.transpose(0, 1))

def main():
    t1 = encode_token("test", alphabet).unsqueeze(0)
    t2 = encode_token("test2", alphabet).unsqueeze(0)
    emb = networks.TwoLayerCNN(C=len(alphabet), M=90, embedding=128, channel=8, mtc_input=1)
    v1 = emb(t1)
    v2 = emb(t2)
    print(pytorch_cos_sim(v1, v2))
    # emb = torch.load("model.torch", map_location=torch.device('cpu')).embedding_net


if __name__ == '__main__':
    main()