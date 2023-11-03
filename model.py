import torch 
import torch.nn as nn
import numpy as np 
import os
from encoder import Encode
from decoder import Decoder


class Encode_Decode(nn.Module):
    def __init__(self):
        super(Encode_Decode, self).__init__()
        
        self.encoder = Encode()
        self.decoder = Decoder()
        
    def forward(self, encode_fp, encode_mol, decode_token):
        decode_token = decode_token
        encode_output, encode_state, decode_mol = self.encoder(encode_fp, encode_mol)
        initial_state = torch.tensor(np.array([item.cpu().detach().numpy() for item in encode_state]))
        initial_state = (initial_state, torch.zeros(3, decode_token.size(0), 256))
        z = encode_output.view(encode_output.size(0), 1, encode_output.size(1)).repeat(1,decode_token.size(1), 1)
        
        decode_mol = decode_mol.view(decode_mol.size(0), 1, decode_mol.size(1)).repeat(1,decode_token.size(1), 1)
        decoder_inputs = torch.cat([decode_token,decode_mol, z],dim=2)
        decoder_output = self.decoder(decoder_inputs, initial_state)
        
        return decoder_output