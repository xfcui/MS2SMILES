import torch 
import torch.nn as nn 

class Decoder(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, numlayers):
        super(Decoder, self).__init__()
        
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.num_layers = numlayers
        self.lstm_layers = nn.ModuleList([
            nn.LSTM(input_size if i == 0 else hidden_size, hidden_size, batch_first=True) for i in range( self.num_layers) 
        ])
        input_size = self.hidden_size
        self.layer_norms = nn.ModuleList([
            nn.LayerNorm(hidden_size) for _ in range(self.num_layers)
        ])
        self.dense = nn.Linear(self.hidden_size, self.output_size)
        
    def forward(self, x, initial_state=None):
        if initial_state is None:
            initial_state = [None] * self.numlayers
        
        for i in range(self.num_layers):
            lstm_layer = self.lstm_layers[i]
            ln_layer = self.layer_norms[i]           
            hx, cx = initial_state   
            lstm_output, (h_n, c_n) = lstm_layer(x, (hx[i].unsqueeze(0), cx[i].unsqueeze(0)))        
            x = ln_layer(lstm_output)
        x = self.dense(x)
        return x