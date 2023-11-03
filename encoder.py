import torch 
import torch.nn as nn 


class Encode(nn.Module):
    def __init__(self, 
                unit1, 
                unit2, 
                layer_encoder,
                states_per_layer,
                fp_dim,
                mf_dim 
                ):
        super(Encode, self).__init__()
        self.unit1 = unit1
        self.unit2 = unit2
        self.layer_encoder = layer_encoder
        self.states_per_layer = states_per_layer
        self.dense1 = nn.Linear(fp_dim, self.unit1)
        self.relu1 = nn.ReLU()
        self.dense2 = nn.Linear(self.unit1,self.unit2)
        self.relu2 = nn.ReLU()
        self.dense3 = nn.Linear(mf_dim, self.unit2)
        self.relu3 = nn.ReLU()
        self.dense4 = nn.Linear(self.unit2, self.unit2)
        self.relu4 = nn.ReLU()
        self.dense5 = nn.Linear(self.unit1, self.unit2)
        self.relu5 = nn.ReLU()
        self.dense6 = nn.Linear(self.unit2, self.unit2)
        self.layernorm1 = nn.LayerNorm(self.unit2)
        self.dense7 = nn.Linear(self.unit2, self.unit2)
        self.layernorm2 = nn.LayerNorm(self.unit2)
        self.dense1 = nn.Linear(fp_dim, self.unit1)
        self.relu1 = nn.ReLU()
        self.dense2 = nn.Linear(self.unit1,self.unit2)
        self.relu2 = nn.ReLU()
        self.fc_layers = nn.ModuleList([nn.Sequential(
            nn.Linear(self.unit2, self.unit2),
            nn.ReLU(),
            nn.Linear(self.unit2, self.unit2),
            nn.ReLU()
        ) for _ in range(3)])    
        self.alpha = nn.Parameter(torch.ones(self.unit2), requires_grad=True)  

    def forward(self, encode_fp, encode_mol):
        
        encode_fp = self.relu1(self.dense1(encode_fp.float()))
        encode_fp = self.relu2(self.dense2(encode_fp))
        encode_fp = self.layernorm1(self.dense6(encode_fp))
        
        encode_mol = self.relu3(self.dense3(encode_mol.float()))
        encode_mol = self.relu4(self.dense4(encode_mol))
        encode_mol = self.layernorm2(self.dense7(encode_mol))
        
        inputs = torch.cat([encode_fp, self.alpha * encode_mol], dim=1)
        z = self.relu5(self.dense5(inputs))
        x = z
        state = []
        for fc_layer in self.fc_layers:
            x = fc_layer(x)
            state.append(x)
        return z, state, encode_mol
    