import torch
from torch import nn, Tensor
import torch.nn.functional as F
import math

class TransformerNetwork(nn.Module):

    def __init__(self, n_layers, n_heads, demb):
        super().__init__()

        #self.positional_encoder = PositionalEncoding(demb,max_len = 10)
        
        #self.compute_nodes_start = nn.Embedding(compute_nodes,demb)
        
        encoder_layer = lambda : nn.TransformerEncoderLayer(d_model=demb, nhead=n_heads)
        self.transformer_encoder = nn.Sequential(*[encoder_layer() for i in range(n_layers)])
        #self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        #self.widden = nn.Linear(3,demb)
        #decoder_layer = nn.TransformerDecoderLayer(d_model=demb, nhead=n_heads)
        #self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=n_layers)
        self.demb = demb
        self.ffwd = nn.Sequential(
            nn.Linear(demb,200),
            nn.ReLU(),
            nn.Linear(200,2),
        )
        #self.flatten = nn.Flatten()
        self.apply(self._init_weights)
       # self.compute_nodes_start = nn.Parameter(torch.randn(compute_nodes,demb,device=device).unsqueeze(0),requires_grad=True)

    def forward(self, x):
        x = torch.cat([positional_encoding(x[:,:,0],int(self.demb/3)),positional_encoding(x[:,:,1],int(self.demb/3)),positional_encoding(x[:,:,2],int(self.demb/3))],axis=2)#self.widden(x) + positional_encoding(x[:,:,0],demb)
        #x = torch.cat([x,self.compute_nodes_start(torch.arange(compute_nodes,device=device)).expand(batch_size,compute_nodes,demb)],dim=1)
        x = self.transformer_encoder(x)
        x = self.ffwd(torch.max(x,1)[0])
        #x = self.ffwd(x).mean(dim=1)
        return x
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

def positional_encoding(position: Tensor, demb, max_len=2) -> Tensor:

    #div_term = torch.exp(torch.arange(0, demb, 2,device=device) * (-math.log(2.0) / demb))
    #div_term = torch.exp(-(torch.arange(0, demb, 2,device=device))/demb)
    div_term = torch.exp((demb-torch.arange(0, demb, 2,device=position.device))*math.log(1.1))*2*torch.pi/(4*max_len)

    #pe = torch.zeros(*position.shape, demb,device=device)
    pe = torch.zeros(*position.shape, demb,device=position.device)
    pe[:,:, 0::2] = torch.sin(position.unsqueeze(2) * div_term)
    pe[:,:, 1::2] = torch.cos(position.unsqueeze(2) * div_term)
    
    return pe