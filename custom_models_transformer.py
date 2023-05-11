# transformer model precision improvement (tmp)

import math

import torch
import torch.nn as nn
from torch import Tensor

from custom_models import DenseModel_based_on_FNN_SpikeDeeptector

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 200): # max_len equals maximum window size
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)

class TransformerModel(nn.Module):
    def __init__(self, input_dim: int, hidden_size: int, num_classes: int, num_layers: int, num_heads: int,
                 dropout: float, device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')):
        super(TransformerModel, self).__init__()

        self.input_dim = input_dim
        self.hidden_size = hidden_size
        self.num_classes = num_classes
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.dropout = dropout
        self.device = device

        self.embedding = nn.Linear(input_dim, hidden_size)
        self.pos_encoder = PositionalEncoding(hidden_size, dropout)
        encoder_layer = nn.TransformerEncoderLayer(hidden_size, num_heads, dim_feedforward=hidden_size*4, dropout=dropout, activation='gelu')
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers)
        self.fc = nn.Linear(hidden_size, num_classes)
        self.softmax = nn.Softmax(dim=1)
        self.sigmoid = nn.Sigmoid()
        self.selu = nn.SELU()

        #self.dense = DenseModel_based_on_FNN_SpikeDeeptector(in_features=hidden_size, out_features=num_classes)

    def forward(self, src: Tensor) -> Tensor:
        batch_size, seq_len = src.shape[0], src.shape[1]
        src = src.view(batch_size, seq_len, self.input_dim)
        src = src.transpose(0, 1).contiguous()
        src = self.embedding(src)
        src = self.pos_encoder(src)
        output = self.transformer_encoder(src)
        output = output.mean(dim=0)
        #output = self.dense(output)
        output = self.fc(output)
        #output = self.softmax(output)
        #output = self.sigmoid(output)
        #pred = (output >= 0.9).float()
        return output


class TransformerModel_without_embedding(nn.Module):
    def __init__(self, input_dim: int, hidden_size: int, num_classes: int, num_layers: int, num_heads: int,
                 dropout: float, device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')):
        super(TransformerModel_without_embedding, self).__init__()

        self.input_dim = input_dim
        self.hidden_size = hidden_size
        self.num_classes = num_classes
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.dropout = dropout
        self.device = device

        #self.embedding = nn.Linear(input_dim, hidden_size)
        self.pos_encoder = PositionalEncoding(hidden_size, dropout)
        encoder_layer = nn.TransformerEncoderLayer(hidden_size, num_heads, dim_feedforward=hidden_size*4, dropout=dropout, activation='gelu')
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers)
        self.fc = nn.Linear(hidden_size, num_classes)
        self.softmax = nn.Softmax(dim=1)
        self.sigmoid = nn.Sigmoid()
        self.selu = nn.SELU()

        #self.dense = DenseModel_based_on_FNN_SpikeDeeptector(in_features=hidden_size, out_features=num_classes)

    def forward(self, src: Tensor) -> Tensor:
        batch_size, seq_len = src.shape[0], src.shape[1]
        src = src.view(batch_size, seq_len, self.input_dim)
        src = src.transpose(0, 1).contiguous()
        #src = self.embedding(src)
        src = self.pos_encoder(src)
        output = self.transformer_encoder(src)
        output = output.mean(dim=0)
        #output = self.dense(output)
        output = self.fc(output)
        #output = self.softmax(output)
        #output = self.sigmoid(output)
        #pred = (output >= 0.9).float()
        return output


class StackedTransformers(nn.Module):
    def __init__(self, device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')):
        super(StackedTransformers, self).__init__()

        self.device = device
        self.model1 = TransformerModel(input_dim=1, hidden_size=256, num_classes=2, num_layers=6, num_heads=8,
                                       dropout=0.1)
        self.model2 = TransformerModel_without_embedding(input_dim=1, hidden_size=256, num_classes=2, num_layers=6,
                                                         num_heads=8, dropout=0.1)
        self.fc = nn.Linear(4, 2)
        self.softmax = nn.Softmax(dim=1)
    def forward(self, x):
        x1 = self.model1(x)
        x2 = self.model2(x)
        out_cat = torch.cat([x1, x2], dim=1)
        out_cat = self.fc(out_cat)
        out_cat = self.softmax(out_cat)
        return out_cat


## following architectures are not working properly

import torch
import torch.nn as nn
import math

class TransformerModel_2(nn.Module):
    def __init__(self, input_dim, hidden_size, num_classes, num_layers, num_heads, dropout, device):
        super(TransformerModel_2, self).__init__()

        self.device = device
        self.num_layers = num_layers
        self.embedding = nn.Linear(input_dim, hidden_size)

        # Define positional encoding
        max_len = 20 #window_size
        d_model = hidden_size
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

        self.layers = nn.ModuleList()
        for i in range(num_layers):
            self.layers.append(nn.TransformerEncoderLayer(d_model, num_heads, dim_feedforward=hidden_size, dropout=dropout))

        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        x = self.embedding(x)
        x = x * math.sqrt(self.embedding.weight.shape[-1])
        x = x + self.pe[:x.size(1), :].to(self.device)
        for i in range(self.num_layers):
            x = self.layers[i](x)
        x = x.mean(dim=1)
        x = self.fc(x)
        return x


class TransformerModel_too_complex(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, num_classes: int, num_layers: int, num_heads: int,
                 dropout: float, bias: bool = True, device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
                 dtype=None) -> None:
        super(TransformerModel_too_complex, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_classes = num_classes
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.dropout = dropout
        self.bias = bias
        self.device = device

        self.embedding = nn.Embedding(input_size, hidden_size)
        self.transformer = nn.Transformer(d_model=hidden_size, nhead=num_heads, num_encoder_layers=num_layers,
                                          num_decoder_layers=num_layers, dim_feedforward=hidden_size * 4,
                                          dropout=dropout, activation='gelu', device=device)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        x = x.long().to(self.device)
        x = self.embedding(x)
        x = x.permute(1, 0, 2)
        output = self.transformer(x, x)
        output = output.permute(1, 0, 2)
        output = self.fc(output[:, -1, :])
        return output


