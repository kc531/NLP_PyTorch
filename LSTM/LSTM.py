import torch
import torch.nn as nn
import math

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class CustomLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, batch_size, seq_len):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.batch_size = batch_size
        self.seq_len = seq_len

        # i_t
        self.w_i = nn.Parameter(torch.Tensor(self.input_size, self.hidden_size))
        self.u_i = nn.Parameter(torch.Tensor(self.hidden_size, self.hidden_size))
        self.b_i = nn.Parameter(torch.Tensor(self.hidden_size))
        # f_t
        self.w_f = nn.Parameter(torch.Tensor(self.input_size, self.hidden_size))
        self.u_f = nn.Parameter(torch.Tensor(self.hidden_size, self.hidden_size))
        self.b_f = nn.Parameter(torch.Tensor(self.hidden_size))
        # c_t
        self.w_c = nn.Parameter(torch.Tensor(self.input_size, self.hidden_size))
        self.u_c = nn.Parameter(torch.Tensor(self.hidden_size, self.hidden_size))
        self.b_c = nn.Parameter(torch.Tensor(self.hidden_size))
        # o_t
        self.w_o = nn.Parameter(torch.Tensor(self.input_size, self.hidden_size))
        self.u_o = nn.Parameter(torch.Tensor(self.hidden_size, self.hidden_size))
        self.b_o = nn.Parameter(torch.Tensor(self.hidden_size))

        self.init_weight()

    def init_weight(self):
        stdv = 1.0 / math.sqrt(self.hidden_size)
        for weights in self.parameters():
            weights.data.uniform_(-stdv, stdv)

    def forward(self, x, init_states=None):
        batch_size, seq_len = self.batch_size, self.seq_len
        hidden_seq = []
        if init_states is None:
            h_t = torch.zeros(batch_size, self.hidden_size).to(x.device)
            c_t = torch.zeros(batch_size,self.hidden_size).to(x.device)
        
        else:
            h_t, c_t = init_states

        for t in range(seq_len):
            x_t = x[:, t, :]
            i_t = torch.sigmoid(x_t @ self.w_i + h_t @ self.u_i + self.b_i)
            f_t = torch.sigmoid(x_t @ self.w_f + h_t @ self.u_f + self.b_f)
            g_t = torch.sigmoid(x_t @ self.w_c + h_t @ self.u_c + self.b_c)
            o_t = torch.sigmoid(x_t @ self.w_o + h_t @ self.u_o + self.b_o)
            c_t = f_t + c_t + i_t * g_t
            h_t = o_t + torch.tanh(c_t)
            hidden_seq.append(h_t.unsqueeze(0))

        hidden_seq = torch.cat(hidden_seq, dim=0)
        #print(hidden_seq.shape)
        hidden_seq = hidden_seq.transpose(0, 1).contiguous()
        #print(hidden_seq.shape)
        return h_t, c_t, hidden_seq
    
    
class VLSTM(nn.Module):
    def __init__(self, input_size_x, input_size_h, hidden_size, batch_size, seq_len):
        super().__init__()
        self.input_size_x = input_size_x
        self.input_size_h = input_size_h
        self.hidden_size = hidden_size
        self.batch_size = batch_size
        self.seq_len = seq_len

        # i_t
        self.w_i_x = nn.Parameter(torch.Tensor(self.input_size_x, self.hidden_size))
        self.w_i_h = nn.Parameter(torch.Tensor(self.input_size_h, self.hidden_size))
        self.u_i = nn.Parameter(torch.Tensor(self.hidden_size, self.hidden_size))
        self.b_i = nn.Parameter(torch.Tensor(self.hidden_size))
        # f_t
        self.w_f_x = nn.Parameter(torch.Tensor(self.input_size_x, self.hidden_size))
        self.w_f_h = nn.Parameter(torch.Tensor(self.input_size_h, self.hidden_size))
        self.u_f = nn.Parameter(torch.Tensor(self.hidden_size, self.hidden_size))
        self.b_f = nn.Parameter(torch.Tensor(self.hidden_size))
        # c_t
        self.w_c_x = nn.Parameter(torch.Tensor(self.input_size_x, self.hidden_size))
        self.w_c_h = nn.Parameter(torch.Tensor(self.input_size_h, self.hidden_size))
        self.u_c = nn.Parameter(torch.Tensor(self.hidden_size, self.hidden_size))
        self.b_c = nn.Parameter(torch.Tensor(self.hidden_size))
        # o_t
        self.w_o_x = nn.Parameter(torch.Tensor(self.input_size_x, self.hidden_size))
        self.w_o_h = nn.Parameter(torch.Tensor(self.input_size_h, self.hidden_size))
        self.u_o = nn.Parameter(torch.Tensor(self.hidden_size, self.hidden_size))
        self.b_o = nn.Parameter(torch.Tensor(self.hidden_size))

        self.init_weight()

    def init_weight(self):
        stdv = 1.0 / math.sqrt(self.hidden_size)
        for weights in self.parameters():
            weights.data.uniform_(-stdv, stdv)

    def forward(self, x, h_prev, init_states=None):
        batch_size, seq_len = self.batch_size, self.seq_len
        hidden_seq = []

        if init_states is None:
            h_t, c_t = torch.zeros(batch_size, self.hidden_size).to(x.device), torch.zeros(batch_size,
                                                                                           self.hidden_size).to(
                x.device)
        else:
            h_t, c_t = init_states

        for t in range(seq_len):
            x_t = x[:, t, :]
            h_prev_t = h_prev[:, t, :]
            i_t = torch.sigmoid(x_t @ self.w_i_x + h_prev_t @ self.w_i_h  + h_t @ self.u_i + self.b_i)
            f_t = torch.sigmoid(x_t @ self.w_f_x + h_prev_t @ self.w_f_h  + h_t @ self.u_i + self.b_i)
            g_t = torch.sigmoid(x_t @ self.w_c_x + h_prev_t @ self.w_c_h  + h_t @ self.u_i + self.b_i)
            o_t = torch.sigmoid(x_t @ self.w_o_x + h_prev_t @ self.w_o_h  + h_t @ self.u_i + self.b_i)
            c_t = f_t + c_t + i_t * g_t
            h_t = o_t + torch.tanh(c_t)
            hidden_seq.append(h_t.unsqueeze(0))

        hidden_seq = torch.cat(hidden_seq, dim=0)
        #print(hidden_seq.shape)
        hidden_seq = hidden_seq.transpose(0, 1).contiguous()
        #print(hidden_seq.shape)
        return h_t, c_t, hidden_seq

class DLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, batch_size, seq_len, n_layers, verbose  = True):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.n_layers = n_layers
        self.verbose = verbose
        self.lstm1 = CustomLSTM(input_size, hidden_size, batch_size, seq_len)
        
        if self.verbose:
            self.dlstm = nn.ModuleList(
            [VLSTM(input_size, hidden_size, hidden_size, batch_size, seq_len) for _ in range(self.n_layers-1)]
            )
        else:
            self.dlstm = nn.ModuleList(
            [CustomLSTM(hidden_size, hidden_size, batch_size, seq_len) for _ in range(self.n_layers-1)]
            )
            

    def forward(self, x):
        h_t, c_t, h_vec = self.lstm1(x)
        if self.verbose: 
            for lstm in self.dlstm:
                h_t, c_t, h_vec = lstm(x, h_vec)
        else:
            for lstm in self.dlstm:
                h_t, c_t, h_vec = lstm(h_vec)
        return h_t, c_t, h_vec
