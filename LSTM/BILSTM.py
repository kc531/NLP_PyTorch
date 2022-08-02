import torch
import torch.nn as nn
import math

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class BILSTM(nn.Module):
    def __init__(self, input_size, hidden_size, batch_size, seq_len):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.batch_size = batch_size
        self.seq_len = seq_len

        #Forward
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


        #Forward
        # i_t
        self.w_i_b = nn.Parameter(torch.Tensor(self.input_size, self.hidden_size))
        self.u_i_b = nn.Parameter(torch.Tensor(self.hidden_size, self.hidden_size))
        self.b_i_b = nn.Parameter(torch.Tensor(self.hidden_size))
        # f_t
        self.w_f_b = nn.Parameter(torch.Tensor(self.input_size, self.hidden_size))
        self.u_f_b = nn.Parameter(torch.Tensor(self.hidden_size, self.hidden_size))
        self.b_f_b = nn.Parameter(torch.Tensor(self.hidden_size))
        # c_t
        self.w_c_b = nn.Parameter(torch.Tensor(self.input_size, self.hidden_size))
        self.u_c_b = nn.Parameter(torch.Tensor(self.hidden_size, self.hidden_size))
        self.b_c_b = nn.Parameter(torch.Tensor(self.hidden_size))
        # o_t
        self.w_o_b = nn.Parameter(torch.Tensor(self.input_size, self.hidden_size))
        self.u_o_b = nn.Parameter(torch.Tensor(self.hidden_size, self.hidden_size))
        self.b_o_b = nn.Parameter(torch.Tensor(self.hidden_size))

        self.init_weight()

    def init_weight(self):
        stdv = 1.0 / math.sqrt(self.hidden_size)
        for weights in self.parameters():
            weights.data.uniform_(-stdv, stdv)

    def forward(self, x, init_states=None):
        batch_size, seq_len = self.batch_size, self.seq_len
        hidden_seq = []
        hidden_seq_b = []
        y_net = []
        if init_states is None:
            h_t = torch.zeros(batch_size, self.hidden_size).to(x.device)
            c_t = torch.zeros(batch_size,self.hidden_size).to(x.device)
            h_t_b = torch.zeros(batch_size, self.hidden_size).to(x.device)
            c_t_b = torch.zeros(batch_size,self.hidden_size).to(x.device)

        else:
            h_t, c_t, h_t_b, c_t_b = init_states

        for t in range(seq_len):
            x_t = x[:, t, :]
            i_t = torch.sigmoid(x_t @ self.w_i + h_t @ self.u_i + self.b_i)
            f_t = torch.sigmoid(x_t @ self.w_f + h_t @ self.u_f + self.b_f)
            g_t = torch.sigmoid(x_t @ self.w_c + h_t @ self.u_c + self.b_c)
            o_t = torch.sigmoid(x_t @ self.w_o + h_t @ self.u_o + self.b_o)
            c_t = f_t + c_t + i_t * g_t
            h_t = o_t + torch.tanh(c_t)

            x_t_b = x[:, seq_len-t-1, :]
            i_t_b = torch.sigmoid(x_t_b @ self.w_i_b + h_t_b @ self.u_i_b + self.b_i_b)
            f_t_b = torch.sigmoid(x_t_b @ self.w_f_b + h_t_b @ self.u_f_b + self.b_f_b)
            g_t_b = torch.sigmoid(x_t_b @ self.w_c_b + h_t_b @ self.u_c_b + self.b_c_b)
            o_t_b = torch.sigmoid(x_t_b @ self.w_o_b + h_t_b @ self.u_o_b + self.b_o_b)
            c_t_b = f_t_b + c_t_b + i_t_b * g_t_b
            h_t_b = o_t_b + torch.tanh(c_t_b)

            hidden_seq.append(h_t.unsqueeze(0))
            hidden_seq_b.append(h_t_b.unsqueeze(0))

        hidden_seq = torch.cat(hidden_seq, dim=0)
        hidden_seq_b = torch.cat(hidden_seq_b, dim=0)

        hidden_seq = hidden_seq.transpose(0, 1).contiguous()
        hidden_seq_b = hidden_seq_b.transpose(0, 1).contiguous()
        hidden_seq_b = torch.flip(hidden_seq_b,dims=(1,))

        y_net = torch.cat((hidden_seq,hidden_seq_b),dim=2)
        y_t = torch.cat((h_t,h_t_b),dim=1)

        return y_t, y_net


class BVLSTM(nn.Module):
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

        #Backward
        # i_t
        self.w_i_x_b = nn.Parameter(torch.Tensor(self.input_size_x, self.hidden_size))
        self.w_i_h_b = nn.Parameter(torch.Tensor(self.input_size_h, self.hidden_size))
        self.u_i_b  = nn.Parameter(torch.Tensor(self.hidden_size, self.hidden_size))
        self.b_i_b  = nn.Parameter(torch.Tensor(self.hidden_size))
        # f_t
        self.w_f_x_b  = nn.Parameter(torch.Tensor(self.input_size_x, self.hidden_size))
        self.w_f_h_b  = nn.Parameter(torch.Tensor(self.input_size_h, self.hidden_size))
        self.u_f_b  = nn.Parameter(torch.Tensor(self.hidden_size, self.hidden_size))
        self.b_f_b  = nn.Parameter(torch.Tensor(self.hidden_size))
        # c_t
        self.w_c_x_b  = nn.Parameter(torch.Tensor(self.input_size_x, self.hidden_size))
        self.w_c_h_b = nn.Parameter(torch.Tensor(self.input_size_h, self.hidden_size))
        self.u_c_b  = nn.Parameter(torch.Tensor(self.hidden_size, self.hidden_size))
        self.b_c_b  = nn.Parameter(torch.Tensor(self.hidden_size))
        # o_t
        self.w_o_x_b  = nn.Parameter(torch.Tensor(self.input_size_x, self.hidden_size))
        self.w_o_h_b  = nn.Parameter(torch.Tensor(self.input_size_h, self.hidden_size))
        self.u_o_b  = nn.Parameter(torch.Tensor(self.hidden_size, self.hidden_size))
        self.b_o_b  = nn.Parameter(torch.Tensor(self.hidden_size))

        self.init_weight()

    def init_weight(self):
        stdv = 1.0 / math.sqrt(self.hidden_size)
        for weights in self.parameters():
            weights.data.uniform_(-stdv, stdv)

    def forward(self, x, h_prev, init_states=None):
        batch_size, seq_len = self.batch_size, self.seq_len
        hidden_seq = []
        hidden_seq_b = []
        y_net = []
        h_prev_b = h_prev[:,:,self.hidden_size:2*self.hidden_size]
        h_prev = h_prev[:,:,0:self.hidden_size]

        if init_states is None:
            h_t, c_t = torch.zeros(batch_size, self.hidden_size).to(x.device), torch.zeros(batch_size,
                                                                                           self.hidden_size).to(
                x.device)
            h_t_b = torch.zeros(batch_size, self.hidden_size).to(x.device)
            c_t_b = torch.zeros(batch_size,self.hidden_size).to(x.device)
        else:
            h_t, c_t, h_t_b, c_t_b = init_states

        for t in range(seq_len):
            x_t = x[:, t, :]
            h_prev_t = h_prev[:, t, :]
            i_t = torch.sigmoid(x_t @ self.w_i_x + h_prev_t @ self.w_i_h  + h_t @ self.u_i + self.b_i)
            f_t = torch.sigmoid(x_t @ self.w_f_x + h_prev_t @ self.w_f_h  + h_t @ self.u_i + self.b_i)
            g_t = torch.sigmoid(x_t @ self.w_c_x + h_prev_t @ self.w_c_h  + h_t @ self.u_i + self.b_i)
            o_t = torch.sigmoid(x_t @ self.w_o_x + h_prev_t @ self.w_o_h  + h_t @ self.u_i + self.b_i)
            c_t = f_t + c_t + i_t * g_t
            h_t = o_t + torch.tanh(c_t)


            x_t_b = x[:, seq_len-t-1, :]
            h_prev_t_b = h_prev_b[:, seq_len-t-1, :]
            i_t_b = torch.sigmoid(x_t_b @ self.w_i_x_b + h_prev_t_b @ self.w_i_h_b  + h_t_b @ self.u_i_b + self.b_i_b)
            f_t_b = torch.sigmoid(x_t_b @ self.w_f_x_b + h_prev_t_b @ self.w_f_h_b  + h_t_b @ self.u_i_b + self.b_i_b)
            g_t_b = torch.sigmoid(x_t_b @ self.w_c_x_b + h_prev_t_b @ self.w_c_h_b  + h_t_b @ self.u_i_b + self.b_i_b)
            o_t_b = torch.sigmoid(x_t_b @ self.w_o_x_b + h_prev_t_b @ self.w_o_h_b  + h_t_b @ self.u_i_b + self.b_i_b)
            c_t_b = f_t_b + c_t_b + i_t_b * g_t_b
            h_t_b = o_t_b + torch.tanh(c_t_b)


            hidden_seq.append(h_t.unsqueeze(0))
            hidden_seq_b.append(h_t_b.unsqueeze(0))

        hidden_seq = torch.cat(hidden_seq, dim=0)
        hidden_seq_b = torch.cat(hidden_seq_b, dim=0)

        hidden_seq = hidden_seq.transpose(0, 1).contiguous()
        hidden_seq_b = hidden_seq_b.transpose(0, 1).contiguous()
        hidden_seq_b = torch.flip(hidden_seq_b,dims=(1,))

        y_net = torch.cat((hidden_seq,hidden_seq_b),dim=2)
        y_t = torch.cat((h_t,h_t_b),dim=1)

        return y_t, y_net


class DBLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, batch_size, seq_len, n_layers, verbose  = True):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.n_layers = n_layers
        self.verbose = verbose
        self.lstm1 = BILSTM(input_size, hidden_size, batch_size, seq_len)

        if self.verbose:
            self.dlstm = nn.ModuleList(
            [BVLSTM(input_size, hidden_size, hidden_size, batch_size, seq_len) for _ in range(self.n_layers-1)]
            )
        else:
            self.dlstm = nn.ModuleList(
            [BILSTM(hidden_size, hidden_size, batch_size, seq_len) for _ in range(self.n_layers-1)]
            )


    def forward(self, x):
        y_t, y_net = self.lstm1(x)
        if self.verbose: 
            for lstm in self.dlstm:
                y_t, y_net = lstm(x, y_net)
        else:
            for lstm in self.dlstm:
                y_t, y_net = lstm(y_net)
        return y_t, y_net
