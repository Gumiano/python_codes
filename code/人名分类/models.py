import torch
import torch.nn as nn

class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=1):
        super(RNN, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.num_layers = num_layers

        # 实例化预定义的RNN模型
        self.rnn = nn.RNN(input_size, hidden_size, num_layers)
        # 实例化nn.Linear，这个线性层用于将nn.RNN的输出维度转换为指定的输出维度
        self.linear = nn.Linear(hidden_size, output_size)
        # 实例化nn.Softmax层，用于从输出层中获取类别结果
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, input, hidden):
        """
        - input代表输入张量，形状是1 x n_letters
        - hidden代表RNN的隐层张量，形状是self.num_layers x 1 x self.hidden_size
        """
        # 预定义的nn.RNN要求输入维度是三维张量，所以使用unsqueeze(0)在第一维度扩展一个维度
        input = input.unsqueeze(0)
        # 将input和hidden输入到传统RNN的实例对象中，如果num_layers=1, rr恒等于hn
        rr, hn = self.rnn(input, hidden)
        # 将从RNN中获得的结果给通过线性变换和softmax返回，同时返回hn作为后续RNN的输入
        return self.softmax(self.linear(rr)), hn

    def initHidden(self):
        """初始化隐层张量"""
        # 初始化一个(self.num_layers, 1, self.hidden_size)形状的0张量
        return torch.zeros(self.num_layers, 1, self.hidden_size)


class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=1):
        super(LSTM, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.num_layers = num_layers

        # 实例化预定义的RNN模型
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers)
        # 实例化nn.Linear，这个线性层用于将nn.RNN的输出维度转换为指定的输出维度
        self.linear = nn.Linear(hidden_size, output_size)
        # 实例化nn.Softmax层，用于从输出层中获取类别结果
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, input, hidden, c):
        """
        - input代表输入张量，形状是1 x n_letters
        - hidden代表RNN的隐层张量，形状是self.num_layers x 1 x self.hidden_size
        """
        # 预定义的nn.RNN要求输入维度是三维张量，所以使用unsqueeze(0)在第一维度扩展一个维度
        input = input.unsqueeze(0)
        # 将input和hidden输入到传统RNN的实例对象中，如果num_layers=1, rr恒等于hn
        rr, (hn, cn) = self.lstm(input, (hidden, c))
        # 将从RNN中获得的结果给通过线性变换和softmax返回，同时返回hn作为后续RNN的输入
        return self.softmax(self.linear(rr)), hn, cn

    def initHiddenAndC(self):
        """初始化隐层张量"""
        c = hidden = torch.zeros(self.num_layers, 1, self.hidden_size)
        return hidden, c


class GRU(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=1):
        super(GRU, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.num_layers = num_layers

        # 实例化预定义的RNN模型
        self.gru = nn.GRU(input_size, hidden_size, num_layers)
        # 实例化nn.Linear，这个线性层用于将nn.RNN的输出维度转换为指定的输出维度
        self.linear = nn.Linear(hidden_size, output_size)
        # 实例化nn.Softmax层，用于从输出层中获取类别结果
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, input, hidden):
        """
        - input代表输入张量，形状是1 x n_letters
        - hidden代表RNN的隐层张量，形状是self.num_layers x 1 x self.hidden_size
        """
        # 预定义的nn.RNN要求输入维度是三维张量，所以使用unsqueeze(0)在第一维度扩展一个维度
        input = input.unsqueeze(0)
        # 将input和hidden输入到传统RNN的实例对象中，如果num_layers=1, rr恒等于hn
        rr, hn = self.gru(input, hidden)
        # 将从RNN中获得的结果给通过线性变换和softmax返回，同时返回hn作为后续RNN的输入
        return self.softmax(self.linear(rr)), hn

    def initHidden(self):
        """初始化隐层张量"""
        return torch.zeros(self.num_layers, 1, self.hidden_size)