import torch
import torch.nn as nn
from utils import grad_clipping

class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=1, criterion=nn.NLLLoss()):
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

        # 其他参数
        self.criterion = criterion

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

    def init_hidden(self):
        """初始化隐层张量。

        Returns:
            隐层张量
        """
        return torch.zeros(self.num_layers, 1, self.hidden_size)

    def __predict_one_name(self, name_tensor):
        """预测并返回一个名字所属的语言类别。

        Args:
            name_tensor: 用于预测的one-hot后的名字张量。
        Returns:
            名字所属的类别张量。
        """
        # 初始化一个隐含层张量
        hidden = self.init_hidden()
        # 将评估数据的每个字符逐个传入RNN中
        for i in range(name_tensor.size()[0]):
            output, hidden = self(name_tensor[i], hidden)
        # 返回整个RNN的输出output
        return output.squeeze(0)

    def step_one_epoch(self, optimizer, category_tensor, name_tensor, clip=False):
        """使用RNN模型训练一个Epoch的数据。

        这里的一个Epoch对应了一个名字。
        损失函数的计算方法是：Softmax -> Log -> NLLLoss，可以直接采用log_softmax函数，但先softmax是为了后面方便直接得到概率。

        Args:
            optimizer: 损失函数优化器。
            category_tensor: 名字对应的类别索引所构成的张量。
            name_tensor: one-hot后的名字张量。
            clip: 是否进行梯度裁剪。
        """
        # print('category_tensor.size():', category_tensor.size())
        # print('name_tensor.size():', name_tensor.size())

        # 梯度清零
        self.zero_grad()
        # hidden_layer的初始参数
        hidden = self.init_hidden()
        # 对每个字母进行学习，time_step=1，将最终的hidden参数传递给下一个名字进行训练，将最终的output作为对该名字在训练阶段的预测结果。
        for i in range(name_tensor.size()[0]):
            # output是由每个类别上的预测概率构成的三维张量(Softmax)
            output, hidden = self(name_tensor[i], hidden)

        # 计算log_softmax
        output = torch.log(output)

        # 计算训练误差
        loss = self.criterion(output.squeeze(0), category_tensor)  # 自动对类别进行one-hot

        # 反向传播，更新梯度
        loss.backward()

        # 梯度裁剪, theta=1
        if clip:
            grad_clipping(self, 1)

        # 调整网络参数。
        optimizer.step()

        # 返回预测结果和训练误差
        return output, loss.item()


class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=1, criterion=nn.NLLLoss()):
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
        # 损失函数
        self.criterion = criterion

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

    def init_hidden_and_c(self):
        """初始化隐层张量和细胞状态。

        对于LSTM模型，隐层张量和细胞状态张量的形状相同。

        Returns:
            隐层张量, 细胞状态张量
        """
        c = hidden = torch.zeros(self.num_layers, 1, self.hidden_size)
        return hidden, c

    def __predict_one_name(self, name_tensor):
        """预测并返回一个名字所属的语言类别。

        Args:
            name_tensor: 用于预测的one-hot后的名字张量。
        Returns:
            名字所属的类别张量。
        """
        # 初始化一个隐含层张量
        hidden, c = self.init_hidden_and_c()
        # 将评估数据的每个字符逐个传入RNN中
        for i in range(name_tensor.size()[0]):
            output, hidden, c = self(name_tensor[i], hidden, c)
        # 返回整个RNN的输出output
        return output.squeeze(0)

    def step_one_epoch(self, optimizer, category_tensor, name_tensor, clip=False):
        """使用LSTM模型训练一个Epoch的数据。

        这里的一个Epoch对应了一个名字。
        损失函数的计算方法是：Softmax -> Log -> NLLLoss，可以直接采用log_softmax函数，但先softmax是为了后面方便直接得到概率。

        Args:
            optimizer: 损失函数优化器。
            category_tensor: 名字对应的类别索引所构成的张量。
            name_tensor: one-hot后的名字张量。
            clip: 是否进行梯度裁剪。
        """
        # 梯度清零
        self.zero_grad()
        # hidden_layer的初始参数
        hidden, c = self.init_hidden_and_c()
        # 对每个字母进行学习，time_step=1，将最终的hidden参数传递给下一个名字进行训练，将最终的output作为对该名字在训练阶段的预测结果。
        for i in range(name_tensor.size()[0]):
            # output是由每个类别上的预测概率构成的三维张量(Softmax)
            output, hidden, c = self(name_tensor[i], hidden, c)

        # 计算log_softmax
        output = torch.log(output)

        # 计算训练误差
        loss = self.criterion(output.squeeze(0), category_tensor)  # 自动对类别进行one-hot

        # 反向传播，更新梯度
        loss.backward()

        # 梯度裁剪, theta=1
        if clip:
            grad_clipping(self, 1)

        # 调整网络参数。
        optimizer.step()

        # 返回预测结果和训练误差
        return output, loss.item()

class GRU(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=1, criterion=nn.NLLLoss()):
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

        # 损失函数
        self.criterion = criterion

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

    def init_hidden(self):
        """初始化隐层张量。

        Returns:
            隐层张量
        """
        return torch.zeros(self.num_layers, 1, self.hidden_size)

    def __predict_one_name(self, name_tensor):
        """预测并返回一个名字所属的语言类别。

        Args:
            name_tensor: 用于预测的one-hot后的名字张量。
        Returns:
            名字所属的类别张量。
        """
        # 初始化一个隐含层张量
        hidden = self.init_hidden()
        # 将评估数据的每个字符逐个传入RNN中
        for i in range(name_tensor.size()[0]):
            output, hidden = self(name_tensor[i], hidden)
        # 返回整个RNN的输出output
        return output.squeeze(0)

    def step_one_epoch(self, optimizer, category_tensor, name_tensor, clip=False):
        """使用GRU模型训练一个Epoch的数据。

        这里的一个Epoch对应了一个名字。
        损失函数的计算方法是：Softmax -> Log -> NLLLoss，可以直接采用log_softmax函数，但先softmax是为了后面方便直接得到概率。

        Args:
            optimizer: 损失函数优化器。
            category_tensor: 名字对应的类别索引所构成的张量。
            name_tensor: one-hot后的名字张量。
            clip: 是否进行梯度裁剪。
        """
        # 梯度清零
        self.zero_grad()
        # hidden_layer的初始参数
        hidden = self.init_hidden()
        # 对每个字母进行学习，time_step=1，将最终的hidden参数传递给下一个名字进行训练，将最终的output作为对该名字在训练阶段的预测结果。
        for i in range(name_tensor.size()[0]):
            # output是由每个类别上的预测概率构成的三维张量(Softmax)
            output, hidden = self(name_tensor[i], hidden)

        # 计算log_softmax
        output = torch.log(output)

        # 计算训练误差
        loss = self.criterion(output.squeeze(0), category_tensor)  # 自动对类别进行one-hot

        # 反向传播，更新梯度
        loss.backward()

        # 梯度裁剪, theta=1
        if clip:
            grad_clipping(self, 1)

        # 调整网络参数。
        optimizer.step()

        # 返回预测结果和训练误差
        return output, loss.item()
