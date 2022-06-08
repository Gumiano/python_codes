import torch.nn.functional

from models import *
from utils import *
from preprocessing import n_letters, n_categories

# 模型参数
input_size = n_letters
n_hidden = 128
output_size = n_categories

# 评价函数
# criterion = nn.CrossEntropyLoss()
criterion = nn.NLLLoss()

# 训练参数
lr = 0.001  # 学习率
n_epochs = 100000  # 训练100000次（可重复的从数据集中抽取100000姓名）
print_every = 1000  # 每训练5000次，打印一次
plot_every = 1000  # 每训练1000次，计算一次训练平均误差


def rnn_step_one_epoch(rnn, optimizer, category_tensor, name_tensor, clip=False):
    """使用RNN模型训练一个Epoch的数据。

    这里的一个Epoch对应了一个名字。
    损失函数的计算方法是：Softmax -> Log -> NLLLoss，可以直接采用log_softmax函数，但先softmax是为了后面方便直接得到概率。

    Args:
        rnn: RNN模型。
        optimizer: 损失函数优化器。
        category_tensor: 名字对应的类别索引所构成的张量。
        name_tensor: one-hot后的名字张量。
        clip: 是否进行梯度裁剪。
    """
    # print('category_tensor.size():', category_tensor.size())
    # print('name_tensor.size():', name_tensor.size())

    # 梯度清零
    rnn.zero_grad()
    # hidden_layer的初始参数
    hidden = rnn.init_hidden()
    # 对每个字母进行学习，time_step=1，将最终的hidden参数传递给下一个名字进行训练，将最终的output作为对该名字在训练阶段的预测结果。
    for i in range(name_tensor.size()[0]):
        # output是由每个类别上的预测概率构成的三维张量(Softmax)
        output, hidden = rnn(name_tensor[i], hidden)

    # 计算log_softmax
    output = torch.log(output)

    # 计算训练误差
    loss = criterion(output.squeeze(0), category_tensor)  # 自动对类别进行one-hot

    # 反向传播，更新梯度
    loss.backward()

    # 梯度裁剪, theta=1
    if clip:
        grad_clipping(rnn, 1)

    # 调整网络参数。
    optimizer.step()

    # 返回预测结果和训练误差
    return output, loss.item()


def train_rnn(is_clip=False):
    rnn = RNN(input_size, n_hidden, output_size)
    optimizer = torch.optim.Adam(params=rnn.parameters(), lr=lr)
    # 误差
    current_loss = 0  # 初始误差为0
    all_losses = []  # 记录平均误差

    # 训练开始时间点
    start = time.time()
    for epoch in range(1, n_epochs + 1):
        # 随机的获取训练数据name和对应的language
        category, name, category_tensor, name_tensor = make_random_sample()
        output, loss = rnn_step_one_epoch(rnn, optimizer, category_tensor, name_tensor, clip=is_clip)
        current_loss += loss

        # 每训练5000次，预测一个姓名，并打印预测情况
        if epoch % print_every == 0:
            guess, guess_i = get_best_category_from_output(output)
            correct = '✓' if guess == category else '✗ (%s)' % category
            print('%d %d%% (%s) %.4f %s / %s %s' % (
            epoch, epoch / n_epochs * 100, time_since(start), loss, name, guess, correct))

        # 每训练5000次，计算一个训练平均误差，方便后面可视化误差曲线图
        if epoch % plot_every == 0:
            all_losses.append(current_loss / plot_every)
            current_loss = 0

    # 保存模型参数
    if is_clip:
        torch.save(rnn.state_dict(), './model/model_rnn_with_clip.pth')
    else:
        torch.save(rnn.state_dict(), './model/model_rnn_no_clip.pth')

    return all_losses, int(time.time() - start)


def lstm_step_one_epoch(lstm: LSTM, optimizer, category_tensor, name_tensor, clip=False):
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
    lstm.zero_grad()
    # hidden_layer的初始参数
    hidden, c = lstm.init_hidden_and_c()
    # 对每个字母进行学习，time_step=1，将最终的hidden参数传递给下一个名字进行训练，将最终的output作为对该名字在训练阶段的预测结果。
    for i in range(name_tensor.size()[0]):
        # output是由每个类别上的预测概率构成的三维张量(Softmax)
        output, hidden, c = lstm(name_tensor[i], hidden, c)

    # 计算log_softmax
    output = torch.log(output)

    # 计算训练误差
    loss = criterion(output.squeeze(0), category_tensor)  # 自动对类别进行one-hot

    # 反向传播，更新梯度
    loss.backward()

    # 梯度裁剪, theta=1
    if clip:
        grad_clipping(lstm, 1)

    # 调整网络参数。
    optimizer.step()

    # 返回预测结果和训练误差
    return output, loss.item()


def gru_step_one_epoch(gru, optimizer, category_tensor, name_tensor, clip=False):
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
    gru.zero_grad()
    # hidden_layer的初始参数
    hidden = gru.init_hidden()
    # 对每个字母进行学习，time_step=1，将最终的hidden参数传递给下一个名字进行训练，将最终的output作为对该名字在训练阶段的预测结果。
    for i in range(name_tensor.size()[0]):
        # output是由每个类别上的预测概率构成的三维张量(Softmax)
        output, hidden = gru(name_tensor[i], hidden)

    # 计算log_softmax
    output = torch.log(output)

    # 计算训练误差
    loss = criterion(output.squeeze(0), category_tensor)  # 自动对类别进行one-hot

    # 反向传播，更新梯度
    loss.backward()

    # 梯度裁剪, theta=1
    if clip:
        grad_clipping(gru, 1)

    # 调整网络参数。
    optimizer.step()

    # 返回预测结果和训练误差
    return output, loss.item()


def train_lstm(is_clip=False):
    lstm = LSTM(input_size, n_hidden, output_size)
    optimizer = torch.optim.Adam(params=lstm.parameters(), lr=lr)
    # 误差
    current_loss = 0  # 初始误差为0
    all_losses = []  # 记录平均误差

    # 训练开始时间点
    start = time.time()

    for epoch in range(1, n_epochs + 1):
        # 随机的获取训练数据name和对应的language
        category, name, category_tensor, name_tensor = make_random_sample()
        output, loss = lstm_step_one_epoch(lstm, optimizer, category_tensor, name_tensor, clip=is_clip)
        current_loss += loss

        # 每训练5000次，预测一个姓名，并打印预测情况
        if epoch % print_every == 0:
            guess, guess_i = get_best_category_from_output(output)
            correct = '✓' if guess == category else '✗ (%s)' % category
            print('%d %d%% (%s) %.4f %s / %s %s' % (
            epoch, epoch / n_epochs * 100, time_since(start), loss, name, guess, correct))

        # 每训练5000次，计算一个训练平均误差，方便后面可视化误差曲线图
        if epoch % plot_every == 0:
            all_losses.append(current_loss / plot_every)
            current_loss = 0

    # 保存模型参数
    if is_clip:
        torch.save(lstm.state_dict(), './model/model_lstm_with_clip.pth')
    else:
        torch.save(lstm.state_dict(), './model/model_lstm_no_clip.pth')
    return all_losses, int(time.time() - start)


def train_gru(is_clip=False):
    gru = GRU(input_size, n_hidden, output_size)
    optimizer = torch.optim.Adam(params=gru.parameters(), lr=lr)
    # 误差
    current_loss = 0  # 初始误差为0
    all_losses = []  # 记录平均误差

    # 训练开始时间点
    start = time.time()

    for epoch in range(1, n_epochs + 1):
        # 随机的获取训练数据name和对应的language
        category, name, category_tensor, name_tensor = make_random_sample()
        output, loss = gru_step_one_epoch(gru, optimizer, category_tensor, name_tensor, clip=is_clip)
        current_loss += loss

        # 每训练5000次，预测一个姓名，并打印预测情况
        if epoch % print_every == 0:
            guess, guess_i = get_best_category_from_output(output)
            correct = '✓' if guess == category else '✗ (%s)' % category
            print('%d %d%% (%s) %.4f %s / %s %s' % (
            epoch, epoch / n_epochs * 100, time_since(start), loss, name, guess, correct))

        # 每训练5000次，计算一个训练平均误差，方便后面可视化误差曲线图
        if epoch % plot_every == 0:
            all_losses.append(current_loss / plot_every)
            current_loss = 0

    # 保存模型参数
    if is_clip:
        torch.save(gru.state_dict(), './model/model_gru_with_clip.pth')
    else:
        torch.save(gru.state_dict(), './model/model_gru_no_clip.pth')
    return all_losses, int(time.time() - start)