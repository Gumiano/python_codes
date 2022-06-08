import torch.nn.functional

from models import *
from utils import *
from preprocessing import n_letters, n_categories

# 模型参数
input_size = n_letters
n_hidden = 128
output_size = n_categories

# 评价函数
criterion = nn.NLLLoss()

# 模型参数
input_size = n_letters
n_hidden = 128
output_size = n_categories
def train_models(is_clip=False):
    """统一训练多个循环神经网络模型。

    Args:
        is_clip: 是否采用梯度裁剪。
    """
    # 训练参数
    lr = 0.001  # 学习率
    n_epochs = 100000  # 训练100000次（可重复的从数据集中抽取100000姓名）
    print_every = 1000  # 每训练1000次，打印一次
    plot_every = 1000  # 每训练1000次，计算一次训练平均误差

    model_names = ['rnn', 'lstm', 'gru']
    models = [RNN(input_size, n_hidden, output_size),  # RNN
              LSTM(input_size, n_hidden, output_size),  # LSTM
              GRU(input_size, n_hidden, output_size),  # GRU
              ]

    # 统一添加优化器
    n_models = len(model_names)
    losses = [[] for _ in range(n_models)]
    periods = [[] for _ in range(n_models)]
    accuracies = [[] for _ in range(n_models)]
    train_accuracies = [[] for _ in range(n_models)]
    for idx, model in enumerate(models):
        optimizer = torch.optim.Adam(params=model.parameters(), lr=lr)
        # 记录每个模型的训练误差
        current_loss = 0  # 初始误差为0
        current_accuracy = 0 # 初始准确率为0
        all_losses = []  # 记录平均误差
        all_accuracies = [] # 记录平均准确率

        # 训练开始时间点
        start = time.time()
        for epoch in range(1, n_epochs + 1):
            # 随机的获取训练数据name和对应的language
            # TODO: 把训练集想办法改到前面
            category, name, category_tensor, name_tensor = make_random_sample()
            output, loss = model.step_one_epoch(optimizer, category_tensor, name_tensor, clip=is_clip)
            # 累加损失值
            current_loss += loss

            guess, guess_i = get_best_category_from_output(output)
            is_correct = (guess == category)
            current_accuracy += int(is_correct)

            # 每训练print_every次，预测一个姓名，并打印预测情况
            if epoch % print_every == 0:
                correct = '✓' if is_correct else '✗ (%s)' % category
                print('%d %d%% (%s) %.4f %s / %s %s mean accuracy:%.4f' % (
                    # epoch, epoch / n_epochs * 100, time_since(start), loss, name, guess, correct, ))
                    epoch, epoch / n_epochs * 100, time_since(start), current_loss/print_every, name, guess, correct, current_accuracy/print_every))

            # 每训练plot_every次，计算一个训练平均误差和一个准确率，方便后面可视化误差曲线图
            if epoch % plot_every == 0:
                all_losses.append(current_loss / plot_every)
                all_accuracies.append(current_accuracy / plot_every)
                # 重置
                current_loss = 0
                current_accuracy = 0

        # 保存模型参数
        if is_clip:
            torch.save(model.state_dict(), f'./model/model_{model_names[idx]}_with_clip.pth')
        else:
            torch.save(model.state_dict(), f'./model/model_{model_names[idx]}_no_clip.pth')
        losses[idx].extend(all_losses)
        periods[idx] = int(time.time() - start)
        accuracies[idx].extend(all_accuracies)


    return losses, periods, accuracies