import random
import torch
import time
import math
import torch.nn as nn
import yaml

from preprocessing import all_categories, category_names, name_to_tensor

def get_best_category_from_output(output):
    """获取输出中概率最大的类别。

    从模型输出张量中获取top 1个值和索引，并根据索引查找对应类别，

    Returns:
        (概率最大的类别, 该类别在all_categories类别向量表中的索引)，例如：

        (French, 5)
    """
    # 从输出张量中返回最大的值和索引对象
    top_n, top_i = output.topk(1)
    # 从top_i对象中取出索引的值
    category_i = top_i[0].item()
    # 根据索引值获取对应语言类别，返回语言类别和索引值
    # print('(all_categories[category_i], category_i):', all_categories[category_i], category_i)
    return all_categories[category_i], category_i

def make_random_sample():
    """返回随机抽取的训练样本。

    Returns:
        一个代表了类别数据，人名，one-hot处理后的类别张量，one-hot处理后的人名张量的元组，对应了一个随机产生的样本。
    """
    # 随机选取一个类别
    category = random.choice(all_categories)
    # 获取category类别对应的名字列表，并随机选取一个名字
    line = random.choice(category_names[category])
    # 将这个类别在类别列表中的索引封装成tensor，得到类别张量
    category_tensor = torch.tensor([all_categories.index(category)], dtype=torch.long)
    # 最后，将随机取到的名字通过函数lineToTensor转换为one-hot张量表示
    line_tensor = name_to_tensor(line)
    return category, line, category_tensor, line_tensor

def grad_clipping(net, theta):
    """对刚刚更新的梯度进行梯度裁剪。

    计算公式为 g <- min (1, theta/||g||) * g，该函数应该放在损失函数backward()之后，优化器step()之前

    Args:
        net: 神经网络模型
        theta: 梯度裁剪公式所需的超参数
    """
    if isinstance(net, nn.Module):
        params = [p for p in net.parameters() if p.requires_grad]
    else:
        params = net.params

    norm = torch.sqrt(sum(torch.sum((p.grad ** 2)) for p in params))
    if norm > theta:
        for param in params:
            param.grad[:] *= theta / norm

def time_since(since):
    """计算训练使用的时间。"""
    now = time.time()
    s = now - since
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)

def create_test_data(num_samples):
    """创建随机产生的测试集。"""
    categories = []
    names = []
    for i in range(num_samples):
        category, name, category_tensor, name_tensor = make_random_sample()
        categories.append(category)
        names.append(name)
    return categories, names

def load_config(config_file_path):
    """加载神经网络配置文件。

    文件为.yaml格式，使用PyYaml模块加载配置文件。

    Returns:
        神经网络训练所需要的参数。
    """
    # 模型参数
    input_size = n_letters
    n_hidden = 128
    output_size = n_categories

    # 评价函数
    criterion = nn.CrossEntropyLoss()

    # 训练参数
    lr = 0.001  # 学习率
    n_epochs = 100000  # 训练100000次（可重复的从数据集中抽取100000姓名）
    print_every = 1000  # 每训练5000次，打印一次
    plot_every = 1000  # 每训练1000次，计算一次训练平均误差

