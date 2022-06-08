import random
import torch
import time
import math
import torch.nn as nn

from preprocessing import all_categories, category_lines, lineToTensor

def categoryFromOutput(output):
    """获取输出中概率最大的类别

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
    print('(all_categories[category_i], category_i):', all_categories[category_i], category_i)
    return all_categories[category_i], category_i

def randomTrainingExample():
    """返回随机抽取的训练样本。"""
    # 首先使用random.choice随机选取一个类别
    category = random.choice(all_categories)
    # 然后通过category_lines字典获取category类别对应的名字列表，并随机选取一个名字
    line = random.choice(category_lines[category])
    # 将这个类别在类别列表中的索引封装成tensor，得到类别张量
    category_tensor = torch.tensor([all_categories.index(category)], dtype=torch.long)
    # 最后，将随机取到的名字通过函数lineToTensor转换为one-hot张量表示
    line_tensor = lineToTensor(line)
    return category, line, category_tensor, line_tensor

def grad_clipping(net, theta):
    """对刚刚更新的梯度进行梯度裁剪。

    公式为：g <- min (1, theta/||g||) * g
    该函数应该放在损失函数backward()之后，优化器step()之前
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
    #计算训练使用的时间
    now = time.time()
    s = now - since
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)

def create_test_data(num_samples):
    """ 测试时只需要category和line这两个数据 """
    categorys = []
    lines = []
    for i in range(num_samples):
        category, line, category_tensor, line_tensor = randomTrainingExample()
        categorys.append(category)
        lines.append(line)
    return categorys, lines


