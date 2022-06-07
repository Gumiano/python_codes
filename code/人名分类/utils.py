import random
import torch
import time
import math

import torch.nn as nn
from preprocessing import all_categories, category_lines, lineToTensor

def categoryFromOutput(output):
    """从输出结果中获得指定类别，参数为输出张量output"""
    # 从输出张量中返回最大的值和索引对象
    top_n, top_i = output.topk(1)
    # 从top_i对象中取出索引的值
    category_i = top_i[0].item()
    # 根据索引值获取对应语言类别，返回语言类别和索引值
    return all_categories[category_i], category_i

def randomTrainingExample():
    """用于随机产生训练数据"""
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
    """
    梯度裁剪
    g <- min (1, theta/||g||) * g
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


