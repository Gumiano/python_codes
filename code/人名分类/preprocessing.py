from io import open
import string
import os
import unicodedata
import torch
import glob

# 获取常用ascii字符和标点
all_letters = string.ascii_letters + " .,;'"

# 获取常用字符数量
n_letters = len(all_letters)


def unicode_to_ascii(s):
    """将Unicode字符转换为Ascii字符。"""
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn' and c in all_letters
    )


# 从持久化文件中读取内容到内存中
data_path = './data/names/'


def read_lines(filename):
    """从文件中读取每一行，并将其转换为Ascii编码。"""
    # 打开指定文件并读取所有内容，使用strip()去除两侧空白符，然后以'\n'进行切分
    lines = open(filename, encoding='utf-8').read().strip().split('\n')
    # 对每一个lines列表中的名字转换为Ascii编码进行规范化，最后返回一个名字列表
    return [unicode_to_ascii(line) for line in lines]


# 构建类别与人名列表一一对应的字典
# 形如：{"English": ["Lily", "Susan", "Kobe"], ...}
category_names = {}

# all_categories形如：{"English", ..., "Chinese"}
all_categories = []

# 使用glob可以使用正则表达式匹配文件
for filename in glob.glob(data_path + "*.txt"):
    # 获取所有可能的类别
    category = os.path.splitext(os.path.basename(filename))[0]  # 获取每个文件的文件名，也就是每个名字对应的类别名
    all_categories.append(category)  # 追加到all_categories列表中
    # 获取每个类别文件所有的名字列表
    names = read_lines(filename)
    category_names[category] = names  # 按照对应的类别，将名字列表写入category_lines字典中

# 所有可能的类别数
n_categories = len(all_categories)


# 将人名转换为对应的one-hot张量表示
def name_to_tensor(name):
    """将人名转换为张量。

    先创建zeros张量，形状为(len(name), 1, n_letters)，然后将人名中的每个字符依照all_letters进行one-hot编码。

    Returns:
        one-hot后的人名张量。
    """
    tensor = torch.zeros(len(name), 1, n_letters)
    # 遍历人名中的每个字符
    for idx, letter in enumerate(name):
        # 使用find()找到每个字符在all_letters中的索引
        tensor[idx][0][all_letters.find(letter)] = 1
    return tensor