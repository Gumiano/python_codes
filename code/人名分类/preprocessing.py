from io import open
import string
import os
import unicodedata
import torch
import glob

# 获取常用字母和标点
all_letters = string.ascii_letters + " .,;'"
# 获取常用字符数量
n_letters = len(all_letters)
print('n_letter:', n_letters)

def unicodeToAscii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn' and c in all_letters
    )

# 从持久化文件中读取内容到内存中
data_path = './data/names/'

def readLines(filename):
    """从文件中读取每一行加载到内存中形成列表"""
    # 打开指定文件并读取所有内容，使用strip()去除两侧空白符，然后以'\n'进行切分
    lines = open(filename, encoding='utf-8').read().strip().split('\n')
    # 对每一个lines列表中的名字转换为Ascii编码进行规范化，最后返回一个名字列表
    return [unicodeToAscii(line) for line in lines]
# 构建类别与人名列表一一对应的字典
# category_lines形如：{"English": ["Lily", "Susan", "Kobe"], ...}
category_lines = {}

# all_categories形如：{"English", ..., "Chinese"}
all_categories = []

# 使用glob可以使用正则表达式匹配文件
for filename in glob.glob(data_path + "*.txt"):
    # 获取每个文件的文件名，也就是每个名字对应的类别名
    category = os.path.splitext(os.path.basename(filename))[0]
    # 追加到all_categories列表中
    all_categories.append(category)
    # 读取每个文件的内容，形成名字列表
    lines = readLines(filename)
    # 按照对应的类别，将名字列表写入category_lines字典中
    category_lines[category] = lines

n_categories = len(all_categories)
print("n_categories:", n_categories)
print(category_lines['Italian'][:5])

# 将人名转换为对应的one-hot张量表示
def lineToTensor(line):
    """参数line是输入的人名"""
    # 初始化一个0张量，形状为（len(line), 1, n_letters)
    # 即人名中的每个字母用(1 x n_letters)的张量来表示
    tensor = torch.zeros(len(line), 1, n_letters)
    # 遍历人名中的每个字符
    for li, letter in enumerate(line):
        # 使用find()找到每个字符在all_letters中的索引
        tensor[li][0][all_letters.find(letter)] = 1
    return tensor