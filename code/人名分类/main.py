import itertools
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, roc_curve, auc, roc_auc_score
from sklearn.preprocessing import label_binarize

from train_models import *
from utils import *


def train_and_plot(is_clip=False):
    """训练并绘制损失函数和耗时曲线。

    Args:
        is_clip: 是否采用梯度裁剪。
    """
    losses, periods = train_models(is_clip=is_clip)
    # all_losses1, period1 = train_rnn(is_clip=is_clip)
    # all_losses2, period2 = train_lstm(is_clip=is_clip)
    # all_losses3, period3 = train_gru(is_clip=is_clip)
    # all_losses = [all_losses1, all_losses2, all_losses3]
    # periods = [period1, period2, period3]
    model_names = ['RNN', 'LSTM', 'GRU']

    # 绘制损失函数曲线
    plot_loss_curve(losses, model_names, with_clip=is_clip)
    # 绘制训练耗时柱状图
    plot_bar(model_names, periods, with_clip=is_clip)


def plot_loss_curve(losses, model_names, with_clip=False):
    """绘制模型损失值随epoch变化的曲线。

    Args:
        losses: 一个列表，每个元素对应了一个模型的损失函数值列表。
        model_names：一个列表，每个元素对应了一个模型名称。
        with_clip: 是否采用了梯度裁剪，如果采用了，标题就是'loss curve with clipping'，否则就是'loss curve without clipping'。
    """
    for idx in range(len(losses)):
        plt.figure(0)
        plt.plot(losses[idx], label=model_names[idx])

    plt.legend(loc='upper left')

    # 标题设置
    if with_clip:
        plt.title('loss curve with clipping')
        plt.savefig('loss_curve_clip')
    else:
        plt.title('loss curve without clipping')
        plt.savefig('loss_curve_no_clip')

    plt.show()


def plot_total_roc_curve(y_test, y_probs: np.ndarray, model_names):
    """绘制总的ROC曲线"""
    for i in range(len(model_names)):
        y_one_hot = label_binarize(y_test, classes=all_categories)
        print('y_one_hot shape:', y_one_hot.shape)
        # 计算MICRO
        # print(roc_auc_score(y_one_hot, y_probs, average='micro'))
        # 计算假正率，真正率
        fpr, tpr, thresholds = roc_curve(y_one_hot.ravel(), y_probs[i].ravel())
        auc_score = auc(fpr, tpr)
        print('micro:', roc_auc_score(y_one_hot, y_probs[i], average='micro'))
        print('thresholds:', thresholds)
        print('auc:', auc_score)
        # 绘制图像，FPR就是横坐标,TPR就是纵坐标
        plt.plot(fpr, tpr, lw=2, alpha=0.7, label=f'{model_names[i]} AUC=%.3f' % auc_score)
        plt.plot((0, 1), (0, 1), lw=1, ls='--', alpha=0.7)
    plt.xlim((-0.01, 1.02))
    plt.ylim((-0.01, 1.02))
    plt.xticks(np.arange(0, 1.1, 0.1))
    plt.yticks(np.arange(0, 1.1, 0.1))
    plt.xlabel('False Positive Rate', fontsize=13)
    plt.ylabel('True Positive Rate', fontsize=13)
    plt.grid(visible=True, ls=':')
    plt.legend(loc='lower right', fancybox=True, framealpha=0.8, fontsize=12)
    plt.title(f'ROC曲线', fontsize=17)
    plt.savefig('all_roc_curve')
    plt.show()


def plot_single_class_roc_curve(y_test, y_probs: np.ndarray, model_name):
    """针对每个类别绘制ROC曲线"""
    pass


def plot_bar(x_data, y_data, with_clip=False):
    """
    绘制柱状图
    """
    plt.figure(1)
    # 绘制训练耗时对比柱状图
    if with_clip:
        plt.title('loss curve with clipping')
        plt.savefig('bar_plot_clip')
    else:
        plt.title('loss curve without clipping')
        plt.savefig('bar_plot_no_clip')
    plt.bar(range(len(x_data)), y_data, tick_label=x_data)
    plt.show()


def load_models():
    """预加载已训练好的模型参数，返回已经加载好参数的模型列表。
    """
    model_names = ['rnn', 'lstm', 'gru']
    models = [RNN(input_size, n_hidden, output_size),  # RNN
              LSTM(input_size, n_hidden, output_size),  # LSTM
              GRU(input_size, n_hidden, output_size),  # GRU
              ]

    for idx, model_name in enumerate(model_names):
        # 将已经训练好的参数从文件加载到模型中
        models[idx].load_state_dict(torch.load(f'./model/model_{model_name}.pth'))
    return models


def evaluateRNN(rnn, name_tensor):
    """返回模型对一个名字预测得到的类别张量。

    Args:
        rnn: RNN模型
        name_tensor: 用于预测的one-hot后的名字张量
    """
    # 初始化一个隐含层张量
    hidden = rnn.init_hidden()
    # 将评估数据的每个字符逐个传入RNN中
    for i in range(name_tensor.size()[0]):
        output, hidden = rnn(name_tensor[i], hidden)
    # 返回整个RNN的输出output
    return output.squeeze(0)


def evaluateLSTM(lstm, line_tensor):
    """line_tensor代表名字的张量表示"""
    # 初始化一个隐含层张量
    hidden, c = lstm.init_hidden_and_c()
    # 将评估数据的每个字符逐个传入RNN中
    for i in range(line_tensor.size()[0]):
        output, hidden, c = lstm(line_tensor[i], hidden, c)
    # 返回整个RNN的输出output
    return output.squeeze(0)


def evaluateGRU(gru, line_tensor):
    """line_tensor代表名字的张量表示"""
    # 初始化一个隐含层张量
    hidden = gru.init_hidden()
    # 将评估数据的每个字符逐个传入RNN中
    for i in range(line_tensor.size()[0]):
        output, hidden = gru(line_tensor[i], hidden)
    # 返回整个RNN的输出output
    return output.squeeze(0)


def predict(model, input_line, evaluate, n_predictions=3):
    """预测函数
        输入参数input_line代表输入的名字
        n_predictions代表最优可能的top_n个国家
    """
    # 首先打印输出
    # print('\n> %s' % input_line)

    # 在模型预测时不能够更新模型参数
    with torch.no_grad():
        # 将输入名字转换为张量表示，并获取预测输出
        output = evaluate(model, name_to_tensor(input_line))
        # 样本预测在各个类别的概率分布
        probs = output.squeeze().numpy()

        # 从预测的输出中取前3个最大的值以及索引
        topv, topi = output.topk(n_predictions, 1, True)
        # 创建存放结果的列表
        predictions = []
        # 遍历n_predictions
        for i in range(n_predictions):
            # 从topv中取出output值
            value = topv[0][i].item()
            # 取出索引并找到对应类别
            category_index = topi[0][i].item()
            # 打印ouptut值和对应的类别
            # print('(%.2f) %s' % (value, all_categories[category_index]))
            # 将结果装进predictions列表中
            predictions.append([value, all_categories[category_index]])

        return predictions, probs


def acc(test_y, pred_y):
    """计算模型在给定测试集上的准确率
    """
    num_samples = len(test_y)
    actual = np.array(test_y)
    pred = np.array(pred_y)

    accuracy = np.sum(actual == pred) / num_samples
    print('accuracy:', accuracy)
    return accuracy


def get_predictions(model, names, eval_func):
    """返回模型在测试集上的预测结果。

    Args:
        model:
        test_data 形如[categories, lines]
    """
    pred = []
    probs = []
    num_samples = len(names)
    for i in range(num_samples):
        out = predict(model, names[i], eval_func, n_predictions=1)
        probs.append(out[1])
        pred.append(out[0][0][1])
    return pred, probs


def plot_confusion_matrix(y_test, y_pred, model_name):
    """绘制单个模型预测结果的混淆矩阵。

    Args:
        y_test: 测试集标签。
        y_pred: 模型在测试集上的预测类别。
        model_name: 模型名称。
    """
    plt.figure(figsize=(16, 9), dpi=100)  # 调整图像大小
    cm = confusion_matrix(y_test, y_pred, labels=all_categories)
    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    np.set_printoptions(formatter={'float': '{: 0.2f}'.format})
    plt.imshow(cm, cmap=plt.cm.get_cmap('coolwarm'))
    plt.colorbar()
    plt.title(f'{model_name} Confusion Matrix')
    tick_marks = np.arange(len(all_categories))
    plt.xticks(tick_marks, all_categories, rotation=45)
    plt.yticks(tick_marks, all_categories)
    plt.ylim(len(all_categories) - 0.5, -0.5)
    fmt = '.2f'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black", fontsize=8)
    plt.tight_layout()
    plt.xlabel('True labels')
    plt.ylabel('Predicted labels')
    plt.savefig(f'./imgs/{model_name} Confusion Matrix')
    plt.show()


def eval_predictions(y_test, y_preds, y_probs, model_names):
    """对训练好的模型进行统一评估。

    Args:
        y_test: 测试集标签。
        y_preds: 一个列表，每个元素对应了一个模型在测试集上的预测类别。
        y_probs: 一个列表，每个元素对应了一个模型在测试集上每个样本在各个类别的预测概率
        model_names: 一个列表，每个元素对应了一个用到的模型名称
    """
    accuracies = []  # 记录每个模型的准确率
    # 对每个模型
    for i in range(len(model_names)):
        # 绘制混淆矩阵
        plot_confusion_matrix(y_test, y_preds[i], model_names[i])
        # 计算准确率
        accuracies.append(acc(y_test, y_preds[i]))
        # 绘制每个模型在每个类别上的ROC曲线
        # plot_roc_curve(y_test, y_probs[i], model_names[i])
    plot_total_roc_curve(y_test, y_probs, model_names)
    # 绘制准确率的对比柱状图
    plot_bar(model_names, accuracies)


def eval_models():
    # 模型名称
    model_names = ['RNN', 'LSTM', 'GRU']
    # 模型评价函数
    evaluate_functions = [evaluateRNN, evaluateLSTM, evaluateGRU]
    # 测试集样本数
    num_test_samples = 100
    # 随机创建测试集
    categories, lines = create_test_data(num_test_samples)
    # 对每个模型
    predictions = []
    probs = []
    for idx, model in enumerate(load_models()):
        # 得到评价函数
        eval_func = evaluate_functions[idx]
        # 获取预测结果
        prediction, prob = get_predictions(model, lines, eval_func)
        predictions.append(prediction)
        probs.append(np.array(prob))
    # 对预测结果进行评价
    eval_predictions(categories, predictions, probs, model_names)


if __name__ == '__main__':
    # eval_models()
    train_and_plot(is_clip=False)
