import sys,os,copy
import warnings

warnings.filterwarnings('ignore')
sys.path.append(os.getcwd())
os.environ["CUDA_VISIBLE_DEVICES"] = '0'

import numpy as np
import torch as th
import torch.nn.functional as F
from torch_scatter import scatter_mean
from torch_geometric.nn import GCNConv
from torch_geometric.data import DataLoader
from tqdm import tqdm
from Process.rand5fold import *
from tools.evaluate import *
from tools.earlystopping2class import EarlyStopping

# TDGCN
class TDrumorGCN(th.nn.Module):

    # 初始化函数 输入特征 隐藏特征 输出特征
    def __init__(self, in_feats, hid_feats, out_feats):
        super(TDrumorGCN, self).__init__()
        # 输入特征丨隐藏特征
        self.conv1 = GCNConv(in_feats, hid_feats)
        # 隐藏特征 + 输入特征丨输出特征
        self.conv2 = GCNConv(hid_feats + in_feats, out_feats)

    # 前向传播
    def forward(self, data):
        # 输入特征 x 丨 图的边索引 edge_index
        x, edge_index = data.x, data.edge_index
        # 输入特征 x 丨 转化为浮点型保存到 x1
        x1 = copy.copy(x.float())
        # 输入特征 x 丨 进行图卷积
        x = self.conv1(x, edge_index)
        # 图卷积后的保存到 x2
        x2 = copy.copy(x)
        # 根博客id root-id
        rootindex = data.rootindex
        # 保留根节点的特征信息
        root_extend = th.zeros(len(data.batch), x1.size(1)).to(device)
        batch_size = max(data.batch) + 1
        for num_batch in range(batch_size):
            index = (th.eq(data.batch, num_batch))
            root_extend[index] = x1[rootindex[num_batch]]
        # 特征维度连接 x + root_extend
        x = th.cat((x, root_extend), 1)
        # ReLU 激活函数
        x = F.relu(x)
        # 根据 self.training 来判断是否处于训练阶段
        x = F.dropout(x, training = self.training)
        # 输入特征 x 丨 进行图卷积
        x = self.conv2(x, edge_index)
        # ReLU 激活函数
        x=F.relu(x)
        root_extend = th.zeros(len(data.batch), x2.size(1)).to(device)
        for num_batch in range(batch_size):
            index = (th.eq(data.batch, num_batch))
            root_extend[index] = x2[rootindex[num_batch]]
        x = th.cat((x, root_extend), 1)
        # 特征汇聚 (mean)平均汇聚
        x = scatter_mean(x, data.batch, dim = 0)
        # 输出特征x
        return x

# BUGCN
class BUrumorGCN(th.nn.Module):

    # 初始化函数 输入特征 隐藏特征 输出特征
    def __init__(self, in_feats, hid_feats, out_feats):
        super(BUrumorGCN, self).__init__()
        # 输入特征丨隐藏特征
        self.conv1 = GCNConv(in_feats, hid_feats)
        # 隐藏特征 + 输入特征丨输出特征
        self.conv2 = GCNConv(hid_feats + in_feats, out_feats)

    # 前向传播
    def forward(self, data):
        # 输入特征 x 丨 图的边索引 BU_edge_index
        x, edge_index = data.x, data.BU_edge_index
        # 输入特征 x 丨 转化为浮点型保存到 x1
        x1 = copy.copy(x.float())
        # 输入特征 x 丨 进行图卷积
        x = self.conv1(x, edge_index)
        # 图卷积后的保存到 x2
        x2 = copy.copy(x)
        # 根博客id root-id
        rootindex = data.rootindex
        # 保留根节点的特征信息
        root_extend = th.zeros(len(data.batch), x1.size(1)).to(device)
        batch_size = max(data.batch) + 1
        for num_batch in range(batch_size):
            index = (th.eq(data.batch, num_batch))
            root_extend[index] = x1[rootindex[num_batch]]
        # 特征维度连接 x + root_extend
        x = th.cat((x, root_extend), 1)
        # ReLU 激活函数
        x = F.relu(x)
        # 根据 self.training 来判断是否处于训练阶段
        x = F.dropout(x, training=self.training)
        # 输入特征 x 丨 进行图卷积
        x = self.conv2(x, edge_index)
        # ReLU 激活函数
        x = F.relu(x)
        root_extend = th.zeros(len(data.batch), x2.size(1)).to(device)
        for num_batch in range(batch_size):
            index = (th.eq(data.batch, num_batch))
            root_extend[index] = x2[rootindex[num_batch]]
        x = th.cat((x, root_extend), 1)
        # 特征汇聚 (mean)平均汇聚
        x = scatter_mean(x, data.batch, dim = 0)
        # 输出特征x
        return x

# Module
class Net(th.nn.Module):
    # 初始化函数 输入特征 隐藏特征 输出特征
    def __init__(self, in_feats, hid_feats, out_feats):
        super(Net, self).__init__()
        # TDGCN Layer
        self.TDrumorGCN = TDrumorGCN(in_feats, hid_feats, out_feats)
        # BUGCN Layer
        self.BUrumorGCN = BUrumorGCN(in_feats, hid_feats, out_feats)
        # Linear Layer 输入维度: 输出特征+隐藏特征 输出维度: 2
        self.fc = th.nn.Linear((out_feats + hid_feats)*2, 2)

    def forward(self, data):
        TD_x = self.TDrumorGCN(data)
        BU_x = self.BUrumorGCN(data)
        # 特征维度连接
        x = th.cat((BU_x, TD_x), 1)
        # 线性层 线性变化
        x = self.fc(x)
        # Log Softmax 在维度1上进行对数Softmax
        x = F.log_softmax(x, dim = 1)
        return x

# Train 函数
def train_GCN(treeDic, x_test, x_train, TDdroprate, BUdroprate, lr, weight_decay, patience, n_epochs, batchsize, dataname, iter):
    # 定义一个模型 输入特征维度5000 隐藏特征维度64 输出特征维度64
    model = Net(5000,64,64).to(device)
    BU_params  = list(map(id, model.BUrumorGCN.conv1.parameters()))
    BU_params += list(map(id, model.BUrumorGCN.conv2.parameters()))

    # 去除BUGCN中Conv1/Conv2的参数
    base_params=filter(lambda p:id(p) not in BU_params, model.parameters())

    # 定义优化器 Adam 模型参数分为3组并指定不同的学习率
    optimizer = th.optim.Adam([
        {'params': base_params},
        {'params': model.BUrumorGCN.conv1.parameters(), 'lr': lr/5},
        {'params': model.BUrumorGCN.conv2.parameters(), 'lr': lr/5}
    ], lr=lr, weight_decay=weight_decay)

    # 模型 训练模式
    model.train()
    
    # 训练集损失Loss 验证集损失Loss 训练集准确率ACC 验证机准确率ACC
    train_losses, val_losses, train_accs, val_accs = [], [], [], []
    
    # 防止过拟合 
    # patience: 如果在'patience'轮内验证集性能没有提升, 就会触发早停机制。
    # verbose: 每次早停条件被满足时打印相关信息, 帮助了解早停的过程
    early_stopping = EarlyStopping(patience = patience, verbose = True)
    
    # 循环
    for epoch in range(n_epochs):
        # FIXME: loadBiData函数处理好的数据 需要观察如何加载的数据
        # 数据集名称如Twitter15/Twitter16 包含树形结构数据 训练数据 测试数据 TDGCN丢弃参数比例 BUGCN丢弃参数比例
        traindata_list, testdata_list = loadBiData(dataname, treeDic, x_train, x_test, TDdroprate, BUdroprate)
        
        # 构建数据加载器
        train_loader = DataLoader(traindata_list, batch_size=batchsize, shuffle=False, num_workers = 10)
        test_loader = DataLoader(testdata_list, batch_size=batchsize, shuffle=True, num_workers = 10)
        
        # AVG Loss/ACC 记录
        avg_loss, avg_acc = [], []
        batch_idx = 0

        # 循环进度条
        tqdm_train_loader = tqdm(train_loader)
        
        # 循环 划分小批量进行处理
        for Batch_data in tqdm_train_loader:
            Batch_data.to(device)
            # 获取模型输出
            out_labels = model(Batch_data)
            # 计算损失 NLL负对数似然损失
            loss = F.nll_loss(out_labels, Batch_data.y)
            # 梯度清空 反向传播
            optimizer.zero_grad()
            loss.backward()
            # 当前Loss放入AVG.Loss中用于后期计算
            avg_loss.append(loss.item())
            optimizer.step()
            # 得到神经网络对当前批次样本的预测结果
            _, pred = out_labels.max(dim = -1)
            # 计算模型在当前批次中预测正确的样本数量
            correct = pred.eq(Batch_data.y).sum().item()
            # 计算准确率
            train_acc = correct / len(Batch_data.y)
            # 当前ACC放入AVG.ACC中用于后期计算
            avg_acc.append(train_acc)
            # 输出信息
            postfix = "Iter {:03d} | Epoch {:05d} | Batch{:02d} | Train_Loss {:.4f}| Train_Accuracy {:.4f}".format(iter, epoch, batch_idx, loss.item(), train_acc)
            tqdm_train_loader.set_postfix_str(postfix)
            batch_idx = batch_idx + 1
        
        # 当前批次Loss/ACC存储用于后续Loss/ACC分析和可视化
        train_losses.append(np.mean(avg_loss))
        train_accs.append(np.mean(avg_acc))

        temp_val_losses, temp_val_accs, temp_val_Acc_all = [], [], [] 

        # ACC 准确率 丨 PREC 精确度 丨 REC 召回率 丨 F1 F1分数
        temp_val_Acc1, temp_val_Prec1, temp_val_Recll1, temp_val_F1 = [], [], [], []
        temp_val_Acc2, temp_val_Prec2, temp_val_Recll2, temp_val_F2 = [], [], [], []
        
        # 模型 评估模式
        model.eval()
        tqdm_test_loader = tqdm(test_loader)

        for Batch_data in tqdm_test_loader:
            Batch_data.to(device)
            val_out = model(Batch_data)
            val_loss = F.nll_loss(val_out, Batch_data.y)
            temp_val_losses.append(val_loss.item())
            _, val_pred = val_out.max(dim=1)
            correct = val_pred.eq(Batch_data.y).sum().item()
            val_acc = correct / len(Batch_data.y)

            Acc_all, \
            Acc1, Prec1, Recll1, F1, \
            Acc2, Prec2, Recll2, F2 = evaluationclass(val_pred, Batch_data.y)

            temp_val_Acc_all.append(Acc_all), \
            temp_val_Acc1.append(Acc1), temp_val_Prec1.append(Prec1), temp_val_Recll1.append(Recll1), temp_val_F1.append(F1), \
            temp_val_Acc2.append(Acc2), temp_val_Prec2.append(Prec2), temp_val_Recll2.append(Recll2), temp_val_F2.append(F2)
            temp_val_accs.append(val_acc)

        # 当前批次Loss/ACC存储用于后续Loss/ACC分析和可视化    
        val_losses.append(np.mean(temp_val_losses))
        val_accs.append(np.mean(temp_val_accs))

        print("Epoch {:05d} | Val_Loss {:.4f} | Val_Accuracy {:.4f}".format(epoch, np.mean(temp_val_losses), np.mean(temp_val_accs)))

        res = ['ACC: {:.4f}'.format(np.mean(temp_val_Acc_all)),
               'C1: {:.4f}, {:.4f}, {:.4f}, {:.4f}'.format(np.mean(temp_val_Acc1), np.mean(temp_val_Prec1), np.mean(temp_val_Recll1), np.mean(temp_val_F1)),
               'C2: {:.4f}, {:.4f}, {:.4f}, {:.4f}'.format(np.mean(temp_val_Acc2), np.mean(temp_val_Prec2), np.mean(temp_val_Recll2), np.mean(temp_val_F2))]
        
        print('Results:', res)
        
        early_stopping(np.mean(temp_val_losses), 
                       np.mean(temp_val_Acc_all), 
                       np.mean(temp_val_Acc1),
                       np.mean(temp_val_Acc2), 
                       np.mean(temp_val_Prec1),
                       np.mean(temp_val_Prec2), 
                       np.mean(temp_val_Recll1), 
                       np.mean(temp_val_Recll2),
                       np.mean(temp_val_F1),
                       np.mean(temp_val_F2), model, 'BiGCN', "weibo")
        
        accs = np.mean(temp_val_Acc_all)
        
        acc1 = np.mean(temp_val_Acc1)
        acc2 = np.mean(temp_val_Acc2)
        
        pre1 = np.mean(temp_val_Prec1)
        pre2 = np.mean(temp_val_Prec2)
        
        rec1 = np.mean(temp_val_Recll1)
        rec2 = np.mean(temp_val_Recll2)
        
        F1 = np.mean(temp_val_F1)
        F2 = np.mean(temp_val_F2)
        
        if early_stopping.early_stop:
            print("Early Stopping")
            accs = early_stopping.accs
            acc1 = early_stopping.acc1
            acc2 = early_stopping.acc2
            pre1 = early_stopping.pre1
            pre2 = early_stopping.pre2
            rec1 = early_stopping.rec1
            rec2 = early_stopping.rec2
            F1 = early_stopping.F1
            F2 = early_stopping.F2
            break

    return train_losses, val_losses, train_accs, val_accs, accs, acc1, pre1, rec1, F1, acc2, pre2, rec2, F2

if __name__ == '__main__':
    
    # 学习率
    lr=5e-4
    
    # 权重衰减
    weight_decay=1e-4
    patience=10

    # Epoch
    n_epochs=200
    batchsize=16

    tddroprate=0
    budroprate=0

    datasetname="Weibo"
    iterations=int(100)
    
    model = "BiGCN"
    device = th.device('cuda:0' if th.cuda.is_available() else 'cpu')

    test_accs, F1, F2 = [], [], []
    ACC1, PRE1, REC1 = [], [], []
    ACC2, PRE2, REC2 = [], [], []

    for iter in range(iterations):
        # 五折交叉验证
        fold0_x_test, fold0_x_train, \
        fold1_x_test, fold1_x_train, \
        fold2_x_test, fold2_x_train, \
        fold3_x_test, fold3_x_train, \
        fold4_x_test, fold4_x_train = load5foldData(datasetname)
        
        # 加载了数据集的树状结构信息
        treeDic=loadTree(datasetname)

        train_losses, val_losses, train_accs, val_accs, accs_0, acc1_0, pre1_0, rec1_0, F1_0, acc2_0, pre2_0, rec2_0, F2_0 \
            = train_GCN(treeDic, fold0_x_test, fold0_x_train, tddroprate, budroprate, lr, weight_decay, patience, n_epochs, batchsize, datasetname, iter)
        
        train_losses, val_losses, train_accs, val_accs, accs_1, acc1_1, pre1_1, rec1_1, F1_1, acc2_1, pre2_1, rec2_1, F2_1 \
            = train_GCN(treeDic, fold1_x_test, fold1_x_train, tddroprate, budroprate, lr, weight_decay, patience, n_epochs, batchsize, datasetname, iter)
        
        train_losses, val_losses, train_accs, val_accs, accs_2, acc1_2, pre1_2, rec1_2, F1_2, acc2_2, pre2_2, rec2_2, F2_2 \
            = train_GCN(treeDic, fold2_x_test, fold2_x_train, tddroprate, budroprate, lr, weight_decay, patience, n_epochs, batchsize, datasetname, iter)
        
        train_losses, val_losses, train_accs, val_accs, accs_3, acc1_3, pre1_3, rec1_3, F1_3, acc2_3, pre2_3, rec2_3, F2_3 \
            = train_GCN(treeDic, fold3_x_test, fold3_x_train, tddroprate, budroprate, lr, weight_decay, patience, n_epochs, batchsize, datasetname, iter)
        
        train_losses, val_losses, train_accs, val_accs, accs_4, acc1_4, pre1_4, rec1_4, F1_4, acc2_4, pre2_4, rec2_4, F2_4 \
            = train_GCN(treeDic, fold4_x_test, fold4_x_train, tddroprate, budroprate, lr, weight_decay, patience, n_epochs, batchsize, datasetname, iter)
        
        # AVG ACC
        test_accs.append((accs_0+accs_1+accs_2+accs_3+accs_4) / 5)

        ACC1.append((acc1_0 + acc1_1 + acc1_2 + acc1_3 + acc1_4) / 5)
        ACC2.append((acc2_0 + acc2_1 + acc2_2 + acc2_3 + acc2_4) / 5)
        PRE1.append((pre1_0 + pre1_1 + pre1_2 + pre1_3 + pre1_4) / 5)
        PRE2.append((pre2_0 + pre2_1 + pre2_2 + pre2_3 + pre2_4) / 5)
        REC1.append((rec1_0 + rec1_1 + rec1_2 + rec1_3 + rec1_4) / 5)
        REC2.append((rec2_0 + rec2_1 + rec2_2 + rec2_3 + rec2_4) / 5)

        F1.append((F1_0 + F1_1 + F1_2 + F1_3 + F1_4) / 5)
        F2.append((F2_0 + F2_1 + F2_2 + F2_3 + F2_4) / 5)

    print("Weibo: | Total_Test_ Accuracy: {:.4f} | ACC1: {:.4f} | ACC2: {:.4f} | PRE1: {:.4f} | PRE2: {:.4f} | REC1: {:.4f} | REC2: {:.4f} | F1: {:.4f} | F2: {:.4f}".format(sum(test_accs) / iterations, sum(ACC1) / iterations, sum(ACC2) / iterations, sum(PRE1) / iterations, sum(PRE2) /iterations, sum(REC1) / iterations, sum(REC2) / iterations, sum(F1) / iterations, sum(F2) / iterations))
