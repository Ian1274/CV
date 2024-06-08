# 使用inception resnet v1 训练lfw_faces，LOSS下降则表示学习成功
# 1.读取数据
# 2.预处理数据
# 3.定义网络

# 导入相关包
import time
import torch
import numpy as np
import matplotlib.pyplot as plt
import torchvision.datasets as tdst
# 预处理
import torchvision.transforms as transforms
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data.sampler import SubsetRandomSampler
# 导入InceptionResnetV1模型
from model.FACENET import InceptionResnetV1
# 模型架构可视化
from torchsummary import summary
# 使用tensorboard记录参数
from torch.utils.tensorboard import SummaryWriter

img_preprocess = transforms.Compose([
    # 裁剪
    transforms.Resize((112,112)),
    # PIL图像转为tensor，归一化到[0,1]：Converts a PIL Image or numpy.ndarray (H x W x C) in the range [0, 255] to a torch.FloatTensor of shape (C x H x W) in the range [0.0, 1.0]
    transforms.ToTensor(),
    # 规范化至 [-1,1]
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])

# 从文件夹读取图片
dataset_clean = tdst.ImageFolder(root='./chinese_faces_cleaned/',transform=img_preprocess)
dataset_masked = tdst.ImageFolder(root='./chinese_faces_masked/',transform=img_preprocess)

# 组合两份数据
dataset_all = torch.utils.data.ConcatDataset([dataset_clean, dataset_masked])

# 将数据分为训练集和测试集

# 批次
BATCH_SIZE = 196
# 测试集占比
validation_split = 0.2

# 数据集大小
dataset_size = len(dataset_all)
# 生成索引list
indices = list(range(dataset_size))
# 分割线
split = int(np.floor(validation_split * dataset_size))
# 打乱索引
np.random.shuffle(indices)
# 训练集和测试集索引
train_indices, val_indices = indices[split:], indices[:split]

#
train_sampler = SubsetRandomSampler(train_indices)
valid_sampler = SubsetRandomSampler(val_indices)

# 各自的loader
train_loader = torch.utils.data.DataLoader(dataset_all, batch_size=BATCH_SIZE, 
                                            sampler=train_sampler)
validation_loader = torch.utils.data.DataLoader(dataset_all, batch_size=BATCH_SIZE,
                                                sampler=valid_sampler)

# device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# 查看类别数量
num_classes=len(dataset_clean.class_to_idx)

# 实例化
facenet = InceptionResnetV1(is_train=True,embedding_length=128,num_classes=num_classes).to(device)

# 定义损失
loss_fn = torch.nn.CrossEntropyLoss()
# 定义优化器
optimizer = optim.Adam(facenet.parameters(), lr=0.001)
# 动态减少LR
scheduler = ReduceLROnPlateau(optimizer, 'min')

# 记录变量
writer = SummaryWriter(log_dir='./log')

# 评估
def check_accuracy(loader, model):
    num_correct = 0
    num_samples = 0
    model.eval()
    # 不记录梯度
    with torch.no_grad():
        for i, (x, y) in enumerate(loader):
            x = x.to(device=device)
            y = y.to(device=device)
            output = model(x)
            # 预测类别
            _, predictions = output.max(1)
            # 预测正确的数量
            num_correct += (predictions == y).sum() 
            # 样本总数
            num_samples += predictions.size(0)
    model.train()
    return num_correct.item(), num_samples

# 训练100个epoch
EPOCH_NUM = 200
# 记录最好的测试acc
best_test_acc = 0

for epoch in range(EPOCH_NUM):
    # 获取批次图像
    start_time = time.time()
    loss = 0
    for i, (x, y) in enumerate(train_loader):
        #----------------------训练D：真实数据标记为1------------------------
        # ！！！每次update前清空梯度
        facenet.zero_grad()
        # 获取数据
        # 图片
        x = x.to(device)
        # 标签
        y = y.to(device)
        # 预测值
        y_pred = facenet(x)
        #计算损失
        loss_batch = loss_fn(y_pred, y)
        
        # 计算梯度
        loss_batch.backward()
        optimizer.step()
        optimizer.zero_grad()
        
        # 记录每个batch的train loss
        loss_batch = loss_batch.detach().cpu()
        # 打印
        print(loss_batch.item())
        loss += loss_batch
    
    # 每个epoch的loss
    loss = loss / len(train_loader)
    
    # 如果降低LR：如果loss连续10个epoch不再下降，就减少LR
    scheduler.step(loss)

    
    # tensorboard 记录 Loss/train
    writer.add_scalar('Loss/train', loss, epoch)

    # 在测试集上评估
    num_correct, num_samples = check_accuracy(validation_loader, facenet)
    test_acc = num_correct/num_samples
    
    # 记录最好的测试准确率，并保存模型
    if best_test_acc < test_acc:
        best_test_acc = test_acc
        # 保存模型
        torch.save(facenet.state_dict(), './save_model/facenet_best.pt')
        print('第{}个EPOCH达到最好ACC:{}'.format(epoch,best_test_acc))
        
    
    # tensorboard 记录 ACC/test
    writer.add_scalar('ACC/test', test_acc, epoch)
    
    # 打印信息
    print('第{}个epoch执行时间：{}s，train loss为：{}，test acc为：{}/{}={}'.format(
        epoch,
        time.time()-start_time,
        loss,
        num_correct,
        num_samples,
        test_acc
    ) )

# 保存模型
torch.save(facenet.state_dict(), './save_model/facenet_latest.pt')