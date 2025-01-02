# CV01_imgSearch
## 以图搜图
### 简介
针对小样本数据集，运用迁移学习技术，提取图像特征，建立faiss特征向量数据库，通过比较特征向量来预测类别。



# CV02_siteProtection
## 工地防护检测
### 简介
检测工地工人是否穿戴了防护马甲以及安全帽（区分不同颜色）
### 效果
![gif_cv02_01](https://github.com/Ian1274/CV/blob/main/CV02_siteProtection/results/result.gif)

# CV03_faceMask
## 口罩人脸识别
### 简介
通过torch简单复现resnet18；

- 数据处理：数据集进行人脸裁剪+随机添加口罩；
- 模型训练：使用InceptionResnetV1进行训练，检测时将output处理成一个128维的embedding；
- 测试预测：将测试图片与对比库都进行embedding后，比较测试图片与对比库的欧氏距离来判断类别；

### 效果
![gif_cv03_01](https://github.com/Ian1274/CV/blob/main/CV03_faceMasks/3.demo/results/result.gif)

