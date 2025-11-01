项目概述
本项目实现了一个完整的中文多模态情感分析系统，旨在通过融合文本、音频和视频三种模态信息，实现对人类情感状态的准确识别与分类。项目基于CH-SIMS中文多模态情感分析数据集，系统性地比较了传统机器学习方法与深度学习方法在多模态情感分析任务中的性能表现。

研究背景与意义
多模态情感分析(Multimodal Sentiment Analysis, MSA)是计算机视觉、自然语言处理和语音信号处理交叉领域的重要研究方向。在实际的人际交流中，情感信息往往通过多种渠道同时传递，包括语言内容、语音韵律特征以及面部表情等视觉信息。单一模态的情感分析方法往往无法充分捕捉这种复杂的情感表达模式。

本项目的主要项目包括：

构建了完整的中文多模态情感分析实验框架
系统性比较了多种机器学习与深度学习方法的性能
复现了多种先进的多模态融合算法
技术架构
数据处理流程
特征提取：分别对文本、音频、视频三种模态进行特征提取
模型训练：使用传统机器学习和深度学习方法进行单模态和多模态建模
融合策略：采用多种融合方法整合不同模态的预测结果
性能评估：通过多种评价指标对模型性能进行全面评估
核心算法
传统机器学习方法
决策树(Decision Tree)：基于信息增益的树形分类器
随机森林(Random Forest)：集成多个决策树的Bagging方法
支持向量机(SVM)：基于最大间隔原理的分类器
梯度提升树(GBDT)：基于梯度提升的集成学习方法
AdaBoost：自适应增强的集成学习算法
Stacking：多层集成学习框架
深度学习方法
文本模态：
BERT：基于Transformer的预训练语言模型
BERT+TextCNN：结合卷积神经网络的文本分类模型
音频模态：
TIM-NET：时间感知双向多尺度网络
CAM++：改进的通道注意力机制网络
视频模态：
YOLOv8：用于人脸检测和情感分类的目标检测网络
多模态融合方法
MTFN(Multi-Task Fusion Network)：多任务融合网络
MLMF(Multimodal Low-rank Fusion)：多模态低秩融合方法
数据集说明
本项目使用CH-SIMS中文多模态情感分析数据集，该数据集具有以下特点：

数据规模：2281个视频片段，训练集1824个样本，测试集457个样本
数据来源：电影、电视剧、综艺节目片段
语言特征：纯中文普通话内容
视频特征：1-10秒时长，单人出镜
标注方式：三分类情感标签(负面、中性、正面)
特征提取规范
文本特征：768维BERT词向量或100维传统词向量
音频特征：33维声学特征(MFCC、基频、谱特征等)
视频特征：709维面部表情特征(面部关键点、动作单元、头部姿态等)
项目结构
├── src/                          # 核心源代码
│   ├── text/                     # 文本处理模块
│   │   ├── bert/                 # BERT模型实现
│   │   └── bert+textcnn/         # BERT+TextCNN模型
│   ├── audio/                    # 音频处理模块
│   │   ├── TIM-Net_SER-main/     # TIM-NET实现
│   │   ├── AudioClassification_PaddlePaddle/  # 音频分类
│   │   └── LSTM/                 # LSTM音频模型
│   ├── video/                    # 视频处理模块
│   ├── ML/                       # 机器学习算法
│   ├── config.json               # 配置文件
│   ├── mutil.py                  # 多模态融合主程序
│   └── predict.py                # 预测接口
├── features/                     # 特征提取模块
│   ├── feature_exrt.py           # 文本特征提取
│   ├── audio_feature_extract.py  # 音频特征提取
│   ├── video_feature_extract.py  # 视频特征提取
│   └── extract_all_features.py   # 统一特征提取接口
├── dataset/                      # 数据集目录
├── results/                      # 实验结果
├── Models/                       # 训练好的模型
├── report/                       # 技术报告
└── README.md                     # 项目说明文档
环境配置
系统要求
Python 3.7+
CUDA 10.2+ (GPU训练可选)
核心依赖
# 基础科学计算库
pip install numpy pandas scikit-learn

# 深度学习框架
pip install torch torchvision transformers

# 多模态处理库
pip install librosa opencv-python dlib

# 中文文本处理
pip install jieba

# 多模态分析框架
pip install MMSA-FET
使用指南
1. 特征提取
# 提取所有模态特征
cd features
python extract_all_features.py

# 单独提取特定模态特征
python extract_all_features.py --modality text
python extract_all_features.py --modality audio
python extract_all_features.py --modality video
2. 模型训练
传统机器学习方法
cd src/ML
python ml_train.py
深度学习方法
# 文本模态
cd src/text/bert
python main.py

# 音频模态
cd src/audio/TIM-Net_SER-main
python main.py

# 视频模态
cd src/video
python model_train_video.py
多模态融合
cd src
python mutil.py
3. 模型预测
cd src
python predict.py
实验结果
单模态性能
模态	方法	准确率	F1分数
文本	BERT	85.2%	84.8%
文本	BERT+TextCNN	86.1%	85.7%
音频	TIM-NET	78.3%	77.9%
视频	YOLOv8	72.6%	71.8%
多模态融合性能
融合方法	准确率	F1分数	提升幅度
MTFN	88.7%	88.3%	+2.6%
MLMF	87.9%	87.5%	+1.8%
to-do list
算法优化
 实现Transformer-based多模态融合模型
 优化音频特征提取算法，提升音频模态性能
 探索注意力机制在多模态融合中的应用
数据扩展
 扩展到更大规模的中文多模态数据集
 增加细粒度情感分析(7分类或连续值回归)
系统完善
 开发Web界面进行实时情感分析演示
 优化模型推理速度，支持实时处理
参考文献
核心论文
Yu, W., Xu, H., Meng, F., Zhu, Y., Ma, Y., Wu, J., Zou, J., & Yang, K. (2020). CH-SIMS: A Chinese Multimodal Sentiment Analysis Dataset with Fine-grained Annotation of Modality. Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics, 3718-3727.
Zadeh, A., Chen, M., Poria, S., Cambria, E., & Morency, L. P. (2017). Tensor Fusion Network for Multimodal Sentiment Analysis. Proceedings of the 2017 Conference on Empirical Methods in Natural Language Processing, 1103-1114.
Liu, Z., Shen, Y., Lakshminarasimhan, V. B., Liang, P. P., Zadeh, A., & Morency, L. P. (2018). Efficient Low-rank Multimodal Fusion with Modality-Specific Factors. arXiv preprint arXiv:1806.00064.
Devlin, J., Chang, M. W., Lee, K., & Toutanova, K. (2019). BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding. Proceedings of the 2019 Conference of the North American Chapter of the Association for Computational Linguistics, 4171-4186.
相关综述
Baltrušaitis, T., Ahuja, C., & Morency, L. P. (2018). Multimodal machine learning: A survey and taxonomy. IEEE Transactions on Pattern Analysis and Machine Intelligence, 41(2), 423-443.
Poria, S., Cambria, E., Bajpai, R., & Hussain, A. (2017). A review of affective computing: From unimodal analysis to multimodal fusion. Information Fusion, 37, 98-125.
相关项目
开源框架
MMSA: 多模态情感分析统一框架，支持15种MSA模型
MMSA-FET: 多模态特征提取工具包，兼容MMSA框架
M-SENA: 多模态情感分析集成平台
OpenFace: 面部行为分析工具包
Transformers: Hugging Face预训练模型库
数据集
CH-SIMS: 中文多模态情感分析数据集
CMU-MOSI: 英文多模态情感分析数据集
CMU-MOSEI: 大规模多模态情感分析数据集
相关实现
TensorFusionNetwork: 张量融合网络官方实现
Low-rank-Multimodal-Fusion: 低秩多模态融合实现
Multimodal-Transformer: 多模态Transformer实现
