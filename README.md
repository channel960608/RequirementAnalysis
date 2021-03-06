﻿# VSM模型使用案例  

### 简介  
本项目为武汉大学需求工程课程作业  
内容为使用VSM模型建立需求到相关制品的需求跟踪链接  

### 运行环境  
- Python 3.*  
- nltk  
- numpy  
- gensim  
除了python，其他包均可通过pip工具安装  

### 代码说明  
本项目入口文件为__vsm.py__, 环境配置完成后运行该文件即可  
具体命令___python vsm.py_  
本项目只用的两个数据集为  
- CCHIT  
- SMOS  
如果需要更换使用的数据集，只需要更改 __vsm.py__ 文件主函数 __run(dataset)__ 的参数为数据集名字（小写）  
#### 文档预处理  
使用nltk进行分词，将单词转化为小写，去除标点以及停用词, 词干化处理 
#### 计算tf与idf  
利用gensim构建语料，并计算tf和idf  
建立空间向量
#### 相似度计算  
实现了余弦相似度计算
#### 模型评估与优化  
主要从以下几个方面  
- 选取不同的阈值k，来对召回率与精度进行比较
- 计算样本中所有标定链接的相似度，查看其相似度的分布情况
- 针对标定链接中计算得相似度较低的样本，找到其相似度计算比较低的原因对文档预处理进行优化

### 输出内容  
- 控制台 数据集中的标定为跟踪链接的文档对，以及他们的文档相似度，根据相似度排序
- 控制台 相似度阈值k从0到1的选择情况下，样本的精度和召回率
- Figure1 数据集中标定链接文档之间相似度的密度分布柱状图，横坐标为相似度，纵坐标为数量
- Figure2 选择不同k值情况下的召回率和精度情况，绿色为召回率，红色为精度
- Figure3 模型精度与召回率的关系，横坐标为召回率，纵坐标为精度 
