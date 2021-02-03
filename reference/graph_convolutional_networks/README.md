# Graph Convolutional Networks

+ **A Comprehensive Survey on Graph Neural Networks**
> 一篇图神经网络的综述文章，将图神经网络分为4类——图循环神经网络、图卷积神经网络、图自编码器和图时空神经网络，将图卷积神经网络分为基于谱和基于空间两种。

+ **Graph Convolutional Networks: Algorithms, Applications and Open Challenges**
> 一篇图卷积神经网络的综述文章，介绍了基于谱和基于空间的两种图卷积方式以及常见模型，同时介绍了图卷积神经网络的应用场景。

+ **Spectral Networks and Deep Locally Connected Networks on Graphs**
> 最早提出图卷积的文章，介绍了基于谱和基于空间的两种图卷积方式，基于空间的方式是将每一层结点的聚类结果作为下一层的神经元，基于谱的方式利用了拉普拉斯矩阵的特征向量进行运算。

+ **Convolutional Neural Networks on Graphs with Fast Localized Spectral Filtering**
> 提出一种基于Chebyshev多项式简化的图卷积方法，多项式的阶次表示了近邻结点的最大跳数，同时提出了图粗化和池化的方法。

+ **Semi-Supervised Classification with Graph Convolutional Networks**
> 基于ChebyNet进行一阶估计（即只考虑图中一步距离的结点）和简化实现一种高效的半监督结点分类方法，计算高效且能够避免过拟合，但无法适用于有向图和边具有特征的图。

+ **Neural Network for Graphs: A Contextual Constructive Approach**
> 提出一种基于空域的图卷积网络NN4G，将某一结点邻居结点以及标签信息进行运算，得到下一层结点特征，NN4G能够适用于多种类型的图。

+ **Diffusion-Convolutional Neural Networks**
> 提出扩散卷积网络，能够将图中结点间信息基于一定概率进行传播，可用于结点分类、边分类、图分类等多种任务场景。