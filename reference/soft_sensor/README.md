# Soft Sensor

+ **Bayes Estimates for the Linear Model**
> 相比于最小二乘线性回归，认为模型参数服从一个分布，取决于先验知识（先验概率）和训练数据（数据似然），当训练数据越来越多时，结果和最小二乘线性回归越接近。

+ **Gaussian Processes for Regression**
> 利用核函数表征样本间的相关性，计算得到测试样本预测结果满足的分布。

+ **Long Short-Term Memory**
> 通过记忆单元的引入实现时序关系的挖掘与学习。

+ **A Framework and Modeling Method of Data-Driven Soft Sensors Based on Semisupervised Gaussian Regression**
> 数据缺失包括自变量缺失和预测变量缺失两种，自变量缺失有多种方法弥补，半监督主要解决预测变量缺失，对高斯过程回归改进以使得能够同时应用大量无标签数据和少量有标签数据建立软测量模型。

+ **Semi-Supervised and Unsupervised Extreme Learning Machines**
> 极限学习机随机化设置输入层到隐含层的参数，采用岭回归计算隐含层到输出层的参数，具有训练效率高预测性能强的特点，针对极限学习机进行改进以使得应用于半监督和无监督场景，在半监督场景下将流行正则框架加入目标函数以实现对无标签数据的利用，在无监督场景下去掉目标函数中的误差项即可，改进后的极限学习机能够在保持原有特点的基础上应用于更多场景。

+ **Nonlinear dynamic soft sensor modeling with supervised long short-term memory network**
> 采用有监督的LSTM实现软测量，隐含层的输入不仅包含上一时刻隐含层输出和样本特征向量，还包含样本标签。

+ **基于慢特征分析的软测量建模方法研究**
> 采用SFA提取慢特征，可采用核函数实现非线性扩展，之后对时滞进行估计，利用GPR实现软测量。

+ **A comprehensive experimental evaluation of orthogonal polynomial expanded random vector functional link neural networks for regression**
> 首先将原始特征向量采用正交多项式进行扩展，输入到隐含层，再利用隐含层输出和扩展向量得出输出，其中输入到隐含层的连接权重随机初始化，只学习输出的连接权重。

+ **Active learning for modeling and prediction of dynamical fluid processes**
> 本文采用GPR对往复多相泵的液体流量进行预测，基于GPR的预测方差利用基于相对方差的主动学习策略实现小样本下模型训练，相较于基于方差的主动学习和等间隔采样的传统训练方式效果更好，适用于具有非线性和动态性的对象。

+ **Deep quality-related feature extraction for soft sensing modeling: A deep learning approach with hybrid VW-SAE**
> 采用栈式自编码器提取特征，基于皮尔逊相关系数和斯皮尔曼相关系数得到混合相关系数，利用混合相关系数建立变量加权的自编码器目标函数，进而实现和目标变量线性和非线性相关的变量提取，提升软测量性能。

+ **Lightly trained support vector data description for novelty detection**
> 改进SVDD（支持向量数据描述，寻找超球体半径尽可能小且包含尽可能多的数据点）使其计算效率提高，用于异常点检测。

+ **基于注意力LSTM的多阶段发酵过程集成质量预测**
> 基于PLS和encoder-decoder模型分别提取得分矩阵和动态特征用以表征静态特性和动态特性，再采用AP聚类方法得到不同的分段结果（稳定阶段和过渡阶段），在不同段内建立注意力机制LSTM集成模型。

+ **基于机器学习的软测量建模及其应用**
> 工业软测量中存在显著非线性、多工况和时变等特点，文中提出了四种方法：1、首先采用AP聚类算法对数据进行聚类划分，不同类别建立各自的GPR模型，采用人工鱼群算法对聚类算法中的超参数进行优化；2、在第一种方法的基础上，在聚类完成之后利用SVDD根据内外边界对聚类结果不明确的样本进行精细化处理；3、采用GPR建立Bagging集成模型，各基学习器依据正则化互信息原则选择与预测变量最相关且信息不冗余的变量进行建模；4、采用局部加权的KPLS模型，其中核函数采用混合核函数。

+ **软测量横型的变量选择方法研究**
> 针对软测量中的变量选择问题，提出三种方法：1、将过滤式和包裹式选择方法相结合，先采用蒙特卡洛无信息变量消除选择部分变量，再采用基于遗传算法的PLS选择变量；2、基于线性回归和变量选择建立MIQP优化问题；3、基于SVR和变量选择建立MILP优化问题。（后两种方法通过引入0-1决策变量实现变量选择）

+ **Dynamic Soft Sensor Development Based on Convolutional Neural Networks**
> 提出两种基于CNN的软测量模型，第一种直接利用时序扩展后的过程数据，采用CNN框架实现软测量建模，第二种在第一种的基础上，首先采用FIR提取反映过程动态特性的特征，再利用CNN实现软测量预测。

+ **Deep learning for quality prediction of nonlinear dynamic processes with variable attention-based long short-term memory network**
> 在LSTM模型中加入变量注意力机制，对不同时刻的不同过程变量赋予不同的权重。

+ **Deep learning with spatiotemporal attention-based LSTM for industrial soft sensor model development**
>在LSTM模型中同时加入空间注意力（过程变量）和时间注意力（样本），在质量变量的预测过程中更关注有助于预测的过程变量和样本。

+ **Gated Stacked Target-Related Autoencoder: A Novel Deep Feature Extraction and Layerwise Ensemble Method for Industrial Soft Sensor Application**
> 为了使得自编码器提取的特征能够与目标直接关联，在解码器部分补充对标签的重构，因而提取到的特征能够蕴含和目标相关的成分，称之为目标相关的自编码器（TAE）。在此基础上将TAE堆叠，每层特征在不同程度上蕴含和目标相关的成分，利用每层特征通过门结构（类似attention机制）得到预测结果的总和作为最终预测结果。

+ **Supervised deep belief network for quality prediction in industrial processes**
> 通过堆叠逐层训练的有监督受限玻尔兹曼机，即同时将过程变量和质量变量作为输入，以提取和质量变量相关的隐层特征，再通过整体网络模型微调得到软测量模型，由于在线测试时无法直接获取质量变量，因此采用上一时刻的测量结果或预测结果作为当前时刻质量变量的估计值。整体思想类似于SLSTM，感觉问题在于训练时采用当前时刻的质量变量预测当前时刻的质量变量，而测试时采用上一时刻的质量变量预测当前时刻的质量变量，两者并不是同样的关系。

+ **Hierarchical quality-relevant feature representation for soft sensor modeling: a novel deep learning strategy**
> 采用堆栈自编码的结构建立软测量模型，相比于普通的自编码器，在重构输入的同时引入对质量变量的重构能力，从而使得逐层提取的特征能够保留对质量变量的预测能力，最终基于逐层提取的特征再建立回归器。由于质量变量个数通常较少，因此可在重构损失中增加质量变量重构的损失权重，以避免模型对质量变量的重构能力较弱。

+ **Semi-Supervised Regression with Co-Training**
> 提出一种用于回归预测的半监督协同训练方法——Coreg。协同训练要求构建两个充分且多样的模型，各自挑选预测置信度最高的无标签样本加入对方的训练数据中，迭代更新训练两个模型，直至两个模型的训练数据均不发生变化或达到训练轮数。Coreg采用两个使用不同距离度量方式的KNN作为回归器，预测置信度的评价使用无标签样本引入前后模型对有标签样本预测误差的变化情况，若能使得模型对有标签样本预测误差最大化减小，则该无标签样本的预测置信度最高。由于每个轮次针对每个无标签样本都需要遍历一遍所有有标签样本，计算复杂度较高，因此针对每个无标签样本，仅近似使用它k个近邻的有标签样本进行置信度评价。最终的预测结果由两个回归器的平均给出。

+ **Co-training partial least squares model for semi-supervised soft sensor development**
> 一种采用PLS实现协同训练的半监督软测量模型，将所有自变量划分为两组分别用于两个PLS模型，即两个模型的多样性所在。由于两个模型都无法使用全部的自变量，因此另定义两个包含全部自变量的数据集用于对应存放两个模型的训练数据。

+ **Just-in-time semi-supervised soft sensor for quality prediction in industrial rubber mixers**
> 采用引入图拉普拉斯正则项的极限学习机作为半监督模型，同时引入即时学习思想，将所有样本点首先进行聚类，同一簇的样本作为建立局部模型的训练数据。

+ **Deep Learning of Semi-supervised Process Data with Hierarchical Extreme Learning Machine and Soft Sensor Application**
> 首先采用层次自编码极限学习机进行逐层无监督的特征提取，再利用引入图拉普拉斯正则项的极限学习机实现半监督的软测量建模。

+ **Ensemble deep learning based semi-supervised soft sensor modeling method and its application on quality prediction for coal preparation process**
> 首先使用有标签数据和无标签数据利用SAE进行无监督的特征提取，再将提取的隐层特征和标签信息一起输入双向LSTM中进行时序关系的挖掘，最后通过全连接层得到预测结果。

+ **Soft Sensor Modeling Method Based on Semisupervised Deep Learning and Its Application to Wastewater Treatment Plant**
> 通过流形正则项的引入实现半监督软测量，流行正则项的约束可以加在网络的任意隐含层上，不仅局限于对标签进行约束。

+ **A novel semi-supervised pre-training strategy for deep networks and its application for quality variable prediction in industrial processes**
> 相较于传统首先利用无标签数据预训练SAE，再利用有标签数据fine-tuning整个网络的方法，此方法在预训练阶段同时使用有标签数据和无标签数据重构过程变量和质量变量，使得预训练阶段所提取的特征能够和质量变量密切相关，从而提升半监督软测量的精度。

+ **Mixture Semisupervised Principal Component Regression Model and Soft Sensor Application**
> 传统的半监督概率PCR方法同时考虑有标签样本输入输出的联合似然和无标签样本输入的似然，通过最大化两部分似然函数之和对模型参数进行求解，因而能够同时使用有标签数据和无标签数据。混合半监督概率PCR在此基础上引入多个PCR模型，将多个PCR模型的加权输出作为结果，所采用的损失函数和之前一样，采用EM算法进行求解，适用于非线性、多模态过程半监督软测量建模。

+ **Nonlinear industrial soft sensor development based on semi-supervised probabilistic mixture of extreme learning machines**
> 提出了一种使用概率混合ELM的半监督软测量算法，通过多个子ELM模型解决复杂非线性问题，每个样本对不同子ELM模型有不同的隶属度，最终使用VBEM对模型参数进行概率求解，推理时也融合了多个子ELM模型的预测结果。

+ **Semisupervised Bayesian Method for Soft Sensor Modeling with Unlabeled Data Samples**
> 在PCR的概率生成框架下，通过最大化有标签样本输入输出的联合似然和无标签样本输入的似然，实现半监督软测量建模。为了更方便地选择所提取潜变量个数，即降维后维度，引入贝叶斯正则项控制模型的复杂度，最终通过EM算法进行求解。

+ **Semisupervised learning for probabilistic partial least squares regression model and soft sensor application**
> 基于概率PLS建立半监督的软测量模型，概率PLS的模型构建和求解过程与概率PCR基本一样，不同之处在于概率PLS提取的主成分一部分用于解释质量变量，另一部分用于解释自身信息，通过最大化有标签样本输入输出的联合似然和无标签样本输入的似然，最终利用EM算法进行求解得到模型参数。