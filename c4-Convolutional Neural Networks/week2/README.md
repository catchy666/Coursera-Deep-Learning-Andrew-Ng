<h1 align="center">Week2: 深度卷积网络：实例探究</h1>

## 目录
* [笔记](#笔记)
   * [目录](#目录)
   * [为什么要进行实例探究](#为什么要进行实例探究)
   * [经典卷积网络](#经典卷积网络)
       - [LeNet-5](#lenet-5)
       - [AlexNet](#alexnet)
       - [VGG](#vgg)
   * [残差网络](#残差网络)
       - [残差网络有效的原因](#残差网络有效的原因)
   * [1×1卷积](#1x1卷积)
   * [Inception网络](#inception网络)
       - [计算成本问题](#计算成本问题)
       - [完整的Inception网络](#完整的Inception网络)
   * [使用开源的实现方案](#使用开源的实现方案)
   * [迁移学习](#迁移学习)
   * [数据扩增](#数据扩增)
   * [计算机视觉现状](#计算机视觉现状)



## 为什么要进行实例探究

计算机视觉研究中的大量研究都集中在如何把这些基本构件组合起来，形成有效的卷积神经网络。最直观的方式之一就是去看一些案例，就像很多人通过看别人的代码来学习编程一样，通过研究别人构建有效组件的案例是个不错的办法。

实际上在计算机视觉任务中表现良好的神经网络框架往往也适用于其它任务。也就是说，如果有人已经训练或者计算出擅长识别猫、狗、人的神经网络或者神经网络框架，而你的计算机视觉识别任务是构建一个自动驾驶汽车，你完全可以借鉴别人的神经网络框架来解决自己的问题。

## 经典卷积网络

### LeNet-5

![](https://raw.githubusercontent.com/catchy666/Coursera-Deep-Learning-Andrew-Ng/main/c4-Convolutional%20Neural%20Networks/week2/tmp-imgs/01.png)

从左往右看，随着网络越来越深，图像的高度和宽度在缩小，从最初的 32×32 缩小到 28×28，再到 14×14、10×10，最后只有 5×5。与此同时，随着网络层次的加深，通道数量一直在增加，从 1 增加到 6 个，再到 16 个。

相关论文：[LeCun et.al., 1998. Gradient-based learning applied to document recognition](http://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=726791&tag=1)。吴恩达老师建议精读第二段，泛读第三段。

特点：

* LeNet-5 针对灰度图像而训练，因此输入图片的通道数为 1。
* 该模型总共包含了约 6 万个参数，远少于标准神经网络所需。
* 典型的 LeNet-5 结构包含卷积层（CONV layer），池化层（POOL layer）和全连接层（FC layer），排列顺序一般为 CONV layer->POOL layer->CONV layer->POOL layer->FC layer->FC layer->OUTPUT layer。一个或多个卷积层后面跟着一个池化层的模式至今仍十分常用。
* 当 LeNet-5模型被提出时，其池化层使用的是平均池化，而且各层激活函数一般选用 Sigmoid 和 tanh。现在，我们可以根据需要，做出改进，使用最大池化并选用 ReLU 作为激活函数。

### AlexNet

![](https://raw.githubusercontent.com/catchy666/Coursera-Deep-Learning-Andrew-Ng/main/c4-Convolutional%20Neural%20Networks/week2/tmp-imgs/02.png)

特点：

* AlexNet 模型与 LeNet-5 模型类似，但是更复杂，包含约 6000 万个参数。另外，AlexNet 模型使用了 ReLU 函数。
* 当用于训练图像和数据集时，AlexNet 能够处理非常相似的基本构造模块，这些模块往往包含大量的隐藏单元或数据。

相关论文：[Krizhevsky et al.,2012. ImageNet classification with deep convolutional neural networks](http://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf)。这是一篇易于理解并且影响巨大的论文，计算机视觉群体自此开始重视深度学习。

### VGG

![](https://raw.githubusercontent.com/catchy666/Coursera-Deep-Learning-Andrew-Ng/main/c4-Convolutional%20Neural%20Networks/week2/tmp-imgs/03.png)

特点：

* VGG 又称 VGG-16 网络，“16”指网络中包含 16 个卷积层和全连接层。
* 超参数较少，只需要专注于构建卷积层。
* 结构不复杂且规整，都是几个卷积层后面跟着可以压缩图像大小的池化层，池化层缩小图像的高度和宽度。同时，卷积层的过滤器数量变化存在一定的规律，由 64 翻倍变成 128，再到 256 和 512。
* 主要缺点是VGG 需要训练的特征数量巨大，包含多达约 1.38 亿个参数。

相关论文：[Simonvan & Zisserman 2015. Very deep convolutional networks for large-scale image recognition](https://arxiv.org/pdf/1409.1556.pdf)。

## 残差网络

因为存在梯度消失和梯度爆炸问题，网络越深，就越难以训练成功。**残差网络（Residual Networks，简称为 ResNets）**可以有效解决这个问题。

![](https://raw.githubusercontent.com/catchy666/Coursera-Deep-Learning-Andrew-Ng/main/c4-Convolutional%20Neural%20Networks/week2/tmp-imgs/04.jpg)

上图结构称为**残差块（Residual block）**。通过**捷径（Short cut，或者称跳远连接，Skip connections）**可以将 $a^{[l]}$添加到第二个 ReLU 过程中，直接建立 $a^{[l]}$与 $a^{[l+2]}$之间的隔层联系。表达式如下：

![](https://raw.githubusercontent.com/catchy666/Coursera-Deep-Learning-Andrew-Ng/main/c4-Convolutional%20Neural%20Networks/week2/tmp-imgs/08.jpg)

- Linear：
$$
z^{[l+1]} = W^{[l+1]}a^{[l]} + b^{[l+1]}
$$

- Relu：
$$
a^{[l+1]} = g(z^{[l+1]})
$$

- Linear：
$$
z^{[l+2]} = W^{[l+2]}a^{[l+1]} + b^{[l+2]}
$$

- Relu：
$$
a^{[l+2]} = g(z^{[l+2]} + a^{[l]})
$$

构建一个残差网络就是将许多残差块堆积在一起，形成一个深度网络。

![](https://raw.githubusercontent.com/catchy666/Coursera-Deep-Learning-Andrew-Ng/main/c4-Convolutional%20Neural%20Networks/week2/tmp-imgs/05.jpg)

为了便于区分，在 ResNets 的论文[He et al., 2015. Deep residual networks for image recognition](https://arxiv.org/pdf/1512.03385.pdf)中，非残差网络被称为**普通网络（Plain Network）**。将它变为残差网络的方法是加上所有的跳远连接。

![](https://raw.githubusercontent.com/catchy666/Coursera-Deep-Learning-Andrew-Ng/main/c4-Convolutional%20Neural%20Networks/week2/tmp-imgs/06.jpg)

在理论上，随着网络深度的增加，性能应该越来越好。但实际上，对于一个普通网络，随着神经网络层数增加，训练错误会先减少，然后开始增多。但残差网络的训练效果显示，即使网络再深，其在训练集上的表现也会越来越好。

![](https://raw.githubusercontent.com/catchy666/Coursera-Deep-Learning-Andrew-Ng/main/c4-Convolutional%20Neural%20Networks/week2/tmp-imgs/07.jpg)

残差网络有助于解决梯度消失和梯度爆炸问题，使得在训练更深的网络的同时，又能保证良好的性能。

###  残差网络有效的原因

假设有一个大型神经网络，其输入为 $X$，输出为 $a^{[l]}$。给这个神经网络额外增加两层，输出为 $a^{[l+2]}$。将这两层看作一个具有跳远连接的残差块。为了方便说明，假设整个网络中都选用 ReLU 作为激活函数，包括输入 X 的非零异常值，因此输出的所有激活值都大于等于 0。

![](https://raw.githubusercontent.com/catchy666/Coursera-Deep-Learning-Andrew-Ng/main/c4-Convolutional%20Neural%20Networks/week2/tmp-imgs/09.jpg)

则有

$$
\begin{equation}
\begin{split}
 a^{[l+2]} &= g(z^{[l+2]}+a^{[l]})  
     \\\ &= g(W^{[l+2]}a^{[l+1]}+b^{[l+2]}+a^{[l]})
\end{split}
\end{equation}
$$

当发生梯度消失时，$W^{[l+2]}\approx0$，$b^{[l+2]}\approx0$，则有：

$$
a^{[l+2]} = g(a^{[l]}) = ReLU(a^{[l]}) = a^{[l]}
$$

因此，这两层额外的残差块不会降低网络性能。而如果没有发生梯度消失时，训练得到的非线性关系会使得表现效果进一步提高。

*（只要让这两项为0，就可以达到增加网络深度却不影响网络性能的目的，而是否把这两个参数置为0就要看反向传播，网络最终能够知道到底要不要skip）*

注意，如果$a[l]$与 $a[l+2]$的维度不同，需要引入矩阵 $W_s$与 $a[l]$相乘，使得二者的维度相匹配。参数矩阵 $W_s$既可以通过模型训练得到，也可以作为固定值，仅使 $a[l]$截断或者补零。

下图展示了通过添加跳跃连接来将普通网络转化为残差网络。这个网络有很多层 3×3 卷积，而且它们大多都是 same 卷积，这就是添加等维特征向量的原因。所以这些都是卷积层，而不是全连接层，因为它们是 same 卷积，维度得以保留，这也解释了添加项$z^{[l+2]}+a^{[l]}$（维度相同所以能够相加)

![](https://raw.githubusercontent.com/catchy666/Coursera-Deep-Learning-Andrew-Ng/main/c4-Convolutional%20Neural%20Networks/week2/tmp-imgs/10.png)

普通网络和 ResNets 网络常用的结构是：卷积层-卷积层-卷积层-池化层-卷积层-卷积层-卷积层-池化层……依此重复。直到最后，有一个通过 softmax 进行预测的全连接层。

## 1x1卷积

1x1 卷积（1x1 convolution，或称为 Network in Network）指过滤器的尺寸为 1。当通道数为 1 时，1x1 卷积意味着卷积操作等同于乘积操作。

![](https://raw.githubusercontent.com/catchy666/Coursera-Deep-Learning-Andrew-Ng/main/c4-Convolutional%20Neural%20Networks/week2/tmp-imgs/11.png)

如果是一张 6×6×32 的图片，那么使用 1×1 过滤器进行卷积效果更好。具体来说，1×1 卷积所实现的功能是遍历这 36 个单元格，计算左图中 32 个数字(通道)和过滤器中 32 个数字(通道)的元素积之和，然后应用 ReLU 非线性函数。

![](https://raw.githubusercontent.com/catchy666/Coursera-Deep-Learning-Andrew-Ng/main/c4-Convolutional%20Neural%20Networks/week2/tmp-imgs/12.png)

这个 1×1×32 过滤器中的 32 个数字可以这样理解，一个神经元的输入是 32 个数字（输入图片中左下角位置 32 个通道中的数字），即相同高度和宽度上某一切片上的 32 个数字，这 32 个数字具有不同通道，乘以 32 个权重（将过滤器中的 32 个数理解为权重），然后应用 ReLU 非线性函数，在这里输出相应的结果。

如果过滤器不止一个，而是多个（filters个），就好像有多个输入单元，输出结果是 6×6x#filters（过滤器数量）。

池化能压缩数据的高度（$n_H$）及宽度（$n_W$），而 1×1 卷积能压缩数据的通道数（$n_C$）。在如下图所示的例子中，用 32 个大小为 1×1×192 的滤波器进行卷积，就能使原先数据包含的 192 个通道压缩为 32 个。

![](https://raw.githubusercontent.com/catchy666/Coursera-Deep-Learning-Andrew-Ng/main/c4-Convolutional%20Neural%20Networks/week2/tmp-imgs/13.png)

虽然论文[Lin et al., 2013. Network in network](https://arxiv.org/pdf/1312.4400.pdf)中关于架构的详细内容并没有得到广泛应用，但是 1x1 卷积的理念十分有影响力，许多神经网络架构（包括 Inception 网络）都受到它的影响。

## Inception网络

在之前的卷积网络中，我们只能选择单一尺寸和类型的滤波器。而 **Inception 网络的作用** 即是代替人工来确定卷积层中的过滤器尺寸与类型，或者确定是否需要创建卷积层或池化层。

基本思想是 Inception 网络不需要人为决定使用哪个过滤器或者是否需要池化，而是由网络自行确定这些参数，你可以给网络添加这些参数的所有可能值，然后把这些输出连接起来，让网络自己学习它需要什么样的参数，采用哪些过滤器组合。

![](https://raw.githubusercontent.com/catchy666/Coursera-Deep-Learning-Andrew-Ng/main/c4-Convolutional%20Neural%20Networks/week2/tmp-imgs/14.jpg)

相关论文：[Szegedy et al., 2014, Going Deeper with Convolutions](https://arxiv.org/pdf/1409.4842.pdf)

### 计算成本问题

在提升性能的同时，Inception 网络有着较大的计算成本。下图是一个例子：

![](https://raw.githubusercontent.com/catchy666/Coursera-Deep-Learning-Andrew-Ng/main/c4-Convolutional%20Neural%20Networks/week2/tmp-imgs/15.png)

图中有 32 个滤波器，每个滤波器的大小为 $5\times5\times192$。输出大小为 $28\times28\times32$，所以需要计算 $28\times28\times32 $个数字，对于每个数，都要执行 $5\times5\times192$ 次乘法运算。加法运算次数与乘法运算次数近似相等。因此，可以看作这一层的计算量为 $28\times28\times32\times5\times5\times192 \approx 1.2$亿。

为了解决计算量大的问题，可以引入 1x1 卷积来减少其计算量。

![](https://raw.githubusercontent.com/catchy666/Coursera-Deep-Learning-Andrew-Ng/main/c4-Convolutional%20Neural%20Networks/week2/tmp-imgs/16.png)

对于同一个例子，我们使用 1x1 卷积把输入数据从 192 个通道减少到 16 个通道，然后对这个较小层运行 5x5 卷积，得到最终输出。这个 1x1 的卷积层通常被称作 **瓶颈层（Bottleneck layer）** 。

改进后的计算量为 $28\times28\times16\times192 + 28\times28\times32\times5\times5\times16 = 1.24 $千万，减少了约 90%。

**只要合理构建瓶颈层，就可以既显著缩小计算规模，又不会降低网络性能。**

### 完整的Inception网络

![](https://raw.githubusercontent.com/catchy666/Coursera-Deep-Learning-Andrew-Ng/main/c4-Convolutional%20Neural%20Networks/week2/tmp-imgs/17.jpg)

上图是引入 1x1 卷积后的 Inception 模块。值得注意的是，为了将所有的输出组合起来，红色的池化层使用 Same 类型的填充（padding）来池化使得输出的宽高不变，通道数也不变。这就是一个 Inception 模块，而 Inception 网络所做的就是将这些模块都组合到一起。

多个 Inception 模块组成一个完整的 Inception 网络（被称为 GoogLeNet，以向 LeNet 致敬），如下图所示：

![](https://raw.githubusercontent.com/catchy666/Coursera-Deep-Learning-Andrew-Ng/main/c4-Convolutional%20Neural%20Networks/week2/tmp-imgs/18.jpg)

注意绿色笔圈出的隐藏层，这些分支都是 Softmax 的输出层，可以用来参与特征的计算及结果预测，起到调整并防止发生过拟合的效果。

经过研究者们的不断发展，Inception 模型的 V2、V3、V4 以及引入残差网络的版本被提出，这些变体都基于 Inception V1 版本的基础思想上。顺便一提，Inception 模型的名字来自电影《盗梦空间》。

## 使用开源的实现方案

很多神经网络复杂细致，并充斥着参数调节的细节问题，因而很难仅通过阅读论文来重现他人的成果。想要搭建一个同样的神经网络，查看开源的实现方案会快很多。

## 迁移学习

在Course3中，[迁移学习]()已经被提到过。计算机视觉是一个经常用到迁移学习的领域。在搭建计算机视觉的应用时，相比于从头训练权重，下载别人已经训练好的网络结构的权重，用其做**预训练**，然后转换到自己感兴趣的任务上，有助于加速开发。

社区经常使用得数据集：比如 ImageNet，或者 MS COCO，或者 Pascal。

在我们自己的任务中，常常遇到的情况是，在做某类物体的识别分类时，面临着数据集不够的情况，这个时候通过应用迁移学习，应用社区研究者建立的模型和参数，用少量的数据仅训练最后的自定义的softmax网络，从而能够在小数据集上达到很好的效果。

不同的深度学习编程框架有不同的方式，允许你指定是否训练特定层的权重。在这个例子中，你只需要训练 Softmax 层的权重，把前面这些层的权重都冻结（而冻结的层由于不需要改变和训练，可以看作一个固定函数。可以将这个固定函数存入硬盘，以便后续使用，而不必每次再使用训练集进行训练了。）。

![](https://raw.githubusercontent.com/catchy666/Coursera-Deep-Learning-Andrew-Ng/main/c4-Convolutional%20Neural%20Networks/week2/tmp-imgs/19.jpg)

1. 取输入图像 x，然后把它映射到这层（softmax的前一层，紫色线条标记的）的激活函数，计算特征或者激活值。
2. 将这个固定函数存入硬盘（softmax 层之前的所有层视为一个固定映射）。
3. 在此之上训练 softmax 分类器。

如果我们在自己的问题上也拥有大量的数据集，我们可以多训练后面的几层。总之随着数据集的增加，我们需要“（冻结） freeze”的层数越来越少。最后如果我们有十分庞大的数据集，那么我们可以训练网络模型的所有参数，将其他研究者训练的模型参数作为参数的初始化来替代随机初始化，来加速我们模型的训练。

![](https://raw.githubusercontent.com/catchy666/Coursera-Deep-Learning-Andrew-Ng/main/c4-Convolutional%20Neural%20Networks/week2/tmp-imgs/20.jpg)

## 数据扩增

计算机视觉领域的应用都需要大量的数据。当数据不够时，**数据扩增（Data Augmentation）** 就有帮助。常用的数据扩增包括：

- 镜像翻转；
- 随机裁剪；
- 色彩转换：给 R、G 和 B 三个通道上加上不同的失真值，改变图片色调。
    - **PCA 颜色增强** ：指更有针对性地对图片的 RGB 通道进行主成分分析（Principles Components Analysis，PCA），对主要的通道颜色进行增加或减少，比如说，如果你的图片呈现紫色，即主要含有红色和蓝色，绿色很少，然后 PCA 颜色增强算法就会对红色和蓝色增减很多，绿色变化相对少一点，所以使总体的颜色保持一致。

## 计算机视觉现状

通常我们的学习算法有两种知识来源：
1. 被标记的数据，就像（x，y）应用在监督学习。
2. 手工工程，有很多方法去建立一个手工工程系统，它可以是源于精心设计的特征，手工精心设计的网络体系结构或者是系统的其他组件。

当然，当数据量不够的时候，我们还有“迁移学习”这一个有效的办法。

另外，在模型研究或者竞赛方面，有一些方法能够有助于提升神经网络模型的性能：

* 集成（Ensembling）：独立地训练几个神经网络，并平均它们的输出。假设你的 7 个神经网络，它们有 7 个不同的预测，然后平均他们，这可能会让你在基准上提高 1%，2%或者更好。
* Multi-crop at test time：将数据扩增应用到测试集，对结果进行平均。

但是由于这些方法计算和内存成本较大，一般不适用于构建实际的生产项目。

