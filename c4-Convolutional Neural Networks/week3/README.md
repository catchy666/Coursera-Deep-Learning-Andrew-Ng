<h1 align="center">Week3: 目标检测</h1>



## 目录 

* [笔记](#笔记)
   * [目录](#目录)
   * [目标定位(Object localization)](#目标定位object-localization)
   * [特征点检测(Landmark detection)](#特征点检测landmark-detection)
   * [目标检测(Object detection)](#目标检测object-detection)
   * [卷积的滑动窗口实现(Convolutional implementation of sliding windows)](#卷积的滑动窗口实现convolutional-implementation-of-sliding-windows)
   * [Bounding Box预测(Bounding box predictions)](#bounding-box预测bounding-box-predictions)
   * [交并比(Intersection over union)](#交并比-intersection-over-union)
   * [非极大值抑制(Non-max suppression)](#非极大值抑制non-max-suppression)
   * [Anchor Boxes](#anchor-boxes)
   * [YOLO 算法(Putting it together: YOLO algorithm)](#yolo-算法putting-it-together-yolo-algorithm)
   * [候选区域(选修)(Region proposals (Optional))](#候选区域选修region-proposals-optional)

目标检测是计算机视觉领域中一个新兴的应用方向，其任务是对输入图像进行分类的同时，检测图像中是否包含某些目标，并对他们准确定位并标识。

-----

## 目标定位(Object localization)

定位分类问题不仅要求判断出图片中物体的种类，还要在图片中标记出它的具体位置，用**边界框（Bounding Box）**把物体圈起来。一般来说，定位分类问题通常只有一个较大的对象位于图片中间位置；而在对象检测问题中，图片可以含有多个对象，甚至单张图片中会有多个不同分类的对象。

![](https://raw.githubusercontent.com/catchy666/Coursera-Deep-Learning-Andrew-Ng/main/c4-Convolutional%20Neural%20Networks/week3/tmp-imgs/01.jpg)

- 为了定位图片中汽车的位置，可以让神经网络多输出 4 个数字，标记为 $b_x$、$b_y$、$b_h$、$b_w$。将图片左上角标记为 (0, 0)，右下角标记为 (1, 1)，则有：

    * 红色方框的中心点：($b_x$，$b_y$)
    * 边界框的高度：$b_h$
    * 边界框的宽度：$b_w$

- 同时还输出$P_c$，表示矩形区域是目标的概率，数值在0-1之间。

因此，训练集不仅包含对象分类标签，还包含表示边界框的四个数字。定义目标标签$ Y$如下：

$$
\left[\begin{matrix}P_c\\\ b_x\\\ b_y\\\ b_h\\\ b_w\\\ c_1\\\ c_2\\\ c_3\end{matrix}\right]
,
$$
则有：
$$
when \ P_c=1:\left[\begin{matrix}1\\\ b_x\\\ b_y\\\ b_h\\\ b_w\\\ c_1\\\ c_2\\\ c_3\end{matrix}\right] ,
$$
其中，$c_n$表示存在第$n$个种类的概率；若$P_c=0$，表示没有检测到目标，则输出label后面的7个参数都可以忽略(用$ ? $来表示)。
$$
when \ P_c=0:\left[\begin{matrix}0\\\ ?\\\ ?\\\ ?\\\ ?\\\ ?\\\ ?\\\ ?\end{matrix}\right]
$$
损失函数可以表示为 $L(\hat y, y)$，如果使用平方误差形式，对于不同的 $P_c$有不同的损失函数（注意下标 $i$指标签的第 $i$个值）：

1. $P_c=1$，即$y_1=1$：

    $L(\hat y,y)=(\hat y_1-y_1)^2+(\hat y_2-y_2)^2+\cdots+(\hat y_8-y_8)^2$

    损失值就是不同元素的平方误差和。

2. $P_c=0$，即$y_1=0$：

    $L(\hat y,y)=(\hat y_1-y_1)^2$

    *对于这种情况，不用考虑其它元素，只需要关注神经网络输出的准确度即可。*

除了使用平方误差，也可以使用逻辑回归损失函数，类标签 $c_1,c_2,c_3$ 也可以通过 Softmax 输出。相比较而言，平方误差已经能够取得比较好的效果。

## 特征点检测(Landmark detection)

神经网络可以像标识目标的中心点位置那样，通过输出图片上的特征点，来实现对目标特征的识别。在标签中，这些特征点以多个二维坐标的形式表示。

举个例子：假设需要定位一张人脸图像，同时检测其中的64个特征点，这些点可以帮助我们定位眼睛、嘴巴等人脸特征。

![](https://raw.githubusercontent.com/catchy666/Coursera-Deep-Learning-Andrew-Ng/main/c4-Convolutional%20Neural%20Networks/week3/tmp-imgs/02.jpg)

- 具体的做法是：准备一个卷积网络和一些特征集，将人脸图片输入卷积网络，输出1或0（1表示有人脸，0表示没有人脸）然后输出$(l_{1x},l_{1y}),\dots,  (l_{64x},l_{64y})$。这里用$l$代表一个特征，这里有129个输出单元，其中1表示图片中有人脸，因为有64个特征，64×2=128，所以最终输出128+1=129个单元，由此实现对图片的人脸检测和定位。

通过检测人脸特征点可以进行情绪分类与判断，或者应用于 AR 领域等等。也可以透过检测姿态特征点来进行人体姿态检测。

## 目标检测(Object detection)

想要实现目标检测，可以采用 **基于滑动窗口的目标检测（Sliding Windows Detection）** 算法。

该算法的步骤如下：

![](https://raw.githubusercontent.com/catchy666/Coursera-Deep-Learning-Andrew-Ng/main/c4-Convolutional%20Neural%20Networks/week3/tmp-imgs/03.jpg)

1. 训练集上搜集相应的各种目标图片和非目标图片，样本图片要求尺寸较小，相应目标居于图片中心位置并基本占据整张图片。

	- 训练集X：将有汽车的图片进行适当的剪切，剪切成整张几乎都被汽车占据的小图或者没有汽车的小图
	- 训练集Y：对X中的图片进行标注，有汽车的标注1，没有汽车的标注0

2. 使用训练集构建 CNN 模型，使得模型有较高的识别率。
3. 选择大小适宜的窗口与合适的固定步幅，对测试图片进行从左到右、从上倒下的滑动遍历。每个窗口区域使用已经训练好的 CNN 模型进行识别判断。
4. 可以选择更大的窗口，然后重复第三步的操作。

![](https://raw.githubusercontent.com/catchy666/Coursera-Deep-Learning-Andrew-Ng/main/c4-Convolutional%20Neural%20Networks/week3/tmp-imgs/04.jpg)

滑动窗算法的**优点**是原理简单，且不需要人为选定目标区域（检测出目标的滑动窗即为目标区域）;**缺点**是其滑动窗的大小和步进长度都需要人为直观设定。滑动窗过小或过大，步进长度过大均会降低目标检测正确率。另外，**每次滑动都要进行一次 CNN 网络计算，如果滑动窗口和步幅较小，计算成本往往很大。**

所以，滑动窗口目标检测算法虽然简单，但是性能不佳，效率较低。

## 卷积的滑动窗口实现(Convolutional implementation of sliding windows)

相比从较大图片多次截取，在卷积层上应用滑动窗口目标检测算法可以提高运行速度，节约重复运算成本。

那么滑动窗口算法卷积实现的**第一步**就是将全连接层转变成为卷积层，如下图所示：

![](https://raw.githubusercontent.com/catchy666/Coursera-Deep-Learning-Andrew-Ng/main/c4-Convolutional%20Neural%20Networks/week3/tmp-imgs/05.jpg)

全连接层转变成卷积层的操作很简单，只需要使用与上一层尺寸一致的滤波器进行卷积运算。最终得到的输出层维度是$1 \times 1 \times 4$，代表4类输出值。

1. 事实上，我们不用像上一节中那样，自己去滑动窗口截取图片的一小部分然后检测，卷积这个操作就可以实现滑动窗口。

2. 我们假设输入的图像是$16 \times 16 \times 3$，而窗口大小是$14 \times 14 \times 3$，我们要做的是把蓝色区域输入卷积网络，生成0或1分类；接着向右滑动2个元素，形成的新区域输入卷积网络，生成0或1分类，然后接着滑动，重复操作。<u>我们在$16 \times 16\times 3$的图像上卷积了4次，输出了4个标签，我们会发现这4次卷积里很多计算是重复的</u>。

3. 而实际上，直接对这个$16 \times 16 \times 3$的图像进行卷积，蓝色区域就是我们初始时用来卷积的第一块区域，到最后它变成了$2\times 2\times 4$的左上角那一块。我们可以看到最后输出的$2 \times 2$，刚好就是4个输出，对应我们上面说的输出4个标签。

同样的，当图片大小是$28 \times 28 \times 3$的时候，CNN网络得到的输出层为$8 \times 8 \times 4$，共64个窗口结果。

![](https://raw.githubusercontent.com/catchy666/Coursera-Deep-Learning-Andrew-Ng/main/c4-Convolutional%20Neural%20Networks/week3/tmp-imgs/06.jpg)
*蓝色窗口表示卷积窗，黄色的表示图片*

**运行速度提高的原理：**在滑动窗口的过程中，需要重复进行 CNN 正向计算。因此，不需要将输入图片分割成多个子集，分别执行向前传播，而是将它们作为一张图片输入给卷积网络进行一次 CNN 正向计算。这样，公共区域的计算可以共享，以降低运算成本。

相关论文：[**Sermanet et al., 2014. OverFeat: Integrated Recognition, Localization and Detection using Convolutional Networks**](https://arxiv.org/pdf/1312.6229.pdf)

-----

## Bounding Box预测(Bounding box predictions)

卷积方式实现的滑动窗口算法，使得在预测时计算的效率大大提高。但是其存在的问题是：不能输出最精准的边界框（Bounding Box）。

假设窗口滑动到蓝色方框的地方，这不是一个能够完美匹配汽车位置的窗口，所以我们需要寻找更加精确的边界框。

![](https://raw.githubusercontent.com/catchy666/Coursera-Deep-Learning-Andrew-Ng/main/c4-Convolutional%20Neural%20Networks/week3/tmp-imgs/07.jpg)

**YOLO（You Only Look Once）算法**可以解决这类问题，得到更精确的边框。

YOLO算法首先将原始图片分割成$n\times n$网格grid，每个网格代表一块区域。为简化说明，下图中将图片分成$3 \times 3$网格。

![](https://raw.githubusercontent.com/catchy666/Coursera-Deep-Learning-Andrew-Ng/main/c4-Convolutional%20Neural%20Networks/week3/tmp-imgs/08.jpg)

然后，利用上一节卷积形式实现滑动窗口算法的思想，对该原始图片构建CNN网络，得到的的输出层维度为$3 \times 3 \times 8$。其中，$3 \times 3$对应9个网格，每个网格的输出包含8个元素：
$$
y=\left[
\begin{matrix}
P_c\\\ b_x\\\ b_y\\\ b_h\\\ b_w\\\ c_1\\\ c_2\\\ c_3
\end{matrix}
\right]
$$

如果目标中心坐标 $(b_x,b_y)$ 不在当前网格内，则当前网格$P_c=0$；相反，则当前网格$P_c=1$（即只看中心坐标是否在当前网格内）。判断有目标的网格中， $b_x,b_y,b_h,b_w$ 限定了目标区域。

- 值得注意的是，当前网格左上角坐标设定为$(0, 0)$，右下角坐标设定为$(1, 1)$， $(b_x,b_y)$ 表示坐标值，范围限定在$[0,1]$之间，
- 但是 $b_h,b_w$ 表示比例值，可以大于 1。因为目标可能超出该网格，横跨多个区域.

**总结：**

- 首先这和图像分类和定位算法非常像，就是它显式地输出边界框坐标，所以这能让神经网络输出边界框，可以具有任意宽高比，并且能输出更精确的坐标，不会受到滑窗分类器的步长大小限制。

- 其次，这是一个卷积实现，你并没有在$3 \times 3$网格上跑9次算法，或者，如果用的是$19 \times 19$的网格，所以不需要让同一个算法跑361次。相反，这是单次卷积实现，**效率很高，甚至可以达到实时识别**。

## 交并比(Intersection over union)

**交并比（IoU, Intersection Over Union）**函数用于评价目标检测算法，它计算预测边框和实际边框交集（I）与并集（U）之比：

![](https://raw.githubusercontent.com/catchy666/Coursera-Deep-Learning-Andrew-Ng/main/c4-Convolutional%20Neural%20Networks/week3/tmp-imgs/09.jpg)

如上图所示，红色方框为真实目标区域，蓝色方框为检测目标区域。两块区域的交集为绿色部分，并集为紫色部分。蓝色方框与红色方框的接近程度可以用IoU比值来定义：

$$
IoU=\frac{I}{U}
$$

IoU 的值在 0～1 之间，且越接近 1 表示目标的定位越准确。

IoU $\geq 0.5$ 时，一般可以认为预测边框是正确的，当然也可以更加严格地要求一个更高的阈值。

## 非极大值抑制(Non-max suppression)

对于汽车目标检测的例子中，我们将图片分成很多精细的格子。最终预测输出的结果中，可能会有相邻的多个格子里均检测出都具有同一个对象。

![](https://raw.githubusercontent.com/catchy666/Coursera-Deep-Learning-Andrew-Ng/main/c4-Convolutional%20Neural%20Networks/week3/tmp-imgs/10.png)

对于每个格子都运行一次，所以1号格子可能会认为这辆车中点应该在格子内部，这几个格子（编号2、3）也会这么认为。对于左边的车子也一样，4号格子会认为它里面有车，格子（编号5）和这个格子（编号6）也会这么认为。

那如何判断哪个网格最为准确呢？方法是使用非极大值抑制算法。

![](https://raw.githubusercontent.com/catchy666/Coursera-Deep-Learning-Andrew-Ng/main/c4-Convolutional%20Neural%20Networks/week3/tmp-imgs/11.jpg)

- 将包含目标中心坐标的可信度 $P_c$小于阈值（例如 0.6）的网格丢弃；
- 选取拥有最大 $P_c$ 的网格；
- 分别计算该网格和其他所有网格的 IoU，将 IoU 超过预设阈值的网格丢弃；
- 重复第 2~3 步，直到不存在未处理的网格。

上述步骤适用于单类别目标检测。进行多个类别目标检测时，对于每个类别，应该单独做一次非极大值抑制。

## Anchor Boxes

到目前为止，我们讨论的情况都是一个网格只检测一个对象。如果要将算法运用在多目标检测上，需要用到 Anchor Boxes。例如一个人站在一辆车前面，该如何使用YOLO算法进行检测呢？

方法就是一个网格的标签中将包含多个 Anchor Box，相当于存在多个用以标识不同目标的边框。



![](https://raw.githubusercontent.com/catchy666/Coursera-Deep-Learning-Andrew-Ng/main/c4-Convolutional%20Neural%20Networks/week3/tmp-imgs/12.jpg)

在上图示例中，我们希望同时检测人和汽车。因此，每个网格的的标签中含有两个 Anchor Box（如Anchor box 1检测人，Anchor box 2检测车）。输出的标签结果大小从 3×3×8 变为 3×3×16。每个Anchor box都有一个$P_c$值，若两个$P_c$都大于预设阈值，则说明检测到了两个目标。

如下面的图片，里面有行人和汽车，在经过了极大值抑制操作之后，最后保留了两个边界框（Bounding Box）。对于行人形状更像Anchor box 1，汽车形状更像Anchor box 2，所以我们将人和汽车分配到不同的输出位置。具体分配，对应下图颜色。

![](https://raw.githubusercontent.com/catchy666/Coursera-Deep-Learning-Andrew-Ng/main/c4-Convolutional%20Neural%20Networks/week3/tmp-imgs/13.jpg)

当然，如果格子中只有汽车的时候，我们使用了两个Anchor box，那么此时我们的目标向量就成为：

$y_{i} = \left[ 0\ ?\ ?\ ?\ ?\ ?\ ?\ ?\ 1\ b_{x}\ b_{y}\ b_{h}\ b_{w}\ 0\ 1\ 0\right]^T$



- Anchor Boxes 也有局限性，对于同一网格有三个及以上目标，或者两个目标的 Anchor Box 高度重合的情况处理不好。

- Anchor Box 的形状一般通过人工选取。高级一点的方法是用 k-means 将两类对象形状聚类，选择最具代表性的 Anchor Box。

## YOLO 算法(Putting it together: YOLO algorithm)

这节将上述关于YOLO算法组件组装在一起构成YOLO对象检测算法。

假设我们要在图片中检测三种目标：行人、汽车和摩托车，同时使用两种不同的Anchor box。

1. 构造训练集：
    - 根据工程目标，将训练集做如下规划。
    - 输入X：同样大小的完整图片；
    - 目标Y：使用 $3\times3$ 网格划分，输出大小 $3\times3\times2\times8$(其中3 × 3表示3×3个网格，2是anchor box的数量，8是向量维度) ，或者 $3\times3\times16$。
    - 对不同格子中的小图，定义目标输出向量Y，如下图示例。
        - 对于格子1的目标y就是这样的$y = \left[ 0\ ?\ ?\ ?\ ?\ ?\ ?\ ?\ 0\ ?\ ?\ ?\ ?\ ?\ ?\ ?\right]^T$。
        - 而对于格子2的目标y则应该是这样：$y = \left[ 0\ ?\ ?\ ?\ ?\ ?\ ?\ ?\ 1\ b_{x}\ b_{y}\ b_{h}\ b_{w}\ 0\ 1\ 0\right]^T$。
        - 训练集中，对于车子有这样一个边界框（编号3），水平方向更长一点。所以如果这是你的anchor box，这是anchor box 1（编号4），这是anchor box 2（编号5），然后红框和anchor box 2的交并比更高，那么车子就和向量的下半部分相关。要注意，这里和anchor box 1有关的$P_c$是0，剩下这些分量都是don’t care-s，然后你的第二个 ，然后你要用这些($b_x,b_y,b_h,b_w$)来指定红边界框的位置

![](https://raw.githubusercontent.com/catchy666/Coursera-Deep-Learning-Andrew-Ng/main/c4-Convolutional%20Neural%20Networks/week3/tmp-imgs/14.png)

2. 模型预测：
    - 输入与训练集中相同大小的图片，同时得到每个格子中不同的输出结果： $3\times3\times2\times8$ 。
    - 输出的预测值，以下图为例：
        - 对于左上的格子（编号1）对应输出预测y（编号3）
        - 对于中下的格子（编号2）对应输出预测y（编号4）

![](https://raw.githubusercontent.com/catchy666/Coursera-Deep-Learning-Andrew-Ng/main/c4-Convolutional%20Neural%20Networks/week3/tmp-imgs/15.png)

3. 运行非最大值抑制（NMS）(为展示效果，换一张复杂的图)：

    - （编号1）假设使用了2个Anchor box，那么对于每一个网格，我们都会得到预测输出的2个bounding boxes，其中一个$P_c$比较高；
    - （编号2）抛弃概率$P_c$值低的预测bounding boxes；
    - （编号3）对每个对象（如行人、汽车、摩托车）分别使用NMS算法得到最终的预测边界框。

![](https://raw.githubusercontent.com/catchy666/Coursera-Deep-Learning-Andrew-Ng/main/c4-Convolutional%20Neural%20Networks/week3/tmp-imgs/16.png)

## 候选区域(选修)(Region proposals (Optional))

前面介绍的滑动窗口目标检测算法对一些明显没有目标的区域也进行了扫描，这降低了算法的运行效率。为了解决这个问题，**R-CNN（Region CNN，带区域的 CNN）**被提出。通过对输入图片运行**图像分割算法**，在不同的色块上找出**候选区域（Region Proposal）**，就只需要在这些区域上运行分类器。

![](https://raw.githubusercontent.com/catchy666/Coursera-Deep-Learning-Andrew-Ng/main/c4-Convolutional%20Neural%20Networks/week3/tmp-imgs/17.png)

R-CNN 的缺点是运行速度很慢，所以有一系列后续研究工作改进。例如 Fast R-CNN（与基于卷积的滑动窗口实现相似，但得到候选区域的聚类步骤依然很慢）、Faster R-CNN（使用卷积对图片进行分割）。不过大多数时候还是比 YOLO 算法慢。

相关论文：

- R-CNN：[Girshik et al., 2013. Rich feature hierarchies for accurate object detection and semantic segmentation](https://arxiv.org/pdf/1311.2524.pdf)
- Fast R-CNN：[Girshik, 2015. Fast R-CNN](https://arxiv.org/pdf/1504.08083.pdf)
- Faster R-CNN：[Ren et al., 2016. Faster R-CNN: Towards real-time object detection with region proposal networks](https://arxiv.org/pdf/1506.01497v3.pdf)

- [Joseph Redmon, Santosh Divvala, Ross Girshick, Ali Farhadi - You Only Look Once: Unified, Real-Time Object Detection (2015)](https://arxiv.org/abs/1506.02640)

- [Joseph Redmon, Ali Farhadi - YOLO9000: Better, Faster, Stronger (2016)](https://arxiv.org/abs/1612.08242)

- [Allan Zelener - YAD2K: Yet Another Darknet 2 Keras](https://github.com/allanzelener/YAD2K)

- [The official YOLO website](https://pjreddie.com/darknet/yolo/)