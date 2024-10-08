---
title: 信息论:什么是信息
comments: true
---

## KL散度 {#KL散度}[^1]

相对熵, 又称为KL散度. 如果我们对于同一个随机变量$X$有两个单独的概率分布$P(X)$和$Q(X)$(我们通常用$P$来表示真实的概率分布, $Q$来表示模型所预测的概率分布), 我们可以使用KL散度(Kullback-Leibler Divergence)来衡量这两个分布之间的差异. 在机器学习中, 我们需要衡量标签和预测值之间的差距, 用KL散度就刚刚好, 即$D(y||\hat{y})$.

KL散度的计算公式为: $D(p||q)=\sum_{i=1}^n p(x_i)\log_2 \frac{p(x_i)}{q(x_i)}$, 其中$n$为事件的所有可能性. KL散度的值越小, 表示$Q$分布和$P$分布越接近. 当KL散度等于$0$的时候, 意味着概率分布$Q$和概率分布$P$是相同的.

## 交叉熵 {#交叉熵}[^1]

对与上述KL散度的计算公式进行变形:

$D(p||q)=\sum_{i=1}^n p(x_i)\log_2 \frac{p(x_i)}{q(x_i)} = \sum_{i=1}^n p(x_i)\log_2 p(x_i)-\sum_{i=1}^np(x_i)\log_2 q(x_i)=-H(p(x))+[-\sum_{i=1}^n p(x_i)\log_2 q(x_i)]$

可以观察到等式的前一部分恰好就是$p$的熵的负数$-H(p(x))$, 等式的后一部分, 就是交叉熵: $G(p||q)=-\sum_{i=1}^n p(x_i)\log_2 q(x_i)$. 由于在优化的过程中, 真实的概率分布永远是不变的, 即$-H(p(x))$永远保持不变, 所以一般只需要关注交叉熵就可以了.

也就是说, 交叉熵和KL散度衡量的都是你使用一个错误的概率分布的时候付出的"额外代价". KL散度/交叉熵绝对值越大, 说明必须额外传输更多的信息才能进行正确编码数据, 因为你得到的概率分布和实际的分布不符.

## 互信息 {#互信息}[^2]

互信息, Mutual Information. 是描述一个随机变量中包含的关于另一个随机变量的信息. 用Venn图表示如下(注意中间的那个部分):

![](https://img.ricolxwz.io/9cd4d2a8fc2d099426208f752376e8b7.png)

从Venn图中, 可以看出:

- $I(X;Y)=H(X)+H(Y)-H(X,Y)$
- $I(X;Y)=H(X)-H(X|Y)$
- $I(X;Y)=H(Y)-H(Y|X)$
- $I(X;Y)=I(Y;X)$

有如下属性:

- $0\leq I(X;Y)\leq min(H(X), H(Y))$
- 在$X$和$Y$中是对称的
- $I(X;Y)=H(X)\rightarrow H(X|Y)=0$
- 是非线性的

???+ example "例子"

    可以用赌博来解释互信息.

    如果你对比赛的结果没有任何额外的信息, 那么最好的策略就是根据结果$X=x$的概率$p(x)$来分配你的赌注. 例如, 如果某个结果的$x$的概率更高, 那么你应该在这个结果中投入更多的资金. 如果有额外的信息$Y$, 最佳策略是根据$p(X|Y)$分配投资, 为什么? 因为如果你有额外的信息$Y$, 并且这个$Y$和$X$有关系, 即$Y$中包含了$X$的一些信息, 互信息大于$0$. 换句话说, 互信息 $I(X;Y)$量化了利用额外信息$Y$进行投资相比于没有额外信息时的优势.

### 自互信息

一个随机变量和自己的互信息就是它的熵.

$I(X;X)=H(X)+H(X)-H(X,X)$, 可以简化为$I(X;X)=H(X)$. 这种情况下, 互信息测量的就是变量$X$对自身提供的信息量. 因为$X$完全了解自己, 所以它对自身的自互信息就等于它的熵$H(X)$. 即这个随机变量包含这个随机变量的信息就是它自己包含的信息.

![](https://img.ricolxwz.io/944041a23df4e876dd8050b45aed57ca.png)

### KL散度和互信息的关系 {#KL散度和互信息的关系}

来看两个随机变量, 对于两个随机变量$X$和$Y$, 我们要探究这两个随机变量之间的关联关系, 即这两个随机变量是否是独立的, 即在Venn图中是否有相交的部分, 我们知道, 这个相交的程度就是互信息. 我们将会尝试将KL散度的那套理论搬到这里, 真实的概率分布$P$就是真实情况下的联合分布, 现在假设这两个变量之间是独立的, 即$p(x, y)=p(x)p(y)$. 所以KL散度$D(p(x, y)||p(x)p(y))=\sum_{x\in A_x, y\in A_y}p(x, y)\log_2 \frac{p(x, y)}{p(x)p(y)}$. 套用贝叶斯公式, $\frac{p(x|y)}{p(x)}=\frac{p(x, y)}{p(x)p(y)}$, 可以得到KL散度$D(p(x, y)||p(x)p(y))=\sum_{x\in A_x, y\in A_y}p(x, y)\log_2\frac{p(x|y)}{p(x)}$. 而这个KL散度就是我们要求的互信息, 即$I(X; Y)=D(p(x, y)||p(x)p(y))$, 因为根据KL散度的定义: KL散度衡量的是一个错误的概率分布的时候付出的"额外代价", 即我们假设$X$和$Y$独立的"惩罚", 我们的错误的概率分布假设的是它们之间没有关系, 所以计算出来的KL散度就是那部分"额外代价", 即互信息. 若这两个变量之间没有关系, "额外代价"等于$0$, 互信息等于$0$, 这两个量之间是独立的. 

???+ tip "Tip"

    我们也可以这么来理解KL散度和互信息: 如果$p(x, y)/p(x)p(y)$这个值较大的话, 说明两个变量之间的依赖程度较高, 然后在乘上一个这种情况发生的概率$p(x, y)$, 做一个加权平均, 就得到了一个总的依赖性. 

### 点互信息

上面我们讨论的是一个两个随机变量的互信息, 而现在我们要讨论的是两个随机变量结果的互信息, 即点互信息, 用$i(x;y)$表示. 它的含义是某个随机变量的结果中包含的关于另一个随机变量的结果的信息. 

它的属性和随机变量的差不多:

- $i(x;y)=h(x)+h(y)-h(x,y)$
- $i(x;y)=h(x)-h(x|y)$
- $i(x;y)=h(y)-h(y|x)$
- $i(x;y)=\log_2 \frac{p(x|y)}{p(x)}$
- $I(X;Y)=<i(x;y)>$

随机变量的互信息一定大于等于$0$, 但是某个随机变量结果的互信息可以大于$0$, 也可以小于$0$:

- $i(x;y)>0$说明$y$的发生增加了$x$发生的可能
- $i(x;y)<0$说明$y$的发生减小了$x$发生的可能
- $i(x;y)=0$说明$y$的发生和$x$没有关系, 没有增加也没有减少$x$发生的可能

### 条件互信息

条件互信息$I(X;Y|Z)$是给出另一个随机变量$Z$的分布的前提下, 一个随机变量包含另一个随机变量的信息.

属性如下:

- $I(X;Y|Z)=H(X|Z)+H(Y|Z)-H(X,Y|Z)$
- $I(X;Y|Z)=H(X|Z)-H(X|Y,Z)$
- $I(X;Y|Z)=H(Y|Z)-H(Y|X,Z)$
- $I(X;Y|Z)=I(Y;X|Z)$
- $I(X;Y|Z)=I(X;Y,Z)-I(X;Z)$
- $0\leq I(X;Y|Z)\leq min(H(X|Z), H(Y|Z))$
- $I(X;Y|Z)=H(X|Z)\rightarrow H(X|Y,Z)=0$

用Venn图可以表示如下:

![](https://img.ricolxwz.io/b9216e1ae398ea0593d69d0331a7f719.png)

使用KL散度可以解释为:

- $I(X;Y|Z)=\sum_{x\in A_x, y\in A_y, z\in A_z}p(x, y, z)\log_2\frac{p(x, y|z)}{p(x|z)p(y|z)}$
- $I(X;Y|Z)=\sum_{x\in A_x, y\in A_y, z\in A_z}p(x, y, z)\log_2\frac{p(x|y, z)}{p(x|z)}$
- $I(X;Y|Z)=D(p(x, y|z)||p(x|z)p(y|z))$

$CMI$是衡量两个随机变量$X$和$Y$在给定的第三个变量$Z$条件下的依赖程度. 如果$X$和$Y$在$Z$的条件下是独立的, 那么$CMI(X;Y|Z)=0$, 如果这个假设是错误的, 那么我们就要付出额外的代价, 这个就是条件互信息, 即"惩罚". 所以, 参考[KL散度和互信息的关系](#KL散度和互信息的关系). 我们可以假设在$Z$发生的条件下, $X$和$Y$是独立的, 即$p(x,y|z)=p(x|z)p(y|z)$, 那么我们要计算的就是真实的概率分布$p(x,y|z)$和我们在假设条件下概率分布的差异, 即KL散度$D(p(x, y|z)||p(x|z)p(y|z))=\sum_{x\in A_x, y\in A_y, z\in A_z}p(x,y,z)\log_2\frac{p(x, y|z)}{p(x|z)p(y|z)}$. 细心的同学可以发现, 这个公式中间是$p(x,y,z)$而不是$p(x,y|z)$, 这是因为$z$的取值会对$x$和$y$的依赖程度造成影响, 所以我们要做的就是取一个加权平均数, 权就是$z$的概率$p(z)$, 也就是说不同的$z$对互信息的贡献是不同的, 需要取一个平均值: $I(X;Y|Z)=\sum_{z\in A_z}p(z)I(X;Y|Z=z)$.

使用统计学可以解释为:

- $I(X;Y|Z)=\sum_{x\in A_x, y\in A_y, z\in A_z}p(x, y, z)\log_2\frac{p(x|y, z)}{p(x|z)}$

#### 点互信息

某个随机变量结果的条件互信息, 即条件点互信息:

- $i(x;y|z)=h(x|z)+h(y|z)-h(x,y|z)$
- $i(x;y|z)=h(x|z)-h(x|y,z)$
- $i(x;y|z)=h(y|z)-h(y|x,z)$
- $i(x;y|z)=\log_2\frac{p(x|y,z)}{p(x|z)}$
- $I(X;Y|Z)=<i(x;y|z)>$

#### 冗余和协同

条件互信息实际上就是在互信息的基础上加了一个$Z$的条件. 那么如果我们在互信息上施加$Z$这个条件, 与$I(X;Y)$相比, 会发生什么变化呢, 是增大, 还是减少, 还是不变? 

- 无影响: 如果$X, Y, Z$是相互独立的话, 那么$I(X;Y)$和$I(X;Y|Z)$是没有区别的
- 冗余: $I(X;Y|Z)$相比于$I(X;Y)$会减少, 因为$Z$可以解释掉一些$Y$中关于$X$的信息, 导致$Y$对$X$的贡献减少. 例如, 如果$X$, $Y$和$Z$是独立同分布的随机比特, 即$X=Y=Z$, 则$I(X;Y|Z)=0$. 由于$X$和$Y$是相同的, 所以它们共享全部的信息量, 这就是为什么$I(X;Y)=1$; 当我们引入条件$Z$的时候, 由于$Z$已经完全描述了$X$和$Y$, 所以在已知$Z$的情况下, $Y$不再提供任何关于$X$的新信息, 即$I(X;Y|Z)=0$
- 协同: $I(X;Y|Z)$相比于$I(X;Y)$会增加. 这种情况发生在$Y$和$Z$一起提供关于$X$的信息, 单独知道任何一个都无法获取到关于$X$的信息. 例如, $X=Y\oplus Z$, $Y$和$Z$是独立同分布的随机比特, $\oplus$表示异或运算, 对于两个二进制位, 如果不同, 结果为$1$, 否则为$0$. $I(X;Y)=0$, 因为仅仅知道$Y$的值, 无法获取到关于$X$的任何信息. $I(X;Y|Z)=1$, 因为在已知$Z$的情况下, $Y$提供了关于$X$的全部信息, 也就是说一旦我们知道了$Z$的值, $Y$的值就完全决定了$X$的值

#### 链式法则

从Venn图中, 我们可以清晰的观察到链式法则.

- $I(X;Y,Z)=I(X;Y)+I(X;Z|Y)$
- $I(X;Y,Z)=I(X;Z)+I(X;Y|Z)$

更一般的形式: $I(X_1, X_2, ..., X_n; Y)=\sum_{i=1}^n I(X_i;Y|X_1, ..., X_{i-1})$. 这就是一个信息回归. 

### 点互信息

点互信息衡量的是在给定$y$的条件下, 观察到$x$这一具体值提供了多少信息, 公式为$i(x;y)=\log_2\frac{p(x|y)}{p(x)}$. 这里, $p(x|y)$是在给定$y$条件下出现$x$的概率, $p(x)$是$x$自身的概率. 

随机变量之间的互信息可以通过点互信息取期望得到. 即$I(X;Y)=<i(x;y)>$. 

点互信息满足4个公理:

1. 可微性: $i(x;y)$对$p(x)$和$p(y)$是可微的
2. 条件形式: $i(x;y)$和$i(x;y|z)$的形式相同, 即它们两的计算公式形式相同. 不同的是$i(x;y|z)$中所有的概率都基于$z$条件化了
3. 可加性: 即$i(x;y;z)=i(x;z)+i(x;y|z)$
4. 独立集合的分离: 对于独立的集合$(x,y)$和$(u,v)$, 如果$p(x,y,u,v)=p(x,y)p(u,v)$, 那么点互信息也满足这种分离性, 即$i(x,u;y,v)=i(x;y)+i(u;v)$

### 应用

互信息可以用于:

- 变量之间的关系检测
- 机器学习中的特征选择, 比如决策树, 互信息和信息增量是一个概念
- ...

[^1]: 一文搞懂交叉熵在机器学习中的使用，透彻理解交叉熵背后的直觉-CSDN博客. (n.d.). Retrieved August 14, 2024, from https://blog.csdn.net/tsyccnh/article/details/79163834
[^2]: 什么是互信息Mutual information-CSDN博客. (n.d.). Retrieved August 14, 2024, from https://blog.csdn.net/qq_40210586/article/details/131699236