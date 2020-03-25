# 一、基本框架
Transformer模型是Google在2017年的论文《Attention  is all you need》中提出的一种模型。Transformer之前的Seq2Seq的模型中，Encoder和Decoder中的基本单元结构是RNN系列（如LSTM，GRU等）的单元。但在Transformer中并没有使用这些单元结构。

首先来说一下transformer和LSTM的最大区别, 就是LSTM的训练是迭代的, 是一个接一个字的来, 当前这个字过完LSTM单元, 才可以进下一个字, 而transformer的训练是并行了, 就是所有字是全部同时训练的, 这样就大大加快了计算效率, transformer使用了位置嵌入(𝑝𝑜𝑠𝑖𝑡𝑖𝑜𝑛𝑎𝑙 𝑒𝑛𝑐𝑜𝑑𝑖𝑛𝑔)来理解语言的顺序, 使用自注意力机制和全连接层来进行计算, 这些后面都会详细讲解.

transformer模型主要分为两大部分, 分别是编码器和解码器, 编码器负责把自然语言序列映射成为隐藏层(下图中第2步用九宫格比喻的部分), 含有自然语言序列的数学表达. 然后解码器把隐藏层再映射为自然语言序列, 从而使我们可以解决各种问题, 如情感分类, 命名实体识别, 语义关系抽取, 摘要生成, 机器翻译等等, 下面我们简单说一下下图的每一步都做了什么:
<img src="https://img-blog.csdnimg.cn/20200306210839699.png" width = 50% height = 50%   />

1、输入自然语言序列到编码器: Why do we work?(为什么要工作);
2、编码器输出编码后的隐藏层, 再输入到解码器;
3、输入<𝑠𝑡𝑎𝑟𝑡>(起始)符号到解码器;
4、得到第一个字"为";
5、将得到的第一个字"为"落下来再输入到解码器;
6、得到第二个字"什";
7、将得到的第二字再落下来, 直到解码器输出<𝑒𝑛𝑑>(终止符), 即序列生成完成.

# 二、编码结构
编码过程是一个把自然语言序列映射称为相关数学表达的一个过程，属于上游任务。
上游任务：将自然语言序列编码为数学表达；
下游任务：将编码映射为自然语言序列如情感分类, 命名实体识别, 语义关系抽取, 摘要生成, 机器翻译等等。

Transformer的编码结构如下，可以分为4个部分，分别是：
1、𝑝𝑜𝑠𝑖𝑡𝑖𝑜𝑛𝑎𝑙 𝑒𝑛𝑐𝑜𝑑𝑖𝑛𝑔, 即位置嵌入(或位置编码);
2、𝑠𝑒𝑙𝑓 𝑎𝑡𝑡𝑒𝑛𝑡𝑖𝑜𝑛 , 自注意力机制;
3、Add & Norm 
4、Feed Forward.

<img src="https://img-blog.csdnimg.cn/20200306211453910.png" width = 70% height = 70%   />

## 2.1 、𝑝𝑜𝑠𝑖𝑡𝑖𝑜𝑛𝑎𝑙 𝑒𝑛𝑐𝑜𝑑𝑖𝑛𝑔
transformer模型没有循环神经网络的迭代操作，所以必须提供每个字的位置信息给transformer, 这样才能识别出语言中的顺序关系。

**但要怎么编码呢？**

一种简单的想法是直接按顺序编码：E=pos=0,1,2,….,T-1  ，但这样会导致数值过大、干扰字嵌入结果和模型 。
那就给上面的编码统一除以长度，变成：E=pos/(T-1)，这样就到了[0,1]区间之内，但也不行，因为这样的话长短文本的编码步长就会存在较大的差异。

经过前面2种实验，我们发现位置编码有两个要求：
1、要体现不同位置的编码区别；
2、编码的差异不应该依赖于文本的长度。


在这里论文中使用了$sine$和$cosine$函数的线性变换来提供给模型位置信息:   
$$PE_{(pos,2i)} = sin(pos / 10000^{2i/d_{\text{model}}})$$

$$PE_{(pos,2i+1)} = cos(pos / 10000^{2i/d_{\text{model}}})$$

上式中$pos$指的是句中字的位置, 取值范围是$[0, \ max \ sequence \ length)$, $i$指的是词向量的维度, 取值范围是$[0, \ embedding \ dimension)$。

上面有$sin$和$cos$一组公式, 也就是对应着$embedding \ dimension$维度的一组奇数和偶数的序号的维度, 例如$0, 1$一组, $2, 3$一组, 分别用上面的$sin$和$cos$函数做处理, 从而产生不同的周期性变化, 而位置嵌入在$embedding \ dimension$维度上随着维度序号增大, 周期变化会越来越慢, 而产生一种包含位置信息的纹理。
![在这里插入图片描述](https://img-blog.csdnimg.cn/20200306211927450.png)
上图展示了编码维度和语句长度之间的关系。

## 2.2 、self attention机制
Self-Attention 的 Query=Key=Value，即 Q，K，V 三个矩阵都来自同一个输入，而 Attention 计算过程如何呢？
![在这里插入图片描述](https://img-blog.csdnimg.cn/20200306212959524.jpg)
Attention 机制实质上就是一个寻址过程，通过给定一个任务相关的查询 Query 向量 Q，通过计算与 Key 的注意力分布并附加在 Value 上，从而计算 Attention Value，这个过程实际上是 Attention 缓解神经网络复杂度的体现，不需要将所有的 N 个输入都输入到神经网络进行计算，而是选择一些与任务相关的信息输入神经网络，与 RNN 中的门控机制思想类似。


Attention 机制实质上就是一个寻址过程，通过给定一个任务相关的查询 Query 向量 Q，通过计算与 Key 的注意力分布并附加在 Value 上，从而计算 Attention Value，这个过程实际上是 Attention 缓解神经网络复杂度的体现，不需要将所有的 N 个输入都输入到神经网络进行计算，而是选择一些与任务相关的信息输入神经网络，与 RNN 中的门控机制思想类似。

Attention 机制计算过程大致可以分成三步：

>① 信息输入：将 Q，K，V 输入模型
>用 $X=[x_1,x_2,\cdots,x_n]$表示输入权重向量
>
>② 计算注意力分布 α：通过计算 Q 和 K 进行点积计算相关度，并通过 softmax 计算分数
>令$Q=K=V=X$，通过 softmax 计算注意力权重，$α_i=softmax(s(k_i,q))=softmax(s(x_i, q))$
>我们将$α_i$称之为注意力概率分布，$s(x_i, q)$ 为注意力打分机制，常见的有如下几种： 
>加性模型：$s(x_i,q)=v^Ttanh(Wx_i+Uq)$
>点积模型：$s(x_i,q)=x_i^Tq$
>缩放点积模型：$s(x_i,q)={x_i^Tq}/\sqrt{d_k}$
>双线性模型：$s(x_i,q)=x_i^TWq$
>③ 信息加权平均：注意力分布 $α_i$来解释在上下文查询$q_i$时，第 $i$ 个信息受关注程度。
>$att(q,X)=\sum_{i=1}^N{α_iX_i}$

上面讲述了 Attention 的通用计算过程，也讲述了注意力分数计算的多种选择，那么在 Transformer 中，采用哪种呢？答案就是：Scaled Dot-Product Attention
<img src="https://img-blog.csdnimg.cn/2020030621384969.png" width = 60% height = 60%   />
运算的过程可以表示成下图形式：
<img src="https://img-blog.csdnimg.cn/2020030621402344.png" width = 60% height = 60%   />

如上图展示的一样，我们在得到位置编码后，会生成[batch_size,seq_len,embed_dim]形式的矩阵，假设embed_dim=9，我们拿出来1个具有6个字符(seq_len=6)的语句，它的编码矩阵如上图第一个矩阵所示。

可以看出我们把它根据3个不同权重的矩阵线性的转化为了3个矩阵：Q、K、V。
设置head=3，然后每个矩阵接着继续划分为3个小矩阵（9/3=3），然后使用Q的一个小矩阵乘以一个K的小矩阵的转置，然后经过softmax归一化之后与V的小矩阵相乘。(图中未展示与V相乘的过程)
用于计算注意力权重的等式为：
$\Large{Attention(Q, K, V) = softmax_k(\frac{QK^T}{\sqrt{d_k}}) V}$

<img src="https://img-blog.csdnimg.cn/20200306214237191.png" width = 60% height = 60%   />

## 2.3、Add&Norm 
1). **残差连接**:   
我们在上一步得到了经过注意力矩阵加权之后的$V$, 也就是$Attention(Q, \ K, \ V)$, 我们对它进行一下转置, 使其和$X_{embedding}$的维度一致, 也就是$[batch \ size, \ sequence \ length, \ embedding \ dimension]$, 然后把他们加起来做残差连接, 直接进行元素相加, 因为他们的维度一致:   
$$X_{embedding} + Attention(Q, \ K, \ V)$$
在之后的运算里, 每经过一个模块的运算, 都要把运算之前的值和运算之后的值相加, 从而得到残差连接, 训练的时候可以使梯度直接走捷径反传到最初始层:
$$X + SubLayer(X)$$

2). $LayerNorm$:   
$Layer Normalization$的作用是把神经网络中隐藏层归一为标准正态分布, 也就是$i.i.d$独立同分布, 以起到加快训练速度, 加速收敛的作用:
$$\mu_{i}=\frac{1}{m} \sum^{m}_{i=1}x_{ij}$$
上式中以矩阵的行$(row)$为单位求均值;
$$\sigma^{2}_{j}=\frac{1}{m} \sum^{m}_{i=1}
(x_{ij}-\mu_{j})^{2}$$
上式中以矩阵的行$(row)$为单位求方差;
$$LayerNorm(x)=\alpha \odot \frac{x_{ij}-\mu_{i}}
{\sqrt{\sigma^{2}_{i}+\epsilon}} + \beta$$
然后用**每一行**的**每一个元素**减去**这行的均值**, 再除以**这行的标准差**, 从而得到归一化后的数值, $\epsilon$是为了防止除$0$;   
之后引入两个可训练参数$\alpha, \ \beta$来弥补归一化的过程中损失掉的信息, 注意$\odot$表示元素相乘而不是点积, 我们一般初始化$\alpha$为全$1$, 而$\beta$为全$0$.

## 2.4、Feed Forward
前馈网络由两层全联接层组成，两层之间有一个 ReLU 激活函数。

```python
def point_wise_feed_forward_network(d_model, dff):
    return tf.keras.Sequential([
        tf.keras.layers.Dense(dff,
                              activation='relu'),  # (batch_size, seq_len, dff)
        tf.keras.layers.Dense(d_model)  # (batch_size, seq_len, d_model)
    ])
```

例如上面两个Tensorflow中的Dense层。

# 四、整体结构
我们下面用公式把一个$transformer \ block$的计算过程整理一下:    
1). 字向量与位置编码:   
$$X = EmbeddingLookup(X) + PositionalEncoding$$

$$X \in \mathbb{R}^{batch \ size  \ * \  seq. \ len. \  * \  embed. \ dim.} $$

2). 自注意力机制:   
$$Q = Linear(X) = XW_{Q}$$ 

$$K = Linear(X) = XW_{K}$$

$$V = Linear(X) = XW_{V}$$

$$X_{attention} = SelfAttention(Q, \ K, \ V)$$

3). 残差连接与$Layer \ Normalization$

$$X_{attention} = X + X_{attention}$$  

$$X_{attention} = LayerNorm(X_{attention})$$

4). 下面进行$transformer \ block$结构图中的**第4部分**, 也就是$FeedForward$, 其实就是两层线性映射并用激活函数激活, 比如说$ReLU$:   
$$X_{hidden} = Activate(Linear(Linear(X_{attention})))$$

5). 重复3).:
$$X_{hidden} = X_{attention} + X_{hidden}$$

$$X_{hidden} = LayerNorm(X_{hidden})$$

$$X_{hidden} \in \mathbb{R}^{batch \ size  \ * \  seq. \ len. \  * \  embed. \ dim.} $$

# 五、实战
**完整代码**：[https://github.com/LIANGQINGYUAN/NLP-Notebook](https://github.com/LIANGQINGYUAN/NLP-Notebook)

参考链接：
Transformer介绍与展示：https://jalammar.github.io/illustrated-transformer/
Transformer讲解与实现：  https://github.com/aespresso/a_journey_into_math_of_ml
Transformer中的注意力机制：https://zhuanlan.zhihu.com/p/109983672