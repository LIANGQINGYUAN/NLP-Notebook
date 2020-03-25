Attention是一种用于提升基于RNN（LSTM或GRU）的Encoder + Decoder模型效果的机制（Mechanism），一般称为Attention Mechanism。Attention给模型赋予了区分辨别的能力，例如，在机器翻译、语音识别应用中，为句子中的每个词赋予不同的权重，使神经网络模型的学习变得更加灵活（soft），同时Attention本身可以做为一种对齐关系，解释翻译输入/输出句子之间的对齐关系，解释模型到底学到了什么知识。
![在这里插入图片描述](https://img-blog.csdnimg.cn/20200303212440869.png)
上图显示了在图像标注中的attention可视化。

Attention Mechanism与人类对外界事物的观察机制很类似，当人类观察外界事物的时候，一般不会把事物当成一个整体去看，往往倾向于根据需要选择性的去获取被观察事物的某些重要部分，比如我们看到一个人时，往往先Attention到这个人的脸，然后再把不同区域的信息组合起来，形成一个对被观察事物的整体印象。因此，Attention Mechanism可以帮助模型对输入的X每个部分赋予不同的权重，抽取出更加关键及重要的信息，使模型做出更加准确的判断，同时不会对模型的计算和存储带来更大的开销，这也是Attention Mechanism应用如此广泛的原因。

# 一、Attention Mechanism原理
## 1.1 Attention Mechanism主要需要解决的问题
《Sequence to Sequence Learning with Neural Networks》介绍了一种基于RNN的Seq2Seq模型，基于一个Encoder和一个Decoder来构建基于神经网络的End-to-End的机器翻译模型，其中，Encoder把输入X编码成一个固定长度的隐向量C，Decoder基于隐向量C解码出目标输出Y。这是一个非常经典的序列到序列的模型，但是却存在两个明显的问题：
1、把输入X的所有信息有压缩到一个固定长度的隐向量C，忽略了输入X的长度，当输入句子长度很长，特别是比训练集中最初的句子长度还长时，模型的性能急剧下降。
2、把输入X编码成一个固定的长度，对于句子中每个词都赋予相同的权重，这样做是不合理的，比如，在机器翻译里，对输入的每个词赋予相同权重，这样做没有区分度，往往使模型性能下降

同样的问题也存在于图像识别领域，卷积神经网络CNN对输入的图像每个区域做相同的处理，这样做没有区分度，特别是当处理的图像尺寸非常大时，问题更明显。因此，2015年，Dzmitry Bahdanau等人在《Neural machine translation by jointly learning to align and translate》提出了Attention Mechanism，用于对输入X的不同部分赋予不同的权重，进而实现软区分的目的。

## 1.2 Attention Mechanism原理
2014年在论文《Sequence to Sequence Learning with Neural Networks》中使用LSTM来搭建Seq2Seq模型。随后，2015年，Kyunghyun Cho等人在论文《Learning Phrase Representations using RNN Encoder–Decoder for Statistical Machine Translation》提出了基于GRU的Seq2Seq模型。两篇文章所提出的Seq2Seq模型，想要解决的主要问题是，如何把机器翻译中，变长的输入X映射到一个变长输出Y的问题，主要结构如下：
![在这里插入图片描述](https://img-blog.csdnimg.cn/20200303212602165.png)

Encoder把一个变成的输入序列x1，x2，x3....xt编码成一个固定长度隐向量（背景向量，或上下文向量context）c，c有两个作用：
1、做为初始向量初始化Decoder的模型，做为decoder模型预测y1的初始向量。
2、做为背景向量，指导y序列中每一个step的y的产出。Decoder主要基于背景向量c和上一步的输出yt-1解码得到该时刻t的输出yt，直到碰到结束标志（\<EOS\>）为止。

在上述的模型中，Encoder-Decoder 框架将输入X都编码转化为语义表示 C，这就导致翻译出来的序列的每一个字都是同权地考虑了输入中的所有的词。例如输入的英文句子是：Tom chase Jerry，目标的翻译结果是：汤姆追逐杰瑞。在未考虑注意力机制的模型当中，模型认为“汤姆 ”这个词的翻译受到 Tom，chase 和 Jerry 这三个词的同权重的影响。但是实际上显然不应该是这样处理的，“汤姆 ”这个词应该受到输入的 Tom 这个词的影响最大，而其它输入的词的影响则应该是非常小的。显然，在未考虑注意力机制的 Encoder-Decoder 模型中，这种不同输入的重要程度并没有体现处理，一般称这样的模型为 分心模型。

而带有 Attention 机制的 Encoder-Decoder 模型则是要从序列中学习到每一个元素的重要程度，然后按重要程度将元素合并。因此，注意力机制可以看作是 Encoder 和 Decoder 之间的接口，它向 Decoder 提供来自每个 Encoder 隐藏状态的信息。通过该设置，模型能够选择性地关注输入序列的有用部分，从而学习它们之间的“对齐”。这就表明，在 Encoder 将输入的序列元素进行编码时，得到的不在是一个固定的语义编码 C ，而是存在多个语义编码，且不同的语义编码由不同的序列元素以不同的权重参数组合而成。一个简单地体现 Attention 机制运行的示意图如下：
![在这里插入图片描述](https://img-blog.csdnimg.cn/20200303212643111.png)
在 Attention 机制下，语义编码 C 就不在是输入X序列的直接编码了，而是各个元素按其重要程度加权求和得到的，即：
$$C_i=\sum_{j=0}^{T_x}{a_{ij}f(x_j)}$$
在公式（6）中，参数 𝑖 表示时刻， 𝑗表示序列中的第 𝑗个元素， 𝑇𝑥 表示序列的长度， 𝑓(⋅) 表示对元素 𝑥𝑗的编码。𝑎𝑖𝑗可以看作是一个概率，反映了元素 ℎ𝑗 对 𝐶𝑖的重要性，可以使用 softmax 来表示：
$$a_{ij}=\frac{exp(e_{ij})}{\sum_{k=1}^{T_x}exp(e_{ik})}$$
这里 𝑒𝑖𝑗正是反映了待编码的元素和其它元素之间的匹配度，当匹配度越高时，说明该元素对其的影响越大，则 𝑎𝑖𝑗的值也就越大。

因此，得出 𝑎𝑖𝑗的过程如下图：
![在这里插入图片描述](https://img-blog.csdnimg.cn/20200303213002855.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FxXzMzNDE0Mjcx,size_16,color_FFFFFF,t_70)
其中，ℎi 表示 Encoder 的转换函数，𝐹(ℎ𝑗,𝐻𝑖)表示预测与目标的匹配打分函数。将以上过程串联起来，则注意力模型的结构如下图所示：
![在这里插入图片描述](https://img-blog.csdnimg.cn/20200303213035764.png)

>① 对 RNN 的输出计算注意程度，通过计算最终时刻的向量与任意 i 时刻向量的权重，通过 softmax 计算出得到注意力偏向分数，如果对某一个序列特别注意，那么计算的偏向分数将会比较大。  
>② 计算 Encoder 中每个时刻的隐向量
>③ 将各个时刻对于最后输出的注意力分数进行加权，计算出每个时刻 i 向量应该赋予多少注意力
>④ decoder 每个时刻都会将 ③ 部分的注意力权重输入到 Decoder 中，此时 Decoder 中的输入有：经过注意力加权的隐藏层向量，Encoder 的输出向量，以及 Decoder 上一时刻的隐向量
>⑤ Decoder 通过不断迭代，Decoder 可以输出最终翻译的序列。
>![在这里插入图片描述](https://img-blog.csdnimg.cn/20200306221151994.jpg?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FxXzMzNDE0Mjcx,size_16,color_FFFFFF,t_70)

# 二、 NMT领域Attention
几十年来，统计机器翻译一直是占主导地位的翻译模型，直到神经机器翻译 (NMT)的诞生。NMT是一种新兴的机器翻译方法，它试图构建和训练单个大型的神经网络，来读取输入文本并输出对应的翻译。
NMT的先驱是Kalchbrenner and Blunsom (2013)， Sutskever et. al (2014)和Cho. et. al (2014b)，其中比较熟悉的框架是来自Sutskever et. al.的序列到序列(seq2seq)模型。
![在这里插入图片描述](https://img-blog.csdnimg.cn/20200303213116290.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FxXzMzNDE0Mjcx,size_16,color_FFFFFF,t_70)
上述seq2seq输入长度为4输出长度为3。
seq2seq的问题是，解码器从编码器接收到的唯一信息是编码器的最后隐藏状态（图中的红色向量）这是一个向量表示，类似于输入序列的数值摘要。在长文本中，我们期望解码器只使用这一个向量表示(希望它“充分描述输入序列”)来输出翻译是不现实的。这可能会导致灾难性的遗忘。

如果我们做不到，那么我们就不应该对解码器如此残忍。那么，如果不光给一个向量表示，同时还给解码器一个来自每个编码器时间步长的向量表示，这样它就可以做出具有充足信息的翻译了，这个想法怎么样？让我们进入注意力机制。

注意力机制是编码器和解码器之间的接口，它向解码器提供来自每个编码器隐藏状态的信息。通过这个设置，模型能够选择性地关注输入序列的有用部分，从而学习它们之间的“对齐”。这有助于模型有效地处理长输入语句。
![在这里插入图片描述](https://img-blog.csdnimg.cn/20200306215126380.gif)
有两种注意类型，使用所有编码器隐藏状态的注意力类型也称为“全局注意力”。相反，“局部注意力”只使用编码器隐藏状态的子集。由于本文的范围是全局attention，因此本文中提到的“attention”均被视为“全局attention”。


引入 Attention 的 Encoder-Decoder 框架下，完成机器翻译任务的大致流程如下：
![在这里插入图片描述](https://img-blog.csdnimg.cn/2020030621560365.gif)

注意力集中在不同的单词上，给每个单词打分。然后，使用softmax之后分数，我们使用编码器隐藏状态的加权和来聚合编码器隐藏状态，得到上下文向量。

# 三、主要代码实现
## 3.1 Encoder

```python
class Encoder(tf.keras.Model):
    def __init__(self, hidden_size=1024, max_sequence_len=30, batch_size=batch_size, embedding_dim=256, vocab_size=5000):
        super(Encoder, self).__init__()
        self.embedding_dim = embedding_dim
        self.vocab_size = vocab_size
        self.max_sequence_len = max_sequence_len
        self.hidden_size = hidden_size
        self.batch_size = batch_size

        self.embedding_layer = Embedding(
            input_dim=self.vocab_size, output_dim=self.embedding_dim)
        self.GRU_1 = GRU(units=hidden_size, return_sequences=True)
        self.GRU_2 = GRU(units=hidden_size,
                         return_sequences=True, return_state=True)

    def initial_hidden_state(self):
        return tf.zeros(shape=(self.batch_size, self.hidden_size))

    def call(self, x, initial_state, training=False):
        x = self.embedding_layer(x)
        x = self.GRU_1(x, initial_state=initial_state)
        x, hidden_state = self.GRU_2(x)
        return x, hidden_state
```
## 3.2 Attention


```python
class Attention(tf.keras.Model):
    def __init__(self, hidden_size=256):
        super(Attention, self).__init__()
        self.fc1 = Dense(units=hidden_size)
        self.fc2 = Dense(units=hidden_size)
        self.fc3 = Dense(units=1)

    def call(self, encoder_output, hidden_state, training=False):
        '''hidden_state : h(t-1)'''
        y_hidden_state = tf.expand_dims(hidden_state, axis=1)
        y_hidden_state = self.fc1(y_hidden_state)
        y_enc_out = self.fc2(encoder_output)
        
        #get a_ij
        y = tf.keras.backend.tanh(y_enc_out + y_hidden_state)
        attention_score = self.fc3(y)
        attention_weights = tf.keras.backend.softmax(attention_score, axis=1)
        
        #get c_i
        context_vector = tf.multiply(encoder_output, attention_weights)
        context_vector = tf.reduce_sum(context_vector, axis=1)

        return context_vector, attention_weights
```

## 3.3 Decoder

```python
class Decoder(tf.keras.Model):
    def __init__(self, hidden_size=1024, max_sequence_len=30, batch_size=batch_size, embedding_dim=256, vocab_size=5000):
        super(Decoder, self).__init__()
        self.embedding_dim = embedding_dim
        self.vocab_size = vocab_size
        self.max_sequence_len = max_sequence_len
        self.hidden_size = hidden_size
        self.batch_size = batch_size
    
        self.embedding_layer = Embedding(
            input_dim=self.vocab_size, output_dim=self.embedding_dim)
        self.GRU = GRU(units=hidden_size,
                       return_sequences=True, return_state=True)
        self.attention = Attention(hidden_size=self.hidden_size)
        self.fc = Dense(units=self.vocab_size)

    def initial_hidden_state(self):
        return tf.zeros(shape=(self.batch_size, self.hidden_size))

    def call(self, x, encoder_output, hidden_state, training=False):
        x = self.embedding_layer(x)
        context_vector, attention_weights = self.attention(
            encoder_output, hidden_state, training=training)
        contect_vector = tf.expand_dims(context_vector, axis=1)
        x = tf.concat([x, contect_vector], axis=-1)
        x, curr_hidden_state = self.GRU(x)
        x = tf.reshape(x, shape=[self.batch_size, -1])
        x = self.fc(x)
        return x, curr_hidden_state, attention_weights
```
最终结果：
![在这里插入图片描述](https://img-blog.csdnimg.cn/2020030321370532.png)
![在这里插入图片描述](https://img-blog.csdnimg.cn/20200303213607681.png)

**完整代码**：[https://github.com/LIANGQINGYUAN/NLP-Notebook](https://github.com/LIANGQINGYUAN/NLP-Notebook)
欢迎star～


参考链接：
模型汇总24 - 深度学习中Attention Mechanism详细介绍：原理、分类及应用：
https://zhuanlan.zhihu.com/p/31547842
浅谈 Attention 机制的理解：https://www.cnblogs.com/ydcode/p/11038064.html
Attention可视化：https://jalammar.github.io/visualizing-neural-machine-translation-mechanics-of-seq2seq-models-with-attention/
Intuitive Understanding of Attention Mechanism in Deep Learning：
https://towardsdatascience.com/intuitive-understanding-of-attention-mechanism-in-deep-learning-6c9482aecf4f
kaggle翻译例子1：https://www.kaggle.com/nikhilxavier/english-to-hindi-machine-translation-attention
kaggle翻译例子2：https://www.kaggle.com/harishreddy18/english-to-french-translation
Go from the basics - Attention mechanism, transformers, BERT：
https://www.kaggle.com/c/tensorflow2-question-answering/discussion/115676#711847
机器翻译语料库：http://www.manythings.org/anki/
 Transformer 系列一：https://zhuanlan.zhihu.com/p/109585084