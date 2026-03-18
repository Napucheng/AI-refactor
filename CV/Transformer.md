# Attention&Transformer

## Attention in ML

$$
query
\\
key
\\
value
$$

### Nadaraya-Watson 核回归

#### Non parameter

​	If a key is closer to the given query, then the attention weight assigned to the corresponding value of this key will be greater.
$$
\begin{aligned}
f(x) & =\sum_{i=1}^n\alpha(x,x_i)y_i \\
 & =\sum_{i=1}^n\frac{\exp\left(-\frac{1}{2}(x-x_i)^2\right)}{\sum_{j=1}^n\exp\left(-\frac{1}{2}(x-x_j)^2\right)}y_i \\
 & =\sum_{i=1}^n\mathrm{softmax}\left(-\frac{1}{2}(x-x_i)^2\right)y_i.
\end{aligned}
$$

#### Parameterized attention aggregation

​	在这里，加入参数w，则需要学习训练出w。
$$
\begin{aligned}
f(x) & =\sum_{i=1}^n\alpha(x,x_i)y_i \\
 & =\sum_{i=1}^n\frac{\exp\left(-\frac{1}{2}((x-x_i)w)^2\right)}{\sum_{j=1}^n\exp\left(-\frac{1}{2}((x-x_i)w)^2\right)}y_i \\
 & =\sum_{i=1}^n\mathrm{softmax}\left(-\frac{1}{2}((x-x_i)w)^2\right)y_i.
\end{aligned}
$$

##### Notes: batch matrix multiplication

```python
torch.bmm(A,B)
A:(b,n,m)
B:(b,n,p)
```

```python
# X_tile的形状:(n_train，n_train)，每一行都包含着相同的训练输入
X_tile = x_train.repeat((n_train, 1))
# Y_tile的形状:(n_train，n_train)，每一行都包含着相同的训练输出
Y_tile = y_train.repeat((n_train, 1))
# keys的形状:('n_train'，'n_train'-1)
keys = X_tile[(1 - torch.eye(n_train)).type(torch.bool)].reshape((n_train, -1))
# values的形状:('n_train'，'n_train'-1)
values = Y_tile[(1-torch.eye(n_train)).type(torch.bool)].reshape((n_train, -1))
```

将keys和values的自己使用mask矩阵去掉

##### Notes: torch.repeat((x,1))

```
Y_tile = y_train.repeat((n_train, 1))
if:
y_train = [y1,y2,y3,y4]
then:
Y_tile =
[y1 y2 y3 y4
 y1 y2 y3 y4
 y1 y2 y3 y4
 y1 y2 y3 y4]
```

通过损失函数、SGD优化、反向传播、可视化完成对参数w的训练；最后带参数的注意力汇聚会比在非参数注意力汇聚的绘制图变得不光滑。

<img src="C:\Users\napuc\AppData\Roaming\Typora\typora-user-images\image-20260311172534361.png" alt="image-20260311172534361" style="zoom:50%;" />

### Attention Scoring Function

![../_images/attention-output.svg](https://zh.d2l.ai/_images/attention-output.svg)
$$
f(\mathbf{q},(\mathbf{k}_1,\mathbf{v}_1),\ldots,(\mathbf{k}_m,\mathbf{v}_m))=\sum_{i=1}^m\alpha(\mathbf{q},\mathbf{k}_i)\mathbf{v}_i\in\mathbb{R}^v,\
$$

```python
#@save
def masked_softmax(X, valid_lens):
    """通过在最后一个轴上掩蔽元素来执行softmax操作"""
    # X:3D张量，valid_lens:1D或2D张量
    if valid_lens is None:
        return nn.functional.softmax(X, dim=-1)
    else:
        shape = X.shape
        if valid_lens.dim() == 1:
            valid_lens = torch.repeat_interleave(valid_lens, shape[1])
        else:
            valid_lens = valid_lens.reshape(-1)
        # 最后一轴上被掩蔽的元素使用一个非常大的负值替换，从而其softmax输出为0
        X = d2l.sequence_mask(X.reshape(-1, shape[-1]), valid_lens,
                              value=-1e6)
        return nn.functional.softmax(X.reshape(shape), dim=-1)
```

Notes: 上述函数将没有意义的特殊词元，如‘eos’‘pad’等等过滤掩蔽。

#### Additive Attention

$$
a(\mathbf{q},\mathbf{k})=\mathbf{w}_v^\top\tanh(\mathbf{W}_q\mathbf{q}+\mathbf{W}_k\mathbf{k})\in\mathbb{R}
$$

​	 将查询和键连结起来后输入到一个多层感知机（MLP）中， 感知机包含一个隐藏层，其隐藏单元数是一个超参数h。 通过使用tanh作为激活函数，并且禁用偏置项。

##### Notes: Unsqueeze

使用广播机制避免循环的有效方式，提升计算效率。
$$
queries.unsqueeze(2) + keys.unsqueeze(1)
\\
(batch, query\_num, key\_num, hidden)
$$

## Bahdanau注意力

示意图：

![../_images/seq2seq-attention-details.svg](https://zh.d2l.ai/_images/seq2seq-attention-details.svg)

这是一个基于RNN，在解码器部分使用transformer完成和所有输入序列的隐状态进行注意力汇聚，得到softmax相关性，最后输出结果。


$$
\mathbf{c}_{t^{\prime}}=\sum_{t=1}^T\alpha(\mathbf{s}_{t^{\prime}-1},\mathbf{h}_t)\mathbf{h}_t
$$
d2l代码库：

```python
class Seq2SeqAttentionDecoder(AttentionDecoder):
    def __init__(self, vocab_size, embed_size, num_hiddens, num_layers,
                 dropout=0, **kwargs):
        super(Seq2SeqAttentionDecoder, self).__init__(**kwargs)
        self.attention = d2l.AdditiveAttention(
            num_hiddens, num_hiddens, num_hiddens, dropout)
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.rnn = nn.GRU(
            embed_size + num_hiddens, num_hiddens, num_layers,
            dropout=dropout)
        self.dense = nn.Linear(num_hiddens, vocab_size)
```

- 预定义一些层

```python
def init_state(self, enc_outputs, enc_valid_lens, *args):
        # outputs的形状为(batch_size，num_steps，num_hiddens).
        # hidden_state的形状为(num_layers，batch_size，num_hiddens)
        outputs, hidden_state = enc_outputs
        return (outputs.permute(1, 0, 2), hidden_state, enc_valid_lens)
```

- 返回的state包含了：
  - 编码器的所有输出
  - 编码器的最终hidden
  - 有效长度(方便Mask)

| 参数                     | 形状  | 含义                 |
| ------------------------ | ----- | -------------------- |
| outputs                  | T,B,H | 每个时间步的隐藏状态 |
| hidden_state             | L,B,H | 最终                 |
| outputs.permute(1, 0, 2) | B,T,H | attention需要        |
| enc_valid_lens           | B,    | 真实有效长度         |

#### Notes: RNN/GRU hidden state

![../_images/deep-rnn.svg](https://zh.d2l.ai/_images/deep-rnn.svg)

```python
def forward(self, X, state):
        # enc_outputs的形状为(batch_size,num_steps,num_hiddens).
        # hidden_state[0]的形状为(num_layers,batch_size,
        # num_hiddens)
        enc_outputs, hidden_state, enc_valid_lens = state
        # 输出X的形状为(num_steps,batch_size,embed_size)
        X = self.embedding(X).swapaxes(0, 1)
        outputs, self._attention_weights = [], []
        for x in X:
            # query的形状为(batch_size,1,num_hiddens)
            query = np.expand_dims(hidden_state[0][-1], axis=1)
            # context的形状为(batch_size,1,num_hiddens)
            context = self.attention(
                query, enc_outputs, enc_outputs, enc_valid_lens)
            # 在特征维度上连结
            x = np.concatenate((context, np.expand_dims(x, axis=1)), axis=-1)
            # 将x变形为(1,batch_size,embed_size+num_hiddens)
            out, hidden_state = self.rnn(x.swapaxes(0, 1), hidden_state)
            outputs.append(out)
            self._attention_weights.append(self.attention.attention_weights)
        # 全连接层变换后，outputs的形状为
        # (num_steps,batch_size,vocab_size)
        outputs = self.dense(np.concatenate(outputs, axis=0))
        return outputs.swapaxes(0, 1), [enc_outputs, hidden_state,
                                        enc_valid_lens]
```

outputs保存每个时间步的decoder输出

self._attention_weights保存权重

context是query和所有hidden state的加权平均

- 实例

```python
encoder = d2l.Seq2SeqEncoder(vocab_size=10, embed_size=8, num_hiddens=16,
                             num_layers=2)
encoder.initialize()
decoder = Seq2SeqAttentionDecoder(vocab_size=10, embed_size=8, num_hiddens=16,
                                  num_layers=2)
decoder.initialize()
X = np.zeros((4, 7))  # (batch_size,num_steps)
state = decoder.init_state(encoder(X), None)
output, state = decoder(X, state)
output.shape, len(state), state[0].shape, len(state[1]), state[1][0].shape
```

#### Notes: Seq2seq Prediction

```

```

## Multihead Attention

![../_images/multi-head-attention.svg](https://zh.d2l.ai/_images/multi-head-attention.svg)

```python
#@save
class MultiHeadAttention(nn.Block):
    """多头注意力"""
    def __init__(self, num_hiddens, num_heads, dropout, use_bias=False,
                 **kwargs):
        super(MultiHeadAttention, self).__init__(**kwargs)
        self.num_heads = num_heads
        self.attention = d2l.DotProductAttention(dropout)
        self.W_q = nn.Dense(num_hiddens, use_bias=use_bias, flatten=False)
        self.W_k = nn.Dense(num_hiddens, use_bias=use_bias, flatten=False)
        self.W_v = nn.Dense(num_hiddens, use_bias=use_bias, flatten=False)
        self.W_o = nn.Dense(num_hiddens, use_bias=use_bias, flatten=False)

    def forward(self, queries, keys, values, valid_lens):
        # queries，keys，values的形状:
        # (batch_size，查询或者“键－值”对的个数，num_hiddens)
        # valid_lens　的形状:
        # (batch_size，)或(batch_size，查询的个数)
        # 经过变换后，输出的queries，keys，values　的形状:
        # (batch_size*num_heads，查询或者“键－值”对的个数，
        # num_hiddens/num_heads)
        queries = transpose_qkv(self.W_q(queries), self.num_heads)
        keys = transpose_qkv(self.W_k(keys), self.num_heads)
        values = transpose_qkv(self.W_v(values), self.num_heads)

        if valid_lens is not None:
            # 在轴0，将第一项（标量或者矢量）复制num_heads次，
            # 然后如此复制第二项，然后诸如此类。
            valid_lens = valid_lens.repeat(self.num_heads, axis=0)

        # output的形状:(batch_size*num_heads，查询的个数，
        # num_hiddens/num_heads)
        output = self.attention(queries, keys, values, valid_lens)

        # output_concat的形状:(batch_size，查询的个数，num_hiddens)
        output_concat = transpose_output(output, self.num_heads)
        return self.W_o(output_concat)
```

### 形状变换

```python
def transpose_qkv(X, num_heads):
    """为了多注意力头的并行计算而变换形状"""
    # 输入X的形状:(batch_size，查询或者“键－值”对的个数，num_hiddens)
    # 输出X的形状:(batch_size，查询或者“键－值”对的个数，num_heads，
    # num_hiddens/num_heads)
    X = X.reshape(X.shape[0], X.shape[1], num_heads, -1)

    # 输出X的形状:(batch_size，num_heads，查询或者“键－值”对的个数,
    # num_hiddens/num_heads)
    X = X.zhege transpose(0, 2, 1, 3)

    # 最终输出的形状:(batch_size*num_heads,查询或者“键－值”对的个数,
    # num_hiddens/num_heads)
    return X.reshape(-1, X.shape[2], X.shape[3])
```

原始输入：[ Batch, Seq, Hidden ]
            ↓
拆分 Hidden: [ Batch, Seq, Heads, Head_Dim ]
            ↓
交换维度：[ Batch, Heads, Seq, Head_Dim ]  <-- 让 Heads 独立出来
            ↓
合并 Batch: [ Batch * Heads, Seq, Head_Dim ] <-- 伪装成更大的 Batch

#### Notes: Valid_lens&Multihead

​	用于忽略<pad>，计算注意力分数时，转换成mask矩阵，其中填充部分变成负无穷。在变换多头的过程中，batch变成了batch_size*num_heads。

​	故使用torch.repeat_interleave复制。

```python
torch.repeat_interleave(input, repeats, dim=None, *, output_size=None)
```

## Self-Attention

### Positional Encoding

​	自注意力机制要有位置编码从而得到序列的顺序信息。
$$
\mathbf{X}\in\mathbb{R}^{n\times d},\mathbf{P}\in\mathbb{R}^{n\times d}
\\
X+P
$$

### Layer **Normalization**

### Masking

#### Encoder

掩蔽padding，不掩蔽上下文，使用valid_len

#### Decoder

掩蔽padding和下文



### 

|      |      |      |
| ---- | ---- | ---- |
|      |      |      |
|      |      |      |
