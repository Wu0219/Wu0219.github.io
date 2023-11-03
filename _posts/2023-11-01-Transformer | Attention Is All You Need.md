---
layout:     post
title:      Transformer | Attention Is All You Need
subtitle:   变形金刚随时变形状
date:       2023-11-01
author:     Yuhang
header-img: img/post/transformer.jpg
catalog: true
tags:
    - CV 
    - transformer
---







## 注意力机制

$Q（query）,K（key）,V（value）$分别为一个输入的原始矩阵和一个可学习的矩阵进行叉乘得来的，假设原有的输入矩阵为$X$,则$Q = X \times W_Q$。 



可以在感性上对QKV进行以下理解：

- Query(to match others)：可以看作是在搜索引擎中输入查找的关键字，其中包含了需要查询的信息，也可以理解成借阅者在图书馆需要借到的书的信息
- Key(to be matched)：需要被查询的信息，如搜索引擎中原始信息片段的描述信息或者是一张可以描述书籍的索引卡
- Attention(Q, K)，即$Q \cdot K$, 可以看成是QK之间的匹配程度，即按照匹配程度给予相应的Q，K对一定的评分，在经过SoftMax之后该评分转化成为相应位置的QK匹配成功的可能性（能这么理解吗？）
- Value(information to be extracted)：信息本身，即搜索引擎指向的网址或者是图书馆的书藉本身



在已经拥有了Q，K，V之后，首先对Q，K进行点乘，此处可以联系向量乘法的意义（向量$A\cdot B$可以看作A在B上的投影长度与B长度的乘积，代表着两个向量在同一方向上的匹配度），即使用点乘计算query和key之间的匹配度，得到的Attention(Q, K)可以用作后续的提取信息。将Attention(Q, K)与V进行点积就可以得出提取之后的信息，即理想中的有用信息。



下图中除了上述过程，还增加了SoftMax


## test1
## test2
## test3
## test4





![image-20231101005051329](https://i.imgur.com/CUY9bvM.png)