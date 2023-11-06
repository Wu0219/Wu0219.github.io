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
    - transformerz
---







## 自注意力机制

$Q（query）,K（key）,V（value）$分别为一个输入的原始矩阵和一个可学习的矩阵进行叉乘得来的，假设原有的输入矩阵为$X$,则$Q = X \times W_Q$。 



可以在感性上对QKV进行以下理解：

- Query(to match others)：可以看作是在搜索引擎中输入查找的关键字，其中包含了需要查询的信息，也可以理解成借阅者在图书馆需要借到的书的信息
- Key(to be matched)：需要被查询的信息，如搜索引擎中原始信息片段的描述信息或者是一张可以描述书籍的索引卡
- Attention(Q, K)，即$Q \cdot K$, 可以看成是QK之间的匹配程度，即按照匹配程度给予相应的Q，K对一定的评分，在经过SoftMax之后该评分转化成为相应位置的QK匹配成功的可能性（能这么理解吗？）
- Value(information to be extracted)：信息本身，即搜索引擎指向的网址或者是图书馆的书藉本身



在已经拥有了Q，K，V之后，首先对Q，K进行点乘，此处可以联系向量乘法的意义（向量$A\cdot B$可以看作A在B上的投影长度与B长度的乘积，代表着两个向量在同一方向上的匹配度），即使用点乘计算query和key之间的匹配度，得到的Attention(Q, K)可以用作后续的提取信息。将Attention(Q, K)与V进行点积就可以得出提取之后的信息，即理想中的有用信息。

本文中注意力的计算公式为：

$$
Attention(Q,K,V)=softmax(\frac{QK^T}{\sqrt{d_k}})V
$$

其解释说明为： *Dot-product attention is identical to our algorithm, except for the scaling factor of $\frac{1}{\sqrt{d_k}}$*

下面让我们理解一下为什么Dot-product attention is identical to our algorithm

假设矩阵Q、K、V都是一个由很多向量构成的矩阵（如一个由许多词向量构成的句子），如下所示Q：

$$
\begin{equation}Q=
 \left[
 \begin{array}{ccc}
     q_{00} & q_{01} & q_{02}  \\
     q_{10} & q_{11} & q_{12}  \\
     q_{20} & q_{21} & q_{22} 
 \end{array}
 \right]        
 \end{equation}
$$

 其中每一行可以看作一个向量（也就是一个单词），矩阵可以被记为：

$$
\begin{equation}Q=
 \left[
 \begin{array}{ccc}
     q_{0}  \\
     q_{1}  \\
     q_{2} 
 \end{array}
 \right]        
 \end{equation}
$$

$$
\begin{equation}K=
 \left[
 \begin{array}{ccc}
     k_{0}  \\
     k_{1}  \\
     k_{2} 
 \end{array}
 \right]        
 \end{equation}
$$

$$
\begin{equation}V=
 \left[
 \begin{array}{ccc}
     v_{0}  \\
     v_{1}  \\
     v_{2} 
 \end{array}
 \right]        
 \end{equation}
$$

此时：

$$
\begin{equation}Q\times K^T =
 \left[\begin{array}{ccc}
     q_{0}  \\
     q_{1}  \\
     q_{2} 
 \end{array}\right]\times\left[\begin{array}{ccc}
     k_{0}  &
     k_{1}  &
     k_{2} 
 \end{array}\right]=\left[\begin{array}{ccc}
     q_{0}\cdot k_{0}  &q_{0}\cdot k_{1}&q_{0}\cdot k_{2}\\
     q_{1}\cdot k_{0}  &q_{1}\cdot k_{1}&q_{1}\cdot k_{2}\\
     q_{2}\cdot k_{0}  &q_{2}\cdot k_{1}&q_{2}\cdot k_{2}
 \end{array}\right]
 =
 \left[\begin{array}{ccc}
     a_{00}  &a_{01}&a_{02}\\
     a_{10}&a_{11}&a_{12}\\
     a_{20}&a_{21}&a_{22}
 \end{array}\right]\end{equation}
$$

在生成的$3\times3$矩阵中每一项都是原有的一个词向量与自己的dot product，即$Attention(Q_n, K_n)$，整个矩阵实现了Q，K两个矩阵的每一向量的两两点乘。

假设原有的矩阵$X$是一句话:「**I love cats**」

$$
\begin{equation}Q/K/V=
 \left[
 \begin{array}{ccc}
     q_{0}/k_{0}/v_{0}  \\
     q_{1}/k_{1}/v_{1}  \\
     q_{2}/k_{2}/v_{2} 
 \end{array}
 \right] \underleftrightarrow{\text{  对应  }}\left[
 \begin{array}{ccc}
     I  \\
     love  \\
     cats 
 \end{array}
 \right]\end{equation}
$$

那么$Q\times K^T$为：

|          | I                      | love                   | cats           |
| -------- | ---------------------- | ---------------------- | -------------- |
| **I**    | $q_0\cdot k_0(a_{00})$ | $q_0\cdot k_1$         | $q_0\cdot k_2$ |
| **love** | $q_1\cdot k_0$         | $q_1\cdot k_1$         | $q_1\cdot k_2$ |
| **cats** | $q_2\cdot k_0$         | $q_2\cdot k_1(a_{21})$ | $q_2\cdot k_2$ |



此时$q_0k_0(a_{00})$即为单词「**I**」与「**I**」的注意力分数(query和key的相似度)，同理$q_2k_1(a_{21})$则为单词「**cats**」与「**love**」的注意力分数



至此我们已经成功得到了$Attention(Q, K)$, 先暂时忽略$\sqrt{d_k}和softmax$，让我们看看$Q\times K^T\times V$:

结合（6）我们可以得出：



$$
\begin{equation} Q\times K^T \times V=
 \left[\begin{array}{ccc}
     a_{00}  &a_{01}&a_{02}\\
     a_{10}&a_{11}&a_{12}\\
     a_{20}&a_{21}&a_{22}
 \end{array}\right]
 \times 
 \left[
 \begin{array}{ccc}
     v_{0}  \\
     v_{1}  \\
     v_{2} 
 \end{array}
 \right]
 = 
 \left[\begin{array}{ccc}
     a_{00}\cdot v_0  +a_{01}\cdot v_1+a_{02}\cdot v_2\\
     a_{10}\cdot v_0  +a_{11}\cdot v_1+a_{12}\cdot v_2\\
     a_{20}\cdot v_0  +a_{21}\cdot v_1+a_{22}\cdot v_2
 \end{array}\right] \underleftrightarrow{\text{  关联最强(最需要注意)  }}\left[
 \begin{array}{ccc}
     I  \\
     love  \\
     cats 
 \end{array}
 \right]\end{equation}
$$



其结果的每一行可以看作是对Value不同项的加权求和，每一行的输出为与X关联最强的元素，在上述例子中， $a_{00}\cdot v_0  +a_{01}\cdot v_1+a_{02}\cdot v_2$是与单词「**I**」关联最强的元素。



以上过程在[李宏毅老师的ppt](https://speech.ee.ntu.edu.tw/~tlkagk/courses/ML_2019/Lecture/Transformer%20(v5).pptx)中有详细说明



![image-20231101005051329](https://i.imgur.com/CUY9bvM.png)
