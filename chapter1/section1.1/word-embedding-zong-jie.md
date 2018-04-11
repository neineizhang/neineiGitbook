# Word Embedding总结

## Word Embedding总结

### 1. 前言

word representation有两种方式

* 传统方法Knowledge-based representation
* 词的离散表示corpus-based representation
* 词的分布式表达word embedding

#### 1.1 Knowledge-based representation

* **简介**

Knowledge-based representation根据语言学家们制定的 [**WordNet**](https://zh.wikipedia.org/wiki/WordNet) ，其中包含了字与字之间的关联。来对文字进行表示。

![](file:///var/folders/mh/49qly7gj5z3g3dt_vvlpqm8r0000gn/T/abnerworks.Typora/image-201804111615490.png?lastModify=1523449857)

!\[mage-20180411161234\]\(\)

* **调用方法**

> from nltk.corpus import wordnet

* **局限性**
  * 文字会不断地发展
  * 主观性较强，不同的人之间有不同的理解
  * 工作量比较大
  * 字与字之间的相似性很难定义

#### 1.2 Corpus-based representation

**1.2.1 Atomic symbols: one-hot representation**

* **定义**

  One-Hot编码，又称为一位有效编码，独热编码，将所有需计算的文字组成一个向量，给出一个文字，它在向量中的位置标为1，其余都为0。

* **限制**

  无法捕捉两个word之间的关系，也就是没有办法捕捉语义信息

  > 例如：car和motorcycle，无论怎样计算相似度都为0。

* 期望：用另一种方式捕捉真正有关字义的部分。
* 方法：当car与motorcycle以及它们相邻的单词（neighbor）存在某些关系时，可以认为这两个单词之间具有相关性。即Neighbors，如何确定neighbor的范围呢？
  * 方式1，**full document 。**full document可以认为在同一篇文章中出现过，文章中的文字之间可以根据文章确定一个相关的主题。
  * 方式2，**windows** 。windows限定在某个窗口内，可以是几句话或者几个单词之内范围，这种方式可以获得词性等信息。

**1.2.2 High-dimensional sparse word vector**

![](file:///var/folders/mh/49qly7gj5z3g3dt_vvlpqm8r0000gn/T/abnerworks.Typora/image-201804111608082.png?lastModify=1523449857)

* **思想**

  基于neighbour，设置**Co-occurrence Matrix**共现矩阵

* **限制**
  * 随着文章字数增加矩阵的大小也会迅速增加；
  * 计算量会迅速增加。之前one-hot形式，由于只有一列存在非0数字，所以维度即使再大，计算量也不会增加太多。而现在则不同，每列都有可能有数个非0数字；
  * 大部分的有效信息，集中在少数区域。没有有效地“散开”，所以robustness会比较差
  * 当增加一个word的时候整个矩阵都要更新
* **期望**

  找到一个低维向量

**1.2.3 Low-dimensional sparse word vector**

* **思想**

  降维dimension reduction，例如通过SVD等，从k维降为r维

![](file:///var/folders/mh/49qly7gj5z3g3dt_vvlpqm8r0000gn/T/abnerworks.Typora/image-201804111642310.png?lastModify=1523449857)

* **限制**
  * 计算量比较大
  * 新增文字后，需要重建矩阵并重新计算矩阵和降维
* **期望**
  * 直接学习出低维向量，而不是从资料中学习到高维矩阵再进行降维

**1.2.4 Word Embedding**

* **思想**

 directly learn low-dimensional word vectors

* **两种方法**
  * **word2vec** \(Mikolov et al. 2013\)
    * skip-gram
    * CBOW
  * **Glove** \(Pennington et al., 2014\)
* **优点**
  * 任意两个词语之间的相似性很容易计算
  * 可以使用在任何NLP监督任务上作为input，vector包含了语义信息
  * 可以利用NN通过训练更新word representation（word embedding不是固定的，可以根据task微调）

### 2. Word2Vec

 CBOW和Skip-Gram是著名的词嵌入工具word2vec中包含的两种模型，与标准语言模型不同，在上下文​可以同时取左右两边的​个字，而标准语言模型中认为，当前词语​仅仅依赖于前文​󰇄。

#### 2.1 Skip-Gram Model

* **目标**

  给定一个单词​，预测出词汇表中每个词在其上下文中的概率。也就是给定一个word，推算出其周围的word的能力。

  ![](file:///var/folders/mh/49qly7gj5z3g3dt_vvlpqm8r0000gn/T/abnerworks.Typora/image-201804111714309.png?lastModify=1523449857)

  其中,​是target word，​是指定窗口大小 ​范围内的其他 word。

  例如上图中，​ 其实就代表​ ，窗口大小 ​ 为 ​ 。​ 与​就是 ​，共​个。

  在指定的 word 条件下，指定窗口内其他 word 发生的概率计算公式如下：

  根据最大似然估计，应该使得上面的概率结果最大。对上面公式进行取 ​ 后添加负号，得到损失函数的表示公式，目标使损失函数最小。

  其中，

  ​数值越大，越相关，对于同一个word当他是target word的时候和他是context word时表示不一样。模型图如下：

  ![](file:///var/folders/mh/49qly7gj5z3g3dt_vvlpqm8r0000gn/T/abnerworks.Typora/image-201804111733547.png?lastModify=1523449857)

* **模型结构**

  ![](file:///var/folders/mh/49qly7gj5z3g3dt_vvlpqm8r0000gn/T/abnerworks.Typora/image-201804111740537.png?lastModify=1523449857)

  模型的结构很简单，如上图，是一个只有一个隐藏层的神经网络。由于要得到输入的每个 ​出现的概率，所以输入层与输出层神经元数需一致。下图中，输入 ​ 和输出层​ 数量为10000，隐藏层 ​ 神经元数为300。

如果对隐藏层权重​ 转换思想，\(**注意转换的仅仅是我们的思想，实际上没有对隐藏层做任何改变**\)。 上面的例子中，输入的 ​ 是10000行，后面的隐藏层共300个神经元，所以 ​ 是 10000×300 的矩阵。 实际​ 计算的过程，思想其实是像下图左中的样子，​ 每一列与 ​ 对应相乘后再相加。如何将想法转换成下图右中的模式呢？请看接下来的例子。

![](file:///var/folders/mh/49qly7gj5z3g3dt_vvlpqm8r0000gn/T/abnerworks.Typora/image-201804111740249.png?lastModify=1523449857)

由于​ 是one-hot转换后的，所以在指定的word上是1，其余都是0。这样的话，经过计算后，实际的结果中是把 ​ 相应的一行数据给完整保留下来了，其余的都乘以0后都没了，具体见下图。所以，也可以从行的角度看 ​。

综上所述，​ 的计算结果，其实也就是从 ​ 中抽取出来相乘不为0的一行当成结果，可以用向量​ 表示。

![](file:///var/folders/mh/49qly7gj5z3g3dt_vvlpqm8r0000gn/T/abnerworks.Typora/image-201804111743487.png?lastModify=1523449857)

上面说了隐藏层 ​ ，接下来分析输出层​ 。从 ​ 到​ 之间的系数为 ​ 。 上面说道​ 输出层共10000个神经元，每个神经元的计算方法如下：

其中，​是 ​的第 ​列。

得到 ​ 后，还需要做最后的转换才是最终输出的结果。转换公式如下，这也是一般多分类 softmax 的计算方法。

![](file:///var/folders/mh/49qly7gj5z3g3dt_vvlpqm8r0000gn/T/abnerworks.Typora/image-201804111749483.png?lastModify=1523449857)

下图就上对上面所讲的，从隐藏层到最后输出概率的一个总结。​ 与 ​的计算结果，经过公式转换得到最后每个 word 的概率。

![](file:///var/folders/mh/49qly7gj5z3g3dt_vvlpqm8r0000gn/T/abnerworks.Typora/image-201804111750273.png?lastModify=1523449857)

如果把计算过程放在整体上看，如下图。

![](file:///var/folders/mh/49qly7gj5z3g3dt_vvlpqm8r0000gn/T/abnerworks.Typora/image-201804111751078.png?lastModify=1523449857)

* **损失函数和梯度下降**前文有讲到，给定一个target word，Loss Function是：，，

使用梯度下降的方法对损失函数进行优化。从右向左，首先看​ 这边。

![](file:///var/folders/mh/49qly7gj5z3g3dt_vvlpqm8r0000gn/T/abnerworks.Typora/image-201804111807493.png?lastModify=1523449857)

整理后得到如下公式。

![](file:///var/folders/mh/49qly7gj5z3g3dt_vvlpqm8r0000gn/T/abnerworks.Typora/image-201804111809049.png?lastModify=1523449857)

然后是​ 这边，

![](file:///var/folders/mh/49qly7gj5z3g3dt_vvlpqm8r0000gn/T/abnerworks.Typora/image-201804111811552.png?lastModify=1523449857)

整理后得到。

![](file:///var/folders/mh/49qly7gj5z3g3dt_vvlpqm8r0000gn/T/abnerworks.Typora/image-201804111812111.png?lastModify=1523449857)

把公式进行简化，简化之后得到下图中红色部分的公式。

![](file:///var/folders/mh/49qly7gj5z3g3dt_vvlpqm8r0000gn/T/abnerworks.Typora/image-201804111812351.png?lastModify=1523449857)

其中​ 是 输入 ​ 中 words总数量。从上面的公式中可以看出来，计算量和 ​相关。当 ​ 比较大的时候，计算量会非常大。

为了解决计算的问题，有两种常用的方法：**层次化softmax（hierarchical softmax ）**和 **负采样（negative sampling）** 。常用的是后者。

* **层次化softmax**

  ![](file:///var/folders/mh/49qly7gj5z3g3dt_vvlpqm8r0000gn/T/abnerworks.Typora/image-201804111817064.png?lastModify=1523449857)

* **负采样**

  **Word2Vec Variants**

  Word2Vec 还有一些其他的方式，比如CBOW、LM 。具体方法请看下图，其实LM是最早被提出来的，而Skip-gram是不断完善后的样子，所以现在Skip-gram应用是最广泛的。CBOW和Skip-gram正好相反，Skip-gram是给定一个word，预测窗口内其他words，而CBOW是给定窗口内其他words的概率，预测指定word。

