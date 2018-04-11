# Word Embedding总结

## 1. 前言

word representation有两种方式

* 传统方法Knowledge-based representation
* 词的离散表示corpus-based representation
* 词的分布式表达word embedding

### 1.1 Knowledge-based representation

* **简介**

Knowledge-based representation根据语言学家们制定的 [**WordNet**](https://zh.wikipedia.org/wiki/WordNet) ，其中包含了字与字之间的关联。来对文字进行表示。

### 1.2 Corpus-based representation

**1.2.1 Atomic symbols: one-hot representation**

![](file:///var/folders/mh/49qly7gj5z3g3dt_vvlpqm8r0000gn/T/abnerworks.Typora/image-201804111615490.png?lastModify=1523449857)

!\[mage-20180411161234\]\(\)

* **调用方法**

> from nltk.corpus import wordnet

* **局限性**
  * 文字会不断地发展
  * 主观性较强，不同的人之间有不同的理解
  * 工作量比较大
  * 字与字之间的相似性很难定义
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

