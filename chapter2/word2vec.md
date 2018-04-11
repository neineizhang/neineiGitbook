---
description: "CBOW和Skip-Gram是著名的词嵌入工具word2vec中包含的两种模型，与标准语言模型不同，在上下文​可以同时取左右两边的​个字，而标准语言模型中认为，当前词语​仅仅依赖于前文​\U001001C4。"
---

# Word2Vec

## 2. Word2Vec

 CBOW和Skip-Gram是著名的词嵌入工具word2vec中包含的两种模型，与标准语言模型不同，在上下文​可以同时取左右两边的​个字，而标准语言模型中认为，当前词语​仅仅依赖于前文​󰇄。

### 2.1 Skip-Gram Model

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

