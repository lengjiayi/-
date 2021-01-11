# 2020自然语言处理

#### 语⾔

是⼀种由三部分组成的符号交流系统：记号，意义和连接两者的符码。

由组合语法规则制约、旨在传达语义的记号形式系统。

- 自然语言：⼈类语⾔，通常是指一种自然地随文化演化的语言。[汉语](https://zh.wikipedia.org/wiki/華語)、[英语](https://zh.wikipedia.org/wiki/英语)、[法语](https://zh.wikipedia.org/wiki/法語)、[西班牙语](https://zh.wikipedia.org/wiki/西班牙語)、[葡萄牙文](https://zh.wikipedia.org/wiki/葡萄牙語)、[日语](https://zh.wikipedia.org/wiki/日语)、[韩语](https://zh.wikipedia.org/wiki/朝鮮語)、[意大利文](https://zh.wikipedia.org/wiki/意大利语)、[德文](https://zh.wikipedia.org/wiki/德语)为自然语言的例子。

- 自然语言处理
    - 利⽤计算机为⼯具对⾃然语⾔进⾏各种加⼯处理、信息提取及应⽤的技术。
    - 自然语言理解：强调对语⾔含义和意图的深层次解释
    - 计算语⾔学：强调可计算的语⾔理论

#### 自然语言处理的难点

- 歧义处理
- 语言知识的表示、获取和运用
- 成语和惯用型的处理
- 对语言的灵活性和动态性的处理
    - 灵活性：同一个意图的不同表达，甚至包含错误的语法等
    - 动态性：语言在不断的变化，如：新词等
- 对常识等与语言无关的知识的利用和处理

#### 汉语处理的难点

- 缺乏计算语言学的句法/语义理论，大都借用基于西方语言的句法/语义理论
- 资源（语料库）缺乏
- 词法分析：词之间没有分隔符，词性标注没有词形变化
- 句法分析：主动词识别难，词法分类与句法功能对应差
- 语义分析：句法结构与句义对应差
- 时态确定困难

### 汉语分词

分词是指根据某个分词规范，把一个“字”串划分成“词”串。

#### 切分歧义

- 交集型歧义：自主-和-平等，战争-与-和平-等问题
- 组合型歧义：他-骑-在-马-上，他-马上-过来
- 混合型歧义：今晚得-到达-南京，得到-达克宁-了，我得-到-达克宁-公司-去

#### 分词方法

<img src="https://raw.githubusercontent.com/lengjiayi/Lecture-Notes/main/assets/image-20210110194403674.png" alt="image-20210110194403674" style="zoom:67%;" />

##### 分词带来的问题

丢失信息、错误的分词、不同分词规范之间的兼容性

## 机器翻译

### 基于规则的机器翻译

由语⾔学⽅⾯的专家进⾏规则的制订。

模型更新需要专家进⾏（制定新的规则）：

- 保证与原先规则兼容
- 不引⼊新的错误

缺点：需要语⾔学家⼤量的⼯作；维护难度⼤；翻译规则 容易发⽣冲突。

### 基于实例的机器翻译

从语料库中学习翻译实例。

1. 查找接近的翻译实例，并进⾏逐词替换进⾏翻译
2. 利⽤类⽐思想analogy，避免复杂的结构分析

### 统计机器翻译

1. 可以⼀定程度上从数据中⾃动挖掘翻译知识
2. 流程相对复杂，其中各个部分都不断被改进和优化
3. 翻译性能遇到瓶颈，难以⼤幅度提升

### 神经网络机器翻译







## 语言模型

### N-Gram

$$
\Pr(X)=\prod_i\Pr[X_i|X_{i-1}]
$$

满足Zipf法则，$fr=c$，即词频和排位的乘积为定值，词频和排位成反比。

- 大部分词都是稀有的



<img src="https://raw.githubusercontent.com/lengjiayi/Lecture-Notes/main/assets/image-20210109110940297.png" alt="image-20210109110940297" style="zoom:50%;" />

#### 平滑方法

- add counts：$P=\frac{p+\delta}{\sum (p+\delta)}$
- Laplace：$P(w)=\frac{p(W)+mp}{(\sum P(W))+m}$
- 线性插值平滑：$P(w_i|w_{i-2}w_{i-1})=\lambda_1P(w_i|w_{i-2}w_{i-1})+\lambda_2P(w_i|w_{i-1})+\lambda_3P(w_i)$

### 对数线性模型

$$
P(Y=y|X=x)=\frac{\exp w\cdot\phi(x,y)}{\sum_y\exp w\cdot\phi(x,y)}
$$

其中，w为权重

### 神经网络语言模型

- RNN
- Bidirectional Language Model
- Masked LM
- Permutation LM

## 文本分类

### Naïve Bayes

$$
argmax_c P(c|Doc)=argmax_c \frac{P(D|c)P(c)}{P(D)}=argmax_c P(D|c)P(c)
$$

对于所有未出现的单词，使用加一平滑。

#### Bernoulli 文本模型

- 通过BOW生成每个文档的feature vector $D_i$，$D_{it}$表示其中第t个单词是否出现，即$D_i\in\{0,1\}^l$

$$
P(D_i|c)=\prod_{t\in V}P(D_{it}|c)=\prod (D_{it}p+(1-D_{it})(1-p))
$$

#### Multinomial 文本模型

- BOW中每一位为代表词频的整数

$$
argmax_c P(D_i|c)=\frac{n_i!}{\prod_{t\in V}D_{it}!}\prod_{t\in  V}P(w_t|c_t)^{D_{it}} \\
\Leftrightarrow argmax_c P(c)\prod_t P(w_t|c_t)^{D_{it}}
$$

其中，$n_i$表示$D_i$中单词数量

### Feature utility measures

- 停词
- Frequency – select the most frequent terms
- Mutual information – select the terms with the highest mutual information (mutual information is also called information gain in this context
    - $I(X,Y)=\sum_y\sum_xp(x,y)log(\frac{p(x,y)}{p(x)p(y)})$
- Χ2 (Chi-square)
    - test independence of two events
    - $\chi^2(D,t,c)=\sum_{e_t\in\{0,1\}}\sum_{e_c\in\{0,1\}}\frac{(N_{e_te_c}-E_{e_te_c})^2}{E_{e_te_c}}$

### Text representation

- Feature Vector
- Features Engineering
- Weight
    - 0-1 vectors
    - Count vectors
    - tf-idf
        - 用于评价词汇i对文档j的重要性
        - tf: frequency of term, $tf_{i,j}=n_{i,j}/|D_j|$，即词汇i在文档j中出现的频率。
        - idf: inverse document frequency，$idf_i = log(\frac{N}{\#\ i\in N})$，即文档数比出现词汇i的文档数的log。常在分母加一防止为零。

## 情感分类

1. extract phrases containing adjectives or adverbs,
2. estimate the semantic orientation of each phrase,
3. classify the review based on the average semantic orientation of the phrases.

## 表示学习

### Eymbolic Encoding

- One-Hot encoding：对词语的相似性没有假设。
- Bag-of-Words：0和1表示单词是否出现。简单，内存开销小。没有考虑词语出现顺序，稀疏，没有考虑语音关联（相似度）。
    - N-Gram BOW：对每一个词典中的n-gram表示为vector的一位。

### Latent Semantic Index 潜在语义索引

矩阵$X\in R^{m\times n}$，$X(i,j)$表示文本 i 在文档 j 中的出现次数，则使用SVD分解，可以将X表示成：$X=U\Sigma V$。通过对$\Sigma$ 排序，舍弃较小的特征值（取前K个）实现降维度。通过列之间的乘积（cosine）代表相似度。

#### 缺点

- 线性模型，对非线性依赖不是最优解
- 如何选取K
- 没有考虑出现顺序
- 添加新单词或文本需要重新计算。

### Word2Vec

CBOW：通过上下文（window_size * 2）预测当前单词，每个词汇采用One-Hot编码。

<img src="https://raw.githubusercontent.com/lengjiayi/Lecture-Notes/main/assets/image-20210109143927921.png" alt="image-20210109143927921" style="zoom:50%;" />

Skip-Gram：通过当前词预测上下文。

使用RNN完成文本任务。

## 词性标注和隐马尔科夫模型*



词性又称词类，是指词汇基本的语法属性。

- 划分依据：根据词的形态、词的语法功能、词的语法意义划分。
- 汉语：借用英文的词类体系；缺乏词性的变化
- 词性标注：
    - 给某种语言的词标注上其所属的词类
    - 对兼类词消歧
- 例：

<img src="https://raw.githubusercontent.com/lengjiayi/Lecture-Notes/main/assets/image-20210109152427918.png" alt="image-20210109152427918" style="zoom:50%;" />



- 词性标注歧义（兼类词）：一个词具有两个或者两个以上的词性，如“上锁”和“门锁”。
- 当前方法正确率可以达到97%
    - Baseline：给每个词标上它最常见的词性，所有的未登录词标上名词词性

### 马尔可夫模型

$$
P(S,w)=\prod_i P(S_i|S_{i-1})P(w_i|S_i)
$$

其中，S表示状态，w表示词汇。

基于两点假设：

- 有限视野：n-1阶马尔可夫模型的视野和n-gram一致。
- 时间独立性

转移矩阵$A_{n\times n}$，其中$a_{ij}=P(S_j|S_i)$

### 隐马尔可夫模型

- 隐状态满足一阶马尔科夫模型，即当前状态仅与上一个状态有关。
- 输出具有独立性
- 何谓“隐”？
    - 状态（序列）是不可见的（隐藏的）
- 什么样的问题需要HMM模型?
    - 我们的问题是基于序列的，比如时间序列，或者状态序列。
    - 我们的问题中有两类数据，一类序列数据是可以观测到的，即观测序列；而另一类数据是不能观察到的，即隐藏状态序列，简称状态序列。

对于隐马尔可夫模型$HMM:\lambda=(S,V,A,B,\Pi)$

S为状态集，V为观察集

隐状态序列$Q=q_1q_2...q_T$，观察序列$O=o_1o_2...o_T$

隐状态转移分布$A=[a_{ij}],a_{ij}=P(q_t=s_j|q_{t-1}=s_i)$，即从隐状态 i 转移到隐状态 j 的概率。

观察值生成概率$B[b_j(v_k)]=P(o_k=v_k|q_t=s_t)$，即当前隐状态j下生成观察值$o_k$的概率

$\Pi=[\Pi_i],\Pi_i=P(q_1=s_j)$为初始状态概率分布

例：

<img src="https://raw.githubusercontent.com/lengjiayi/Lecture-Notes/main/assets/image-20210109154952224.png" alt="image-20210109154952224" style="zoom:50%;" />

#### Viterbi算法

求解观测序列，最可能的状态状态序列（模型解码）。Viterbi算法为动态规划方法。

定义$\delta_t(i)$为t时刻时，隐状态为$s_i$，输出为序列$w_1...w_t$概率最大的隐藏状态路径：
$$
\delta_t(i)=\max_{s_1s_2...s_{t-1}} P(Pos_1...Pos_{t-1},Pos_t=s_i,w_1...w_t)
$$
算法如下：

>初始化：$\delta_1(i)=\max P(Pos_1=s_i,w_1)=\pi_ib_i(w_1)$
>
>迭代：
>$$
>\begin{align}
>\delta_{t+1}(j)&=\max P(Pos_1...Pos_t,Pos_{t+1}=s_j,w1...w_{t+1}) \\
>&=\max_i [a_{ij}b_j(w_{t+1})\delta_t(i)]
>\end{align}
>$$

令$|S|=N$，则每计算一个$\delta_t(i)$，要计算从t-1时刻所有状态到$s_i$的概率，时间复杂度为$O(N)$，每个时刻t需要计算N个状态$\delta(s_1...s_N)$，时间复杂度为$O(N^2)$，且序列长为T，因此算法时间复杂度为$O(N^2T)$



#### MEMM 最大熵马尔可夫模型



## 句法分析*

### 上下文无关文法

对于文法$G=(N,T,S,R)$

- N是非终结符号集合
- T是终结符号集合
- S是开始符号
- R是产生式规则

和编译原理一样，按照语法规则产生语法树。

### 自顶向下句法分析

>算法：
>
>1. 取 ((S) 1)作为**当前状态**（初始状态），**后备状态**为空。
>
>2. 若当前状态为空，则失败，算法结束，
>
>3. 否则，若当前状态的符号表为空，
>    1. 位置计数器值处于句子末尾，则成功，算法结束
>    2. 位置计数器值处于句子中间，转5
>
>4. 否则，进行**状态转换**，若转换成功，则转2
>
>5. 否则，**回溯**，转2。

例子：

<img src="https://raw.githubusercontent.com/lengjiayi/Lecture-Notes/main/assets/image-20210109220545590.png" alt="image-20210109220545590" style="zoom:50%;" />

<img src="https://raw.githubusercontent.com/lengjiayi/Lecture-Notes/main/assets/image-20210109220613777.png" alt="image-20210109220613777" style="zoom:50%;" />





#### 歧义

程序设计语言可以通过约束避免歧义，如LL(1)语法，LR语法。而自然语言无法避免歧义。

### 概率上下文无关文法

每一个产生式都有其概率，语法树的概率即所有边的概率乘积。

### 分析算法

$argmax_G P(W|G)$，分配所有语法规则的概率

#### 向内算法

对于规则$A\rightarrow BC$
$$
\alpha_{i,j}(A)=P(w_i...w_j|A)=\sum_{B,C,k}P(A\rightarrow BC)\alpha_{i,k}(B)\alpha_{k+1,j}(C) \\
\alpha_{i,i}=P(A\rightarrow w_i)
$$
词串概率：
$$
P(S\rightarrow w_1...w_n|G)=\alpha_{1,n}(S)
$$


#### 向外算法

定义$\beta_{i,j}(A)$为通过语法G，通过A推出$w_i...w_j$的概率，且有语法规则$C\rightarrow AB,C\rightarrow BA$则有
$$
\beta_{1,n}(A)=(A=S)?1\ else\ 0 \\
\beta_{i,j}(A)=\sum_{B,C,j<k}\beta_{i,k}(C)P(C\rightarrow AB)\alpha_{j+1,k}(B)+\sum_{B,C,h<i}\beta_{h,j}(C)P(C\rightarrow BA)\alpha_{h,i-1}(B)
$$
词串概率：
$$
P(w_{1n}|G)=\sum_{A,k}\beta(A)P(A\rightarrow w_k)
$$

## 信息抽取

### Relation Classification

通过文本预测两个实体之间的关系。如果使用监督学习，则需要大量的标记数据。

### Distant Supervision

自动生成大量标记数据。

- 方法：如果两个实体在KB（知识库）中具有关系R，则所有包含两个实体的句子都被标记为关系R。

- 缺点：不可避免的会有噪声（标签错误）。

### Noise 噪声

- 假阳性：两个实体没有被标记的关系
- 假阴性：本来具有关系（且在数据集中表现出来）的实体对被标记为没有关系（NA）

#### Suppress Noise 压缩噪声

降低假阳性的重要性，为每一条句子分配权重。

- 假设：
    - 至少一个句子会提及两个实体的关系。
- Multi-Instance Learning：所有提及相同对实体的句子会被装进一个bag中，在bag之间进行训练。
    - 为bag中的句子分配权重，权重和为1
- 缺点：
    - 无法处理bag中没有描输两个实体关系的情况。（不满足假设的情况）
    - 假阴性没有处理

#### Removing Noise 移除噪声

将假阳性句子移除数据集。

- 方法：通过reinforcement learning学习一个判别器，通过classifer的反馈训练。
- 缺点:
    - 假阴性没有处理

#### Rectify Noise 修正噪声

将错误标签更正为正确标签，假阳性修正为正确标签或NA，假阴性从NA修正为阳性。



