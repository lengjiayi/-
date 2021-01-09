

---

## 第一讲 线性空间与线性算子

### 线性空间

- <img src="https://raw.githubusercontent.com/lengjiayi/Lecture-Notes/main/assets/image-20201223141946922.png" alt="image-20201223141946922" style="zoom:50%;" /><img src="https://raw.githubusercontent.com/lengjiayi/Lecture-Notes/main/assets/image-20201223142006381.png" alt="image-20201223142006381" style="zoom:50%;" />

### 向量系等价

- <img src="https://raw.githubusercontent.com/lengjiayi/Lecture-Notes/main/assets/image-20201223142053785.png" alt="image-20201223142053785" style="zoom:50%;" />

### 极大线性无关组（列向量）

- 秩rank(A)
- spark(A)，组成线性相关向量系最小向量个数<img src="https://raw.githubusercontent.com/lengjiayi/Lecture-Notes/main/assets/image-20201223143117657.png" alt="image-20201223143117657" style="zoom:50%;" />

- 基底
- 子空间<img src="https://raw.githubusercontent.com/lengjiayi/Lecture-Notes/main/assets/image-20201223143714574.png" alt="image-20201223143714574" style="zoom:50%;" />
    - 生成线性子空间$span\{x_1,x_2,...x_n\}$
    - 交空间
    - 和空间（每个元素和）
    - 零空间
    - 像空间
    - 直接和<img src="https://raw.githubusercontent.com/lengjiayi/Lecture-Notes/main/assets/image-20201223144620472.png" alt="image-20201223144620472" style="zoom:50%;" />
    - 超平面
- 转移矩阵$B=AP\Rightarrow x=Py$

### 线性算子<img src="https://raw.githubusercontent.com/lengjiayi/Lecture-Notes/main/assets/image-20201223145847490.png" alt="image-20201223145847490" style="zoom:50%;" />

- 同构算子<img src="https://raw.githubusercontent.com/lengjiayi/Lecture-Notes/main/assets/image-20201223151036192.png" alt="image-20201223151036192" style="zoom:50%;" />

### 向量在给定基底下的表示

- $T(\alpha_1,\alpha_2,...,\alpha_n)=(\beta_1,\beta_2,...,\beta_m)A$<img src="https://raw.githubusercontent.com/lengjiayi/Lecture-Notes/main/assets/image-20201223151934242.png" alt="image-20201223151934242" style="zoom:50%;" />
    - 其中，$T(\alpha_i)=\sum_{j=1}^m\alpha_{ji}\beta_j$，为$V^m$空间的一个向量
- 矩阵相似<img src="https://raw.githubusercontent.com/lengjiayi/Lecture-Notes/main/assets/image-20201223174109052.png" alt="image-20201223174109052" style="zoom:50%;" />

---

## 第二讲 内积空间与等积变换

### 内积<img src="https://raw.githubusercontent.com/lengjiayi/Lecture-Notes/main/assets/image-20201223174816610.png" alt="image-20201223174816610" style="zoom:50%;" />

### 度量矩阵（Gram矩阵）<img src="https://raw.githubusercontent.com/lengjiayi/Lecture-Notes/main/assets/image-20201224102419882.png" alt="image-20201224102419882" style="zoom:50%;" /><img src="https://raw.githubusercontent.com/lengjiayi/Lecture-Notes/main/assets/image-20201224102434176.png" alt="image-20201224102434176" style="zoom:50%;" /><img src="https://raw.githubusercontent.com/lengjiayi/Lecture-Notes/main/assets/image-20201224102419882.png" alt="image-20201224102419882" style="zoom:50%;" /><img src="https://raw.githubusercontent.com/lengjiayi/Lecture-Notes/main/assets/image-20201224102434176.png" alt="image-20201224102434176" style="zoom:50%;" />

### 正交基

- Schmidt正交化<img src="https://raw.githubusercontent.com/lengjiayi/Lecture-Notes/main/assets/image-20201224103744541.png" alt="image-20201224103744541" style="zoom:50%;" />
- <img src="https://raw.githubusercontent.com/lengjiayi/Lecture-Notes/main/assets/image-20201224104346503.png" alt="image-20201224104346503" style="zoom:50%;" />

- 复内积线性空间（酉空间）<img src="https://raw.githubusercontent.com/lengjiayi/Lecture-Notes/main/assets/image-20201224104537577.png" alt="image-20201224104537577" style="zoom:50%;" />
- 线性泛函<img src="https://raw.githubusercontent.com/lengjiayi/Lecture-Notes/main/assets/image-20201224104845692.png" alt="image-20201224104845692" style="zoom:50%;" />
- 伴随变换<img src="https://raw.githubusercontent.com/lengjiayi/Lecture-Notes/main/assets/image-20201224110005808.png" alt="image-20201224110005808" style="zoom:50%;" />
- 正交变换<img src="https://raw.githubusercontent.com/lengjiayi/Lecture-Notes/main/assets/image-20201224141057802.png" alt="image-20201224141057802" style="zoom:50%;" />
    - <img src="https://raw.githubusercontent.com/lengjiayi/Lecture-Notes/main/assets/image-20201224143856456.png" alt="image-20201224143856456" style="zoom:50%;" />

---

## 第三讲 赋范线性空间与范数

- 范数<img src="https://raw.githubusercontent.com/lengjiayi/Lecture-Notes/main/assets/image-20201224152656763.png" alt="image-20201224152656763" style="zoom:50%;" />
    - <img src="https://raw.githubusercontent.com/lengjiayi/Lecture-Notes/main/assets/image-20201224152829567.png" alt="image-20201224152829567" style="zoom:50%;" /><img src="https://raw.githubusercontent.com/lengjiayi/Lecture-Notes/main/assets/image-20201225103015096.png" alt="image-20201225103015096" style="zoom:50%;" />
- 赋范空间 <img src="https://raw.githubusercontent.com/lengjiayi/Lecture-Notes/main/assets/image-20201225105350621.png" alt="image-20201225105350621" style="zoom:50%;" />
- <img src="https://raw.githubusercontent.com/lengjiayi/Lecture-Notes/main/assets/image-20201225105537998.png" alt="image-20201225105537998" style="zoom:50%;" />

### 矩阵范数

- <img src="https://raw.githubusercontent.com/lengjiayi/Lecture-Notes/main/assets/image-20201225110658028.png" alt="image-20201225110658028" style="zoom:50%;" />
- <img src="https://raw.githubusercontent.com/lengjiayi/Lecture-Notes/main/assets/image-20201225110725865.png" alt="image-20201225110725865" style="zoom:50%;" />
    - <img src="https://raw.githubusercontent.com/lengjiayi/Lecture-Notes/main/assets/image-20201225110738804.png" alt="image-20201225110738804" style="zoom:50%;" /><img src="https://raw.githubusercontent.com/lengjiayi/Lecture-Notes/main/assets/image-20201225111011112.png" alt="image-20201225111011112" style="zoom:50%;" />

#### 谱半径<img src="https://raw.githubusercontent.com/lengjiayi/Lecture-Notes/main/assets/image-20201225111711852.png" alt="image-20201225111711852" style="zoom:50%;" /><img src="https://raw.githubusercontent.com/lengjiayi/Lecture-Notes/main/assets/image-20201225111722369.png" alt="image-20201225111722369" style="zoom:50%;" />

- <img src="https://raw.githubusercontent.com/lengjiayi/Lecture-Notes/main/assets/image-20201225112001311.png" alt="image-20201225112001311" style="zoom:50%;" />
- <img src="https://raw.githubusercontent.com/lengjiayi/Lecture-Notes/main/assets/image-20201225112219066.png" alt="image-20201225112219066" style="zoom:50%;" />

#### 条件数<img src="https://raw.githubusercontent.com/lengjiayi/Lecture-Notes/main/assets/image-20201225113414983.png" alt="image-20201225113414983" style="zoom:50%;" />

---

## 第四讲 矩阵的特征值与奇异值分解

### 特征值

- <img src="https://raw.githubusercontent.com/lengjiayi/Lecture-Notes/main/assets/image-20201225161824195.png" alt="image-20201225161824195" style="zoom:50%;" />
- <img src="https://raw.githubusercontent.com/lengjiayi/Lecture-Notes/main/assets/image-20201225161833283.png" alt="image-20201225161833283" style="zoom:50%;" />
- <img src="https://raw.githubusercontent.com/lengjiayi/Lecture-Notes/main/assets/image-20201225161842535.png" alt="image-20201225161842535" style="zoom:50%;" />
- <img src="https://raw.githubusercontent.com/lengjiayi/Lecture-Notes/main/assets/image-20201225161848688.png" alt="image-20201225161848688" style="zoom:50%;" />
- <img src="https://raw.githubusercontent.com/lengjiayi/Lecture-Notes/main/assets/image-20201225162009281.png" alt="image-20201225162009281" style="zoom:50%;" />
- <img src="https://raw.githubusercontent.com/lengjiayi/Lecture-Notes/main/assets/image-20201225170346381.png" alt="image-20201225170346381" style="zoom:50%;" />即![image-20201225170900528](https://raw.githubusercontent.com/lengjiayi/Lecture-Notes/main/assets/image-20201225170900528.png)

### 相似矩阵<img src="https://raw.githubusercontent.com/lengjiayi/Lecture-Notes/main/assets/image-20201225171825505.png" alt="image-20201225171825505" style="zoom:50%;" />

- <img src="https://raw.githubusercontent.com/lengjiayi/Lecture-Notes/main/assets/image-20201225172016849.png" alt="image-20201225172016849" style="zoom:50%;" />
- <img src="https://raw.githubusercontent.com/lengjiayi/Lecture-Notes/main/assets/image-20201225172027687.png" alt="image-20201225172027687" style="zoom:50%;" />
- <img src="https://raw.githubusercontent.com/lengjiayi/Lecture-Notes/main/assets/image-20201225172037536.png" alt="image-20201225172037536" style="zoom:50%;" />

### 实对称矩阵的特征值问题

- <img src="https://raw.githubusercontent.com/lengjiayi/Lecture-Notes/main/assets/image-20210105153845585.png" alt="image-20210105153845585" style="zoom:50%;" />
- <img src="https://raw.githubusercontent.com/lengjiayi/Lecture-Notes/main/assets/image-20210105153858921.png" alt="image-20210105153858921" style="zoom:50%;" />![image-20210105153926997](https://raw.githubusercontent.com/lengjiayi/Lecture-Notes/main/assets/image-20210105153926997.png)

#### 谱分解



<img src="https://raw.githubusercontent.com/lengjiayi/Lecture-Notes/main/assets/image-20210105153941636.png" alt="image-20210105153941636" style="zoom:50%;" />



### Rayleigh 商

- <img src="https://raw.githubusercontent.com/lengjiayi/Lecture-Notes/main/assets/image-20210105154034443.png" alt="image-20210105154034443" style="zoom:50%;" />
- 

### Schur定理和正规矩阵

<img src="https://raw.githubusercontent.com/lengjiayi/Lecture-Notes/main/assets/image-20210105154728687.png" alt="image-20210105154728687" style="zoom:50%;" />

### 奇异值分解SVD

- <img src="https://raw.githubusercontent.com/lengjiayi/Lecture-Notes/main/assets/image-20210105171252978.png" alt="image-20210105171252978" style="zoom:50%;" />
- <img src="https://raw.githubusercontent.com/lengjiayi/Lecture-Notes/main/assets/image-20210105171303188.png" alt="image-20210105171303188" style="zoom:50%;" />
- <img src="https://raw.githubusercontent.com/lengjiayi/Lecture-Notes/main/assets/image-20210105171312607.png" alt="image-20210105171312607" style="zoom:50%;" />![image-20210105171323460](https://raw.githubusercontent.com/lengjiayi/Lecture-Notes/main/assets/image-20210105171323460.png)
- <img src="https://raw.githubusercontent.com/lengjiayi/Lecture-Notes/main/assets/image-20210105171312607.png" alt="image-20210105171312607" style="zoom:50%;" />![image-20210105171323460](https://raw.githubusercontent.com/lengjiayi/Lecture-Notes/main/assets/image-20210105171323460.png)
- <img src="https://raw.githubusercontent.com/lengjiayi/Lecture-Notes/main/assets/image-20210105171332848.png" alt="image-20210105171332848" style="zoom:50%;" />

#### 数值性质

- <img src="https://raw.githubusercontent.com/lengjiayi/Lecture-Notes/main/assets/image-20210105172523364.png" alt="image-20210105172523364" style="zoom:50%;" />

- <img src="https://raw.githubusercontent.com/lengjiayi/Lecture-Notes/main/assets/image-20210105172705971.png" alt="image-20210105172705971" style="zoom:50%;" />

## 投影分析

### 投影算子

<img src="https://raw.githubusercontent.com/lengjiayi/Lecture-Notes/main/assets/image-20210105194443534.png" alt="image-20210105194443534" style="zoom:50%;" />



<img src="https://raw.githubusercontent.com/lengjiayi/Lecture-Notes/main/assets/image-20210105195201016.png" alt="image-20210105195201016" style="zoom:50%;" />

<img src="https://raw.githubusercontent.com/lengjiayi/Lecture-Notes/main/assets/image-20210105201859491.png" alt="image-20210105201859491" style="zoom:50%;" />

<img src="https://raw.githubusercontent.com/lengjiayi/Lecture-Notes/main/assets/image-20210105202112537.png" alt="image-20210105202112537" style="zoom:50%;" />

### 典型投影

- ![image-20210105203059638](https://raw.githubusercontent.com/lengjiayi/Lecture-Notes/main/assets/image-20210105203059638.png)

- <img src="https://raw.githubusercontent.com/lengjiayi/Lecture-Notes/main/assets/image-20210105203133502.png" alt="image-20210105203133502" style="zoom:80%;" />

## 矩阵分解与广义逆矩阵

### 满秩分解

<img src="https://raw.githubusercontent.com/lengjiayi/Lecture-Notes/main/assets/image-20210105203226226.png" alt="image-20210105203226226" style="zoom:50%;" />



### 三角分解

<img src="https://raw.githubusercontent.com/lengjiayi/Lecture-Notes/main/assets/image-20210105203702834.png" alt="image-20210105203702834" style="zoom:50%;" />

<img src="https://raw.githubusercontent.com/lengjiayi/Lecture-Notes/main/assets/image-20210105203853533.png" alt="image-20210105203853533" style="zoom:50%;" />

<img src="https://raw.githubusercontent.com/lengjiayi/Lecture-Notes/main/assets/image-20210105204000906.png" alt="image-20210105204000906" style="zoom:50%;" />

<img src="https://raw.githubusercontent.com/lengjiayi/Lecture-Notes/main/assets/image-20210105204052405.png" alt="image-20210105204052405" style="zoom:50%;" />

### 正交分解

<img src="https://raw.githubusercontent.com/lengjiayi/Lecture-Notes/main/assets/image-20210105204118282.png" alt="image-20210105204118282" style="zoom:50%;" />

### 广义逆矩阵

<img src="https://raw.githubusercontent.com/lengjiayi/Lecture-Notes/main/assets/image-20210105205508431.png" alt="image-20210105205508431" style="zoom:50%;" />

#### 左逆/右逆

- <img src="https://raw.githubusercontent.com/lengjiayi/Lecture-Notes/main/assets/image-20210105205531623.png" alt="image-20210105205531623" style="zoom:50%;" />



- <img src="https://raw.githubusercontent.com/lengjiayi/Lecture-Notes/main/assets/image-20210105205616024.png" alt="image-20210105205616024" style="zoom:50%;" />



#### 广义逆

<img src="https://raw.githubusercontent.com/lengjiayi/Lecture-Notes/main/assets/image-20210105205649304.png" alt="image-20210105205649304" style="zoom:50%;" />

<img src="https://raw.githubusercontent.com/lengjiayi/Lecture-Notes/main/assets/image-20210105205910860.png" alt="image-20210105205910860" style="zoom:50%;" />



<img src="https://raw.githubusercontent.com/lengjiayi/Lecture-Notes/main/assets/image-20210105205948652.png" alt="image-20210105205948652" style="zoom:50%;" />



<img src="https://raw.githubusercontent.com/lengjiayi/Lecture-Notes/main/assets/image-20210105210121867.png" alt="image-20210105210121867" style="zoom:50%;" />



