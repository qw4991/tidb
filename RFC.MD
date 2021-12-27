# TisorFlow

+ 作者: 张原嘉、徐逸凡、兰海

## 项目简介
TisorFlow 的目标是扩展 TiDB 的计算能力，使其可以用来进行比较通用的分布式机器学习模型训练，如分布式的训练 MLP、CNN 等神经网络，用来进行图像识别。

## 背景 & 动机
TiDB 良好的代码结构，原生的分布式计算能力，使得我们能很方便的将分布式模型训练实现在其之上。
同时结合 SQL 接口，及已有的分布式存储能力，能让用户以 SQL 的方式来读取、预处理数据及训练模型。

## 项目设计
项目分为三部分：1）SQL 接口，2）分布式训练框架，3）分布式模型训练；

SQL 接口我们引入下面 2 种语法及新的内置函数 Apply：
+ 创建模型的语句 `CREATE MODEL name WITH 'type'='CNN', 'k1'='v1', 'k2'='v2'`: 用来创建和定义模型参数;
+ 训练模型的语句 `TRAIN MODEL name WITH 'select ...'`: 用来预处理数据及进行模型训练;
+ 内置函数 `Apply`: 用来使用模型，进行宠物图片识别 `select * from pets where apply('cat-reg-model', pets.img)='cat'`;

模型训练会利用多个 TiDB 节点进行分布式训练，接收到 TRAIN MODEL 指令的 TiDB 会成 Master，进行训练任务的初始化话；
然后利用 TiDB 已有的 CopTask 接口，进行对其他 TiDB 实例的任务下发；

每次训练会进行多轮，每一轮 Master 把当前的模型参数及训练数据传递给 Slaver，Slaver 训练完成后返回当前的梯度；
Master 会汇总所有 Slaver 的梯度，对模型进行迭代，然后用迭代后的模型进行下一轮训练，直至收敛。
