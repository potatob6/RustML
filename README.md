# RustML: A Simple Rust Machine Learning Framework

RustML 是一个用 Rust 实现的轻量级机器学习框架，专注于张量操作和自动微分。以下通过一个具体示例演示如何构建神经网络、使用均方误差（MSE）训练模型，并执行反向传播和参数优化。


## 目录
- [安装](#安装)
- [示例：2输入层、2隐藏层、4输出层的神经网络](#示例-2输入层-2隐藏层-4输出层的神经网络)
  - [1. 定义网络架构](#1-定义网络架构)
  - [2. 创建训练数据](#2-创建训练数据)
  - [3. 初始化模型参数](#3-初始化模型参数)
  - [4. 前向传播](#4-前向传播)
  - [5. 用MSE计算损失](#5-用mse计算损失)
  - [6. 反向传播与梯度下降](#6-反向传播与梯度下降)
  - [7. 完整训练循环](#7-完整训练循环)
- [核心函数参考](#核心函数参考)
- [注意事项](#注意事项)


## 安装
在 `Cargo.toml` 中添加依赖：
```toml
[dependencies]
rand = "0.9.0"
rustml = { path = "path/to/RustML" }  # 本地路径
```

## 示例：2 输入层、2 隐藏层、4 输出层的神经网络
### 1. 定义网络架构
构建如下结构的神经网络：
- 输入层：2 个特征
- 隐藏层 1：3 个神经元（ReLU 激活）
- 隐藏层 2：4 个神经元（ReLU 激活）
- 输出层：4 个神经元（线性激活）

### 2. 创建训练数据
生成随机输入数据和合成标签（以回归任务为例）：
```rust
use rustml::tensor::{Tensor, Float};

// 输入数据：10个样本，2个特征（形状：[10, 2]）
let x = Tensor::rand(1.0, &vec![10, 2]);

// 目标标签：10个样本，4个输出（形状：[10, 4]）
let y = Tensor::rand(1.0, &vec![10, 4]);
```

### 3. 初始化模型参数
使用随机初始化各层的权重和偏置：
```rust
// 第1层：输入（2）→ 隐藏层1（3）
let w1 = Tensor::rand(0.1, &vec![2, 3]);  // 权重矩阵：[2, 3]
let b1 = Tensor::zeros(&vec![3]);         // 偏置：[3]

// 第2层：隐藏层1（3）→ 隐藏层2（4）
let w2 = Tensor::rand(0.1, &vec![3, 4]);  // 权重矩阵：[3, 4]
let b2 = Tensor::zeros(&vec![4]);         // 偏置：[4]

// 输出层：隐藏层2（4）→ 输出（4）
let w_out = Tensor::rand(0.1, &vec![4, 4]);  // 权重矩阵：[4, 4]
let b_out = Tensor::zeros(&vec![4]);         // 偏置：[4]
```

### 4. 前向传播
通过矩阵乘法和激活函数实现前向传播：
```rust
// 第1层前向传播
let z1 = x.mul(&w1).add(&b1);  // 矩阵乘法 + 偏置
let a1 = z1.relu();             // 应用ReLU激活函数

// 第2层前向传播
let z2 = a1.mul(&w2).add(&b2);
let a2 = z2.relu();             // 应用ReLU激活函数

// 输出层（线性激活）
let logits = a2.mul(&w_out).add(&b_out);
```


### 5. 用 MSE 计算损失
计算预测值与真实标签的均方误差（MSE）：
```rust
let loss = logits.mse(&y);  // 计算MSE损失
```

### 6. 反向传播与梯度下降
反向传播：计算损失对所有参数的梯度
参数优化：使用梯度下降更新参数
```rust
// 1. 反向传播计算梯度（从损失张量开始）
loss.backward(loss.raw_first());

// 2. 梯度下降更新参数（学习率lr=0.01）
let lr = 0.01;
w1.optimize(lr);  // 更新权重w1
b1.optimize(lr);  // 更新偏置b1
w2.optimize(lr);  // 更新权重w2
b2.optimize(lr);  // 更新偏置b2
w_out.optimize(lr);  // 更新输出层权重
b_out.optimize(lr);  // 更新输出层偏置

// 3. 清除梯度（为下一次迭代做准备）
w1.clear_grad();
b1.clear_grad();
w2.clear_grad();
b2.clear_grad();
w_out.clear_grad();
b_out.clear_grad();
```

### 7. 完整训练循环
将上述步骤整合到训练循环中（以 1000 次迭代为例）：
```rust
for epoch in 0..1000 {
    // 前向传播
    let z1 = x.mul(&w1).add(&b1);
    let a1 = z1.relu();
    let z2 = a1.mul(&w2).add(&b2);
    let a2 = z2.relu();
    let logits = a2.mul(&w_out).add(&b_out);
    
    // 计算损失
    let loss = logits.mse(&y);
    
    // 反向传播与参数更新
    loss.backward(loss.raw_first());
    w1.optimize(lr);
    b1.optimize(lr);
    w2.optimize(lr);
    b2.optimize(lr);
    w_out.optimize(lr);
    b_out.optimize(lr);
    
    // 清除梯度
    w1.clear_grad();
    b1.clear_grad();
    w2.clear_grad();
    b2.clear_grad();
    w_out.clear_grad();
    b_out.clear_grad();
    
    // 打印训练进度
    if epoch % 100 == 0 {
        println!("Epoch {}: Loss = {}", epoch, loss.raw_data()[0]);
    }
}
```

## 核心函数参考
|函数名|描述|
|-|-|
|Tensor::rand(range, shape)|创建随机值张量，取值范围 [0, range)，形状由 shape 指定（如 &[m, n]）|
|Tensor::zeros(shape)|创建全零张量|
|Tensor::mul(other)|矩阵乘法（仅支持 2D 张量，形状需满足矩阵乘法规则：左张量列数 = 右张量行数）|
|Tensor::add(other)|元素级加法（支持广播机制，形状需相容，如 [m, n] + [n] 或 [1, n]）|
|Tensor::relu()|对张量所有元素应用 ReLU 激活函数（\(f(x) = \max(0, x)\)）|
|Tensor::mse(label)|计算当前张量与标签张量的均方误差（MSE），公式为 \(\frac{1}{n} \sum (x_i - y_i)^2\)|
|Tensor::backward(start)|从指定节点（通常是损失张量）开始反向传播，递归计算所有依赖节点的梯度|
|Tensor::optimize(lr)|使用梯度下降更新参数，公式为 \(\text{param} -= \text{lr} \times \text{gradient}\)|
|Tensor::clear_grad()|清除张量的梯度值，将存储梯度的内部数组重置为零，避免梯度累加|

## 注意事项
### 1. 参数初始化
过小的初始权重可能导致梯度消失，过大可能导致梯度爆炸，建议使用 Xavier/Glorot 初始化或 He 初始化（可扩展框架实现）。

### 2. 性能优化
当前实现为简单演示，未使用并行计算或 GPU 加速，实际项目中可考虑集成 ndarray 或 rust-cuda 提升性能。

### 3. 错误处理
张量操作会检查形状合法性，若出现 panic 或寻址错误，请检查输入张量的维度是否符合函数要求。

### 4. 暂无Dropout过拟合解决方案
该项目是初创项目，暂时不支持Dropout解决方案，当然可以手动实现，后续版本会实现。

### 5. 暂未实现Adam等其他优化器的算法
每次Backward算法后会存储梯度信息，可以手动处理梯度信息然后更新。
RustML是暂时是轻量级机器学习框架，欢迎提交Issue或PR为RustML实现各种优化器。

---
通过以上示例，你可以快速掌握 RustML 的核心工作流。框架设计简洁，易于扩展，适合作为机器学习入门实践或小型项目的底层库。欢迎提交 Issue 或 PR，共同完善 RustML！
