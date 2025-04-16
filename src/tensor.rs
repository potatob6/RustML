// AI framework by potatob
// Copyright by potatob 2025-04-03

use std::{cell::RefCell, collections::VecDeque};
use std::rc::Rc;
use rand::Rng;

pub type Float = f64;

#[macro_export]
macro_rules! tensor {
    ($a: expr) => {
        Tensor::new($a)
    };

    ($($x: expr),+ $(,)?) => {
        Tensor::from_vec(vec![$($x),+])
    };
}

#[derive(Debug)]
pub struct ValueData {
    pub is_const: bool,      // 是否常量
    pub data: Float,         // 当前结点计算值
    pub grad: Float,         // 当前结点梯度
    pub op: Option<Op>,      // 当前结点运算类型
    pub parents: Vec<Value>, // 双亲结点
}

#[derive(Debug, Clone, Copy)]
pub enum Op {
    Add,       // 加法结点
    Mul,       // 乘法结点
    ReLU,      // ReLU结点 ReLU(x)
    Recip,     // 倒数结点  1/x
    Sigmoid,   // sigmoid结点 sigmoid(x)
    Rev,       // 相反数结点 -x
    Exp,       // 自然指数结点 Exp(x)
    Ln,        // 自然对数结点 ln(x)
}

pub type Value = Rc<RefCell<ValueData>>;

impl ValueData {
    pub fn new(data: Float) -> Self {
        ValueData {
            is_const: false,
            data,
            grad: 0.0,
            op: None,
            parents: Vec::new(),
        }
    }

    pub fn to_const(self) -> Self {
        Self {
            is_const: true,
            ..self
        }
    }

    pub fn to_var(self) -> Self {
        Self {
            is_const: false,
            ..self
        }
    }

    pub fn clear_grad(&mut self) {
        self.grad = 0.0;
    }
}

// 显式乘法函数
pub fn value_mul(a: Value, b: Value) -> Value {
    let result = Rc::new(RefCell::new(ValueData {
        is_const: a.borrow().is_const && b.borrow().is_const,
        data: a.borrow().data * b.borrow().data,
        grad: 0.0,
        op: Some(Op::Mul),
        parents: vec![a.clone(), b.clone()],
    }));
    result
}

// 显式加法函数
pub fn value_add(a: Value, b: Value) -> Value {
    let result = Rc::new(RefCell::new(ValueData {
        is_const: a.borrow().is_const && b.borrow().is_const,
        data: a.borrow().data + b.borrow().data,
        grad: 0.0,
        op: Some(Op::Add),
        parents: vec![a.clone(), b.clone()],
    }));
    result
}

// 结点梯度更新
pub fn value_optimize(a: Value, lr: Float) {
    let now_grad = a.borrow().grad;

    a.borrow_mut().data -= lr * now_grad;
}

// 显式ReLU激活
pub fn value_relu(a: Value) -> Value {
    Rc::new(RefCell::new(ValueData {
        is_const: a.borrow().is_const,
        data: if a.borrow().data >= 0.0 { a.borrow().data } else { 0.0 },
        grad: 0.0,
        op: Some(Op::ReLU),
        parents: vec![a.clone()],
    }))
}

// 相反数
pub fn value_rev(a: Value) -> Value {
    Rc::new(RefCell::new(ValueData {
        is_const: a.borrow().is_const,
        data: -a.borrow().data,
        grad: 0.0,
        op: Some(Op::Rev),
        parents: vec![a.clone()],
    }))
}

// 显式Sigmoid激活
pub fn value_sigmoid(a: Value) -> Value {
    Rc::new(RefCell::new(ValueData {
        is_const: a.borrow().is_const,
        data: sigmoid(a.borrow().data),
        grad: 0.0,
        op: Some(Op::Sigmoid),
        parents: vec![a.clone()]
    }))
}

// 倒数
pub fn value_recip(a: Value) -> Value {
    Rc::new(RefCell::new(ValueData {
        is_const: a.borrow().is_const,
        data: 1.0 / a.borrow().data,
        grad: 0.0,
        op: Some(Op::Recip),
        parents: vec![a.clone()],
    }))
}

// Exp(x)
pub fn value_exp(a: Value) -> Value {
    Rc::new(RefCell::new(ValueData {
        is_const: a.borrow().is_const,
        data: E.powf(a.borrow().data),
        grad: 0.0,
        op: Some(Op::Exp),
        parents: vec![a.clone()],
    }))
}

// ln(x)
pub fn value_ln(a: Value) -> Value {
    Rc::new(RefCell::new(ValueData {
        is_const: a.borrow().is_const,
        data: a.borrow().data.ln(),
        grad: 0.0,
        op: Some(Op::Ln),
        parents: vec![a.clone()],
    }))
}

pub fn backward(v: Value) {
    let mut queue = VecDeque::new();
    // 初始化输出节点梯度
    v.borrow_mut().grad = 1.0;
    if v.borrow().is_const {
        return;
    }

    queue.push_back(v);
    
    while !queue.is_empty() {
        // 逆序反向传播
        let node_ref = queue.pop_front().unwrap();
        let node_ref = node_ref.borrow();

        for parent in &node_ref.parents {
            if !parent.borrow().is_const {
                queue.push_back(parent.clone());
            }
        }
        if let Some(op) = node_ref.op {
            match op {
                Op::Add => {
                    let grad = node_ref.grad;
                    node_ref.parents[0].borrow_mut().grad += grad;
                    node_ref.parents[1].borrow_mut().grad += grad;
                }
                Op::Mul => {
                    let grad = node_ref.grad;
                    let a = &node_ref.parents[0];
                    let b = &node_ref.parents[1];
                    let a_data = a.borrow().data;
                    let b_data = b.borrow().data;
                    a.borrow_mut().grad += grad * b_data;
                    b.borrow_mut().grad += grad * a_data;
                }
                Op::ReLU => {
                    let grad = node_ref.grad;
                    let a = &node_ref.parents[0];
                    if a.borrow().data >= 0.0 {
                        a.borrow_mut().grad += grad;
                    }
                }
                Op::Recip => {
                    let grad = node_ref.grad;
                    let a = &node_ref.parents[0];
                    let direv_data = -1.0 / (a.borrow().data * a.borrow().data);
                    a.borrow_mut().grad += grad * direv_data;
                }
                Op::Sigmoid => {
                    let grad = node_ref.grad;
                    let a = &node_ref.parents[0];
                    let a_data = a.borrow().data;
                    let now_grad = sigmoid(a_data) * (1.0 - sigmoid(a_data));
                    a.borrow_mut().grad += grad * now_grad;
                }
                Op::Rev => {
                    let grad = node_ref.grad;
                    let a = &node_ref.parents[0];
                    a.borrow_mut().grad += -1.0 * grad;
                }
                Op::Exp => {
                    let grad = node_ref.grad;
                    let a = &node_ref.parents[0];
                    a.borrow_mut().grad += grad * node_ref.data;
                }
                Op::Ln => {
                    let grad = node_ref.grad;
                    let a = &node_ref.parents[0];
                    a.borrow_mut().grad += grad * ( 1.0 / node_ref.data);
                }
            }
        }
    }
}

use crate::shape::{self, Indexable, Shape, Transposable};

pub struct Tensor {
    pub shape: shape::Shape,
    // pub row_c: usize,
    // pub col_c: usize,
    pub data: Vec<Value>,
}

impl Tensor {
    // 0填充Tensor
    pub fn zeros(shape: &Vec<usize>) -> Self {
        let shape = Shape::with_vec(shape);

        let mut v = Vec::with_capacity(shape.size());
        for _ in 0..shape.size() {
            v.push(Rc::new(RefCell::new(ValueData::new(0.0))));
        }
        Self { 
            shape, 
            data: v,
        }
    }

    pub fn fill(v: Float, shape: &Vec<usize>) -> Self {
        let shape = Shape::with_vec(shape);

        let v = (0..shape.size()).map(|_| {
            Rc::new(RefCell::new(ValueData::new(v)))
        }).collect::<Vec<Value>>();

        Self {
            shape,
            data: v,
        }
    }

    pub fn new(v: Float) -> Self {
        let shape = Shape::scalar();

        Self {
            shape, 
            data: vec![Rc::new(RefCell::new(ValueData::new(v)))],
        }
    }

    pub fn clear_grad(&mut self) {
        self.data.iter().for_each(|x| {
            x.borrow_mut().clear_grad();
        });
    }

    pub fn to_const(self) -> Self {
        self.data.iter().for_each(|x| {
            x.borrow_mut().is_const = true
        });
        self
    }

    pub fn to_var(self) -> Self {
        self.data.iter().for_each(|x| {
            x.borrow_mut().is_const = false
        });
        self
    }

    // Vec导入Tensor
    pub fn from_vec(v: Vec<Float>) -> Self {
        
        let size = v.len();

        let shape = Shape::with_vec(&vec![size]);

        let mut data = Vec::with_capacity(size);
        for value in v {
            data.push(Rc::new(RefCell::new(ValueData::new(value))));
        }
        Self {
            shape,
            data: data,
        }
    }

    // 随机参数初始化Tensor
    pub fn rand(range: Float, shape: &Vec<usize>) -> Self {
        let shape = Shape::with_vec(shape);
        let size = shape.size();

        let mut rand = rand::rng();
        let mut v = Vec::with_capacity(size);
        for _ in 0..size {
            v.push(Rc::new(RefCell::new(ValueData::new(rand.random::<Float>() * range))));
        }
        Self { 
            shape,
            data: v,
        }
    }

    // 向量点积操作
    pub fn dot_mul(&self, other: &Self) -> Self {
        let new_shape = self.shape.shape_range(&other.shape);

        let data = (0..new_shape.size()).map(|x| {
            let v = new_shape.inverse_get(x);

            let self_cell = self.get(&v);
            let other_cell = other.get(&v);

            value_mul(self_cell, other_cell)
        })
        .collect::<Vec<_>>();

        Self {
            shape: new_shape,
            data,
        }
    }

    pub fn shape(&self) -> Vec<usize> {
        self.shape.shape()
    }

    pub fn raw_first(&self) -> Value {
        unsafe { self.data.get_unchecked(0).clone() }
    }

    // 二维坐标获取值
    pub fn get(&self, idx: &Vec<usize>) -> Value {
        unsafe { Rc::clone(self.data.get_unchecked(self.shape.get(idx))) }
    }

    pub fn get_mut(&mut self, idx: &Vec<usize>) -> &mut Value {
        unsafe { self.data.get_unchecked_mut(self.shape.get(idx)) }
    }

    pub fn transpose(self, axis: &Vec<usize>) -> Self {
        Self {
            shape: self.shape.transpose(axis),
            ..self
        }
    }

    // 重塑Tensor形状（连续性保持）
    pub fn reshape(self, new_shape: &Vec<usize>) -> Self {
        let new_shape = Shape::with_vec(new_shape);

        Self {
            shape: new_shape,
            ..self
        }
    }

    // Tensor相加
    pub fn add(&self, other: &Self) -> Self {
        let new_shape = self.shape.shape_range(&other.shape);
        let size = new_shape.size();

        let data = (0..size).map(|x| {
            let v = new_shape.inverse_get(x);

            let self_cell = self.get(&v);
            let other_cell = other.get(&v);

            value_add(self_cell, other_cell)
        })
        .collect::<Vec<_>>();

        Self {
            shape: new_shape,
            data,
        }
    }

    // Tensor相乘
    pub fn mul(&self, other: &Self) -> Self {
        let medium_size = self.shape.get_axis_size(self.shape.num_axis() - 1);
        let before_size = self.shape.get_axis_size(self.shape.num_axis() - 2);
        let after_size = other.shape.get_axis_size(other.shape.num_axis() - 1);

        let tail = vec![before_size, after_size];

        let new_shape = self.shape.shape_range_restrict_tail(&other.shape, &tail);

        let data = (0..new_shape.size()).map(|x| {
            let mut v1 = new_shape.inverse_get(x);
            let mut v2 = v1.clone();
            let axis = v1.len();

            let sum = (0..medium_size).fold(Rc::new(RefCell::new(ValueData::new(0.0))), |acc, x| {
                unsafe {
                    *v1.get_unchecked_mut(axis - 1) = x;
                    *v2.get_unchecked_mut(axis - 2) = x;

                    let self_cell = self.get(&v1);
                    let other_cell = other.get(&v2);

                    value_add(acc.clone(), value_mul(self_cell, other_cell))
                }
            });
            sum
        })
        .collect::<Vec<_>>();

        Self {
            shape: new_shape,
            data,
        }
    }

    // relu函数
    pub fn relu(&self) -> Self {
        let shape = self.shape.clone();

        let data = self.data.iter().map(|x| {
            let node = x.clone();
            value_relu(node)
        }).collect::<Vec<_>>();

        Self {
            shape,
            data,
        }
    }

    pub fn rev(&self) -> Self {
        let shape = self.shape.clone();
        let data = self.data.iter().map(|x| {
            value_rev(x.clone())
        }).collect();

        Self {
            shape,
            data,
        }
    }

    pub fn pow(&self, times: usize) -> Self {
        let shape = self.shape.clone();
        let data = self.data.iter().map(|x| {
            let mut r = x.clone();
            for _ in 1..times {
                r = value_mul(r.clone(), x.clone());
            }
            r
        }).collect();
        Self {
            shape,
            data,
        }
    }

    pub fn ln(&self) -> Self {
        let shape = self.shape.clone();
        let data = self.data.iter().map(|x| {
            value_ln(x.clone())
        }).collect();
        Self {
            shape,
            data,
        }
    }

    // sigmoid函数
    pub fn sigmoid(&self) -> Self {
        let shape = self.shape.clone();

        let data = self.data.iter().map(|x| {
            let node = x.clone();
            value_sigmoid(node)
        }).collect::<Vec<_>>();

        Self {
            shape,
            data,
        }
    }

    // 沿轴折叠函数
    pub fn axis_fold(&self, axis_flag: &Vec<bool>, mut f: impl FnMut(&mut Rc<RefCell<ValueData>>, &Rc<RefCell<ValueData>>)) -> Self {
        let now_shape = self.shape.shape();
        let new_shape = axis_cast(&now_shape, axis_flag);

        let mut s = Self::zeros(&new_shape);

        self.data.iter().enumerate()
            .for_each(|x| {
                let v = self.shape.inverse_get(x.0);
                let c = axis_cast(&v, axis_flag);

                let target = s.get_mut(&c);
                f(target, x.1);
            });

        s
    }

    // 沿轴向求和
    pub fn sum(&self, axis_flag: &Vec<bool>) -> Self {
        Self::axis_fold(&self, axis_flag, |x, y| {
            *x = value_add(x.clone(), y.clone());
        })
    }

    // 倒数加微小误差
    pub fn recip_epsilon(&self, epsilon: Float) -> Self {
        let mut data = Self::add(&self, &Tensor::new(epsilon));
        let data1 = data.data.iter().map(|x| {
            value_recip(x.clone())
        })
        .collect();

        data.data = data1;
        data
    }

    // 倒数，添加微小误差
    pub fn recip(&self) -> Self {
        Self::recip_epsilon(&self, 1e-8)
    }

    // Exp函数
    pub fn exp(&self) -> Self {
        let shape = self.shape.clone();

        let data = self.data.iter().map(|x| {
            value_exp(x.clone())
        })
        .collect();

        Self {
            shape,
            data,
        }
    }

    // softmax函数，添加微小误差
    pub fn softmax(&self) -> Self {
        let shape = self.shape.clone();

        let axis_flag = (0..shape.num_axis())
            .map(|x| {
                if x == shape.num_axis() - 1 {
                    true
                } else {
                    false
                }
            }).collect();

        let self_exp = self.exp();

        let t1 = self_exp.sum(&axis_flag);
        let t1 = t1.recip_epsilon(1e-8);

        self_exp.dot_mul(&t1)
    }


    // softmax函数，自定义微小误差
    pub fn softmax_epsilon(&self, epsilon: Float) -> Self {
        let shape = self.shape.clone();

        let axis_flag = (0..shape.num_axis())
            .map(|x| {
                if x == shape.num_axis() - 1 {
                    true
                } else {
                    false
                }
            }).collect();

        let self_exp = self.exp();

        let t1 = self_exp.sum(&axis_flag);
        let t1 = t1.recip_epsilon(epsilon);

        self_exp.dot_mul(&t1)
    }

    // 打印Tensor计算值
    pub fn raw_data(&self) -> String {
        let mut s = String::new();
        s.push('[');
        self.data.iter().enumerate().for_each(|x| {
            if x.0 == 0 {
                s += &x.1.borrow().data.to_string();
            } else {
                s += &format!(", {}", &x.1.borrow().data.to_string());
            }
        });
        s.push(']');
        s
    }

    // // 打印Tensor梯度
    pub fn raw_grad(&self) -> String {
        let mut s = String::new();
        s.push('[');
        self.data.iter().enumerate().for_each(|x| {
            if x.0 == 0 {
                s += &x.1.borrow().grad.to_string();
            } else {
                s += &format!(", {}", &x.1.borrow().grad.to_string());
            }
        });
        s.push(']');
        s
    }

    pub fn calc<F>(&self, mut f: F) -> Self
        where F: FnMut(&Self) -> Self {
        f(self)
    }

    // 均方误差函数
    pub fn mse(&self, label: &Self) -> Self {
        let t1 = self.add(&label.rev()).pow(2);

        let last_axis_num = t1.shape.get_axis_size(t1.shape.num_axis() - 1);
        let n = Self::new(last_axis_num as Float).to_const();
        let n = n.recip_epsilon(0.0);
        let t1 = t1.dot_mul(&n);
        let t1 = t1.sum(&(0..t1.shape.num_axis()).map(|x| {
            if x == t1.shape.num_axis() - 1 {
                true
            } else {
                false
            }
        }).collect());
        t1
    }

    pub fn cross_entropy(&self, label: &Self) -> Self {
        let t1 = self.ln();
        let t1 = label.dot_mul(&t1);

        let t1 = t1.sum(&(0..t1.shape.num_axis()).map(|x| {
            if x == t1.shape.num_axis() - 1 {
                true
            } else {
                false
            }
        }).collect());
        t1.rev()
    }

    // 梯度下降参数更新法
    pub fn optimize(&mut self, lr: Float) {
        for node in &self.data {
            value_optimize(node.clone(), lr);
        }
    }
}

pub const E: Float = 2.71828182;

// sigmoid函数
pub fn sigmoid(v: Float) -> Float {
    1.0 / (1.0 + E.powf(-v))
}

fn axis_cast(axis: &Vec<usize>, axis_flag: &Vec<bool>) -> Vec<usize> {
    axis.iter()
        .zip(axis_flag.iter())
        .map(|x| {
            if *x.1 {
                1
            } else {
                *x.0
            }
        })
        .collect::<Vec<_>>()
}

// 均方误差函数
// pub fn tensor_mse(label: &Tensor, predict: &Tensor) -> Tensor {
//     assert_eq!(label.row_c, predict.row_c);
//     assert_eq!(label.col_c, predict.col_c);

//     let size = label.row_c * label.col_c;

//     let v = label.data.iter().cloned().zip(
//         predict.data.iter().cloned()
//     )
//     .map(|x| {
//         let t1 = value_add(x.0, value_rev(x.1));
//         value_mul(t1.clone(), t1.clone())
//     })
//     .fold(Rc::new(RefCell::new(ValueData::new(0.0))), |acc, x| {
//         value_add(acc, x).clone()
//     });

//     let size = Rc::new(RefCell::new(ValueData::new(size as Float)));
//     let mse = value_mul(v, value_recip(size));
    
//     Tensor {
//         row_c: 1,
//         col_c: 1,
//         data: vec![mse],
//         extendable: false,
//     }
// }

// // 交叉熵函数
// pub fn tensor_cross_entropy(label: &Tensor, predict: &Tensor) -> Tensor {
//     assert_eq!(label.col_c, predict.col_c);
//     assert_eq!(label.row_c, predict.row_c);

//     let sum = Rc::new(RefCell::new(ValueData::new(0.0)));
//     let zip1 = label
//         .data.iter().cloned()
//         .zip(
//             predict.data.iter().cloned()
//         ).collect::<Vec<(Value, Value)>>();

//     let s = zip1.iter().fold(sum, |acc, x| {
//         value_add(acc.clone(), value_mul(x.0.clone(), value_ln(x.1.clone()))).clone()
//     });

//     let s = value_rev(s.clone());

//     Tensor {
//         row_c: 1,
//         col_c: 1,
//         data: vec![s.clone()],
//         extendable: false,
//     }
// }