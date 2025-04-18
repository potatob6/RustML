#[derive(Debug, Clone)]
pub struct Shape {
    _origin: Vec<usize>,
    _view: Vec<usize>,
    _strides: Vec<usize>,
    _size: usize,
    _restrict_mode: bool,
}

pub trait Indexable<T> {
    fn get(&self, v: &Vec<T>) -> usize;
}

pub trait Transposable<T> {
    fn transpose(self, trans: &Vec<T>) -> Self;
}

impl Shape {
    pub fn scalar() -> Self {
        Self {
            _origin: vec![1],
            _view: vec![0],
            _strides: vec![1],
            _size: 1,
            _restrict_mode: false,
        }
    }

    pub fn with_vec(v: &Vec<usize>) -> Self {
        let mut t_strides = Vec::with_capacity(v.len());

        let mut stride_cum = 1;
        for i in (0..v.len()).rev() {
            t_strides.push(stride_cum);
            stride_cum *= unsafe { v.get_unchecked(i) };
        }
        t_strides.reverse();

        Self {
            _origin: v.clone(),
            _view: (0..v.len()).collect::<Vec<usize>>(),
            _strides: t_strides,
            _size: stride_cum,
            _restrict_mode: false,
        }
    }

    pub fn num_axis(&self) -> usize {
        self._origin.len()
    }

    pub fn size(&self) -> usize {
        self._size
    }

    pub fn transpose_restore(&mut self) {
        self._view = (0..self._origin.len()).collect::<Vec<usize>>();
    }

    pub fn shape(&self) -> Vec<usize> {
        let mut shape = Vec::with_capacity(self._origin.len());
        self._view.iter()
            .for_each(|x| {
                unsafe {
                    shape.push(*self._origin.get_unchecked(*x));
                }
            });
        shape
    }

    pub fn is_scalar(&self) -> bool {
        self._origin.len() == 1 && unsafe { *self._origin.get_unchecked(0) == 1 }
    }

    pub fn is_vector(&self) -> bool {
        self._origin.len() == 1
    }

    pub fn restrict(self) -> Self {
        Self {
            _restrict_mode: true,
            ..self
        }
    }

    pub fn shape_range(&self, other: &Self) -> Self {
        use crate::zip_longest;

        let self_iter = self._view
            .iter()
            .rev()
            .map(|x| {
                unsafe { self._origin.get_unchecked(*x) }
        });
        let other_iter = other._view
            .iter()
            .rev()
            .map(|x| {
                unsafe { other._origin.get_unchecked(*x) }
        });

        let mut clamped_shape = zip_longest::zip_longest(self_iter, other_iter)
            .map(|x| {
                match x.0 {
                    Some(a) => {
                        match x.1 {
                            Some(b) => if a >= b { *a } else { *b }, 
                            None => *a,
                        }
                    }
                    None => {
                        *x.1.unwrap()
                    }
                }
            })
            .collect::<Vec<_>>();

        clamped_shape.reverse();


        let len = clamped_shape.len();
        let mut t_strides = Vec::with_capacity(clamped_shape.len());
        let mut stride_cum = 1;
        for i in (0..clamped_shape.len()).rev() {
            t_strides.push(stride_cum);
            stride_cum *= unsafe { clamped_shape.get_unchecked(i) };
        }
        t_strides.reverse();

        Self {
            _origin: clamped_shape,
            _view: (0..len).collect::<Vec<usize>>(),
            _strides: t_strides,
            _size: stride_cum,
            _restrict_mode: false,
        }
    }

    pub fn shape_range_restrict_tail(&self, other: &Self, tail: &Vec<usize>) -> Self {
        use crate::zip_longest;

        let self_iter = self._view
            .iter()
            .rev()
            .map(|x| {
                unsafe { self._origin.get_unchecked(*x) }
        });
        let other_iter = other._view
            .iter()
            .rev()
            .map(|x| {
                unsafe { other._origin.get_unchecked(*x) }
        });

        let mut clamped_shape = zip_longest::zip_longest(self_iter, other_iter)
            .map(|x| {
                match x.0 {
                    Some(a) => {
                        match x.1 {
                            Some(b) => if a >= b { *a } else { *b }, 
                            None => *a,
                        }
                    }
                    None => {
                        *x.1.unwrap()
                    }
                }
            })
            .collect::<Vec<_>>();

        clamped_shape.reverse();
        clamped_shape.iter_mut().rev()
            .zip(tail.iter().rev())
            .for_each(|x| {
                *x.0 = *x.1;
            });


        let len = clamped_shape.len();
        let mut t_strides = Vec::with_capacity(clamped_shape.len());
        let mut stride_cum = 1;
        for i in (0..clamped_shape.len()).rev() {
            t_strides.push(stride_cum);
            stride_cum *= unsafe { clamped_shape.get_unchecked(i) };
        }
        t_strides.reverse();

        Self {
            _origin: clamped_shape,
            _view: (0..len).collect::<Vec<usize>>(),
            _strides: t_strides,
            _size: stride_cum,
            _restrict_mode: false,
        }    
    }

    pub fn inverse_get(&self, idx: usize) -> Vec<usize> {
        self._view.iter().map(|x| {
            unsafe { (idx / self._strides.get_unchecked(*x)) % self._origin.get_unchecked(*x) }
        })
        .collect::<Vec<_>>()
    }

    pub fn get_axis_size(&self, axis: usize) -> usize {
        unsafe {
            *self._origin.get_unchecked(*self._view.get_unchecked(axis))
        }
    }

    pub fn reshape_tail(self, tail_num: usize,tail_shape: &mut Vec<usize>) -> Self {
        let axis_num = self.num_axis();

        let mut front = self._origin.iter().enumerate().filter(|x| {
            x.0 < axis_num - tail_num
        })
        .map(|x| {
            *x.1
        })
        .collect::<Vec<_>>();

        front.append(tail_shape);
        
        Shape::with_vec(&front)
    }
}

impl Indexable<usize> for Shape {
    fn get(&self, v: &Vec<usize>) -> usize {
        if self.is_scalar() {
            return 0;
        }
        return v.iter().zip(self._view.iter())
            .fold(0, |acc, x| {
                unsafe {
                    acc + (x.0 % self._origin.get_unchecked(*x.1)) * self._strides.get_unchecked(*x.1)
                }
            });
    }
}

impl Transposable<usize> for Shape {
    fn transpose(self, v: &Vec<usize>) -> Self {
        let new_view = v.iter().map(|x| {
            unsafe {
                *self._view.get_unchecked(*x)
            }
        }).collect();
        Self {
            _view: new_view,
            ..self
        }
    }
}