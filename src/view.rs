use godot::prelude::*;
use nalgebra::{DMatrix, Dyn, Scalar, VecStorage};
use strum_macros::AsRefStr;

use crate::mat::{DType, Mat};

#[derive(Clone, AsRefStr)]
pub enum ViewType {
    Identity,
    Real,
    Imag,
}

#[derive(Clone)]
pub enum Shape {
    Identity,
    Rect {
        start: (usize, usize),
        shape: (usize, usize),
    },
    Index {
        indices: Vec<(usize, usize)>,
    },
}

#[derive(Clone)]
pub struct View {
    pub mat: Gd<Mat>,
    pub view_type: ViewType,
    pub shape: Shape
}

impl View {
    pub fn new(mat: Gd<Mat>, view_type: ViewType, shape: Shape) -> Self {
        Self { mat, view_type, shape }
    }
}

impl Shape {
    pub fn intersect(&self, parent: &Shape) -> Shape {
        match (self, parent) {
            (Shape::Identity, _) => parent.clone(),
            (_, Shape::Identity) => self.clone(),

            (Shape::Rect { start: s1, shape }, Shape::Rect { start: s2, .. }) => {
                Shape::Rect {
                    start: (s1.0 + s2.0, s1.1 + s2.1),
                    shape: *shape,
                }
            }
            (Shape::Rect { start, shape }, Shape::Index { indices }) => {
                Shape::Index {
                    indices: indices.iter().skip(start.1).take(shape.1).copied().collect()
                }
            }

            (Shape::Index { indices: i1 }, Shape::Index { indices: i2 }) => {
                Shape::Index {
                    indices: i1.iter().map(|i| i2[i.1]).collect()
                }
            },

            (Shape::Index { indices }, Shape::Rect { start, .. }) => {
                Shape::Index {
                    indices: indices.iter().map(|i| (i.0 + start.0, i.1 + start.1)).collect()
                }
            }
        }
    }

    pub fn iter<'a, T>(&'a self, mat: &'a DMatrix<T>) -> impl Iterator<Item = &'a T>
    where
        T: Scalar
    {
        self.indices(mat.shape()).map(|(i, j)| &mat[(i, j)])
    }

    pub fn shape(&self, in_shape: (usize, usize)) -> (usize, usize) {
        match self {
            Shape::Identity => in_shape,
            Shape::Rect { shape, .. } => *shape,
            Shape::Index { indices } => (1, indices.len()),
        }
    }

    pub fn slice<T>(&self, mat: &DMatrix<T>) -> DMatrix<T>
    where
        T: Scalar
    {
        match self {
            Shape::Identity => mat.clone(),
            Shape::Rect { start, shape } => {
                mat.view(*start, *shape).into()
            },
            Shape::Index { indices } => {
                let values: Vec<T> = indices.iter().map(|i| mat[(i.0, i.1)].clone()).collect();
                let s = VecStorage::new(Dyn(1), Dyn(values.len()), values);
                DMatrix::from_data(s).into()
            },
        }
    }

    pub fn get<'a, T>(&self, mat: &'a DMatrix<T>, row: usize, col: usize) -> &'a T
    where
        T: Scalar
    {
        match self {
            Shape::Identity => &mat[(row, col)],
            Shape::Rect { start, .. } => {
                &mat[(start.0 + row, start.1 + col)]
            },
            Shape::Index { indices } => {
                &mat[(indices[col].0, indices[col].1)]
            },
        }
    }

    pub fn get_mut<'a, T>(&self, mat: &'a mut DMatrix<T>, row: usize, col: usize) -> &'a mut T
    where
        T: Scalar
    {
        match self {
            Shape::Identity => &mut mat[(row, col)],
            Shape::Rect { start, .. } => {
                &mut mat[(start.0 + row, start.1 + col)]
            },
            Shape::Index { indices } => {
                &mut mat[(indices[col].0, indices[col].1)]
            },
        }
    }

    pub fn indices<'a>(
        &'a self,
        in_shape: (usize, usize),
    ) -> Box<dyn Iterator<Item = (usize, usize)> + 'a> {
        match self {
            Shape::Identity => {
                Box::new((0..in_shape.0).flat_map(move |i| (0..in_shape.1).map(move |j| (i, j))))
            }
            Shape::Rect { start, shape } => Box::new(
                (0..shape.0)
                    .flat_map(move |i| (0..shape.1).map(move |j| (start.0 + i, start.1 + j))),
            ),
            Shape::Index { indices } => Box::new(indices.into_iter().map(|i| *i)),
        }
    }

    pub fn transform_index(&self, row: usize, col: usize) -> (usize, usize) {
        match self {
            Shape::Identity => (row, col),
            Shape::Rect { start, .. } => (row + start.0, col + start.1),
            Shape::Index { indices } => indices[col],
        }
    }
}

impl ViewType {
    pub fn intersect(&self, parent: &ViewType) -> Option<ViewType> {
        match (self, parent) {
            (ViewType::Identity, other) => Some(other.clone()),
            (other, ViewType::Identity) => Some(other.clone()),

            (ViewType::Real, ViewType::Real) => Some(ViewType::Real),
            (ViewType::Real, ViewType::Imag) => None,

            (ViewType::Imag, ViewType::Real) => None,
            (ViewType::Imag, ViewType::Imag) => Some(ViewType::Imag),
        }
    }

    pub fn view_dtype(&self, mat_dt: DType) -> Option<DType> {
        match (self, &mat_dt) {
            (ViewType::Identity, _) => Some(mat_dt),
            (
                ViewType::Real,
                DType::Bool | DType::U8 | DType::I32 | DType::I64 | DType::F32 | DType::F64,
            ) => Some(mat_dt),
            (ViewType::Real | ViewType::Imag, DType::C64) => Some(DType::F32),
            (ViewType::Real | ViewType::Imag, DType::C128) => Some(DType::F64),
            _ => None,
        }
    }
}
