use std::error::Error;

use godot::prelude::*;
use nalgebra::{Complex, DMatrix};
use strum_macros::AsRefStr;

use crate::{
    mat::DType,
    view::{Shape, View, ViewType},
};

type Result<T> = std::result::Result<T, Box<dyn std::error::Error>>;

#[derive(AsRefStr, Clone)]
pub enum MData {
    Bool(DMatrix<bool>),
    U8(DMatrix<u8>),
    I32(DMatrix<i32>),
    I64(DMatrix<i64>),
    F32(DMatrix<f32>),
    F64(DMatrix<f64>),
    C64(DMatrix<Complex<f32>>),
    C128(DMatrix<Complex<f64>>),
    View(View),
}

impl MData {
    pub fn zeros(nrows: usize, ncols: usize, dtype: &DType) -> Self {
        match dtype {
            DType::Bool => MData::U8(DMatrix::zeros(nrows, ncols))
                .astype(DType::Bool)
                .unwrap(),
            DType::U8 => MData::U8(DMatrix::zeros(nrows, ncols)),
            DType::I32 => MData::I32(DMatrix::zeros(nrows, ncols)),
            DType::I64 => MData::I64(DMatrix::zeros(nrows, ncols)),
            DType::F32 => MData::F32(DMatrix::zeros(nrows, ncols)),
            DType::F64 => MData::F64(DMatrix::zeros(nrows, ncols)),
            DType::C64 => MData::C64(DMatrix::zeros(nrows, ncols)),
            DType::C128 => MData::C128(DMatrix::zeros(nrows, ncols)),
        }
    }

    pub fn identity(nrows: usize, ncols: usize, dtype: &DType) -> Result<Self> {
        match dtype {
            DType::U8 => Ok(MData::U8(DMatrix::identity(nrows, ncols))),
            DType::I32 => Ok(MData::I32(DMatrix::identity(nrows, ncols))),
            DType::I64 => Ok(MData::I64(DMatrix::identity(nrows, ncols))),
            DType::F32 => Ok(MData::F32(DMatrix::identity(nrows, ncols))),
            DType::F64 => Ok(MData::F64(DMatrix::identity(nrows, ncols))),
            DType::C64 => Ok(MData::C64(DMatrix::identity(nrows, ncols))),
            DType::C128 => Ok(MData::C128(DMatrix::identity(nrows, ncols))),
            _ => Err(format!(
                "Cannot create identity matrix for dtype {}",
                dtype.to_godot()
            )
            .into()),
        }
    }

    pub fn get_index(&self, row: usize, col: usize, vt: &ViewType, s: &Shape) -> Option<Variant> {
        match (self, vt) {
            (MData::Bool(m), ViewType::Identity | ViewType::Real) => {
                Some(s.get(m, row, col).to_variant())
            }
            (MData::U8(m), ViewType::Identity | ViewType::Real) => {
                Some(s.get(m, row, col).to_variant())
            }
            (MData::I32(m), ViewType::Identity | ViewType::Real) => {
                Some(s.get(m, row, col).to_variant())
            }
            (MData::I64(m), ViewType::Identity | ViewType::Real) => {
                Some(s.get(m, row, col).to_variant())
            }
            (MData::F32(m), ViewType::Identity | ViewType::Real) => {
                Some(s.get(m, row, col).to_variant())
            }
            (MData::F64(m), ViewType::Identity | ViewType::Real) => {
                Some(s.get(m, row, col).to_variant())
            }
            (MData::C64(m), ViewType::Identity) => {
                let x = s.get(m, row, col);
                Some(Vector2::new(x.re, x.im).to_variant())
            }
            (MData::C64(m), ViewType::Real) => Some(s.get(m, row, col).re.to_variant()),
            (MData::C64(m), ViewType::Imag) => Some(s.get(m, row, col).im.to_variant()),
            (MData::C128(m), ViewType::Identity) => {
                let x = s.get(m, row, col);
                Some(array![x.re, x.im].to_variant())
            }
            (MData::C128(m), ViewType::Real) => Some(s.get(m, row, col).re.to_variant()),
            (MData::C128(m), ViewType::Imag) => Some(s.get(m, row, col).im.to_variant()),
            (MData::View(v), _) => {
                let vc = vt.intersect(&v.view_type).unwrap();
                let sc = s.intersect(&v.shape);
                let index = s.transform_index(row, col);
                v.mat.bind().m.get_index(index.0, index.1, &vc, &sc)
            }
            _ => None,
        }
    }

    pub fn set_index(
        &mut self,
        row: usize,
        col: usize,
        value: Variant,
        vt: &ViewType,
        s: &Shape,
    ) -> Result<()> {
        let base_type = self.base_dtype().to_godot();
        match (&mut *self, vt) {
            (MData::Bool(m), ViewType::Identity | ViewType::Real) => value
                .try_to()
                .map(|v| *s.get_mut(m, row, col) = v)
                .map_err::<Box<dyn std::error::Error>, _>(|_| {
                    format!(
                        "Incompatible types: matrix of type {} cannot store values of type {:?}",
                        base_type,
                        value.get_type()
                    )
                    .into()
                }),
            (MData::U8(m), ViewType::Identity | ViewType::Real) => value
                .try_to()
                .map_err(|_| {
                    format!(
                        "Incompatible types: matrix of type {} cannot store values of type {:?}",
                        base_type,
                        value.get_type()
                    )
                    .into()
                })
                .map(|v| *s.get_mut(m, row, col) = v),
            (MData::I32(m), ViewType::Identity | ViewType::Real) => value
                .try_to()
                .map_err(|_| {
                    format!(
                        "Incompatible types: matrix of type {} cannot store values of type {:?}",
                        base_type,
                        value.get_type()
                    )
                    .into()
                })
                .map(|v| *s.get_mut(m, row, col) = v),
            (MData::I64(m), ViewType::Identity | ViewType::Real) => value
                .try_to()
                .map_err(|_| {
                    format!(
                        "Incompatible types: matrix of type {} cannot store values of type {:?}",
                        base_type,
                        value.get_type()
                    )
                    .into()
                })
                .map(|v| *s.get_mut(m, row, col) = v),
            (MData::F32(m), ViewType::Identity | ViewType::Real) => value
                .try_to()
                .map_err(|_| {
                    format!(
                        "Incompatible types: matrix of type {} cannot store values of type {:?}",
                        base_type,
                        value.get_type()
                    )
                    .into()
                })
                .map(|v| *s.get_mut(m, row, col) = v),
            (MData::F64(m), ViewType::Identity | ViewType::Real) => value
                .try_to()
                .map_err(|_| {
                    format!(
                        "Incompatible types: matrix of type {} cannot store values of type {:?}",
                        base_type,
                        value.get_type()
                    )
                    .into()
                })
                .map(|v| *s.get_mut(m, row, col) = v),
            (MData::C64(m), ViewType::Identity) => value
                .try_to::<Vector2>()
                .map_err(|_| {
                    format!(
                        "Incompatible types: matrix of type {} cannot store values of type {:?}",
                        base_type,
                        value.get_type()
                    )
                    .into()
                })
                .map(|v| *s.get_mut(m, row, col) = Complex::new(v.x, v.y)),
            (MData::C64(m), ViewType::Real) => value
                .try_to::<f64>()
                .map_err(|_| {
                    format!(
                        "Incompatible types: matrix of type {} cannot store values of type {:?}",
                        base_type,
                        value.get_type()
                    )
                    .into()
                })
                .map(|v| s.get_mut(m, row, col).re = v as f32),
            (MData::C64(m), ViewType::Imag) => value
                .try_to::<f64>()
                .map_err(|_| {
                    format!(
                        "Incompatible types: matrix of type {} cannot store values of type {:?}",
                        base_type,
                        value.get_type()
                    )
                    .into()
                })
                .map(|v| s.get_mut(m, row, col).im = v as f32),
            (MData::C128(m), ViewType::Identity) => {
                let value = value.try_to::<Array<f64>>().map_err(|_| {
                    format!(
                        "Incompatible types: matrix of type {} cannot store values of type {:?}",
                        base_type,
                        value.get_type()
                    )
                })?;

                if value.len() != 2 {
                    return Err(format!(
                        "Expected array of length two, got array of length {}.",
                        value.len()
                    )
                    .into());
                }

                *s.get_mut(m, row, col) = Complex::new(value.get(0), value.get(1));
                Ok(())
            }
            (MData::C128(m), ViewType::Real) => value
                .try_to::<f64>()
                .map_err(|_| {
                    format!(
                        "Incompatible types: matrix of type {} cannot store values of type {:?}",
                        base_type,
                        value.get_type()
                    )
                    .into()
                })
                .map(|v| s.get_mut(m, row, col).re = v),
            (MData::C128(m), ViewType::Imag) => value
                .try_to::<f64>()
                .map_err(|_| {
                    format!(
                        "Incompatible types: matrix of type {} cannot store values of type {:?}",
                        base_type,
                        value.get_type()
                    )
                    .into()
                })
                .map(|v| s.get_mut(m, row, col).im = v),
            (MData::View(v), _) => {
                let vc = vt.intersect(&v.view_type).unwrap();
                let sc = s.intersect(&v.shape);
                let index = s.transform_index(row, col);
                v.mat.bind_mut().m.set_index(index.0, index.1, value, &vc, &sc)
            }
            _ => Err("Invalid view type. This is a bug!".into()),
        }
    }

    pub fn slice(&self, vt: &ViewType, s: &Shape) -> Option<Self> {
        match (self, vt) {
            (MData::Bool(m), ViewType::Identity | ViewType::Real) => Some(MData::Bool(s.slice(m))),
            (MData::U8(m), ViewType::Identity | ViewType::Real) => Some(MData::U8(s.slice(m))),
            (MData::I32(m), ViewType::Identity | ViewType::Real) => Some(MData::I32(s.slice(m))),
            (MData::I64(m), ViewType::Identity | ViewType::Real) => Some(MData::I64(s.slice(m))),
            (MData::F32(m), ViewType::Identity | ViewType::Real) => Some(MData::F32(s.slice(m))),
            (MData::F64(m), ViewType::Identity | ViewType::Real) => Some(MData::F64(s.slice(m))),
            (MData::C64(m), ViewType::Identity) => Some(MData::C64(s.slice(m))),
            (MData::C64(m), ViewType::Real) => Some(MData::F32(s.slice(m).map(|x| x.re))),
            (MData::C64(m), ViewType::Imag) => Some(MData::F32(s.slice(m).map(|x| x.im))),
            (MData::C128(m), ViewType::Identity) => Some(MData::C128(s.slice(m))),
            (MData::C128(m), ViewType::Real) => Some(MData::F64(s.slice(m).map(|x| x.re))),
            (MData::C128(m), ViewType::Imag) => Some(MData::F64(s.slice(m).map(|x| x.im))),
            (MData::View(v), _) => {
                let vc = vt.intersect(&v.view_type).unwrap();
                let sc = s.intersect(&v.shape);
                v.mat.bind().m.slice(&vc, &sc)
            }
            _ => None,
        }
    }

    pub fn set(&mut self, other: &MData, vt1: &ViewType, s1: &Shape, vt2: &ViewType, s2: &Shape) {
        let shape = self.shape();
        match (&mut *self, other, vt1, vt2) {
            (
                MData::Bool(a),
                MData::Bool(b),
                ViewType::Identity | ViewType::Real,
                ViewType::Identity | ViewType::Real,
            ) => s1
                .indices(shape)
                .zip(s2.indices(other.shape()))
                .for_each(|((i1, j1), (i2, j2))| a[(i1, j1)] = b[(i2, j2)]),
            (
                MData::U8(a),
                MData::U8(b),
                ViewType::Identity | ViewType::Real,
                ViewType::Identity | ViewType::Real,
            ) => s1
                .indices(shape)
                .zip(s2.indices(other.shape()))
                .for_each(|((i1, j1), (i2, j2))| a[(i1, j1)] = b[(i2, j2)]),
            (
                MData::I32(a),
                MData::I32(b),
                ViewType::Identity | ViewType::Real,
                ViewType::Identity | ViewType::Real,
            ) => s1
                .indices(shape)
                .zip(s2.indices(other.shape()))
                .for_each(|((i1, j1), (i2, j2))| a[(i1, j1)] = b[(i2, j2)]),
            (
                MData::F32(a),
                MData::F32(b),
                ViewType::Identity | ViewType::Real,
                ViewType::Identity | ViewType::Real,
            ) => s1
                .indices(shape)
                .zip(s2.indices(other.shape()))
                .for_each(|((i1, j1), (i2, j2))| a[(i1, j1)] = b[(i2, j2)]),
            (MData::F32(a), MData::C64(b), ViewType::Identity, ViewType::Real) => s1
                .indices(shape)
                .zip(s2.indices(other.shape()))
                .for_each(|((i1, j1), (i2, j2))| a[(i1, j1)] = b[(i2, j2)].re),
            (MData::F32(a), MData::C64(b), ViewType::Identity, ViewType::Imag) => s1
                .indices(shape)
                .zip(s2.indices(other.shape()))
                .for_each(|((i1, j1), (i2, j2))| a[(i1, j1)] = b[(i2, j2)].im),
            (
                MData::F64(a),
                MData::F64(b),
                ViewType::Identity | ViewType::Real,
                ViewType::Identity | ViewType::Real,
            ) => s1
                .indices(shape)
                .zip(s2.indices(other.shape()))
                .for_each(|((i1, j1), (i2, j2))| a[(i1, j1)] = b[(i2, j2)]),
            (
                MData::F64(a),
                MData::C128(b),
                ViewType::Identity | ViewType::Real,
                ViewType::Real,
            ) => s1
                .indices(shape)
                .zip(s2.indices(other.shape()))
                .for_each(|((i1, j1), (i2, j2))| a[(i1, j1)] = b[(i2, j2)].re),
            (
                MData::F64(a),
                MData::C128(b),
                ViewType::Identity | ViewType::Real,
                ViewType::Imag,
            ) => s1
                .indices(shape)
                .zip(s2.indices(other.shape()))
                .for_each(|((i1, j1), (i2, j2))| a[(i1, j1)] = b[(i2, j2)].im),
            (MData::C64(a), MData::C64(b), ViewType::Identity, ViewType::Identity) => s1
                .indices(shape)
                .zip(s2.indices(other.shape()))
                .for_each(|((i1, j1), (i2, j2))| a[(i1, j1)] = b[(i2, j2)]),
            (MData::C64(a), MData::F32(b), ViewType::Real, ViewType::Identity) => s1
                .indices(shape)
                .zip(s2.indices(other.shape()))
                .for_each(|((i1, j1), (i2, j2))| a[(i1, j1)].re = b[(i2, j2)]),
            (MData::C64(a), MData::F32(b), ViewType::Imag, ViewType::Identity) => s1
                .indices(shape)
                .zip(s2.indices(other.shape()))
                .for_each(|((i1, j1), (i2, j2))| a[(i1, j1)].im = b[(i2, j2)]),
            (MData::C128(a), MData::C128(b), ViewType::Identity, ViewType::Identity) => s1
                .indices(shape)
                .zip(s2.indices(other.shape()))
                .for_each(|((i1, j1), (i2, j2))| a[(i1, j1)] = b[(i2, j2)]),
            (MData::C128(a), MData::F64(b), ViewType::Real, ViewType::Identity) => s1
                .indices(shape)
                .zip(s2.indices(other.shape()))
                .for_each(|((i1, j1), (i2, j2))| a[(i1, j1)].re = b[(i2, j2)]),
            (MData::C128(a), MData::F64(b), ViewType::Imag, ViewType::Identity) => s1
                .indices(shape)
                .zip(s2.indices(other.shape()))
                .for_each(|((i1, j1), (i2, j2))| a[(i1, j1)].im = b[(i2, j2)]),
            (MData::View(v1), MData::View(v2), _, _) => {
                let vc1 = vt1.intersect(&v1.view_type).unwrap();
                let vc2 = vt2.intersect(&v2.view_type).unwrap();
                let sc1 = s1.intersect(&v1.shape);
                let sc2 = s2.intersect(&v2.shape);
                v1.mat
                    .bind_mut()
                    .m
                    .set(&v2.mat.bind().m, &vc1, &sc1, &vc2, &sc2);
            }
            (MData::View(v1), _, _, _) => {
                let vc1 = vt1.intersect(&v1.view_type).unwrap();
                let sc1 = s1.intersect(&v1.shape);
                v1.mat.bind_mut().m.set(other, &vc1, &sc1, vt2, s2);
            }
            (_, MData::View(v2), _, _) => {
                let vc2 = vt2.intersect(&v2.view_type).unwrap();
                let sc2 = s1.intersect(&v2.shape);
                self.set(&v2.mat.bind().m, vt1, s1, &vc2, &sc2);
            }
            _ => {}
        }
    }

    pub fn dtype(&self) -> DType {
        match self {
            MData::Bool(_) => DType::Bool,
            MData::U8(_) => DType::U8,
            MData::I32(_) => DType::I32,
            MData::I64(_) => DType::I64,
            MData::F32(_) => DType::F32,
            MData::F64(_) => DType::F64,
            MData::C64(_) => DType::C64,
            MData::C128(_) => DType::C128,
            MData::View(v) => v.view_type.view_dtype(v.mat.bind().m.dtype()).unwrap(),
        }
    }

    pub fn base_dtype(&self) -> DType {
        match self {
            MData::Bool(_) => DType::Bool,
            MData::U8(_) => DType::U8,
            MData::I32(_) => DType::I32,
            MData::I64(_) => DType::I64,
            MData::F32(_) => DType::F32,
            MData::F64(_) => DType::F64,
            MData::C64(_) => DType::C64,
            MData::C128(_) => DType::C128,
            MData::View(v) => v.mat.bind().m.dtype(),
        }
    }

    pub fn shape(&self) -> (usize, usize) {
        match self {
            MData::Bool(m) => m.shape(),
            MData::U8(m) => m.shape(),
            MData::I32(m) => m.shape(),
            MData::I64(m) => m.shape(),
            MData::F32(m) => m.shape(),
            MData::F64(m) => m.shape(),
            MData::C64(m) => m.shape(),
            MData::C128(m) => m.shape(),
            MData::View(v) => v.shape.shape(v.mat.bind().m.shape()),
        }
    }

    pub fn nrows(&self) -> usize {
        self.shape().0
    }

    pub fn ncols(&self) -> usize {
        self.shape().1
    }

    pub fn tranpose(&self) -> Option<MData> {
        match &self {
            MData::Bool(m) => Some(MData::Bool(m.transpose())),
            MData::U8(m) => Some(MData::U8(m.transpose())),
            MData::I32(m) => Some(MData::I32(m.transpose())),
            MData::I64(m) => Some(MData::I64(m.transpose())),
            MData::F32(m) => Some(MData::F32(m.transpose())),
            MData::F64(m) => Some(MData::F64(m.transpose())),
            MData::C64(m) => Some(MData::C64(m.transpose())),
            MData::C128(m) => Some(MData::C128(m.transpose())),
            MData::View(v) => v
                .mat
                .bind()
                .m
                .slice(&v.view_type, &v.shape)
                .map(|m| m.tranpose())
                .flatten(),
        }
    }

    pub fn adjoint(&self) -> Result<MData> {
        match &self {
            MData::F32(m) => Ok(MData::F32(m.adjoint())),
            MData::F64(m) => Ok(MData::F64(m.adjoint())),
            MData::C64(m) => Ok(MData::C64(m.adjoint())),
            MData::C128(m) => Ok(MData::C128(m.adjoint())),
            MData::View(v) => v
                .mat
                .bind()
                .m
                .slice(&v.view_type, &v.shape)
                .ok_or(format!("Cannot find adjoint of empty view.").into())
                .and_then(|m| m.adjoint()),
            _ => Err(format!("Unsupported type: {}.h()", self.as_ref()).into()),
        }
    }

    pub fn any(&self, vt: &ViewType, s: &Shape) -> Result<bool> {
        match (self, vt) {
            (MData::Bool(m), ViewType::Identity | ViewType::Real) => Ok(s.iter(m).any(|x| *x)),
            (MData::U8(m), ViewType::Identity | ViewType::Real) => Ok(s.iter(m).any(|x| *x > 0)),
            (MData::I32(m), ViewType::Identity | ViewType::Real) => Ok(s.iter(m).any(|x| *x > 0)),
            (MData::I64(m), ViewType::Identity | ViewType::Real) => Ok(s.iter(m).any(|x| *x > 0)),
            (MData::F32(m), ViewType::Identity | ViewType::Real) => Ok(s.iter(m).any(|x| *x > 0.0)),
            (MData::F64(m), ViewType::Identity | ViewType::Real) => Ok(s.iter(m).any(|x| *x > 0.0)),
            (MData::C64(m), ViewType::Real) => Ok(s.iter(m).any(|x| x.re > 0.0)),
            (MData::C64(m), ViewType::Imag) => Ok(s.iter(m).any(|x| x.im > 0.0)),
            (MData::C128(m), ViewType::Real) => Ok(s.iter(m).any(|x| x.re > 0.0)),
            (MData::C128(m), ViewType::Imag) => Ok(s.iter(m).any(|x| x.im > 0.0)),
            (MData::View(v), _) => {
                let vc = vt.intersect(&v.view_type).unwrap();
                let sc = s.intersect(&v.shape);
                v.mat.bind().m.any(&vc, &sc)
            }
            _ => Err(format!(
                "The `any` property of a {:?} view to a matrix of type {} is ill-defined",
                vt.as_ref(),
                self.dtype().to_godot()
            )
            .into()),
        }
    }

    pub fn all(&self, vt: &ViewType, s: &Shape) -> Result<bool> {
        match (self, vt) {
            (MData::Bool(m), ViewType::Identity | ViewType::Real) => Ok(s.iter(m).all(|x| *x)),
            (MData::U8(m), ViewType::Identity | ViewType::Real) => Ok(s.iter(m).all(|x| *x > 0)),
            (MData::I32(m), ViewType::Identity | ViewType::Real) => Ok(s.iter(m).all(|x| *x > 0)),
            (MData::I64(m), ViewType::Identity | ViewType::Real) => Ok(s.iter(m).all(|x| *x > 0)),
            (MData::F32(m), ViewType::Identity | ViewType::Real) => Ok(s.iter(m).all(|x| *x > 0.0)),
            (MData::F64(m), ViewType::Identity | ViewType::Real) => Ok(s.iter(m).all(|x| *x > 0.0)),
            (MData::C64(m), ViewType::Real) => Ok(s.iter(m).all(|x| x.re > 0.0)),
            (MData::C64(m), ViewType::Imag) => Ok(s.iter(m).all(|x| x.im > 0.0)),
            (MData::C128(m), ViewType::Real) => Ok(s.iter(m).all(|x| x.re > 0.0)),
            (MData::C128(m), ViewType::Imag) => Ok(s.iter(m).all(|x| x.im > 0.0)),
            (MData::View(v), _) => {
                let vc = vt.intersect(&v.view_type).unwrap();
                let sc = s.intersect(&v.shape);
                v.mat.bind().m.all(&vc, &sc)
            }
            _ => Err(format!(
                "The `all` property of a {:?} view to a matrix of type {} is ill-defined",
                vt.as_ref(),
                self.base_dtype().to_godot()
            )
            .into()),
        }
    }

    pub fn resize(&self, nrows: usize, ncols: usize, val: Variant) -> Result<Self> {
        match (self, val.get_type()) {
            (MData::Bool(m), VariantType::Int) => Ok(MData::Bool(m.clone().resize(
                nrows,
                ncols,
                val.to::<i64>() == 1,
            ))),
            (MData::U8(m), VariantType::Int) => Ok(MData::U8(m.clone().resize(
                nrows,
                ncols,
                val.to::<i64>() as u8,
            ))),
            (MData::I32(m), VariantType::Int) => Ok(MData::I32(m.clone().resize(
                nrows,
                ncols,
                val.to::<i64>() as i32,
            ))),
            (MData::I64(m), VariantType::Int) => {
                Ok(MData::I64(m.clone().resize(nrows, ncols, val.to::<i64>())))
            }
            (MData::F32(m), VariantType::Float) => Ok(MData::F32(m.clone().resize(
                nrows,
                ncols,
                val.to::<f64>() as f32,
            ))),
            (MData::C64(m), VariantType::Vector2) => {
                let c = Complex::new(val.to::<Vector2>().x, val.to::<Vector2>().y);
                Ok(MData::C64(m.clone().resize(nrows, ncols, c)))
            }
            (MData::C128(m), VariantType::Vector2) => {
                let c = Complex::new(val.to::<Vector2>().x as f64, val.to::<Vector2>().y as f64);
                Ok(MData::C128(m.clone().resize(nrows, ncols, c)))
            }
            (MData::C128(m), VariantType::Array) => {
                let arr = val.try_to::<Array<f64>>();
                arr.map_err(|_| {
                    format!("Array must be of type f64, got {:?}", val.get_type()).into()
                })
                .and_then(|arr| {
                    let length = arr.len();
                    (length == 2)
                        .then(|| arr)
                        .ok_or(format!("Array must have length 2, got length {}", length).into())
                })
                .map(|arr| Complex::new(arr.get(0), arr.get(1)))
                .map(|c| MData::C128(m.clone().resize(nrows, ncols, c)))
            }
            (MData::View(_), _) => Err(format!(
                "A matrix view cannot be resized, considering cloning it first."
            )
            .into()),
            _ => Err(format!(
                "Variant value {} is not supported for matrix of type {}",
                val.to_string(),
                self.as_ref()
            )
            .into()),
        }
    }

    pub fn kron(&self, other: &MData) -> Result<Self> {
        match (&self, other) {
            (MData::U8(a), MData::U8(b)) => Ok(MData::U8(a.kronecker(b))),
            (MData::I32(a), MData::I32(b)) => Ok(MData::I32(a.kronecker(b))),
            (MData::I64(a), MData::I64(b)) => Ok(MData::I64(a.kronecker(b))),
            (MData::F32(a), MData::F32(b)) => Ok(MData::F32(a.kronecker(b))),
            (MData::F64(a), MData::F64(b)) => Ok(MData::F64(a.kronecker(b))),
            (MData::C64(a), MData::C64(b)) => Ok(MData::C64(a.kronecker(b))),
            (MData::C128(a), MData::C128(b)) => Ok(MData::C128(a.kronecker(b))),
            (MData::View(_), MData::View(_)) => self
                .slice(&ViewType::Identity, &Shape::Identity)
                .unwrap()
                .kron(&other.slice(&ViewType::Identity, &Shape::Identity).unwrap()),
            (MData::View(_), _) => self
                .slice(&ViewType::Identity, &Shape::Identity)
                .unwrap()
                .kron(other),
            (_, MData::View(_)) => {
                self.kron(&other.slice(&ViewType::Identity, &Shape::Identity).unwrap())
            }
            _ => Err(format!(
                "Unsupported operand types: {} o {}",
                self.as_ref(),
                other.as_ref()
            )
            .into()),
        }
    }

    pub fn mm(&self, other: &MData) -> Result<Self> {
        match (&self, other) {
            (MData::U8(a), MData::U8(b)) => Ok(MData::U8(a * b)),
            (MData::I32(a), MData::I32(b)) => Ok(MData::I32(a * b)),
            (MData::I64(a), MData::I64(b)) => Ok(MData::I64(a * b)),
            (MData::F32(a), MData::F32(b)) => Ok(MData::F32(a * b)),
            (MData::F64(a), MData::F64(b)) => Ok(MData::F64(a * b)),
            (MData::C64(a), MData::C64(b)) => Ok(MData::C64(a * b)),
            (MData::C128(a), MData::C128(b)) => Ok(MData::C128(a * b)),
            (MData::View(_), MData::View(_)) => self
                .slice(&ViewType::Identity, &Shape::Identity)
                .unwrap()
                .mm(&other.slice(&ViewType::Identity, &Shape::Identity).unwrap()),
            (MData::View(_), _) => self
                .slice(&ViewType::Identity, &Shape::Identity)
                .unwrap()
                .mm(other),
            (_, MData::View(_)) => {
                self.mm(&other.slice(&ViewType::Identity, &Shape::Identity).unwrap())
            }
            _ => Err(format!(
                "Unsupported operand types: {}.{}",
                self.as_ref(),
                other.as_ref()
            )
            .into()),
        }
    }

    pub fn mul(&self, other: &MData) -> Result<Self> {
        match (&self, other) {
            (MData::U8(a), MData::U8(b)) => Ok(MData::U8(a.component_mul(b))),
            (MData::I32(a), MData::I32(b)) => Ok(MData::I32(a.component_mul(b))),
            (MData::I64(a), MData::I64(b)) => Ok(MData::I64(a.component_mul(b))),
            (MData::F32(a), MData::F32(b)) => Ok(MData::F32(a.component_mul(b))),
            (MData::F64(a), MData::F64(b)) => Ok(MData::F64(a.component_mul(b))),
            (MData::C64(a), MData::C64(b)) => Ok(MData::C64(a.component_mul(b))),
            (MData::C128(a), MData::C128(b)) => Ok(MData::C128(a.component_mul(b))),
            (MData::View(_), MData::View(_)) => self
                .slice(&ViewType::Identity, &Shape::Identity)
                .unwrap()
                .mul(&other.slice(&ViewType::Identity, &Shape::Identity).unwrap()),
            (MData::View(_), _) => self
                .slice(&ViewType::Identity, &Shape::Identity)
                .unwrap()
                .mul(other),
            (_, MData::View(_)) => {
                self.mul(&other.slice(&ViewType::Identity, &Shape::Identity).unwrap())
            }
            _ => Err(format!(
                "Unsupported operand types: {} * {}",
                self.as_ref(),
                other.as_ref()
            )
            .into()),
        }
    }

    pub fn add(&self, other: &MData) -> Result<Self> {
        match (&self, other) {
            (MData::U8(a), MData::U8(b)) => Ok(MData::U8(a + b)),
            (MData::I32(a), MData::I32(b)) => Ok(MData::I32(a + b)),
            (MData::I64(a), MData::I64(b)) => Ok(MData::I64(a + b)),
            (MData::F32(a), MData::F32(b)) => Ok(MData::F32(a + b)),
            (MData::F64(a), MData::F64(b)) => Ok(MData::F64(a + b)),
            (MData::C64(a), MData::C64(b)) => Ok(MData::C64(a + b)),
            (MData::C128(a), MData::C128(b)) => Ok(MData::C128(a + b)),
            (MData::View(_), MData::View(_)) => self
                .slice(&ViewType::Identity, &Shape::Identity)
                .unwrap()
                .add(&other.slice(&ViewType::Identity, &Shape::Identity).unwrap()),
            (MData::View(_), _) => self
                .slice(&ViewType::Identity, &Shape::Identity)
                .unwrap()
                .add(other),
            (_, MData::View(_)) => {
                self.add(&other.slice(&ViewType::Identity, &Shape::Identity).unwrap())
            }
            _ => Err(format!(
                "Unsupported operand types: {} + {}",
                self.as_ref(),
                other.as_ref()
            )
            .into()),
        }
    }

    pub fn lt(&self, other: &MData) -> Result<Self> {
        match (&self, other) {
            (MData::U8(a), MData::U8(b)) => Ok(MData::Bool(a.zip_map(b, |a, b| a < b))),
            (MData::I32(a), MData::I32(b)) => Ok(MData::Bool(a.zip_map(b, |a, b| a < b))),
            (MData::I64(a), MData::I64(b)) => Ok(MData::Bool(a.zip_map(b, |a, b| a < b))),
            (MData::F32(a), MData::F32(b)) => Ok(MData::Bool(a.zip_map(b, |a, b| a < b))),
            (MData::F64(a), MData::F64(b)) => Ok(MData::Bool(a.zip_map(b, |a, b| a < b))),
            (MData::View(_), MData::View(_)) => self
                .slice(&ViewType::Identity, &Shape::Identity)
                .unwrap()
                .lt(&other.slice(&ViewType::Identity, &Shape::Identity).unwrap()),
            (MData::View(_), _) => self
                .slice(&ViewType::Identity, &Shape::Identity)
                .unwrap()
                .lt(other),
            (_, MData::View(_)) => {
                self.lt(&other.slice(&ViewType::Identity, &Shape::Identity).unwrap())
            }
            _ => Err(format!(
                "Unsupported operand types: {} < {}",
                self.as_ref(),
                other.as_ref()
            )
            .into()),
        }
    }

    pub fn le(&self, other: &MData) -> Result<Self> {
        match (&self, other) {
            (MData::U8(a), MData::U8(b)) => Ok(MData::Bool(a.zip_map(b, |a, b| a <= b))),
            (MData::I32(a), MData::I32(b)) => Ok(MData::Bool(a.zip_map(b, |a, b| a <= b))),
            (MData::I64(a), MData::I64(b)) => Ok(MData::Bool(a.zip_map(b, |a, b| a <= b))),
            (MData::F32(a), MData::F32(b)) => Ok(MData::Bool(a.zip_map(b, |a, b| a <= b))),
            (MData::F64(a), MData::F64(b)) => Ok(MData::Bool(a.zip_map(b, |a, b| a <= b))),
            (MData::View(_), MData::View(_)) => self
                .slice(&ViewType::Identity, &Shape::Identity)
                .unwrap()
                .le(&other.slice(&ViewType::Identity, &Shape::Identity).unwrap()),
            (MData::View(_), _) => self
                .slice(&ViewType::Identity, &Shape::Identity)
                .unwrap()
                .le(other),
            (_, MData::View(_)) => {
                self.le(&other.slice(&ViewType::Identity, &Shape::Identity).unwrap())
            }
            _ => Err(format!(
                "Unsupported operand types: {} <= {}",
                self.as_ref(),
                other.as_ref()
            )
            .into()),
        }
    }

    pub fn eq(&self, other: &MData) -> Result<Self> {
        match (&self, other) {
            (MData::U8(a), MData::U8(b)) => Ok(MData::Bool(a.zip_map(b, |a, b| a == b))),
            (MData::I32(a), MData::I32(b)) => Ok(MData::Bool(a.zip_map(b, |a, b| a == b))),
            (MData::I64(a), MData::I64(b)) => Ok(MData::Bool(a.zip_map(b, |a, b| a == b))),
            (MData::F32(a), MData::F32(b)) => Ok(MData::Bool(a.zip_map(b, |a, b| a.eq(&b)))),
            (MData::F64(a), MData::F64(b)) => Ok(MData::Bool(a.zip_map(b, |a, b| a.eq(&b)))),
            (MData::C64(a), MData::C64(b)) => Ok(MData::Bool(a.zip_map(b, |a, b| a.eq(&b)))),
            (MData::C128(a), MData::C128(b)) => Ok(MData::Bool(a.zip_map(b, |a, b| a.eq(&b)))),
            (MData::View(_), MData::View(_)) => self
                .slice(&ViewType::Identity, &Shape::Identity)
                .unwrap()
                .eq(&other.slice(&ViewType::Identity, &Shape::Identity).unwrap()),
            (MData::View(_), _) => self
                .slice(&ViewType::Identity, &Shape::Identity)
                .unwrap()
                .eq(other),
            (_, MData::View(_)) => {
                self.eq(&other.slice(&ViewType::Identity, &Shape::Identity).unwrap())
            }
            _ => Err(format!(
                "Unsupported operand types: {} == {}",
                self.as_ref(),
                other.as_ref()
            )
            .into()),
        }
    }

    pub fn ge(&self, other: &MData) -> Result<Self> {
        match (&self, other) {
            (MData::U8(a), MData::U8(b)) => Ok(MData::Bool(a.zip_map(b, |a, b| a >= b))),
            (MData::I32(a), MData::I32(b)) => Ok(MData::Bool(a.zip_map(b, |a, b| a >= b))),
            (MData::I64(a), MData::I64(b)) => Ok(MData::Bool(a.zip_map(b, |a, b| a >= b))),
            (MData::F32(a), MData::F32(b)) => Ok(MData::Bool(a.zip_map(b, |a, b| a >= b))),
            (MData::F64(a), MData::F64(b)) => Ok(MData::Bool(a.zip_map(b, |a, b| a >= b))),
            (MData::View(_), MData::View(_)) => self
                .slice(&ViewType::Identity, &Shape::Identity)
                .unwrap()
                .ge(&other.slice(&ViewType::Identity, &Shape::Identity).unwrap()),
            (MData::View(_), _) => self
                .slice(&ViewType::Identity, &Shape::Identity)
                .unwrap()
                .ge(other),
            (_, MData::View(_)) => {
                self.ge(&other.slice(&ViewType::Identity, &Shape::Identity).unwrap())
            }
            _ => Err(format!(
                "Unsupported operand types: {} >= {}",
                self.as_ref(),
                other.as_ref()
            )
            .into()),
        }
    }

    pub fn gt(&self, other: &MData) -> Result<Self> {
        match (&self, other) {
            (MData::U8(a), MData::U8(b)) => Ok(MData::Bool(a.zip_map(b, |a, b| a > b))),
            (MData::I32(a), MData::I32(b)) => Ok(MData::Bool(a.zip_map(b, |a, b| a > b))),
            (MData::I64(a), MData::I64(b)) => Ok(MData::Bool(a.zip_map(b, |a, b| a > b))),
            (MData::F32(a), MData::F32(b)) => Ok(MData::Bool(a.zip_map(b, |a, b| a > b))),
            (MData::F64(a), MData::F64(b)) => Ok(MData::Bool(a.zip_map(b, |a, b| a > b))),
            (MData::View(_), MData::View(_)) => self
                .slice(&ViewType::Identity, &Shape::Identity)
                .unwrap()
                .gt(&other.slice(&ViewType::Identity, &Shape::Identity).unwrap()),
            (MData::View(_), _) => self
                .slice(&ViewType::Identity, &Shape::Identity)
                .unwrap()
                .gt(other),
            (_, MData::View(_)) => {
                self.gt(&other.slice(&ViewType::Identity, &Shape::Identity).unwrap())
            }
            _ => Err(format!(
                "Unsupported operand types: {} > {}",
                self.as_ref(),
                other.as_ref()
            )
            .into()),
        }
    }

    pub fn astype(&self, dt: DType) -> Result<MData> {
        let m = self
            .slice(&ViewType::Identity, &Shape::Identity)
            .ok_or::<Box<dyn Error>>("Failed to slice matrix. This is a bug!".into())?;
        match (m, &dt) {
            (MData::Bool(m), DType::Bool) => Ok(MData::Bool(m)),
            (MData::Bool(m), DType::U8) => Ok(MData::U8(m.map(|x| x as u8))),
            (MData::Bool(m), DType::I32) => Ok(MData::I32(m.map(|x| x as i32))),
            (MData::Bool(m), DType::I64) => Ok(MData::I64(m.map(|x| x as i64))),
            (MData::Bool(m), DType::F32) => Ok(MData::F32(m.map(|x| x as i32 as f32))),
            (MData::Bool(m), DType::F64) => Ok(MData::F64(m.map(|x| x as i32 as f64))),
            (MData::Bool(m), DType::C64) => {
                Ok(MData::C64(m.map(|x| Complex::new(x as i32 as f32, 0.0))))
            }
            (MData::Bool(m), DType::C128) => {
                Ok(MData::C128(m.map(|x| Complex::new(x as i32 as f64, 0.0))))
            }
            (MData::U8(m), DType::Bool) => Ok(MData::Bool(m.map(|x| x > 0))),
            (MData::U8(m), DType::U8) => Ok(MData::U8(m)),
            (MData::U8(m), DType::I32) => Ok(MData::I32(m.map(|x| x as i32))),
            (MData::U8(m), DType::I64) => Ok(MData::I64(m.map(|x| x as i64))),
            (MData::U8(m), DType::F32) => Ok(MData::F32(m.map(|x| x as f32))),
            (MData::U8(m), DType::F64) => Ok(MData::F64(m.map(|x| x as f64))),
            (MData::U8(m), DType::C64) => Ok(MData::C64(m.map(|x| Complex::new(x as f32, 0.0)))),
            (MData::U8(m), DType::C128) => Ok(MData::C128(m.map(|x| Complex::new(x as f64, 0.0)))),
            (MData::I32(m), DType::Bool) => Ok(MData::Bool(m.map(|x| x > 0))),
            (MData::I32(m), DType::U8) => Ok(MData::U8(m.map(|x| x as u8))),
            (MData::I32(m), DType::I32) => Ok(MData::I32(m)),
            (MData::I32(m), DType::I64) => Ok(MData::I64(m.map(|x| x as i64))),
            (MData::I32(m), DType::F32) => Ok(MData::F32(m.map(|x| x as f32))),
            (MData::I32(m), DType::F64) => Ok(MData::F64(m.map(|x| x as f64))),
            (MData::I32(m), DType::C64) => Ok(MData::C64(m.map(|x| Complex::new(x as f32, 0.0)))),
            (MData::I32(m), DType::C128) => Ok(MData::C128(m.map(|x| Complex::new(x as f64, 0.0)))),
            (MData::I64(m), DType::Bool) => Ok(MData::Bool(m.map(|x| x > 0))),
            (MData::I64(m), DType::U8) => Ok(MData::U8(m.map(|x| x as u8))),
            (MData::I64(m), DType::I32) => Ok(MData::I32(m.map(|x| x as i32))),
            (MData::I64(m), DType::I64) => Ok(MData::I64(m)),
            (MData::I64(m), DType::F32) => Ok(MData::F32(m.map(|x| x as f32))),
            (MData::I64(m), DType::F64) => Ok(MData::F64(m.map(|x| x as f64))),
            (MData::I64(m), DType::C64) => Ok(MData::C64(m.map(|x| Complex::new(x as f32, 0.0)))),
            (MData::I64(m), DType::C128) => Ok(MData::C128(m.map(|x| Complex::new(x as f64, 0.0)))),
            (MData::F32(m), DType::Bool) => Ok(MData::Bool(m.map(|x| x > 0.0))),
            (MData::F32(m), DType::U8) => Ok(MData::U8(m.map(|x| x as u8))),
            (MData::F32(m), DType::I32) => Ok(MData::I32(m.map(|x| x as i32))),
            (MData::F32(m), DType::I64) => Ok(MData::I64(m.map(|x| x as i64))),
            (MData::F32(m), DType::F32) => Ok(MData::F32(m)),
            (MData::F32(m), DType::F64) => Ok(MData::F64(m.map(|x| x as f64))),
            (MData::F32(m), DType::C64) => Ok(MData::C64(m.map(|x| Complex::new(x as f32, 0.0)))),
            (MData::F32(m), DType::C128) => Ok(MData::C128(m.map(|x| Complex::new(x as f64, 0.0)))),
            (MData::F64(m), DType::Bool) => Ok(MData::Bool(m.map(|x| x > 0.0))),
            (MData::F64(m), DType::U8) => Ok(MData::U8(m.map(|x| x as u8))),
            (MData::F64(m), DType::I32) => Ok(MData::I32(m.map(|x| x as i32))),
            (MData::F64(m), DType::I64) => Ok(MData::I64(m.map(|x| x as i64))),
            (MData::F64(m), DType::F32) => Ok(MData::F32(m.map(|x| x as f32))),
            (MData::F64(m), DType::F64) => Ok(MData::F64(m)),
            (MData::F64(m), DType::C64) => Ok(MData::C64(m.map(|x| Complex::new(x as f32, 0.0)))),
            (MData::F64(m), DType::C128) => Ok(MData::C128(m.map(|x| Complex::new(x as f64, 0.0)))),
            (MData::C64(m), DType::C64) => Ok(MData::C64(m)),
            (MData::C64(m), DType::C128) => Ok(MData::C128(
                m.map(|x| Complex::new(x.re as f64, x.im as f64)),
            )),
            (MData::C128(m), DType::C64) => Ok(MData::C64(
                m.map(|x| Complex::new(x.re as f32, x.im as f32)),
            )),
            (MData::C128(m), DType::C128) => Ok(MData::C128(m)),
            _ => Err(format!(
                "Unsupported cast from to {} to {}",
                self.as_ref(),
                dt.to_godot()
            )
            .into()),
        }
    }
}

impl std::fmt::Display for MData {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            MData::Bool(m) => m.fmt(f),
            MData::U8(m) => m.fmt(f),
            MData::I32(m) => m.fmt(f),
            MData::I64(m) => m.fmt(f),
            MData::F32(m) => m.fmt(f),
            MData::F64(m) => m.fmt(f),
            MData::C64(m) => m.fmt(f),
            MData::C128(m) => m.fmt(f),
            MData::View(v) => v
                .mat
                .bind()
                .m
                .slice(&v.view_type, &v.shape)
                .ok_or(std::fmt::Error {})?
                .fmt(f),
        }
    }
}
