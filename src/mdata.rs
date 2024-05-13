use godot::prelude::*;
use nalgebra::{Complex, DMatrix};
use strum_macros::AsRefStr;

use crate::mat::{DType, Mat};

type Result<T> = std::result::Result<T, Box<dyn std::error::Error>>;

#[derive(Copy, Clone)]
pub enum ViewType {
    All,
    Real,
    Imag
}

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
    View {
        mat: Gd<Mat>,
        start: (usize, usize),
        shape: (usize, usize),
        view_type: ViewType
    }
}

impl MData {
    pub fn zeros(nrows: usize, ncols: usize, dtype: &DType) -> Self {
        match dtype {
            DType::Bool => MData::U8(DMatrix::zeros(nrows, ncols)).astype(DType::Bool).unwrap(),
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
            _ => Err(format!("Cannot create identity matrix for dtype {}", dtype.to_godot()).into())
        }
    }

    pub fn get(&self, row: usize, col: usize) -> Option<Variant> {
        match self {
            MData::Bool(m) => m.get((row, col)).map(|x| x.to_variant()),
            MData::U8(m) => m.get((row, col)).map(|x| x.to_variant()),
            MData::I32(m) => m.get((row, col)).map(|x| x.to_variant()),
            MData::I64(m) => m.get((row, col)).map(|x| x.to_variant()),
            MData::F32(m) => m.get((row, col)).map(|x| x.to_variant()),
            MData::F64(m) => m.get((row, col)).map(|x| x.to_variant()),
            MData::C64(m) => m.get((row, col)).map(|x| Vector2::new(x.re, x.im).to_variant()),
            MData::C128(m) => m.get((row, col)).map(|x| array![x.re, x.im].to_variant()),
            MData::View { mat, start, .. } => {
                mat.bind().m.get(start.0 + row, start.1 + col)
            }
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
            MData::View { mat, .. } => mat.bind().m.dtype()
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
            MData::View { shape, .. } => *shape
        }
    }

    pub fn slice(&self, start: (usize, usize), shape: (usize, usize), view_type: ViewType) -> Self {
        match &self {
            MData::Bool(m) => MData::Bool(m.view(start, shape).clone_owned()),
            MData::U8(m) => MData::U8(m.view(start, shape).clone_owned()),
            MData::I32(m) => MData::I32(m.view(start, shape).clone_owned()),
            MData::I64(m) => MData::I64(m.view(start, shape).clone_owned()),
            MData::F32(m) => MData::F32(m.view(start, shape).clone_owned()),
            MData::F64(m) => MData::F64(m.view(start, shape).clone_owned()),
            MData::C64(m) => {
                match view_type {
                    ViewType::All => MData::C64(m.view(start, shape).clone_owned()),
                    ViewType::Real => MData::F32(m.view(start, shape).map(|x| x.re).clone_owned()),
                    ViewType::Imag => MData::F32(m.view(start, shape).map(|x| x.im).clone_owned())
                }
            },
            MData::C128(m) => {
                match view_type {
                    ViewType::All => MData::C128(m.view(start, shape).clone_owned()),
                    ViewType::Real => MData::F64(m.view(start, shape).map(|x| x.re).clone_owned()),
                    ViewType::Imag => MData::F64(m.view(start, shape).map(|x| x.im).clone_owned())
                }
            },
            MData::View { mat, start: p_start, view_type: p_view_type, .. } => {
                match (view_type, p_view_type) {
                    (ViewType::All, ViewType::Real) => {
                        let start = (p_start.0 + start.0, p_start.1 + start.1);
                        mat.bind().m.slice(start, shape, ViewType::Real)
                    },
                    (ViewType::All, ViewType::Imag) => {
                        let start = (p_start.0 + start.0, p_start.1 + start.1);
                        mat.bind().m.slice(start, shape, ViewType::Imag)
                    },
                    (ViewType::Real, ViewType::Imag) => {
                        MData::zeros(shape.0, shape.1, &mat.bind().m.dtype())
                    },
                    (ViewType::Imag, ViewType::Real) => {
                        MData::zeros(shape.0, shape.1, &mat.bind().m.dtype())
                    },
                    _ => {
                        let start = (p_start.0 + start.0, p_start.1 + start.1);
                        mat.bind().m.slice(start, shape, view_type)
                    }
                }
            }
        }
    }

    pub fn tranpose(&self) -> MData {
        match &self {
            MData::Bool(m) => MData::Bool(m.transpose()),
            MData::U8(m) => MData::U8(m.transpose()),
            MData::I32(m) => MData::I32(m.transpose()),
            MData::I64(m) => MData::I64(m.transpose()),
            MData::F32(m) => MData::F32(m.transpose()),
            MData::F64(m) => MData::F64(m.transpose()),
            MData::C64(m) => MData::C64(m.transpose()),
            MData::C128(m) => MData::C128(m.transpose()),
            MData::View { mat, start, shape, view_type } => {
                mat.bind().m.slice(*start, *shape, *view_type).tranpose()
            }
        }
    }

    pub fn adjoint(&self) -> Result<MData> {
        match &self {
            MData::F32(m) => Ok(MData::F32(m.adjoint())),
            MData::F64(m) => Ok(MData::F64(m.adjoint())),
            MData::C64(m) => Ok(MData::C64(m.adjoint())),
            MData::C128(m) => Ok(MData::C128(m.adjoint())),
            MData::View { mat, start, shape, view_type } => {
                mat.bind().m.slice(*start, *shape, *view_type).adjoint()
            }
            _ => {
                Err(format!("Unsupported type: {}.h()", self.as_ref()).into())
            }
        }
    }

    pub fn get_view_type(&self) -> ViewType {
        match self {
            MData::View { view_type, .. } => *view_type,
            _ => ViewType::All
        }
    }

    pub fn any(&self) -> bool {
        match self {
            MData::Bool(m) => m.iter().any(|x| *x),
            MData::U8(m) => m.iter().any(|x| *x == 1),
            MData::I32(m) => m.iter().any(|x| *x == 1),
            MData::I64(m) => m.iter().any(|x| *x == 1),
            MData::F32(m) => m.iter().any(|x| *x == 1.0),
            MData::F64(m) => m.iter().any(|x| *x == 1.0),
            MData::View { mat, start, shape, view_type } => {
                mat.bind().m.slice(*start, *shape, *view_type).any()
            },
            _ => false
        }
    }

    pub fn all(&self) -> bool {
        match self {
            MData::Bool(m) => m.iter().all(|x| *x),
            MData::U8(m) => m.iter().all(|x| *x == 1),
            MData::I32(m) => m.iter().all(|x| *x == 1),
            MData::I64(m) => m.iter().all(|x| *x == 1),
            MData::F32(m) => m.iter().all(|x| *x == 1.0),
            MData::F64(m) => m.iter().all(|x| *x == 1.0),
            MData::View { mat, start, shape, view_type } => {
                mat.bind().m.slice(*start, *shape, *view_type).all()
            },
            _ => false
        }
    }

    pub fn resize(&self, nrows: usize, ncols: usize, val: Variant) -> Result<Self> {
        match (self, val.get_type()) {
            (MData::Bool(m), VariantType::Int) => {
                Ok(MData::Bool(m.clone().resize(nrows, ncols, val.to::<i64>() == 1)))
            },
            (MData::U8(m), VariantType::Int) => {
                Ok(MData::U8(m.clone().resize(nrows, ncols, val.to::<i64>() as u8)))
            },
            (MData::I32(m), VariantType::Int) => {
                Ok(MData::I32(m.clone().resize(nrows, ncols, val.to::<i64>() as i32)))
            }
            (MData::I64(m), VariantType::Int) => {
                Ok(MData::I64(m.clone().resize(nrows, ncols, val.to::<i64>())))
            }
            (MData::F32(m), VariantType::Float) => {
                Ok(MData::F32(m.clone().resize(nrows, ncols, val.to::<f64>() as f32)))
            }
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
                arr.map_err(|_| format!("Array must be of type f64, got {:?}", val.get_type()).into())
                    .and_then(|arr| {
                        let length = arr.len();
                        (length == 2).then(|| arr)
                        .ok_or(format!("Array must have length 2, got length {}", length).into())
                    })
                    .map(|arr| Complex::new(arr.get(0), arr.get(1)))
                    .map(|c| MData::C128(m.clone().resize(nrows, ncols, c)))
            }
            _ => Err(format!("Variant value {} is not supported for matrix of type {}", 
                val.to_string(), self.as_ref()).into())
        }
    }

    pub fn kron(&self, other: &MData) -> Result<Self> {
        match (&self, other) {
            (MData::U8(a), MData::U8(b)) => {
                Ok(MData::U8(a.kronecker(b)))
            },
            (MData::I32(a), MData::I32(b)) => {
                Ok(MData::I32(a.kronecker(b)))
            },
            (MData::I64(a), MData::I64(b)) => {
                Ok(MData::I64(a.kronecker(b)))
            },
            (MData::F32(a), MData::F32(b)) => {
                Ok(MData::F32(a.kronecker(b)))
            },
            (MData::F64(a), MData::F64(b)) => {
                Ok(MData::F64(a.kronecker(b)))
            },
            (MData::C64(a), MData::C64(b)) => {
                Ok(MData::C64(a.kronecker(b)))
            },
            (MData::C128(a), MData::C128(b)) => {
                Ok(MData::C128(a.kronecker(b)))
            },
            _ => {
                Err(format!("Unsupported operand types: {} o {}", self.as_ref(), other.as_ref()).into())
            }
        }
    }

    pub fn mm(&self, other: &MData) -> Result<Self> {
        match (&self, other) {
            (MData::U8(a), MData::U8(b)) => {
                Ok(MData::U8(a * b))
            },
            (MData::I32(a), MData::I32(b)) => {
                Ok(MData::I32(a * b))
            },
            (MData::I64(a), MData::I64(b)) => {
                Ok(MData::I64(a * b))
            },
            (MData::F32(a), MData::F32(b)) => {
                Ok(MData::F32(a * b))
            },
            (MData::F64(a), MData::F64(b)) => {
                Ok(MData::F64(a * b))
            },
            (MData::C64(a), MData::C64(b)) => {
                Ok(MData::C64(a * b))
            },
            (MData::C128(a), MData::C128(b)) => {
                Ok(MData::C128(a * b))
            },
            _ => {
                Err(format!("Unsupported operand types: {}.{}", self.as_ref(), other.as_ref()).into())
            }
        }
    } 

    pub fn mul(&self, other: &MData) -> Result<Self> {
        match (&self, other) {
            (MData::U8(a), MData::U8(b)) => {
                Ok(MData::U8(a.component_mul(b)))
            },
            (MData::I32(a), MData::I32(b)) => {
                Ok(MData::I32(a.component_mul(b)))
            },
            (MData::I64(a), MData::I64(b)) => {
                Ok(MData::I64(a.component_mul(b)))
            },
            (MData::F32(a), MData::F32(b)) => {
                Ok(MData::F32(a.component_mul(b)))
            },
            (MData::F64(a), MData::F64(b)) => {
                Ok(MData::F64(a.component_mul(b)))
            },
            (MData::C64(a), MData::C64(b)) => {
                Ok(MData::C64(a.component_mul(b)))
            },
            (MData::C128(a), MData::C128(b)) => {
                Ok(MData::C128(a.component_mul(b)))
            },
            _ => {
                Err(format!("Unsupported operand types: {} * {}", self.as_ref(), other.as_ref()).into())
            }
        }
    }

    pub fn add(&self, other: &MData) -> Result<Self> {
        match (&self, other) {
            (MData::U8(a), MData::U8(b)) => {
                Ok(MData::U8(a + b))
            },
            (MData::I32(a), MData::I32(b)) => {
                Ok(MData::I32(a + b))
            },
            (MData::I64(a), MData::I64(b)) => {
                Ok(MData::I64(a + b))
            },
            (MData::F32(a), MData::F32(b)) => {
                Ok(MData::F32(a + b))
            },
            (MData::F64(a), MData::F64(b)) => {
                Ok(MData::F64(a + b))
            },
            (MData::C64(a), MData::C64(b)) => {
                Ok(MData::C64(a + b))
            },
            (MData::C128(a), MData::C128(b)) => {
                Ok(MData::C128(a + b))
            },
            _ => {
                Err(format!("Unsupported operand types: {} + {}", self.as_ref(), other.as_ref()).into())
            }
        }
    }

    pub fn lt(&self, other: &MData) -> Result<Self> {
        match (&self, other) {
            (MData::U8(a), MData::U8(b)) => {
                Ok(MData::Bool(a.zip_map(b, |a, b| a < b)))
            },
            (MData::I32(a), MData::I32(b)) => {
                Ok(MData::Bool(a.zip_map(b, |a, b| a < b)))
            },
            (MData::I64(a), MData::I64(b)) => {
                Ok(MData::Bool(a.zip_map(b, |a, b| a < b)))
            },
            (MData::F32(a), MData::F32(b)) => {
                Ok(MData::Bool(a.zip_map(b, |a, b| a < b)))
            },
            (MData::F64(a), MData::F64(b)) => {
                Ok(MData::Bool(a.zip_map(b, |a, b| a < b)))
            }
            _ => {
                Err(format!("Unsupported operand types: {} < {}", self.as_ref(), other.as_ref()).into())
            }
        }
    }

    pub fn le(&self, other: &MData) -> Result<Self> {
        match (&self, other) {
            (MData::U8(a), MData::U8(b)) => {
                Ok(MData::Bool(a.zip_map(b, |a, b| a <= b)))
            },
            (MData::I32(a), MData::I32(b)) => {
                Ok(MData::Bool(a.zip_map(b, |a, b| a <= b)))
            },
            (MData::I64(a), MData::I64(b)) => {
                Ok(MData::Bool(a.zip_map(b, |a, b| a <= b)))
            },
            (MData::F32(a), MData::F32(b)) => {
                Ok(MData::Bool(a.zip_map(b, |a, b| a <= b)))
            },
            (MData::F64(a), MData::F64(b)) => {
                Ok(MData::Bool(a.zip_map(b, |a, b| a <= b)))
            }
            _ => {
                Err(format!("Unsupported operand types: {} <= {}", self.as_ref(), other.as_ref()).into())
            }
        }
    }

    pub fn eq(&self, other: &MData) -> Result<Self> {
        match (&self, other) {
            (MData::U8(a), MData::U8(b)) => {
                Ok(MData::Bool(a.zip_map(b, |a, b| a == b)))
            },
            (MData::I32(a), MData::I32(b)) => {
                Ok(MData::Bool(a.zip_map(b, |a, b| a == b)))
            },
            (MData::I64(a), MData::I64(b)) => {
                Ok(MData::Bool(a.zip_map(b, |a, b| a == b)))
            },
            (MData::F32(a), MData::F32(b)) => {
                Ok(MData::Bool(a.zip_map(b, |a, b| a.eq(&b))))
            },
            (MData::F64(a), MData::F64(b)) => {
                Ok(MData::Bool(a.zip_map(b, |a, b| a.eq(&b))))
            }
            (MData::C64(a), MData::C64(b)) => {
                Ok(MData::Bool(a.zip_map(b, |a, b| a.eq(&b))))
            }
            (MData::C128(a), MData::C128(b)) => {
                Ok(MData::Bool(a.zip_map(b, |a, b| a.eq(&b))))
            }
            _ => {
                Err(format!("Unsupported operand types: {} == {}", self.as_ref(), other.as_ref()).into())
            }
        }
    }

    pub fn ge(&self, other: &MData) -> Result<Self> {
        match (&self, other) {
            (MData::U8(a), MData::U8(b)) => {
                Ok(MData::Bool(a.zip_map(b, |a, b| a >= b)))
            },
            (MData::I32(a), MData::I32(b)) => {
                Ok(MData::Bool(a.zip_map(b, |a, b| a >= b)))
            },
            (MData::I64(a), MData::I64(b)) => {
                Ok(MData::Bool(a.zip_map(b, |a, b| a >= b)))
            },
            (MData::F32(a), MData::F32(b)) => {
                Ok(MData::Bool(a.zip_map(b, |a, b| a >= b)))
            },
            (MData::F64(a), MData::F64(b)) => {
                Ok(MData::Bool(a.zip_map(b, |a, b| a >= b)))
            }
            _ => {
                Err(format!("Unsupported operand types: {} >= {}", self.as_ref(), other.as_ref()).into())
            }
        }
    }

    pub fn gt(&self, other: &MData) -> Result<Self> {
        match (&self, other) {
            (MData::U8(a), MData::U8(b)) => {
                Ok(MData::Bool(a.zip_map(b, |a, b| a > b)))
            },
            (MData::I32(a), MData::I32(b)) => {
                Ok(MData::Bool(a.zip_map(b, |a, b| a > b)))
            },
            (MData::I64(a), MData::I64(b)) => {
                Ok(MData::Bool(a.zip_map(b, |a, b| a > b)))
            },
            (MData::F32(a), MData::F32(b)) => {
                Ok(MData::Bool(a.zip_map(b, |a, b| a > b)))
            },
            (MData::F64(a), MData::F64(b)) => {
                Ok(MData::Bool(a.zip_map(b, |a, b| a > b)))
            }
            _ => {
                Err(format!("Unsupported operand types: {} > {}", self.as_ref(), other.as_ref()).into())
            }
        }
    }

    pub fn astype(&self, dt: DType) -> Result<MData> {
        match (self, &dt) {
            (MData::Bool(_), DType::Bool) => { Ok(self.clone()) },
            (MData::Bool(m), DType::U8) => {  Ok(MData::U8(m.map(|x| x as u8))) },
            (MData::Bool(m), DType::I32) => { Ok(MData::I32(m.map(|x| x as i32))) },
            (MData::Bool(m), DType::I64) => { Ok(MData::I64(m.map(|x| x as i64))) },
            (MData::Bool(m), DType::F32) => { Ok(MData::F32(m.map(|x| x as i32 as f32))) },
            (MData::Bool(m), DType::F64) => { Ok(MData::F64(m.map(|x| x as i32 as f64))) },
            (MData::Bool(m), DType::C64) => { Ok(MData::C64(m.map(|x| Complex::new(x as i32 as f32, 0.0)))) },
            (MData::Bool(m), DType::C128) => { Ok(MData::C128(m.map(|x| Complex::new(x as i32 as f64, 0.0))))},
            (MData::U8(m), DType::Bool) => { Ok(MData::Bool(m.map(|x| x > 0))) },
            (MData::U8(_), DType::U8) => { Ok(self.clone()) },
            (MData::U8(m), DType::I32) => { Ok(MData::I32(m.map(|x| x as i32))) },
            (MData::U8(m), DType::I64) => { Ok(MData::I64(m.map(|x| x as i64))) },
            (MData::U8(m), DType::F32) => { Ok(MData::F32(m.map(|x| x as f32))) },
            (MData::U8(m), DType::F64) => { Ok(MData::F64(m.map(|x| x as f64))) },
            (MData::U8(m), DType::C64) => { Ok(MData::C64(m.map(|x| Complex::new(x as f32, 0.0)))) },
            (MData::U8(m), DType::C128) => { Ok(MData::C128(m.map(|x| Complex::new(x as f64, 0.0))))},
            (MData::I32(m), DType::Bool) => { Ok(MData::Bool(m.map(|x| x > 0))) },
            (MData::I32(m), DType::U8) => {  Ok(MData::U8(m.map(|x| x as u8))) },
            (MData::I32(_), DType::I32) => { Ok(self.clone()) },
            (MData::I32(m), DType::I64) => { Ok(MData::I64(m.map(|x| x as i64))) },
            (MData::I32(m), DType::F32) => { Ok(MData::F32(m.map(|x| x as f32))) },
            (MData::I32(m), DType::F64) => { Ok(MData::F64(m.map(|x| x as f64))) },
            (MData::I32(m), DType::C64) => { Ok(MData::C64(m.map(|x| Complex::new(x as f32, 0.0)))) },
            (MData::I32(m), DType::C128) => { Ok(MData::C128(m.map(|x| Complex::new(x as f64, 0.0))))},
            (MData::I64(m), DType::Bool) => { Ok(MData::Bool(m.map(|x| x > 0))) },
            (MData::I64(m), DType::U8) => {  Ok(MData::U8(m.map(|x| x as u8))) },
            (MData::I64(m), DType::I32) => { Ok(MData::I32(m.map(|x| x as i32))) },
            (MData::I64(_), DType::I64) => { Ok(self.clone()) },
            (MData::I64(m), DType::F32) => { Ok(MData::F32(m.map(|x| x as f32))) },
            (MData::I64(m), DType::F64) => { Ok(MData::F64(m.map(|x| x as f64))) },
            (MData::I64(m), DType::C64) => { Ok(MData::C64(m.map(|x| Complex::new(x as f32, 0.0)))) },
            (MData::I64(m), DType::C128) => { Ok(MData::C128(m.map(|x| Complex::new(x as f64, 0.0))))},
            (MData::F32(m), DType::Bool) => { Ok(MData::Bool(m.map(|x| x > 0.0))) },
            (MData::F32(m), DType::U8) => {  Ok(MData::U8(m.map(|x| x as u8))) },
            (MData::F32(m), DType::I32) => { Ok(MData::I32(m.map(|x| x as i32))) },
            (MData::F32(m), DType::I64) => { Ok(MData::I64(m.map(|x| x as i64))) },
            (MData::F32(_), DType::F32) => { Ok(self.clone()) },
            (MData::F32(m), DType::F64) => { Ok(MData::F64(m.map(|x| x as f64))) },
            (MData::F32(m), DType::C64) => { Ok(MData::C64(m.map(|x| Complex::new(x as f32, 0.0)))) },
            (MData::F32(m), DType::C128) => { Ok(MData::C128(m.map(|x| Complex::new(x as f64, 0.0))))},
            (MData::F64(m), DType::Bool) => { Ok(MData::Bool(m.map(|x| x > 0.0))) },
            (MData::F64(m), DType::U8) => {  Ok(MData::U8(m.map(|x| x as u8))) },
            (MData::F64(m), DType::I32) => { Ok(MData::I32(m.map(|x| x as i32))) },
            (MData::F64(m), DType::I64) => { Ok(MData::I64(m.map(|x| x as i64))) },
            (MData::F64(m), DType::F32) => { Ok(MData::F32(m.map(|x| x as f32))) },
            (MData::F64(_), DType::F64) => { Ok(self.clone()) },
            (MData::F64(m), DType::C64) => { Ok(MData::C64(m.map(|x| Complex::new(x as f32, 0.0)))) },
            (MData::F64(m), DType::C128) => { Ok(MData::C128(m.map(|x| Complex::new(x as f64, 0.0))))},
            (MData::C64(_), DType::C64) => { Ok(self.clone()) },
            (MData::C64(m), DType::C128) => { Ok(MData::C128(m.map(|x| Complex::new(x.re as f64, x.im as f64))))},
            (MData::C128(m), DType::C64) => { Ok(MData::C64(m.map(|x| Complex::new(x.re as f32, x.im as f32)))) },
            (MData::C128(_), DType::C128) => { Ok(self.clone()) },
            (MData::View { mat, start, shape, view_type }, dt) => { 
                mat.bind().m.slice(*start, *shape, *view_type).astype(dt.clone())
            },
            _ => {
                Err(format!("Unsupported cast from to {} to {}", self.as_ref(), dt.to_godot()).into())
            }
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
            MData::View { mat, start, shape, view_type } => {
                mat.bind().m.slice(*start, *shape, *view_type).fmt(f)
            },
        }
    }
}