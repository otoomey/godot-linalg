use godot::prelude::*;
use nalgebra::{Complex, DMatrix};
use rand::{distributions::Uniform, Rng};

use crate::{
    mdata::MData,
    view::{Shape, View, ViewType},
};

type Result<T> = std::result::Result<T, Box<dyn std::error::Error>>;

#[derive(GodotConvert, Var, Export, Clone, PartialEq)]
#[godot(via=GString)]
pub enum DType {
    Bool,
    U8,
    I32,
    I64,
    F32,
    F64,
    C64,
    C128,
}

fn check_dim(nrows: i64, ncols: i64) -> Result<()> {
    if nrows < 1 {
        // godot_script_error!("Number of rows must be greater than 0: {}", nrows);
        return Err(format!("Number of rows must be greater than 0: {}", nrows).into());
    } else if ncols < 1 {
        return Err(format!("Number of columns must be greater than 0: {}", ncols).into());
    }
    return Ok(());
}

fn extract_array<T: FromGodot>(arr: &Array<Variant>) -> Result<Vec<T>> {
    arr.iter_shared()
        .enumerate()
        .map(|(i, v)| {
            v.try_to::<T>()
                .map_err(|_| format!("Expected value {} at index {}.", v, i).into())
        })
        .collect()
}

#[derive(GodotClass)]
#[class(base=RefCounted)]
pub struct Mat {
    pub m: MData,
    base: Base<RefCounted>,
}

#[godot_api]
impl IRefCounted for Mat {
    fn init(base: Base<RefCounted>) -> Self {
        Self {
            m: MData::F32(DMatrix::zeros(1, 1)),
            base,
        }
    }

    fn to_string(&self) -> GString {
        self.m.to_string().into_godot()
    }
}

#[godot_api]
impl Mat {
    #[func]
    fn rand(nrows: i64, ncols: i64) -> Option<Gd<Self>> {
        if let Err(err) = check_dim(nrows, ncols) {
            godot_script_error!("{}", err);
            return None;
        }
        let range = Uniform::new(0.0, 1.0);
        let data = rand::thread_rng()
            .sample_iter(&range)
            .take((nrows * ncols) as usize)
            .collect();
        Some(Gd::from_init_fn(|base| Self {
            m: MData::F64(DMatrix::from_vec(nrows as usize, ncols as usize, data)),
            base,
        }))
    }

    #[func]
    fn from_array(nrows: i64, ncols: i64, data: Variant) -> Option<Gd<Self>> {
        if let Err(err) = check_dim(nrows, ncols) {
            godot_script_error!("{}", err);
            return None;
        }
        match data.get_type() {
            VariantType::Array => {
                let arr = data.to::<Array<Variant>>();
                if nrows * ncols != arr.len() as i64 {
                    godot_script_error!(
                        "Expected data length of {}, got {}.",
                        nrows * ncols,
                        arr.len()
                    );
                    return None;
                }
                match arr.get(0).get_type() {
                    VariantType::Bool => match extract_array(&arr) {
                        Ok(values) => Some(MData::Bool(DMatrix::from_vec(
                            nrows as usize,
                            ncols as usize,
                            values,
                        ))),
                        Err(err) => {
                            godot_script_error!("{}", err);
                            None
                        }
                    },
                    VariantType::Int => match extract_array(&arr) {
                        Ok(values) => Some(MData::I64(DMatrix::from_vec(
                            nrows as usize,
                            ncols as usize,
                            values,
                        ))),
                        Err(err) => {
                            godot_script_error!("{}", err);
                            None
                        }
                    },
                    VariantType::Float => match extract_array(&arr) {
                        Ok(values) => Some(MData::F64(DMatrix::from_vec(
                            nrows as usize,
                            ncols as usize,
                            values,
                        ))),
                        Err(err) => {
                            godot_script_error!("{}", err);
                            None
                        }
                    },
                    VariantType::Vector2 => match extract_array::<Vector2>(&arr) {
                        Ok(values) => {
                            let c64 = values.into_iter().map(|v| Complex::new(v.x, v.y));
                            Some(MData::C64(DMatrix::from_iterator(
                                nrows as usize,
                                ncols as usize,
                                c64,
                            )))
                        }
                        Err(err) => {
                            godot_script_error!("{}", err);
                            None
                        }
                    },
                    _ => None,
                }
            }
            VariantType::PackedByteArray => {
                let arr = data.to::<PackedByteArray>();
                if nrows * ncols != arr.len() as i64 {
                    godot_script_error!(
                        "Expected data length of {}, got {}.",
                        nrows * ncols,
                        arr.len()
                    );
                    return None;
                }
                Some(MData::U8(DMatrix::from_vec(
                    nrows as usize,
                    ncols as usize,
                    arr.as_slice().to_vec(),
                )))
            }
            VariantType::PackedInt32Array => {
                let arr = data.to::<PackedInt32Array>();
                if nrows * ncols != arr.len() as i64 {
                    godot_script_error!(
                        "Expected data length of {}, got {}.",
                        nrows * ncols,
                        arr.len()
                    );
                    return None;
                }
                Some(MData::I32(DMatrix::from_vec(
                    nrows as usize,
                    ncols as usize,
                    arr.as_slice().to_vec(),
                )))
            }
            VariantType::PackedInt64Array => {
                let arr = data.to::<PackedInt64Array>();
                if nrows * ncols != arr.len() as i64 {
                    godot_script_error!(
                        "Expected data length of {}, got {}.",
                        nrows * ncols,
                        arr.len()
                    );
                    return None;
                }
                Some(MData::I64(DMatrix::from_vec(
                    nrows as usize,
                    ncols as usize,
                    arr.as_slice().to_vec(),
                )))
            }
            VariantType::PackedFloat32Array => {
                let arr = data.to::<PackedFloat32Array>();
                if nrows * ncols != arr.len() as i64 {
                    godot_script_error!(
                        "Expected data length of {}, got {}.",
                        nrows * ncols,
                        arr.len()
                    );
                    return None;
                }
                Some(MData::F32(DMatrix::from_vec(
                    nrows as usize,
                    ncols as usize,
                    arr.as_slice().to_vec(),
                )))
            }
            VariantType::PackedFloat64Array => {
                let arr = data.to::<PackedFloat64Array>();
                if nrows * ncols != arr.len() as i64 {
                    godot_script_error!(
                        "Expected data length of {}, got {}.",
                        nrows * ncols,
                        arr.len()
                    );
                    return None;
                }
                Some(MData::F64(DMatrix::from_vec(
                    nrows as usize,
                    ncols as usize,
                    arr.as_slice().to_vec(),
                )))
            }
            VariantType::PackedVector2Array => {
                let arr = data.to::<PackedVector2Array>();
                if nrows * ncols != arr.len() as i64 {
                    godot_script_error!(
                        "Expected data length of {}, got {}.",
                        nrows * ncols,
                        arr.len()
                    );
                    return None;
                }
                Some(MData::C64(DMatrix::from_iterator(
                    nrows as usize,
                    ncols as usize,
                    arr.as_slice().iter().map(|v| Complex::new(v.x, v.y)),
                )))
            }
            _ => {
                godot_script_error!("Expected Array or Packed*Array, got {}", data);
                None
            }
        }
        .map(|m| Gd::from_init_fn(|base| Self { m, base }))
    }

    #[func]
    fn from_diagonal_element(nrows: i64, ncols: i64, data: Variant) -> Option<Gd<Self>> {
        if let Err(err) = check_dim(nrows, ncols) {
            godot_script_error!("{}", err);
            return None;
        }
        match data.get_type() {
            VariantType::Nil => {
                godot_script_error!("Argument data cannot be nil");
                None
            }
            VariantType::Int => Some(MData::I64(DMatrix::from_diagonal_element(
                nrows as usize,
                ncols as usize,
                data.to(),
            ))),
            VariantType::Float => Some(MData::F64(DMatrix::from_diagonal_element(
                nrows as usize,
                ncols as usize,
                data.to(),
            ))),
            VariantType::Vector2 => Some(MData::C64(DMatrix::from_diagonal_element(
                nrows as usize,
                ncols as usize,
                Complex::new(data.to::<Vector2>().x, data.to::<Vector2>().y),
            ))),
            _ => {
                godot_script_error!("Unsupported data type {}", data);
                None
            }
        }
        .map(|m| Gd::from_init_fn(|base| Self { m, base }))
    }

    #[func]
    fn identity(nrows: i64, ncols: i64, dtype: GString) -> Option<Gd<Self>> {
        if let Err(err) = check_dim(nrows, ncols) {
            godot_script_error!("{}", err);
            return None;
        }
        DType::try_from_godot(dtype.clone())
            .map_err(|_| format!("Unknown dtype {}", dtype).into())
            .and_then(|dtype| MData::identity(nrows as usize, ncols as usize, &dtype))
            .map(|m| Gd::from_init_fn(|base| Self { m, base }))
            .map_err(|err| godot_script_error!("{}", err))
            .ok()
    }

    #[func]
    fn zeros(nrows: i64, ncols: i64, dtype: GString) -> Option<Gd<Self>> {
        if let Err(err) = check_dim(nrows, ncols) {
            godot_script_error!("{}", err);
            return None;
        }
        DType::try_from_godot(dtype.clone())
            .map_err(|_| format!("Unknown dtype {}", dtype).into())
            .map(|dtype| {
                Gd::from_init_fn(|base| Self {
                    m: MData::zeros(nrows as usize, ncols as usize, &dtype),
                    base,
                })
            })
            .map_err(|err: Box<dyn std::error::Error>| godot_script_error!("{}", err))
            .ok()
    }

    #[func]
    fn from_fn(nrows: i64, ncols: i64, func: Callable) -> Option<Gd<Self>> {
        if let Err(err) = check_dim(nrows, ncols) {
            godot_script_error!("{}", err);
            return None;
        }
        let arr = (0..nrows)
            .into_iter()
            .map(|r| {
                (0..ncols)
                    .into_iter()
                    .map(move |c| (r, c))
                    .map(|(r, c)| func.callv(array![r.to_variant(), c.to_variant()]))
            })
            .flatten()
            .collect::<VariantArray>();

        if nrows * ncols != arr.len() as i64 {
            godot_script_error!(
                "Expected data length of {}, got {}.",
                nrows * ncols,
                arr.len()
            );
            return None;
        }
        match arr.get(0).get_type() {
            VariantType::Bool => match extract_array(&arr) {
                Ok(values) => Some(MData::Bool(DMatrix::from_vec(
                    nrows as usize,
                    ncols as usize,
                    values,
                ))),
                Err(err) => {
                    godot_script_error!("{}", err);
                    None
                }
            },
            VariantType::Int => match extract_array(&arr) {
                Ok(values) => Some(MData::I64(DMatrix::from_vec(
                    nrows as usize,
                    ncols as usize,
                    values,
                ))),
                Err(err) => {
                    godot_script_error!("{}", err);
                    None
                }
            },
            VariantType::Float => match extract_array(&arr) {
                Ok(values) => Some(MData::F64(DMatrix::from_vec(
                    nrows as usize,
                    ncols as usize,
                    values,
                ))),
                Err(err) => {
                    godot_script_error!("{}", err);
                    None
                }
            },
            VariantType::Vector2 => match extract_array::<Vector2>(&arr) {
                Ok(values) => {
                    let c64 = values.into_iter().map(|v| Complex::new(v.x, v.y));
                    Some(MData::C64(DMatrix::from_iterator(
                        nrows as usize,
                        ncols as usize,
                        c64,
                    )))
                }
                Err(err) => {
                    godot_script_error!("{}", err);
                    None
                }
            },
            _ => None,
        }
        .map(|m| Gd::from_init_fn(|base| Self { m, base }))
    }

    #[func]
    fn add(&self, other: Gd<Mat>) -> Option<Gd<Self>> {
        let m = &other.bind().m;
        self.m
            .add(m)
            .map(|m| Gd::from_init_fn(|base| Self { m, base }))
            .map_err(|err| godot_script_error!("{}", err))
            .ok()
    }

    #[func]
    fn mul(&self, other: Gd<Mat>) -> Option<Gd<Self>> {
        let m = &other.bind().m;
        self.m
            .mul(m)
            .map(|m| Gd::from_init_fn(|base| Self { m, base }))
            .map_err(|err| godot_script_error!("{}", err))
            .ok()
    }

    #[func]
    fn mm(&self, other: Gd<Mat>) -> Option<Gd<Self>> {
        let m = &other.bind().m;
        self.m
            .mm(m)
            .map(|m| Gd::from_init_fn(|base| Self { m, base }))
            .map_err(|err| godot_script_error!("{}", err))
            .ok()
    }

    #[func]
    fn kron(&self, other: Gd<Mat>) -> Option<Gd<Self>> {
        let m = &other.bind().m;
        self.m
            .kron(m)
            .map(|m| Gd::from_init_fn(|base| Self { m, base }))
            .map_err(|err| godot_script_error!("{}", err))
            .ok()
    }

    #[func]
    fn lt(&self, other: Gd<Mat>) -> Option<Gd<Self>> {
        let m = &other.bind().m;
        self.m
            .lt(m)
            .map(|m| Gd::from_init_fn(|base| Self { m, base }))
            .map_err(|err| godot_script_error!("{}", err))
            .ok()
    }

    #[func]
    fn le(&self, other: Gd<Mat>) -> Option<Gd<Self>> {
        let m = &other.bind().m;
        self.m
            .le(m)
            .map(|m| Gd::from_init_fn(|base| Self { m, base }))
            .map_err(|err| godot_script_error!("{}", err))
            .ok()
    }

    #[func]
    fn eq(&self, other: Gd<Mat>) -> Option<Gd<Self>> {
        let m = &other.bind().m;
        self.m
            .eq(m)
            .map(|m| Gd::from_init_fn(|base| Self { m, base }))
            .map_err(|err| godot_script_error!("{}", err))
            .ok()
    }

    #[func]
    fn ge(&self, other: Gd<Mat>) -> Option<Gd<Self>> {
        let m = &other.bind().m;
        self.m
            .ge(m)
            .map(|m| Gd::from_init_fn(|base| Self { m, base }))
            .map_err(|err| godot_script_error!("{}", err))
            .ok()
    }

    #[func]
    fn gt(&self, other: Gd<Mat>) -> Option<Gd<Self>> {
        let m = &other.bind().m;
        self.m
            .gt(m)
            .map(|m| Gd::from_init_fn(|base| Self { m, base }))
            .map_err(|err| godot_script_error!("{}", err))
            .ok()
    }

    #[func]
    fn all(&self) -> Variant {
        self.m
            .all(&crate::view::ViewType::Identity, &Shape::Identity)
            .map_err(|err| godot_script_error!("{}", err))
            .ok()
            .map(|x| x.to_variant())
            .unwrap_or(Variant::nil())
    }

    #[func]
    fn any(&self) -> Variant {
        self.m
            .any(&crate::view::ViewType::Identity, &Shape::Identity)
            .map_err(|err| godot_script_error!("{}", err))
            .ok()
            .map(|x| x.to_variant())
            .unwrap_or(Variant::nil())
    }

    #[func]
    fn t(&self) -> Gd<Self> {
        Gd::from_init_fn(|base| Self {
            m: self.m.tranpose().unwrap(),
            base,
        })
    }

    #[func]
    fn h(&self) -> Option<Gd<Self>> {
        self.m
            .adjoint()
            .map(|m| Gd::from_init_fn(|base| Self { m, base }))
            .map_err(|err| godot_script_error!("{}", err))
            .ok()
    }

    #[func]
    fn shape(&self) -> Array<i64> {
        let s = self.m.shape();
        array![s.0 as i64, s.1 as i64]
    }

    #[func]
    fn resize(&self, nrows: i64, ncols: i64, val: Variant) -> Option<Gd<Self>> {
        if let Err(err) = check_dim(nrows, ncols) {
            godot_script_error!("{}", err);
            return None;
        }
        self.m
            .resize(nrows as usize, ncols as usize, val)
            .map(|m| Gd::from_init_fn(|base| Self { m, base }))
            .map_err(|err| godot_script_error!("{}", err))
            .ok()
    }

    #[func]
    fn get(&self, row: i64, col: i64) -> Variant {
        if row < 0 || row >= self.m.shape().0 as i64 {
            godot_script_error!("`row`={} exceeds mat bounds ({:?})", row, self.m.shape());
            return Variant::nil();
        }
        if col < 0 || col >= self.m.shape().1 as i64 {
            godot_script_error!("`col`={} exceeds mat bounds ({:?})", col, self.m.shape());
            return Variant::nil();
        }
        self.m
            .get_index(
                row as usize,
                col as usize,
                &crate::view::ViewType::Identity,
                &Shape::Identity,
            )
            .unwrap_or(Variant::nil())
    }

    #[func]
    fn set_(&mut self, other: Gd<Mat>) {
        let shape = other.bind().m.shape();
        let dtype = other.bind().m.dtype();
        if shape != self.m.shape() {
            godot_script_error!(
                "Matrix sizes do not match: {:?} != {:?}",
                shape,
                self.m.shape()
            );
            return;
        }
        if dtype != self.m.dtype() {
            godot_script_error!(
                "Matrix types do not match: {} != {}",
                dtype.to_godot(),
                self.m.dtype().to_godot()
            );
            return;
        }
        self.m.set(
            &other.bind().m,
            &ViewType::Identity,
            &Shape::Identity,
            &ViewType::Identity,
            &Shape::Identity,
        );
    }

    #[func]
    fn set_index_(&mut self, row: i64, col: i64, value: Variant) {
        if row < 0 || row >= self.m.shape().0 as i64 {
            godot_script_error!("`row`={} exceeds mat bounds ({:?})", row, self.m.shape());
            return;
        }
        if col < 0 || col >= self.m.shape().1 as i64 {
            godot_script_error!("`col`={} exceeds mat bounds ({:?})", col, self.m.shape());
            return;
        }
        self.m
            .set_index(
                row as usize,
                col as usize,
                value,
                &ViewType::Identity,
                &Shape::Identity,
            )
            .unwrap_or_else(|err| godot_script_error!("{}", err));
    }

    #[func]
    fn rows(&self, i: i64, size: i64) -> Option<Gd<Self>> {
        if i < 0 {
            godot_script_error!("`i` must be greater than 0, got: {}", i);
            return None;
        }
        if size < 1 {
            godot_script_error!("`size` must be greater than or equal to 1, got: {}", size);
            return None;
        }
        if i + size > self.m.shape().0 as i64 {
            godot_script_error!(
                "view is out of bounds for mat of shape: {:?}",
                self.m.shape()
            );
            return None;
        }
        let start = (i as usize, 0);
        let shape = (size as usize, self.m.shape().1);
        let shape = Shape::Rect { start, shape };
        Some(Gd::from_init_fn(|base| Self {
            m: MData::View(View::new(self.to_gd(), ViewType::Identity, shape)),
            base,
        }))
    }

    #[func]
    fn columns(&self, i: i64, size: i64) -> Option<Gd<Self>> {
        if i < 0 {
            godot_script_error!("`i` must be greater than 0, got: {}", i);
            return None;
        }
        if size < 1 {
            godot_script_error!("`size` must be greater than or equal to 1, got: {}", size);
            return None;
        }
        if i + size > self.m.shape().1 as i64 {
            godot_script_error!(
                "view is out of bounds for mat of shape: {:?}",
                self.m.shape()
            );
            return None;
        }
        let start = (0, i as usize);
        let shape = (self.m.shape().0, size as usize);
        let shape = Shape::Rect { start, shape };
        Some(Gd::from_init_fn(|base| Self {
            m: MData::View(View::new(self.to_gd(), ViewType::Identity, shape)),
            base,
        }))
    }

    #[func]
    fn view(&self, start: Array<i64>, shape: Array<i64>) -> Option<Gd<Self>> {
        if start.len() != 2 || start.iter_shared().any(|i| i < 0) {
            godot_script_error!(
                "`start` must be positive semi-definite array of size two, got: {}",
                start
            );
            return None;
        }
        if shape.len() != 2 || shape.iter_shared().any(|i| i < 1) {
            godot_script_error!(
                "`shape` must be positive definite array of size two, got: {}",
                shape
            );
            return None;
        }
        if start.get(0) + shape.get(0) > self.m.shape().0 as i64 {
            godot_script_error!(
                "view is out of bounds for mat of shape: {:?}",
                self.m.shape()
            );
            return None;
        }
        if start.get(1) + shape.get(1) > self.m.shape().1 as i64 {
            godot_script_error!(
                "view is out of bounds for mat of shape: {:?}",
                self.m.shape()
            );
            return None;
        }
        let start = (start.get(0) as usize, start.get(1) as usize);
        let shape = (shape.get(0) as usize, shape.get(1) as usize);
        let shape = Shape::Rect { start, shape };
        Some(Gd::from_init_fn(|base| Self {
            m: MData::View(View::new(self.to_gd(), ViewType::Identity, shape)),
            base,
        }))
    }

    #[func]
    fn real(&self) -> Gd<Self> {
        Gd::from_init_fn(|base| Self {
            m: MData::View(View::new(self.to_gd(), ViewType::Real, Shape::Identity)),
            base,
        })
    }

    #[func]
    fn imag(&self) -> Option<Gd<Self>> {
        match self.m.dtype() {
            DType::C64 | DType::C128 => Some(Gd::from_init_fn(|base| Self {
                m: MData::View(View::new(self.to_gd(), ViewType::Imag, Shape::Identity)),
                base,
            })),
            _ => {
                godot_script_error!("Data type {} is real valued.", self.m.dtype().to_godot());
                None
            }
        }
    }

    #[func]
    fn clone(&self) -> Gd<Self> {
        Gd::from_init_fn(|base| Self {
            m: self.m.slice(&ViewType::Identity, &Shape::Identity).unwrap(),
            base,
        })
    }

    #[func]
    fn dtype(&self) -> GString {
        self.m.dtype().to_godot()
    }

    #[func]
    fn astype(&self, dtype: GString) -> Option<Gd<Self>> {
        let dt = DType::try_from_godot(dtype.clone());
        if dt.is_err() {
            godot_script_error!("Unknown `dtype`: {}", dtype);
            return None;
        }
        let dtype = dt.unwrap();
        self.m
            .astype(dtype)
            .map(|m| Gd::from_init_fn(|base| Self { m, base }))
            .map_err(|err| godot_script_error!("{}", err))
            .ok()
    }

    #[func]
    fn slice(&self, start: Array<i64>, shape: Array<i64>) -> Option<Gd<Self>> {
        if start.len() != 2 || start.iter_shared().any(|i| i < 0) {
            godot_script_error!(
                "`start` must be positive definite array of size two, got: {}",
                start
            );
            return None;
        }
        if shape.len() != 2 || shape.iter_shared().any(|i| i < 0) {
            godot_script_error!(
                "`shape` must be positive definite array of size two, got: {}",
                shape
            );
            return None;
        }
        if start.get(0) + shape.get(0) > self.m.shape().0 as i64 {
            godot_script_error!(
                "view is out of bounds for mat of shape: {:?}",
                self.m.shape()
            );
            return None;
        }
        if start.get(1) + shape.get(1) > self.m.shape().1 as i64 {
            godot_script_error!(
                "view is out of bounds for mat of shape: {:?}",
                self.m.shape()
            );
            return None;
        }
        let start = (start.get(0) as usize, start.get(1) as usize);
        let shape = (shape.get(0) as usize, shape.get(1) as usize);
        let shape = Shape::Rect { start, shape };
        Some(Gd::from_init_fn(|base| Self {
            m: self.m.slice(&ViewType::Identity, &shape).unwrap(),
            base,
        }))
    }
}
