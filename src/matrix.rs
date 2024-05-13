use godot::prelude::*;
use nalgebra::{Complex, DMatrix, Matrix};
use rand::{distributions::Uniform, Rng};
use strum_macros::AsRefStr;

type Result<T> = std::result::Result<T, Box<dyn std::error::Error>>;

#[derive(GodotConvert, Var, Export, Clone, PartialEq)] 
 #[godot(via=GString)] 
enum DType {
    Bool,
    U8,
    I32,
    I64,
    F32,
    F64,
    C64,
    C128
}

#[derive(Copy, Clone)]
enum ViewType {
    All,
    Real,
    Imag
}

#[derive(AsRefStr, Clone)]
enum MType {
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

fn check_dim(nrows: i64, ncols: i64) -> Result<()>{
    if nrows < 1 {
        // godot_script_error!("Number of rows must be greater than 0: {}", nrows);
        return Err(format!("Number of rows must be greater than 0: {}", nrows).into())
    } else if ncols < 1 {
        return Err(format!("Number of columns must be greater than 0: {}", ncols).into())
    }
    return Ok(())
}

fn extract_array<T: FromGodot>(arr: &Array<Variant>) -> Result<Vec<T>> {
    arr.iter_shared()
        .enumerate()
        .map(|(i, v)| v.try_to::<T>()
            .map_err(|_| {
               format!("Expected value {} at index {}.", v, i).into()
            })
        )
        .collect()
}

#[derive(GodotClass)]
#[class(base=RefCounted)]
struct Mat {
    m: MType,
    base: Base<RefCounted>
}

#[godot_api]
impl IRefCounted for Mat {
    fn init(base: Base<RefCounted>) -> Self {
        Self {
            m: MType::F32(DMatrix::zeros(1, 1)),
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
            return None
        }
        let range = Uniform::new(0.0, 1.0);
        let data = rand::thread_rng().sample_iter(&range).take((nrows*ncols) as usize).collect();
        Some(Gd::from_init_fn(|base| {
            Self {
                m: MType::F64(DMatrix::from_vec(nrows as usize, ncols as usize, data)),
                base,
            }
        }))
    }

    #[func]
    fn from_array(nrows: i64, ncols: i64, data: Variant) -> Option<Gd<Self>> {
        if let Err(err) = check_dim(nrows, ncols) {
            godot_script_error!("{}", err);
            return None
        }
        match data.get_type() {
            VariantType::Array => {
                let arr = data.to::<Array<Variant>>();
                if nrows * ncols != arr.len() as i64 {
                    godot_script_error!("Expected data length of {}, got {}.", nrows * ncols, arr.len());
                    return None
                }
                match arr.get(0).get_type() {
                    VariantType::Bool => match extract_array(&arr) {
                        Ok(values) => {
                            Some(MType::Bool(DMatrix::from_vec(
                                nrows as usize, 
                                ncols as usize, 
                                values
                            )))
                        },
                        Err(err) => {
                            godot_script_error!("{}", err);
                            None
                        }
                    },
                    VariantType::Int => match extract_array(&arr) {
                        Ok(values) => {
                            Some(MType::I64(DMatrix::from_vec(
                                nrows as usize, 
                                ncols as usize, 
                                values
                            )))
                        },
                        Err(err) => {
                            godot_script_error!("{}", err);
                            None
                        }
                    },
                    VariantType::Float => match extract_array(&arr) {
                        Ok(values) => {
                            Some(MType::F64(DMatrix::from_vec(
                                nrows as usize, 
                                ncols as usize, 
                                values
                            )))
                        },
                        Err(err) => {
                            godot_script_error!("{}", err);
                            None
                        }
                    },
                    VariantType::Vector2 => match extract_array::<Vector2>(&arr) {
                        Ok(values) => {
                            let c64 = values.into_iter()
                                .map(|v| Complex::new(v.x, v.y));
                            Some(MType::C64(DMatrix::from_iterator(
                                nrows as usize, 
                                ncols as usize, 
                                c64
                            )))
                        },
                        Err(err) => {
                            godot_script_error!("{}", err);
                            None
                        }
                    },
                    _ => None
                }
            },
            VariantType::PackedByteArray => {
                let arr = data.to::<PackedByteArray>();
                if nrows * ncols != arr.len() as i64 {
                    godot_script_error!("Expected data length of {}, got {}.", nrows * ncols, arr.len());
                    return None
                }
                Some(MType::U8(DMatrix::from_vec(
                    nrows as usize, 
                    ncols as usize, 
                    arr.as_slice().to_vec()
                )))
            },
            VariantType::PackedInt32Array => {
                let arr = data.to::<PackedInt32Array>();
                if nrows * ncols != arr.len() as i64 {
                    godot_script_error!("Expected data length of {}, got {}.", nrows * ncols, arr.len());
                    return None
                }
                Some(MType::I32(DMatrix::from_vec(
                    nrows as usize, 
                    ncols as usize, 
                    arr.as_slice().to_vec()
                )))
            },
            VariantType::PackedInt64Array => {
                let arr = data.to::<PackedInt64Array>();
                if nrows * ncols != arr.len() as i64 {
                    godot_script_error!("Expected data length of {}, got {}.", nrows * ncols, arr.len());
                    return None
                }
                Some(MType::I64(DMatrix::from_vec(
                    nrows as usize, 
                    ncols as usize, 
                    arr.as_slice().to_vec()
                )))
            },
            VariantType::PackedFloat32Array => {
                let arr = data.to::<PackedFloat32Array>();
                if nrows * ncols != arr.len() as i64 {
                    godot_script_error!("Expected data length of {}, got {}.", nrows * ncols, arr.len());
                    return None
                }
                Some(MType::F32(DMatrix::from_vec(
                    nrows as usize, 
                    ncols as usize, 
                    arr.as_slice().to_vec()
                )))
            },
            VariantType::PackedFloat64Array => {
                let arr = data.to::<PackedFloat64Array>();
                if nrows * ncols != arr.len() as i64 {
                    godot_script_error!("Expected data length of {}, got {}.", nrows * ncols, arr.len());
                    return None
                }
                Some(MType::F64(DMatrix::from_vec(
                    nrows as usize, 
                    ncols as usize, 
                    arr.as_slice().to_vec()
                )))
            },
            VariantType::PackedVector2Array => {
                let arr = data.to::<PackedVector2Array>();
                if nrows * ncols != arr.len() as i64 {
                    godot_script_error!("Expected data length of {}, got {}.", nrows * ncols, arr.len());
                    return None
                }
                Some(MType::C64(DMatrix::from_iterator(
                    nrows as usize, 
                    ncols as usize, 
                    arr.as_slice().iter()
                        .map(|v| Complex::new(v.x, v.y))
                )))
            },
            _ => {
                godot_script_error!("Expected Array or Packed*Array, got {}", data);
                None
            }
        }.map(|m| Gd::from_init_fn(|base|
            Self {
                m,
                base,
            }
        ))
    }

    #[func]
    fn from_diagonal_element(nrows: i64, ncols: i64, data: Variant) -> Option<Gd<Self>> {
        if let Err(err) = check_dim(nrows, ncols) {
            godot_script_error!("{}", err);
            return None
        }
        match data.get_type() {
            VariantType::Nil => {
                godot_script_error!("Argument data cannot be nil");
                None
            },
            VariantType::Int => Some(MType::I64(DMatrix::from_diagonal_element(
                nrows as usize, 
                ncols as usize, 
                data.to()
            ))),
            VariantType::Float => Some(MType::F64(DMatrix::from_diagonal_element(
                    nrows as usize, 
                    ncols as usize, 
                    data.to()
            ))),
            VariantType::Vector2 => Some(MType::C64(DMatrix::from_diagonal_element(
                nrows as usize, 
                ncols as usize, 
                Complex::new(data.to::<Vector2>().x, data.to::<Vector2>().y)
            ))),
            _ => {
                godot_script_error!("Unsupported data type {}", data);
                None
            }
        }.map(|m| Gd::from_init_fn(|base|
            Self {
                m,
                base,
            }
        ))
    }

    #[func]
    fn identity(nrows: i64, ncols: i64, dtype: GString) -> Option<Gd<Self>> {
        if let Err(err) = check_dim(nrows, ncols) {
            godot_script_error!("{}", err);
            return None
        }
        let dtype = DType::from_godot(dtype);
        match dtype {
                DType::U8 => Some(MType::U8(DMatrix::identity(
                    nrows as usize,
                    ncols as usize
                ))),
                DType::I32 => Some(MType::I32(DMatrix::identity(
                    nrows as usize,
                    ncols as usize
                ))),
                DType::I64 => Some(MType::I64(DMatrix::identity(
                    nrows as usize,
                    ncols as usize
                ))),
                DType::F32 => Some(MType::F32(DMatrix::identity(
                    nrows as usize,
                    ncols as usize
                ))),
                DType::F64 => Some(MType::F64(DMatrix::identity(
                    nrows as usize,
                    ncols as usize
                ))),
                DType::C64 => Some(MType::C64(DMatrix::identity(
                    nrows as usize,
                    ncols as usize
                ))),
                DType::C128 => Some(MType::C128(DMatrix::identity(
                    nrows as usize,
                    ncols as usize
                ))),
                DType::Bool => {
                    godot_script_error!("Unsupported data type {} for identity. Consider Mat.identity(.., U8).astype(Bool).", dtype.to_godot());
                    None
                }
        }.map(|m| Gd::from_init_fn(|base| {
            Self {
                m,
                base,
            }
        }))
    }

    #[func]
    fn from_fn(nrows: i64, ncols: i64, func: Callable) -> Option<Gd<Self>> {
        if let Err(err) = check_dim(nrows, ncols) {
            godot_script_error!("{}", err);
            return None
        }
        let arr = (0..nrows).into_iter()
            .map(|r| (0..ncols).into_iter()
                .map(move |c| (r, c))
                .map(|(r, c)| {
                    func.callv(array![r.to_variant(), c.to_variant()])
                })
            )
            .flatten()
            .collect::<VariantArray>();

        if nrows * ncols != arr.len() as i64 {
            godot_script_error!("Expected data length of {}, got {}.", nrows * ncols, arr.len());
            return None
        }
        match arr.get(0).get_type() {
            VariantType::Bool => match extract_array(&arr) {
                Ok(values) => {
                    Some(MType::Bool(DMatrix::from_vec(
                        nrows as usize, 
                        ncols as usize, 
                        values
                    )))
                },
                Err(err) => {
                    godot_script_error!("{}", err);
                    None
                }
            },
            VariantType::Int => match extract_array(&arr) {
                Ok(values) => {
                    Some(MType::I64(DMatrix::from_vec(
                        nrows as usize, 
                        ncols as usize, 
                        values
                    )))
                },
                Err(err) => {
                    godot_script_error!("{}", err);
                    None
                }
            },
            VariantType::Float => match extract_array(&arr) {
                Ok(values) => {
                    Some(MType::F64(DMatrix::from_vec(
                        nrows as usize, 
                        ncols as usize, 
                        values
                    )))
                },
                Err(err) => {
                    godot_script_error!("{}", err);
                    None
                }
            },
            VariantType::Vector2 => match extract_array::<Vector2>(&arr) {
                Ok(values) => {
                    let c64 = values.into_iter()
                        .map(|v| Complex::new(v.x, v.y));
                    Some(MType::C64(DMatrix::from_iterator(
                        nrows as usize, 
                        ncols as usize, 
                        c64
                    )))
                },
                Err(err) => {
                    godot_script_error!("{}", err);
                    None
                }
            },
            _ => None
        }.map(|m| Gd::from_init_fn(|base|
            Self {
                m,
                base,
            }
        ))
    }

    #[func]
    fn add(&self, other: Gd<Mat>) -> Option<Gd<Self>> {
        let m = &other.bind().m;
        self.m.add(m).map(|m| Gd::from_init_fn(|base|
            Self {
                m,
                base,
            }
        ))
    }

    #[func]
    fn mul(&self, other: Gd<Mat>) -> Option<Gd<Self>> {
        let m = &other.bind().m;
        self.m.mul(m).map(|m| Gd::from_init_fn(|base|
            Self {
                m,
                base,
            }
        ))
    }

    #[func]
    fn mm(&self, other: Gd<Mat>) -> Option<Gd<Self>> {
        let m = &other.bind().m;
        self.m.mm(m).map(|m| Gd::from_init_fn(|base|
            Self {
                m,
                base,
            }
        ))
    } 

    #[func]
    fn kron(&self, other: Gd<Mat>) -> Option<Gd<Self>> {
        let m = &other.bind().m;
        self.m.kron(m).map(|m| Gd::from_init_fn(|base|
            Self {
                m,
                base,
            }
        ))
    }

    #[func]
    fn lt(&self, other: Gd<Mat>) -> Option<Gd<Self>> {
        let m = &other.bind().m;
        self.m.lt(m).map(|m| Gd::from_init_fn(|base|
            Self { m, base, }
        ))
    }

    #[func]
    fn le(&self, other: Gd<Mat>) -> Option<Gd<Self>> {
        let m = &other.bind().m;
        self.m.le(m).map(|m| Gd::from_init_fn(|base|
            Self { m, base, }
        ))
    }

    #[func]
    fn eq(&self, other: Gd<Mat>) -> Option<Gd<Self>> {
        let m = &other.bind().m;
        self.m.eq(m).map(|m| Gd::from_init_fn(|base|
            Self { m, base, }
        ))
    }

    #[func]
    fn ge(&self, other: Gd<Mat>) -> Option<Gd<Self>> {
        let m = &other.bind().m;
        self.m.ge(m).map(|m| Gd::from_init_fn(|base|
            Self { m, base, }
        ))
    }

    #[func]
    fn gt(&self, other: Gd<Mat>) -> Option<Gd<Self>> {
        let m = &other.bind().m;
        self.m.gt(m).map(|m| Gd::from_init_fn(|base|
            Self { m, base, }
        ))
    }

    #[func]
    fn all(&self) -> bool {
        self.m.all()
    }

    #[func]
    fn any(&self) -> bool {
        self.m.any()
    }

    #[func]
    fn t(&self) -> Gd<Self> {
        Gd::from_init_fn(|base|
            Self {
                m: self.m.tranpose(),
                base,
            }
        )
    }

    #[func]
    fn h(&self) -> Option<Gd<Self>> {
        self.m.adjoint().map(|m| Gd::from_init_fn(|base|
            Self { m, base }
        ))
    }

    #[func]
    fn shape(&self) -> Array<i64> {
        let s = self.m.shape();
        array![s.0 as i64, s.1 as i64]
    }

    #[func]
    fn get(&self, row: i64, col: i64) -> Variant {
        if row < 0 || row >= self.m.shape().0 as i64 {
            godot_script_error!("`row`={} exceeds mat bounds ({:?})", row, self.m.shape());
            return Variant::nil()
        }
        if col < 0 || col >= self.m.shape().1 as i64 {
            godot_script_error!("`col`={} exceeds mat bounds ({:?})", col, self.m.shape());
            return Variant::nil()
        }
        self.m.get(row as usize, col as usize).unwrap_or(Variant::nil())
    }

    #[func]
    fn set(&mut self, other: Gd<Mat>) {
        let shape = other.bind().m.shape();
        let dtype = other.bind().m.dtype();
        if shape != self.m.shape() {
            godot_script_error!("Matrix sizes do not match: {:?} != {:?}", shape, self.m.shape());
            return;
        }
        if dtype != self.m.dtype() {
            godot_script_error!("Matrix types do not match: {} != {}", dtype.to_godot(), self.m.dtype().to_godot());
            return;
        }
        self.m = other.bind().m.clone();
    }

    #[func]
    fn rows(&self, i: i64, size: i64) -> Option<Gd<Self>> {
        if i < 0 {
            godot_script_error!("`i` must be greater than 0, got: {}", i);
            return None
        }
        if size < 1 {
            godot_script_error!("`size` must be greater than or equal to 1, got: {}", size);
            return None
        }
        if i + size > self.m.shape().0 as i64 {
            godot_script_error!("view is out of bounds for mat of shape: {:?}", self.m.shape());
            return None
        }
        let start = (i as usize, 0);
        let shape = (size as usize, self.m.shape().1);
        Some(Gd::from_init_fn(|base|
            Self {
                m: MType::View { mat: self.to_gd(), start, shape, view_type: self.m.get_view_type() },
                base,
            }
        ))
    }

    #[func]
    fn columns(&self, i: i64, size: i64) -> Option<Gd<Self>> {
        if i < 0 {
            godot_script_error!("`i` must be greater than 0, got: {}", i);
            return None
        }
        if size < 1 {
            godot_script_error!("`size` must be greater than or equal to 1, got: {}", size);
            return None
        }
        if i + size > self.m.shape().1 as i64 {
            godot_script_error!("view is out of bounds for mat of shape: {:?}", self.m.shape());
            return None
        }
        let start = (0, i as usize);
        let shape = (self.m.shape().0, size as usize);
        Some(Gd::from_init_fn(|base|
            Self {
                m: MType::View { mat: self.to_gd(), start, shape, view_type: self.m.get_view_type() },
                base,
            }
        ))
    }

    #[func]
    fn view(&self, start: Array<i64>, shape: Array<i64>) -> Option<Gd<Self>> {
        if start.len() != 2 || start.iter_shared().any(|i| i < 0) {
            godot_script_error!("`start` must be positive semi-definite array of size two, got: {}", start);
            return None
        }
        if shape.len() != 2 || shape.iter_shared().any(|i| i < 1) {
            godot_script_error!("`shape` must be positive definite array of size two, got: {}", shape);
            return None
        }
        if start.get(0) + shape.get(0) > self.m.shape().0 as i64 {
            godot_script_error!("view is out of bounds for mat of shape: {:?}", self.m.shape());
            return None
        }
        if start.get(1) + shape.get(1) > self.m.shape().1 as i64 {
            godot_script_error!("view is out of bounds for mat of shape: {:?}", self.m.shape());
            return None
        }
        let start = (start.get(0) as usize, start.get(1) as usize);
        let shape = (shape.get(0) as usize, shape.get(1) as usize);
        Some(Gd::from_init_fn(|base|
            Self {
                m: MType::View { mat: self.to_gd(), start, shape, view_type: self.m.get_view_type() },
                base,
            }
        ))
    }

    #[func]
    fn real(&self) -> Gd<Self> {
        Gd::from_init_fn(|base|
            Self {
                m: MType::View { mat: self.to_gd(), start: (0, 0), shape: self.m.shape(), view_type: ViewType::Real },
                base,
            }
        )
    }

    #[func]
    fn imag(&self) -> Option<Gd<Self>> {
        match self.m.dtype() {
            DType::C64 | DType::C128 => Some(Gd::from_init_fn(|base|
                Self {
                    m: MType::View { mat: self.to_gd(), start: (0, 0), shape: self.m.shape(), view_type: ViewType::Imag },
                    base,
                }
            )),
            _ => {
                godot_script_error!("Data type {} is real valued.", self.m.dtype().to_godot());
                None
            }
        }
    }

    #[func]
    fn copy(&self) -> Gd<Self> {
        Gd::from_init_fn(|base|
            Self {
                m: self.m.clone(),
                base,
            }
        )
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
        if let Some(m) = self.m.astype(dtype) {
            return Some(Gd::from_init_fn(|base|
                Self { m, base }
            ))
        }
        None
    }

    #[func]
    fn slice(&self, start: Array<i64>, shape: Array<i64>) -> Option<Gd<Self>> {
        if start.len() != 2 || start.iter_shared().any(|i| i < 0) {
            godot_script_error!("`start` must be positive definite array of size two, got: {}", start);
            return None
        }
        if shape.len() != 2 || shape.iter_shared().any(|i| i < 0) {
            godot_script_error!("`shape` must be positive definite array of size two, got: {}", shape);
            return None
        }
        if start.get(0) + shape.get(0) > self.m.shape().0 as i64 {
            godot_script_error!("view is out of bounds for mat of shape: {:?}", self.m.shape());
            return None
        }
        if start.get(1) + shape.get(1) > self.m.shape().1 as i64 {
            godot_script_error!("view is out of bounds for mat of shape: {:?}", self.m.shape());
            return None
        }
        let start = (start.get(0) as usize, start.get(1) as usize);
        let shape = (shape.get(0) as usize, shape.get(1) as usize);
        Some(Gd::from_init_fn(|base|
            Self {
                m: self.m.slice(start, shape, ViewType::All),
                base,
            }
        ))
    }
}

impl MType {
    fn zeros(nrows: usize, ncols: usize, dtype: &DType) -> Self {
        match dtype {
            DType::Bool => MType::U8(DMatrix::zeros(nrows, ncols)).astype(DType::Bool).unwrap(),
            DType::U8 => MType::U8(DMatrix::zeros(nrows, ncols)),
            DType::I32 => MType::I32(DMatrix::zeros(nrows, ncols)),
            DType::I64 => MType::I64(DMatrix::zeros(nrows, ncols)),
            DType::F32 => MType::F32(DMatrix::zeros(nrows, ncols)),
            DType::F64 => MType::F64(DMatrix::zeros(nrows, ncols)),
            DType::C64 => MType::C64(DMatrix::zeros(nrows, ncols)),
            DType::C128 => MType::C128(DMatrix::zeros(nrows, ncols)),
        }
    }

    fn get(&self, row: usize, col: usize) -> Option<Variant> {
        match self {
            MType::Bool(m) => m.get((row, col)).map(|x| x.to_variant()),
            MType::U8(m) => m.get((row, col)).map(|x| x.to_variant()),
            MType::I32(m) => m.get((row, col)).map(|x| x.to_variant()),
            MType::I64(m) => m.get((row, col)).map(|x| x.to_variant()),
            MType::F32(m) => m.get((row, col)).map(|x| x.to_variant()),
            MType::F64(m) => m.get((row, col)).map(|x| x.to_variant()),
            MType::C64(m) => m.get((row, col)).map(|x| Vector2::new(x.re, x.im).to_variant()),
            MType::C128(m) => m.get((row, col)).map(|x| array![x.re, x.im].to_variant()),
            MType::View { mat, start, .. } => {
                mat.bind().m.get(start.0 + row, start.1 + col)
            }
        }
    }

    fn dtype(&self) -> DType {
        match self {
            MType::Bool(_) => DType::Bool,
            MType::U8(_) => DType::U8,
            MType::I32(_) => DType::I32,
            MType::I64(_) => DType::I64,
            MType::F32(_) => DType::F32,
            MType::F64(_) => DType::F64,
            MType::C64(_) => DType::C64,
            MType::C128(_) => DType::C128,
            MType::View { mat, .. } => mat.bind().m.dtype()
        }
    }

    fn shape(&self) -> (usize, usize) {
        match self {
            MType::Bool(m) => m.shape(),
            MType::U8(m) => m.shape(),
            MType::I32(m) => m.shape(),
            MType::I64(m) => m.shape(),
            MType::F32(m) => m.shape(),
            MType::F64(m) => m.shape(),
            MType::C64(m) => m.shape(),
            MType::C128(m) => m.shape(),
            MType::View { shape, .. } => *shape
        }
    }

    fn slice(&self, start: (usize, usize), shape: (usize, usize), view_type: ViewType) -> Self {
        match &self {
            MType::Bool(m) => MType::Bool(m.view(start, shape).clone_owned()),
            MType::U8(m) => MType::U8(m.view(start, shape).clone_owned()),
            MType::I32(m) => MType::I32(m.view(start, shape).clone_owned()),
            MType::I64(m) => MType::I64(m.view(start, shape).clone_owned()),
            MType::F32(m) => MType::F32(m.view(start, shape).clone_owned()),
            MType::F64(m) => MType::F64(m.view(start, shape).clone_owned()),
            MType::C64(m) => {
                match view_type {
                    ViewType::All => MType::C64(m.view(start, shape).clone_owned()),
                    ViewType::Real => MType::F32(m.view(start, shape).map(|x| x.re).clone_owned()),
                    ViewType::Imag => MType::F32(m.view(start, shape).map(|x| x.im).clone_owned())
                }
            },
            MType::C128(m) => {
                match view_type {
                    ViewType::All => MType::C128(m.view(start, shape).clone_owned()),
                    ViewType::Real => MType::F64(m.view(start, shape).map(|x| x.re).clone_owned()),
                    ViewType::Imag => MType::F64(m.view(start, shape).map(|x| x.im).clone_owned())
                }
            },
            MType::View { mat, start: p_start, view_type: p_view_type, .. } => {
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
                        MType::zeros(shape.0, shape.1, &mat.bind().m.dtype())
                    },
                    (ViewType::Imag, ViewType::Real) => {
                        MType::zeros(shape.0, shape.1, &mat.bind().m.dtype())
                    },
                    _ => {
                        let start = (p_start.0 + start.0, p_start.1 + start.1);
                        mat.bind().m.slice(start, shape, view_type)
                    }
                }
            }
        }
    }

    fn tranpose(&self) -> MType {
        match &self {
            MType::Bool(m) => MType::Bool(m.transpose()),
            MType::U8(m) => MType::U8(m.transpose()),
            MType::I32(m) => MType::I32(m.transpose()),
            MType::I64(m) => MType::I64(m.transpose()),
            MType::F32(m) => MType::F32(m.transpose()),
            MType::F64(m) => MType::F64(m.transpose()),
            MType::C64(m) => MType::C64(m.transpose()),
            MType::C128(m) => MType::C128(m.transpose()),
            MType::View { mat, start, shape, view_type } => {
                mat.bind().m.slice(*start, *shape, *view_type).tranpose()
            }
        }
    }

    fn adjoint(&self) -> Option<MType> {
        match &self {
            MType::F32(m) => Some(MType::F32(m.adjoint())),
            MType::F64(m) => Some(MType::F64(m.adjoint())),
            MType::C64(m) => Some(MType::C64(m.adjoint())),
            MType::C128(m) => Some(MType::C128(m.adjoint())),
            MType::View { mat, start, shape, view_type } => {
                mat.bind().m.slice(*start, *shape, *view_type).adjoint()
            }
            _ => {
                godot_script_error!("Unsupported type: {}.h()", self.as_ref());
                None
            }
        }
    }

    fn get_view_type(&self) -> ViewType {
        match self {
            MType::View { view_type, .. } => *view_type,
            _ => ViewType::All
        }
    }

    fn any(&self) -> bool {
        match self {
            MType::Bool(m) => m.iter().any(|x| *x),
            MType::U8(m) => m.iter().any(|x| *x == 1),
            MType::I32(m) => m.iter().any(|x| *x == 1),
            MType::I64(m) => m.iter().any(|x| *x == 1),
            MType::F32(m) => m.iter().any(|x| *x == 1.0),
            MType::F64(m) => m.iter().any(|x| *x == 1.0),
            MType::View { mat, start, shape, view_type } => {
                mat.bind().m.slice(*start, *shape, *view_type).any()
            },
            _ => false
        }
    }

    fn all(&self) -> bool {
        match self {
            MType::Bool(m) => m.iter().all(|x| *x),
            MType::U8(m) => m.iter().all(|x| *x == 1),
            MType::I32(m) => m.iter().all(|x| *x == 1),
            MType::I64(m) => m.iter().all(|x| *x == 1),
            MType::F32(m) => m.iter().all(|x| *x == 1.0),
            MType::F64(m) => m.iter().all(|x| *x == 1.0),
            MType::View { mat, start, shape, view_type } => {
                mat.bind().m.slice(*start, *shape, *view_type).all()
            },
            _ => false
        }
    }

    fn kron(&self, other: &MType) -> Option<Self> {
        match (&self, other) {
            (MType::U8(a), MType::U8(b)) => {
                Some(MType::U8(a.kronecker(b)))
            },
            (MType::I32(a), MType::I32(b)) => {
                Some(MType::I32(a.kronecker(b)))
            },
            (MType::I64(a), MType::I64(b)) => {
                Some(MType::I64(a.kronecker(b)))
            },
            (MType::F32(a), MType::F32(b)) => {
                Some(MType::F32(a.kronecker(b)))
            },
            (MType::F64(a), MType::F64(b)) => {
                Some(MType::F64(a.kronecker(b)))
            },
            (MType::C64(a), MType::C64(b)) => {
                Some(MType::C64(a.kronecker(b)))
            },
            (MType::C128(a), MType::C128(b)) => {
                Some(MType::C128(a.kronecker(b)))
            },
            _ => {
                godot_script_error!("Unsupported operand types: {} o {}.", 
                    self.as_ref(), other.as_ref());
                return None
            }
        }
    }

    fn mm(&self, other: &MType) -> Option<Self> {
        match (&self, other) {
            (MType::U8(a), MType::U8(b)) => {
                Some(MType::U8(a * b))
            },
            (MType::I32(a), MType::I32(b)) => {
                Some(MType::I32(a * b))
            },
            (MType::I64(a), MType::I64(b)) => {
                Some(MType::I64(a * b))
            },
            (MType::F32(a), MType::F32(b)) => {
                Some(MType::F32(a * b))
            },
            (MType::F64(a), MType::F64(b)) => {
                Some(MType::F64(a * b))
            },
            (MType::C64(a), MType::C64(b)) => {
                Some(MType::C64(a * b))
            },
            (MType::C128(a), MType::C128(b)) => {
                Some(MType::C128(a * b))
            },
            _ => {
                godot_script_error!("Unsupported operand types: {} + {}.", 
                    self.as_ref(), other.as_ref());
                return None
            }
        }
    } 

    fn mul(&self, other: &MType) -> Option<Self> {
        match (&self, other) {
            (MType::U8(a), MType::U8(b)) => {
                Some(MType::U8(a.component_mul(b)))
            },
            (MType::I32(a), MType::I32(b)) => {
                Some(MType::I32(a.component_mul(b)))
            },
            (MType::I64(a), MType::I64(b)) => {
                Some(MType::I64(a.component_mul(b)))
            },
            (MType::F32(a), MType::F32(b)) => {
                Some(MType::F32(a.component_mul(b)))
            },
            (MType::F64(a), MType::F64(b)) => {
                Some(MType::F64(a.component_mul(b)))
            },
            (MType::C64(a), MType::C64(b)) => {
                Some(MType::C64(a.component_mul(b)))
            },
            (MType::C128(a), MType::C128(b)) => {
                Some(MType::C128(a.component_mul(b)))
            },
            _ => {
                godot_script_error!("Unsupported operand types: {} + {}.", 
                    self.as_ref(), other.as_ref());
                return None
            }
        }
    }

    fn add(&self, other: &MType) -> Option<Self> {
        match (&self, other) {
            (MType::U8(a), MType::U8(b)) => {
                Some(MType::U8(a + b))
            },
            (MType::I32(a), MType::I32(b)) => {
                Some(MType::I32(a + b))
            },
            (MType::I64(a), MType::I64(b)) => {
                Some(MType::I64(a + b))
            },
            (MType::F32(a), MType::F32(b)) => {
                Some(MType::F32(a + b))
            },
            (MType::F64(a), MType::F64(b)) => {
                Some(MType::F64(a + b))
            },
            (MType::C64(a), MType::C64(b)) => {
                Some(MType::C64(a + b))
            },
            (MType::C128(a), MType::C128(b)) => {
                Some(MType::C128(a + b))
            },
            _ => {
                godot_script_error!("Unsupported operand types: {} + {}.", 
                self.as_ref(), 
                other.as_ref());
                return None
            }
        }
    }

    fn lt(&self, other: &MType) -> Option<Self> {
        match (&self, other) {
            (MType::U8(a), MType::U8(b)) => {
                Some(MType::Bool(a.zip_map(b, |a, b| a < b)))
            },
            (MType::I32(a), MType::I32(b)) => {
                Some(MType::Bool(a.zip_map(b, |a, b| a < b)))
            },
            (MType::I64(a), MType::I64(b)) => {
                Some(MType::Bool(a.zip_map(b, |a, b| a < b)))
            },
            (MType::F32(a), MType::F32(b)) => {
                Some(MType::Bool(a.zip_map(b, |a, b| a < b)))
            },
            (MType::F64(a), MType::F64(b)) => {
                Some(MType::Bool(a.zip_map(b, |a, b| a < b)))
            }
            _ => {
                godot_script_error!("Unsupported operand types: {} < {}.", 
                    self.as_ref(), other.as_ref());
                return None
            }
        }
    }

    fn le(&self, other: &MType) -> Option<Self> {
        match (&self, other) {
            (MType::U8(a), MType::U8(b)) => {
                Some(MType::Bool(a.zip_map(b, |a, b| a <= b)))
            },
            (MType::I32(a), MType::I32(b)) => {
                Some(MType::Bool(a.zip_map(b, |a, b| a <= b)))
            },
            (MType::I64(a), MType::I64(b)) => {
                Some(MType::Bool(a.zip_map(b, |a, b| a <= b)))
            },
            (MType::F32(a), MType::F32(b)) => {
                Some(MType::Bool(a.zip_map(b, |a, b| a <= b)))
            },
            (MType::F64(a), MType::F64(b)) => {
                Some(MType::Bool(a.zip_map(b, |a, b| a <= b)))
            }
            _ => {
                godot_script_error!("Unsupported operand types: {} <= {}.", 
                self.as_ref(), 
                other.as_ref());
                return None
            }
        }
    }

    fn eq(&self, other: &MType) -> Option<Self> {
        match (&self, other) {
            (MType::U8(a), MType::U8(b)) => {
                Some(MType::Bool(a.zip_map(b, |a, b| a == b)))
            },
            (MType::I32(a), MType::I32(b)) => {
                Some(MType::Bool(a.zip_map(b, |a, b| a == b)))
            },
            (MType::I64(a), MType::I64(b)) => {
                Some(MType::Bool(a.zip_map(b, |a, b| a == b)))
            },
            (MType::F32(a), MType::F32(b)) => {
                Some(MType::Bool(a.zip_map(b, |a, b| a.eq(&b))))
            },
            (MType::F64(a), MType::F64(b)) => {
                Some(MType::Bool(a.zip_map(b, |a, b| a.eq(&b))))
            }
            (MType::C64(a), MType::C64(b)) => {
                Some(MType::Bool(a.zip_map(b, |a, b| a.eq(&b))))
            }
            (MType::C128(a), MType::C128(b)) => {
                Some(MType::Bool(a.zip_map(b, |a, b| a.eq(&b))))
            }
            _ => {
                godot_script_error!("Unsupported operand types: {} == {}.", 
                self.as_ref(), 
                other.as_ref());
                return None
            }
        }
    }

    fn ge(&self, other: &MType) -> Option<Self> {
        match (&self, other) {
            (MType::U8(a), MType::U8(b)) => {
                Some(MType::Bool(a.zip_map(b, |a, b| a >= b)))
            },
            (MType::I32(a), MType::I32(b)) => {
                Some(MType::Bool(a.zip_map(b, |a, b| a >= b)))
            },
            (MType::I64(a), MType::I64(b)) => {
                Some(MType::Bool(a.zip_map(b, |a, b| a >= b)))
            },
            (MType::F32(a), MType::F32(b)) => {
                Some(MType::Bool(a.zip_map(b, |a, b| a >= b)))
            },
            (MType::F64(a), MType::F64(b)) => {
                Some(MType::Bool(a.zip_map(b, |a, b| a >= b)))
            }
            _ => {
                godot_script_error!("Unsupported operand types: {} >= {}.", 
                self.as_ref(), 
                other.as_ref());
                return None
            }
        }
    }

    fn gt(&self, other: &MType) -> Option<Self> {
        match (&self, other) {
            (MType::U8(a), MType::U8(b)) => {
                Some(MType::Bool(a.zip_map(b, |a, b| a > b)))
            },
            (MType::I32(a), MType::I32(b)) => {
                Some(MType::Bool(a.zip_map(b, |a, b| a > b)))
            },
            (MType::I64(a), MType::I64(b)) => {
                Some(MType::Bool(a.zip_map(b, |a, b| a > b)))
            },
            (MType::F32(a), MType::F32(b)) => {
                Some(MType::Bool(a.zip_map(b, |a, b| a > b)))
            },
            (MType::F64(a), MType::F64(b)) => {
                Some(MType::Bool(a.zip_map(b, |a, b| a > b)))
            }
            _ => {
                godot_script_error!("Unsupported operand types: {} > {}.", 
                self.as_ref(), 
                other.as_ref());
                return None
            }
        }
    }

    fn astype(&self, dt: DType) -> Option<MType> {
        match (self, &dt) {
            (MType::Bool(_), DType::Bool) => { Some(self.clone()) },
            (MType::Bool(m), DType::U8) => {  Some(MType::U8(m.map(|x| x as u8))) },
            (MType::Bool(m), DType::I32) => { Some(MType::I32(m.map(|x| x as i32))) },
            (MType::Bool(m), DType::I64) => { Some(MType::I64(m.map(|x| x as i64))) },
            (MType::Bool(m), DType::F32) => { Some(MType::F32(m.map(|x| x as i32 as f32))) },
            (MType::Bool(m), DType::F64) => { Some(MType::F64(m.map(|x| x as i32 as f64))) },
            (MType::Bool(m), DType::C64) => { Some(MType::C64(m.map(|x| Complex::new(x as i32 as f32, 0.0)))) },
            (MType::Bool(m), DType::C128) => { Some(MType::C128(m.map(|x| Complex::new(x as i32 as f64, 0.0))))},
            (MType::U8(m), DType::Bool) => { Some(MType::Bool(m.map(|x| x > 0))) },
            (MType::U8(_), DType::U8) => { Some(self.clone()) },
            (MType::U8(m), DType::I32) => { Some(MType::I32(m.map(|x| x as i32))) },
            (MType::U8(m), DType::I64) => { Some(MType::I64(m.map(|x| x as i64))) },
            (MType::U8(m), DType::F32) => { Some(MType::F32(m.map(|x| x as f32))) },
            (MType::U8(m), DType::F64) => { Some(MType::F64(m.map(|x| x as f64))) },
            (MType::U8(m), DType::C64) => { Some(MType::C64(m.map(|x| Complex::new(x as f32, 0.0)))) },
            (MType::U8(m), DType::C128) => { Some(MType::C128(m.map(|x| Complex::new(x as f64, 0.0))))},
            (MType::I32(m), DType::Bool) => { Some(MType::Bool(m.map(|x| x > 0))) },
            (MType::I32(m), DType::U8) => {  Some(MType::U8(m.map(|x| x as u8))) },
            (MType::I32(_), DType::I32) => { Some(self.clone()) },
            (MType::I32(m), DType::I64) => { Some(MType::I64(m.map(|x| x as i64))) },
            (MType::I32(m), DType::F32) => { Some(MType::F32(m.map(|x| x as f32))) },
            (MType::I32(m), DType::F64) => { Some(MType::F64(m.map(|x| x as f64))) },
            (MType::I32(m), DType::C64) => { Some(MType::C64(m.map(|x| Complex::new(x as f32, 0.0)))) },
            (MType::I32(m), DType::C128) => { Some(MType::C128(m.map(|x| Complex::new(x as f64, 0.0))))},
            (MType::I64(m), DType::Bool) => { Some(MType::Bool(m.map(|x| x > 0))) },
            (MType::I64(m), DType::U8) => {  Some(MType::U8(m.map(|x| x as u8))) },
            (MType::I64(m), DType::I32) => { Some(MType::I32(m.map(|x| x as i32))) },
            (MType::I64(_), DType::I64) => { Some(self.clone()) },
            (MType::I64(m), DType::F32) => { Some(MType::F32(m.map(|x| x as f32))) },
            (MType::I64(m), DType::F64) => { Some(MType::F64(m.map(|x| x as f64))) },
            (MType::I64(m), DType::C64) => { Some(MType::C64(m.map(|x| Complex::new(x as f32, 0.0)))) },
            (MType::I64(m), DType::C128) => { Some(MType::C128(m.map(|x| Complex::new(x as f64, 0.0))))},
            (MType::F32(m), DType::Bool) => { Some(MType::Bool(m.map(|x| x > 0.0))) },
            (MType::F32(m), DType::U8) => {  Some(MType::U8(m.map(|x| x as u8))) },
            (MType::F32(m), DType::I32) => { Some(MType::I32(m.map(|x| x as i32))) },
            (MType::F32(m), DType::I64) => { Some(MType::I64(m.map(|x| x as i64))) },
            (MType::F32(_), DType::F32) => { Some(self.clone()) },
            (MType::F32(m), DType::F64) => { Some(MType::F64(m.map(|x| x as f64))) },
            (MType::F32(m), DType::C64) => { Some(MType::C64(m.map(|x| Complex::new(x as f32, 0.0)))) },
            (MType::F32(m), DType::C128) => { Some(MType::C128(m.map(|x| Complex::new(x as f64, 0.0))))},
            (MType::F64(m), DType::Bool) => { Some(MType::Bool(m.map(|x| x > 0.0))) },
            (MType::F64(m), DType::U8) => {  Some(MType::U8(m.map(|x| x as u8))) },
            (MType::F64(m), DType::I32) => { Some(MType::I32(m.map(|x| x as i32))) },
            (MType::F64(m), DType::I64) => { Some(MType::I64(m.map(|x| x as i64))) },
            (MType::F64(m), DType::F32) => { Some(MType::F32(m.map(|x| x as f32))) },
            (MType::F64(_), DType::F64) => { Some(self.clone()) },
            (MType::F64(m), DType::C64) => { Some(MType::C64(m.map(|x| Complex::new(x as f32, 0.0)))) },
            (MType::F64(m), DType::C128) => { Some(MType::C128(m.map(|x| Complex::new(x as f64, 0.0))))},
            (MType::C64(_), DType::C64) => { Some(self.clone()) },
            (MType::C64(m), DType::C128) => { Some(MType::C128(m.map(|x| Complex::new(x.re as f64, x.im as f64))))},
            (MType::C128(m), DType::C64) => { Some(MType::C64(m.map(|x| Complex::new(x.re as f32, x.im as f32)))) },
            (MType::C128(_), DType::C128) => { Some(self.clone()) },
            (MType::View { mat, start, shape, view_type }, dt) => { 
                mat.bind().m.slice(*start, *shape, *view_type).astype(dt.clone())
            },
            _ => {
                godot_script_error!("Unsupported cast from to {} to {}", self.as_ref(), dt.to_godot());
                None
            }
        }
    }
}

impl std::fmt::Display for MType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            MType::Bool(m) => m.fmt(f),
            MType::U8(m) => m.fmt(f),
            MType::I32(m) => m.fmt(f),
            MType::I64(m) => m.fmt(f),
            MType::F32(m) => m.fmt(f),
            MType::F64(m) => m.fmt(f),
            MType::C64(m) => m.fmt(f),
            MType::C128(m) => m.fmt(f),
            MType::View { mat, start, shape, view_type } => {
                mat.bind().m.slice(*start, *shape, *view_type).fmt(f)
            },
        }
    }
}