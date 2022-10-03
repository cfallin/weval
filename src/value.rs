//! Symbolic and concrete values.

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum WasmVal {
    I32(u32),
    I64(u64),
    F32(u32),
    F64(u64),
    V128(u128),
}

impl std::convert::From<walrus::ir::Value> for WasmVal {
    fn from(val: walrus::ir::Value) -> Self {
        match val {
            walrus::ir::Value::I32(i) => WasmVal::I32(i as u32),
            walrus::ir::Value::I64(i) => WasmVal::I64(i as u64),
            walrus::ir::Value::F32(f) => WasmVal::F32(f.to_bits()),
            walrus::ir::Value::F64(f) => WasmVal::F64(f.to_bits()),
            walrus::ir::Value::V128(v) => WasmVal::V128(v),
        }
    }
}

impl WasmVal {
    pub fn is_truthy(self) -> bool {
        match self {
            WasmVal::I32(i) => i != 0,
            // Only boolean-ish types (i32) can be evaluated for
            // truthiness.
            _ => panic!("Type error: non-i32 used in boolean-ish context"),
        }
    }

    pub fn integer_value(self) -> Option<u64> {
        match self {
            WasmVal::I32(i) => Some(i as u64),
            WasmVal::I64(i) => Some(i),
            _ => None,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Value {
    /// "top" default value; undefined.
    Top,
    /// A value known at specialization time.
    Concrete(WasmVal),
    /// A value only computed at runtime.
    Runtime,
}

impl Value {
    pub fn meet(a: Value, b: Value) -> Value {
        match (a, b) {
            (Value::Top, x) | (x, Value::Top) => x,
            (Value::Concrete(a), Value::Concrete(b)) if a == b => Value::Concrete(a),
            _ => Value::Runtime,
        }
    }
}
