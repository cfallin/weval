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
    ///
    /// May have special "tags" attached, to mark that e.g. derived
    /// values can be used as pointers to read
    /// const-at-specialization-time memory.
    Concrete(WasmVal, ValueTags),
    /// A value only computed at runtime.
    Runtime(ValueTags),
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Default)]
pub struct ValueTags(u32);

/// Constructors for value tags.
impl ValueTags {
    /// All values reached in memory through this value as a pointer
    /// are constant at specialization time.
    pub const fn const_memory() -> Self {
        ValueTags(1)
    }
}

/// Operators on value tags.
impl std::ops::BitOr<ValueTags> for ValueTags {
    type Output = ValueTags;
    fn bitor(self, rhs: ValueTags) -> ValueTags {
        ValueTags(self.0 | rhs.0)
    }
}
impl ValueTags {
    pub fn contains(&self, tags: ValueTags) -> bool {
        (self.0 & tags.0) == tags.0
    }
    pub fn meet(&self, other: ValueTags) -> ValueTags {
        ValueTags(self.0 & other.0)
    }
}

impl Value {
    pub fn tags(&self) -> ValueTags {
        match self {
            &Value::Top => ValueTags::default(),
            &Value::Concrete(_, t) => t,
            &Value::Runtime(t) => t,
        }
    }

    pub fn with_tags(&self, new_tags: ValueTags) -> Value {
        match self {
            &Value::Top => Value::Top,
            &Value::Concrete(k, t) => Value::Concrete(k, t | new_tags),
            &Value::Runtime(t) => Value::Runtime(t | new_tags),
        }
    }

    pub fn meet(a: Value, b: Value) -> Value {
        match (a, b) {
            (Value::Top, x) | (x, Value::Top) => x,
            (Value::Concrete(a, t1), Value::Concrete(b, t2)) if a == b => {
                Value::Concrete(a, t1.meet(t2))
            }
            (a, b) => Value::Runtime(a.tags().meet(b.tags())),
        }
    }

    pub fn is_const_u32(&self) -> Option<u32> {
        match self {
            &Value::Concrete(WasmVal::I32(k), _) => Some(k),
            _ => None,
        }
    }

    pub fn is_const_truthy(&self) -> Option<bool> {
        self.is_const_u32().map(|k| k != 0)
    }
}
