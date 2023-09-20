//! Symbolic and concrete values.

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum WasmVal {
    I32(u32),
    I64(u64),
    F32(u32),
    F64(u64),
    V128(u128),
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

    pub fn from_bits(ty: waffle::Type, bits: u64) -> Option<Self> {
        match ty {
            waffle::Type::I32 => Some(WasmVal::I32(bits as u32)),
            waffle::Type::I64 => Some(WasmVal::I64(bits)),
            waffle::Type::F32 => Some(WasmVal::F32(bits as u32)),
            waffle::Type::F64 => Some(WasmVal::F64(bits)),
            waffle::Type::V128 => Some(WasmVal::V128(bits as u128)),
            waffle::Type::FuncRef => None,
        }
    }
}

impl std::convert::TryFrom<waffle::Operator> for WasmVal {
    type Error = ();
    fn try_from(op: waffle::Operator) -> Result<Self, Self::Error> {
        match op {
            waffle::Operator::I32Const { value } => Ok(WasmVal::I32(value as u32)),
            waffle::Operator::I64Const { value } => Ok(WasmVal::I64(value as u64)),
            waffle::Operator::F32Const { value } => Ok(WasmVal::F32(value)),
            waffle::Operator::F64Const { value } => Ok(WasmVal::F64(value)),
            _ => Err(()),
        }
    }
}

#[derive(Clone, Debug, Default, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum AbstractValue {
    /// "top" default value; undefined.
    #[default]
    Top,
    /// A value known at specialization time.
    ///
    /// May have special "tags" attached, to mark that e.g. derived
    /// values can be used as pointers to read
    /// const-at-specialization-time memory.
    Concrete(WasmVal, ValueTags),
    /// A value only computed at runtime. The instruction that
    /// computed it is specified, if known.
    Runtime(Option<waffle::Value>, ValueTags),
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Default)]
pub struct ValueTags(u32);

/// Constructors for value tags.
///
/// N.B.: the constant values returned below are *bitfields*, i.e.,
/// powers of two.
impl ValueTags {
    /// All values reached in memory through this value as a pointer
    /// are constant at specialization time.
    pub const fn const_memory() -> Self {
        ValueTags(1)
    }

    /// `const_memory` is passed transitively through loads.
    pub const fn const_memory_transitive() -> Self {
        ValueTags(2)
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
        // - const_memory and const_memory_transitive merge as intersection.
        ValueTags((self.0 & 3) & (other.0 & 3))
    }
    /// Get the tags that are "sticky": propagate across all ops.
    pub fn sticky(&self) -> ValueTags {
        ValueTags(0)
    }
}

impl AbstractValue {
    pub fn tags(&self) -> ValueTags {
        match self {
            &AbstractValue::Top => ValueTags::default(),
            &AbstractValue::Concrete(_, t) => t,
            &AbstractValue::Runtime(_, t) => t,
        }
    }

    pub fn with_tags(&self, new_tags: ValueTags) -> AbstractValue {
        match self {
            &AbstractValue::Top => AbstractValue::Top,
            &AbstractValue::Concrete(k, t) => AbstractValue::Concrete(k, t | new_tags),
            &AbstractValue::Runtime(v, t) => AbstractValue::Runtime(v, t | new_tags),
        }
    }

    pub fn meet(a: &AbstractValue, b: &AbstractValue) -> AbstractValue {
        match (a, b) {
            (AbstractValue::Top, x) | (x, AbstractValue::Top) => x.clone(),
            (x, y) if x == y => x.clone(),
            (AbstractValue::Concrete(a, t1), AbstractValue::Concrete(b, t2)) if a == b => {
                AbstractValue::Concrete(*a, t1.meet(*t2))
            }
            (AbstractValue::Runtime(cause1, t1), AbstractValue::Runtime(cause2, t2)) => {
                log::debug!(
                    "runtime({:?} meet runtime({:?}) -> runtime({:?})",
                    cause1,
                    cause2,
                    cause1.or(*cause2)
                );
                AbstractValue::Runtime(cause1.or(*cause2), t1.meet(*t2))
            }
            (AbstractValue::Runtime(cause1, t1), x) | (x, AbstractValue::Runtime(cause1, t1)) => {
                AbstractValue::Runtime(*cause1, t1.meet(x.tags()))
            }
            (av1, av2) => AbstractValue::Runtime(None, av1.tags().meet(av2.tags())),
        }
    }

    pub fn prop_sticky_tags(self, other: &AbstractValue) -> AbstractValue {
        self.with_tags(other.tags().sticky())
    }

    pub fn is_const_u32(&self) -> Option<u32> {
        match self {
            &AbstractValue::Concrete(WasmVal::I32(k), _) => Some(k),
            _ => None,
        }
    }

    pub fn is_const_u64(&self) -> Option<u64> {
        match self {
            &AbstractValue::Concrete(WasmVal::I64(k), _) => Some(k),
            _ => None,
        }
    }

    pub fn is_const_truthy(&self) -> Option<bool> {
        self.is_const_u32().map(|k| k != 0)
    }
}
