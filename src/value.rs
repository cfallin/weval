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
            waffle::Type::FuncRef | waffle::Type::TypedFuncRef(..) => None,
        }
    }

    pub fn as_bits(self) -> u128 {
        match self {
            WasmVal::I32(i) => i.into(),
            WasmVal::I64(i) => i.into(),
            WasmVal::F32(f) => f.into(),
            WasmVal::F64(f) => f.into(),
            WasmVal::V128(v) => v,
        }
    }

    pub fn mask_bits(self, mask: u128) -> Self {
        match self {
            WasmVal::I32(i) => WasmVal::I32(i & (mask as u32)),
            WasmVal::I64(i) => WasmVal::I64(i & (mask as u64)),
            WasmVal::F32(f) => WasmVal::F32(f & (mask as u32)),
            WasmVal::F64(f) => WasmVal::F64(f & (mask as u64)),
            WasmVal::V128(v) => WasmVal::V128(v & mask),
        }
    }

    pub fn maximal_mask(self) -> u128 {
        match self {
            WasmVal::I32(i) => u32::MAX.into(),
            WasmVal::I64(i) => u64::MAX.into(),
            WasmVal::F32(f) => u32::MAX.into(),
            WasmVal::F64(f) => u64::MAX.into(),
            WasmVal::V128(v) => u128::MAX,
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
    /// A value known at specialization time, at least partially. The
    /// `u64` mask indicates which bits are known (the rest are
    /// `Runtime`).
    Concrete(WasmVal, u128),
    /// A value that points to memory known at specialization time,
    /// with the given offset.
    ConcreteMemory(MemoryBufferIndex, u32),
    /// Static memory pointer.
    StaticMemory(u32),
    /// A value only computed at runtime. The instruction that
    /// computed it is specified, if known.
    Runtime(Option<waffle::Value>),
}

/// Memory pointed to by one of the incoming arguments to a
/// specialized function.
#[derive(Clone, Debug, Default, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct MemoryBufferIndex(pub u32);

impl AbstractValue {
    pub fn meet(a: &AbstractValue, b: &AbstractValue) -> AbstractValue {
        match (a, b) {
            (AbstractValue::Top, x) | (x, AbstractValue::Top) => x.clone(),
            (x, y) if x == y => x.clone(),
            (AbstractValue::Concrete(a, mask1), AbstractValue::Concrete(b, mask2))
                if (a.as_bits() & *mask1) == (b.as_bits() & *mask2) =>
            {
                if *mask1 & *mask2 == 0 {
                    AbstractValue::Runtime(None)
                } else {
                    let mask = *mask1 & *mask2;
                    AbstractValue::Concrete(a.mask_bits(mask), mask)
                }
            }
            (AbstractValue::Runtime(cause1), AbstractValue::Runtime(cause2)) => {
                log::debug!(
                    "runtime({:?} meet runtime({:?}) -> runtime({:?})",
                    cause1,
                    cause2,
                    cause1.or(*cause2)
                );
                AbstractValue::Runtime(cause1.or(*cause2))
            }
            (AbstractValue::Runtime(cause1), _x) | (_x, AbstractValue::Runtime(cause1)) => {
                AbstractValue::Runtime(*cause1)
            }
            (_av1, _av2) => AbstractValue::Runtime(None),
        }
    }

    pub fn as_const_u32(&self) -> Option<u32> {
        match self {
            &AbstractValue::Concrete(WasmVal::I32(k), mask) if mask == u32::MAX.into() => Some(k),
            &AbstractValue::StaticMemory(addr) => Some(addr),
            _ => None,
        }
    }

    pub fn as_const_u32_or_mem_offset(&self) -> Option<u32> {
        match self {
            &AbstractValue::Concrete(WasmVal::I32(k), mask) if mask == u32::MAX.into() => Some(k),
            &AbstractValue::ConcreteMemory(_, off) => Some(off),
            _ => None,
        }
    }

    pub fn as_const_u64(&self) -> Option<u64> {
        match self {
            &AbstractValue::Concrete(WasmVal::I64(k), mask) if mask == u64::MAX.into() => Some(k),
            &AbstractValue::StaticMemory(addr) => Some(u64::from(addr)),
            _ => None,
        }
    }

    pub fn as_const_truthy(&self) -> Option<bool> {
        self.as_const_u32().map(|k| k != 0)
    }
}

pub fn mask_for_ty(ty: waffle::Type) -> u128 {
    match ty {
        waffle::Type::I32 => u32::MAX.into(),
        waffle::Type::I64 => u64::MAX.into(),
        waffle::Type::F32 => u32::MAX.into(),
        waffle::Type::F64 => u64::MAX.into(),
        waffle::Type::V128 => u128::MAX,
        waffle::Type::FuncRef => u128::MAX,
    }
}
