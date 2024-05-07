//! Final filter pass to remove intrinsics imports and calls to intrinsics.
//!
//! Needs to do a few things:
//! - Remove any imports from a "weval" module.
//! - Track how removing those imports renumbers other import and
//!   function indices, and rewrite function indices in the code (`call`
//!   instructions) and in table initializers.
//!   - We have to do this by hand, since wasm-encoder doesn't support
//!     larger-than-default leb128s. We generate the opcode (`0x10`)
//!     and a leb128 equal to the original instruction's length.
//! - Rewrite calls to the removed intrinsics to drops or nothing, depending:
//!   - If a return value, then the first arg is returned. Assert that types
//!     match accordingly. Generate a drop (`0x1a`) for all remaining args.
//!   - Otherwise, if any args, generate drops for all args.

use fxhash::FxHashMap;
use wasmparser::{ElementItems, ElementKind, ExternalKind, Parser, Payload, TypeRef, ValType};

#[derive(Clone, Debug)]
enum FuncRemap {
    Index(u32),
    InlinedBytecode(Vec<wasm_encoder::Instruction<'static>>),
}

impl FuncRemap {
    fn as_index(&self) -> anyhow::Result<u32> {
        match self {
            &Self::Index(i) => Ok(i),
            _ => anyhow::bail!("Attempt to refer to index of deleted intrinsic import"),
        }
    }
}

#[derive(Default, Clone, Debug)]
struct Rewrite {
    func_remap: FxHashMap<u32, FuncRemap>,
    func_types: Vec<(Vec<ValType>, Vec<ValType>)>,
}

fn gen_replacement_bytecode(
    args: &[ValType],
    results: &[ValType],
    name: &str,
    weval_globals: u32,
) -> anyhow::Result<Vec<wasm_encoder::Instruction<'static>>> {
    match name {
        // These are polyfilled to access newly-added globals.
        "read.global.0" => Ok(vec![wasm_encoder::Instruction::GlobalGet(weval_globals)]),
        "read.global.1" => Ok(vec![wasm_encoder::Instruction::GlobalGet(
            weval_globals + 1,
        )]),
        "write.global.0" => Ok(vec![wasm_encoder::Instruction::GlobalSet(weval_globals)]),
        "write.global.1" => Ok(vec![wasm_encoder::Instruction::GlobalSet(
            weval_globals + 1,
        )]),
        // These can't be polyfilled so we rewrite them to
        // trap. They're only used in template-specialized variants
        // fed to weval requests.
        "read.specialization.global"
        | "read.reg"
        | "write.reg"
        | "push.stack"
        | "pop.stack"
        | "read.stack"
        | "write.stack"
        | "sync.stack"
        | "read.local"
        | "write.local" => Ok(vec![wasm_encoder::Instruction::Unreachable]),

        // All other intrinsics have "pass through first arg" behavior
        // if they have a return value, and otherwise have no effect.
        _ => {
            anyhow::ensure!(results.len() <= 1);
            anyhow::ensure!(results.len() <= args.len());
            if args.len() > 0 && results.len() > 0 {
                anyhow::ensure!(
                    args[0] == results[0],
                    "Intrinsic's first arg is different type than result"
                );
            }

            let mut insts = vec![];
            for _ in 0..(args.len() - results.len()) {
                insts.push(wasm_encoder::Instruction::Drop);
            }
            Ok(insts)
        }
    }
}

fn parser_to_encoder_ty(ty: wasmparser::ValType) -> wasm_encoder::ValType {
    match ty {
        wasmparser::ValType::I32 => wasm_encoder::ValType::I32,
        wasmparser::ValType::I64 => wasm_encoder::ValType::I64,
        wasmparser::ValType::F32 => wasm_encoder::ValType::F32,
        wasmparser::ValType::F64 => wasm_encoder::ValType::F64,
        wasmparser::ValType::V128 => wasm_encoder::ValType::V128,
        wasmparser::ValType::Ref(wasmparser::RefType::FUNCREF) => {
            wasm_encoder::ValType::Ref(wasm_encoder::RefType::FUNCREF)
        }
        wasmparser::ValType::Ref(wasmparser::RefType::EXTERNREF) => {
            wasm_encoder::ValType::Ref(wasm_encoder::RefType::EXTERNREF)
        }
        wasmparser::ValType::Ref(r) => wasm_encoder::ValType::Ref(wasm_encoder::RefType {
            nullable: r.is_nullable(),
            heap_type: wasm_encoder::HeapType::Concrete(
                r.type_index().unwrap().as_module_index().unwrap(),
            ),
        }),
    }
}

impl Rewrite {
    pub fn process(mut self, module: &[u8]) -> anyhow::Result<Vec<u8>> {
        let parser = Parser::new(0);
        let mut out = wasm_encoder::Module::new();
        let mut orig_func_idx = 0;
        let mut out_func_idx = 0;
        let mut num_funcs = 0;
        let mut num_funcs_emitted = 0;
        let mut out_code_section = wasm_encoder::CodeSection::new();
        let mut weval_globals = 0;

        // Scan globals section once to count globals.
        for payload in parser.clone().parse_all(module) {
            match payload? {
                Payload::GlobalSection(globals) => {
                    for _ in globals.into_iter() {
                        weval_globals += 1;
                    }
                    break;
                }
                _ => {}
            }
        }

        for payload in parser.parse_all(module) {
            let payload = payload?;
            let raw_section = payload.as_section();
            let transcribe = match payload {
                Payload::Version { .. } => false,
                Payload::End(..) => false,

                // Type section: copy all function types so we can refer to them later.
                Payload::TypeSection(types) => {
                    for fty in types.into_iter_err_on_gc_types() {
                        let fty = fty?;
                        let (args, results) = (fty.params().to_vec(), fty.results().to_vec());
                        self.func_types.push((args, results));
                    }
                    true
                }

                // Import section: transcribe manually, removing
                // intrinsic imports and noting remappings for each
                // imported function.
                Payload::ImportSection(imports) => {
                    let mut out_imports = wasm_encoder::ImportSection::new();

                    for import in imports.into_iter() {
                        let import = import?;
                        match import.ty {
                            TypeRef::Func(fty) => {
                                let orig_idx = orig_func_idx;
                                orig_func_idx += 1;

                                if import.module == "weval" {
                                    // Omit the import, and add a rewriting to the func_remap info.
                                    let (args, results) = &self.func_types[fty as usize];
                                    let bytecode = gen_replacement_bytecode(
                                        args,
                                        results,
                                        import.name,
                                        weval_globals,
                                    )?;
                                    self.func_remap
                                        .insert(orig_idx, FuncRemap::InlinedBytecode(bytecode));
                                } else {
                                    // Transcribe the import.
                                    out_imports.import(
                                        import.module,
                                        import.name,
                                        wasm_encoder::EntityType::Function(fty),
                                    );
                                    self.func_remap
                                        .insert(orig_idx, FuncRemap::Index(out_func_idx));
                                    out_func_idx += 1;
                                }
                            }
                            ty => anyhow::bail!("import type {:?} not supported", ty),
                        }
                    }

                    out.section(&out_imports);
                    false
                }

                // Globals section: add two mut i64 globals for {read,write}.global.{0,1}.
                Payload::GlobalSection(globals) => {
                    let mut out_globals = wasm_encoder::GlobalSection::new();
                    for global in globals.into_iter() {
                        let global = global?;
                        let val_type = match global.ty.content_type {
                            wasmparser::ValType::I32 => wasm_encoder::ValType::I32,
                            wasmparser::ValType::I64 => wasm_encoder::ValType::I64,
                            wasmparser::ValType::F32 => wasm_encoder::ValType::F32,
                            wasmparser::ValType::F64 => wasm_encoder::ValType::F64,
                            wasmparser::ValType::V128 => wasm_encoder::ValType::V128,
                            ty => panic!("Unsupported global type: {:?}", ty),
                        };
                        let ty = wasm_encoder::GlobalType {
                            val_type,
                            mutable: global.ty.mutable,
                        };
                        let reader = global.init_expr.get_operators_reader();
                        let mut init_expr = wasm_encoder::ConstExpr::empty();
                        for op in reader {
                            let op = op?;
                            match op {
                                wasmparser::Operator::I32Const { value } => {
                                    init_expr = init_expr.with_i32_const(value);
                                }
                                wasmparser::Operator::I64Const { value } => {
                                    init_expr = init_expr.with_i64_const(value);
                                }
                                wasmparser::Operator::F32Const { value } => {
                                    init_expr =
                                        init_expr.with_f32_const(f32::from_bits(value.bits()));
                                }
                                wasmparser::Operator::F64Const { value } => {
                                    init_expr =
                                        init_expr.with_f64_const(f64::from_bits(value.bits()));
                                }
                                wasmparser::Operator::End => {}
                                op => {
                                    panic!("Unsupported operator in global initializer: {:?}", op)
                                }
                            }
                        }
                        out_globals.global(ty, &init_expr);
                    }

                    for _ in 0..2 {
                        out_globals.global(
                            wasm_encoder::GlobalType {
                                val_type: wasm_encoder::ValType::I64,
                                mutable: true,
                            },
                            &wasm_encoder::ConstExpr::empty().with_i64_const(0),
                        );
                    }

                    out.section(&out_globals);
                    false
                }

                Payload::FunctionSection(funcs) => {
                    for fty in funcs {
                        let _fty = fty?;
                        let orig_idx = orig_func_idx;
                        orig_func_idx += 1;
                        let out_idx = out_func_idx;
                        out_func_idx += 1;
                        self.func_remap.insert(orig_idx, FuncRemap::Index(out_idx));
                    }
                    true
                }

                Payload::ExportSection(exports) => {
                    let mut out_exports = wasm_encoder::ExportSection::new();
                    for export in exports {
                        let export = export?;
                        let (index, kind) = match &export.kind {
                            ExternalKind::Func => (
                                self.func_remap.get(&export.index).unwrap().as_index()?,
                                wasm_encoder::ExportKind::Func,
                            ),
                            ExternalKind::Memory => {
                                (export.index, wasm_encoder::ExportKind::Memory)
                            }
                            ExternalKind::Table => (export.index, wasm_encoder::ExportKind::Table),
                            ExternalKind::Global => {
                                (export.index, wasm_encoder::ExportKind::Global)
                            }
                            ExternalKind::Tag => (export.index, wasm_encoder::ExportKind::Tag),
                        };
                        out_exports.export(export.name, kind, index);
                    }

                    out.section(&out_exports);
                    false
                }

                Payload::ElementSection(elements) => {
                    let mut out_elements = wasm_encoder::ElementSection::new();
                    for element in elements {
                        let element = element?;

                        let mut out_items = vec![];
                        let mut out_exprs = vec![];
                        let out_items = match element.items {
                            ElementItems::Functions(funcs) => {
                                for f in funcs {
                                    let f = f?;
                                    let new = self.func_remap.get(&f).unwrap().as_index()?;
                                    out_items.push(new);
                                }
                                wasm_encoder::Elements::Functions(&out_items[..])
                            }
                            ElementItems::Expressions(ty, exprs) => {
                                let sig = ty.type_index().unwrap().as_module_index().unwrap();
                                for expr in exprs {
                                    let expr = expr?;
                                    let func = expr
                                        .get_operators_reader()
                                        .into_iter()
                                        .map(|op| {
                                            let op = op.unwrap();
                                            match op {
                                                wasmparser::Operator::RefFunc {
                                                    function_index,
                                                } => self
                                                    .func_remap
                                                    .get(&function_index)
                                                    .unwrap()
                                                    .as_index()
                                                    .unwrap(),
                                                _ => panic!("Unsupported op"),
                                            }
                                        })
                                        .next()
                                        .unwrap();
                                    out_exprs.push(wasm_encoder::ConstExpr::ref_func(func));
                                }
                                wasm_encoder::Elements::Expressions(
                                    wasm_encoder::RefType {
                                        nullable: true,
                                        heap_type: wasm_encoder::HeapType::Concrete(sig),
                                    },
                                    &out_exprs[..],
                                )
                            }
                        };

                        match element.kind {
                            ElementKind::Active {
                                table_index,
                                offset_expr,
                            } => {
                                let mut const_expr = None;
                                for op in offset_expr.get_operators_reader() {
                                    let op = op?;
                                    match op {
                                        wasmparser::Operator::I32Const { value } => {
                                            if const_expr.is_none() {
                                                const_expr =
                                                    Some(wasm_encoder::ConstExpr::i32_const(value));
                                            } else {
                                                anyhow::bail!("More than one i32const in active table elem expr");
                                            }
                                        }
                                        wasmparser::Operator::End => {}
                                        _ => anyhow::bail!(
                                            "unexpected operator in active table elem expr"
                                        ),
                                    }
                                }
                                out_elements.active(table_index, &const_expr.unwrap(), out_items);
                            }
                            _ => panic!("Unsupported element kind for element section"),
                        }
                    }

                    out.section(&out_elements);
                    false
                }

                Payload::CodeSectionStart { count, .. } => {
                    num_funcs = count;
                    false
                }

                Payload::CodeSectionEntry(code) => {
                    // Rewrite calls, ref.funcs, and return_calls
                    // according to `func_remap`. (The latter two
                    // become errors; intrinsics can only be used for
                    // ordinary calls.)

                    let mut locals = vec![];
                    for local in code.get_locals_reader()? {
                        let (count, ty) = local?;
                        let ty = parser_to_encoder_ty(ty);
                        locals.push((count, ty));
                    }

                    let mut func = wasm_encoder::Function::new(locals);
                    let mut last_offset = code.range().start;
                    let mut skip = true;
                    for entry in code.get_operators_reader()?.into_iter_with_offsets() {
                        let (op, offset) = entry?;
                        if !skip {
                            func.raw(module[last_offset..offset].iter().cloned());
                        }
                        last_offset = offset;

                        skip = match op {
                            wasmparser::Operator::Call { function_index } => {
                                match self.func_remap.get(&function_index).unwrap() {
                                    FuncRemap::Index(i) => {
                                        func.instruction(&wasm_encoder::Instruction::Call(*i));
                                    }
                                    FuncRemap::InlinedBytecode(ops) => {
                                        for op in ops {
                                            func.instruction(op);
                                        }
                                    }
                                }
                                true
                            }
                            wasmparser::Operator::ReturnCall { function_index } => {
                                match self.func_remap.get(&function_index).unwrap() {
                                    FuncRemap::Index(i) => {
                                        func.instruction(&wasm_encoder::Instruction::ReturnCall(
                                            *i,
                                        ));
                                    }
                                    FuncRemap::InlinedBytecode(ops) => {
                                        for op in ops {
                                            func.instruction(op);
                                        }
                                        func.instruction(&wasm_encoder::Instruction::Return);
                                    }
                                }
                                true
                            }
                            wasmparser::Operator::RefFunc { function_index }
                                if self
                                    .func_remap
                                    .get(&function_index)
                                    .unwrap()
                                    .as_index()
                                    .is_err() =>
                            {
                                anyhow::bail!("ref.func taken of intrinsic");
                            }
                            _ => false,
                        };
                    }
                    if !skip {
                        func.raw(module[last_offset..code.range().end].iter().cloned());
                    }

                    out_code_section.function(&func);
                    num_funcs_emitted += 1;

                    if num_funcs_emitted == num_funcs {
                        out.section(&out_code_section);
                    }

                    false
                }

                Payload::CustomSection(reader) if reader.name() == "name" => {
                    let name_reader =
                        wasmparser::NameSectionReader::new(reader.data(), reader.data_offset());
                    let mut names = wasm_encoder::NameSection::new();
                    let mut func_names = wasm_encoder::NameMap::new();
                    for subsection in name_reader {
                        let subsection = subsection?;
                        match subsection {
                            wasmparser::Name::Function(names) => {
                                for name in names {
                                    let name = name?;
                                    if let Some(&FuncRemap::Index(new_index)) =
                                        self.func_remap.get(&name.index)
                                    {
                                        func_names.append(new_index, name.name);
                                    }
                                }
                            }
                            _ => {}
                        }
                    }
                    names.functions(&func_names);
                    out.section(&names);
                    false
                }
                Payload::CustomSection(..) => false,
                _ => true,
            };

            if transcribe {
                let (id, range) = raw_section.unwrap();
                out.section(&wasm_encoder::RawSection {
                    id,
                    data: &module[range],
                });
            }
        }

        Ok(out.finish())
    }
}

pub fn filter(module: &[u8]) -> anyhow::Result<Vec<u8>> {
    let rewrite = Rewrite::default();
    rewrite.process(module)
}
