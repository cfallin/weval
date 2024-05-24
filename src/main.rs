#![allow(dead_code)]

use std::collections::BTreeSet;
use std::path::PathBuf;
use structopt::StructOpt;

mod constant_offsets;
mod dce;
mod directive;
mod escape;
mod eval;
mod filter;
mod image;
mod intrinsics;
mod liveness;
mod state;
mod stats;
mod value;

const STUBS: &'static str = include_str!("../lib/weval-stubs.wat");

#[derive(Clone, Debug, StructOpt)]
pub enum Command {
    /// Partially evaluate a Wasm module, optionally wizening first.
    Weval {
        /// The input Wasm module.
        #[structopt(short = "i")]
        input_module: PathBuf,

        /// The output Wasm module.
        #[structopt(short = "o")]
        output_module: PathBuf,

        /// Whether to Wizen the module first.
        #[structopt(short = "w")]
        wizen: bool,

        /// A collection of pre-collected weval requests, if any, to
        /// add to the weval'ing and resulting lookup table.
        #[structopt(short = "c")]
        corpus: Option<PathBuf>,

        /// Show stats on specialization code size.
        #[structopt(long = "show-stats")]
        show_stats: bool,

        /// Output IR for generic and specialized functions to files in a directory.
        #[structopt(long = "output-ir")]
        output_ir: Option<PathBuf>,
    },

    /// Pre-compile a Wasm module for weval request collection, using
    /// the appropriate version and configuration of the internal
    /// Wasmtime engine.
    Precompile {
        /// The input Wasm module.
        #[structopt(short = "i")]
        input_module: PathBuf,

        /// The output precompiled Wasm module path.
        #[structopt(short = "o")]
        output_precompiled: PathBuf,
    },

    /// Run a Wasm module normally, collecting all weval requests at
    /// the end.
    Collect {
        /// The input Wasm module.
        #[structopt(short = "i")]
        input_module: PathBuf,

        /// The input Wasm module precompiled, if available, to make instantiation faster.
        #[structopt(short = "p")]
        input_precompiled: Option<PathBuf>,

        /// The output weval request collection.
        #[structopt(short = "o")]
        output_requests: PathBuf,

        /// Which weval sites to collect requests from, as a list of
        /// integers.
        #[structopt(short = "s")]
        site: Vec<u32>,

        /// The rest of the arguments for the program we're running.
        #[structopt(last = true)]
        args: Vec<String>,
    },

    /// Merge collected weval requests into one corpus.
    Merge {
        /// The output file to produce.
        #[structopt(short = "o")]
        output: PathBuf,

        /// The input corpus files from `weval collect`.
        #[structopt(last = true)]
        inputs: Vec<PathBuf>,
    },
}

fn main() -> anyhow::Result<()> {
    let _ = env_logger::try_init();
    let cmd = Command::from_args();

    match cmd {
        Command::Weval {
            input_module,
            output_module,
            wizen,
            corpus,
            show_stats,
            output_ir,
        } => weval(
            input_module,
            output_module,
            wizen,
            corpus,
            show_stats,
            output_ir,
        ),
        Command::Precompile {
            input_module,
            output_precompiled,
        } => precompile(input_module, output_precompiled),
        Command::Collect {
            input_module,
            input_precompiled,
            output_requests,
            site,
            args,
        } => collect(input_module, input_precompiled, output_requests, site, args),
        Command::Merge { output, inputs } => merge(output, inputs),
    }
}

fn wizen(raw_bytes: Vec<u8>) -> anyhow::Result<Vec<u8>> {
    let mut w = wizer::Wizer::new();
    w.allow_wasi(true)?;
    w.inherit_env(true);
    w.dir(".");
    w.wasm_bulk_memory(true);
    w.preload_bytes("weval", STUBS.as_bytes().to_vec())?;
    w.func_rename("_start", "wizer.resume");
    w.run(&raw_bytes[..])
}

fn weval(
    input_module: PathBuf,
    output_module: PathBuf,
    do_wizen: bool,
    corpus: Option<PathBuf>,
    show_stats: bool,
    output_ir: Option<PathBuf>,
) -> anyhow::Result<()> {
    let raw_bytes = std::fs::read(&input_module)?;

    // Optionally, Wizen the module first.
    let module_bytes = if do_wizen {
        wizen(raw_bytes)?
    } else {
        raw_bytes
    };

    // Load module.
    let mut frontend_opts = waffle::FrontendOptions::default();
    frontend_opts.debug = true;
    let module = waffle::Module::from_wasm_bytes(&module_bytes[..], &frontend_opts)?;

    // Build module image.
    let mut im = image::build_image(&module, None)?;

    // Collect directives.
    let directives = directive::collect(&module, &mut im)?;
    log::debug!("Directives: {:?}", directives);

    // Get any corpus of pre-collected directives as well.
    let corpus = match corpus {
        Some(path) => {
            let bytes = std::fs::read(&path)?;
            let directives: Vec<directive::Directive> = bincode::deserialize(&bytes[..])?;
            directives
        }
        None => {
            vec![]
        }
    };

    // Make sure IR output directory exists.
    if let Some(dir) = &output_ir {
        std::fs::create_dir_all(dir)?;
    }

    // Partially evaluate.
    let progress = indicatif::ProgressBar::new(0);
    let mut result = eval::partially_evaluate(
        module,
        &mut im,
        &directives[..],
        &corpus[..],
        Some(progress),
        output_ir,
    )?;

    // Update memories in module.
    image::update(&mut result.module, &im);

    log::debug!("Final module:\n{}", result.module.display());

    if show_stats {
        for stats in result.stats {
            eprintln!(
                "Function {}: {} blocks, {} insts)",
                stats.generic, stats.generic_blocks, stats.generic_insts,
            );
            eprintln!(
                "   specialized ({} times): {} blocks, {} insts",
                stats.specializations, stats.specialized_blocks, stats.specialized_insts
            );
            eprintln!(
                "   virtstack: {} reads ({} mem), {} writes ({} mem)",
                stats.virtstack_reads,
                stats.virtstack_reads_mem,
                stats.virtstack_writes,
                stats.virtstack_writes_mem
            );
            eprintln!(
                "   locals: {} reads ({} mem), {} writes ({} mem)",
                stats.local_reads,
                stats.local_reads_mem,
                stats.local_writes,
                stats.local_writes_mem
            );
            eprintln!(
                "   live values at block starts: {} ({} per block)",
                stats.live_value_at_block_start,
                (stats.live_value_at_block_start as f64) / (stats.specialized_blocks as f64),
            );
        }
    }

    let bytes = result.module.to_wasm_bytes()?;

    let bytes = filter::filter(&bytes[..])?;

    std::fs::write(&output_module, &bytes[..])?;

    Ok(())
}

fn precompile(input_module: PathBuf, output_precompiled: PathBuf) -> anyhow::Result<()> {
    let engine = wasmtime::Engine::new(&wasmtime::Config::default())?;
    let module = wasmtime::Module::from_file(&engine, &input_module)?;
    let bytes = module.serialize()?;
    std::fs::write(&output_precompiled, &bytes[..])?;
    Ok(())
}

fn collect(
    input_module: PathBuf,
    input_precompiled: Option<PathBuf>,
    output_requests: PathBuf,
    site: Vec<u32>,
    args: Vec<String>,
) -> anyhow::Result<()> {
    let raw_bytes = std::fs::read(&input_module)?;
    let engine = wasmtime::Engine::new(&wasmtime::Config::default())?;
    let module = if let Some(p) = &input_precompiled {
        unsafe { wasmtime::Module::deserialize_file(&engine, p)? }
    } else {
        wasmtime::Module::from_file(&engine, &input_module)?
    };

    let mut wasi = wasmtime_wasi::WasiCtxBuilder::new();
    wasi.inherit_stdin()
        .inherit_stdout()
        .inherit_stderr()
        .inherit_env()
        .preopened_dir(
            ".",
            ".",
            wasmtime_wasi::DirPerms::READ,
            wasmtime_wasi::FilePerms::READ,
        )
        .unwrap();
    for arg in &args {
        wasi.arg(arg);
    }
    let wasi = wasi.build_p1();

    let mut linker = wasmtime::Linker::new(&engine);
    let mut store = wasmtime::Store::new(&engine, wasi);
    wasmtime_wasi::preview1::add_to_linker_sync(&mut linker, |s| s)?;
    let stubs_module = wasmtime::Module::new(&engine, STUBS.as_bytes())?;
    let stubs = wasmtime::Instance::new(&mut store, &stubs_module, &[])?;
    linker.instance(&mut store, "weval", stubs)?;
    let instance = linker.instantiate(&mut store, &module)?;
    let func = instance.get_typed_func::<(), ()>(&mut store, "_start")?;
    func.call(&mut store, ())?;
    let memory = instance
        .exports(&mut store)
        .filter_map(|e| e.into_memory())
        .next()
        .expect("no exported memory");
    let bytes = memory.data(&store)[..].to_vec();

    let mut frontend_opts = waffle::FrontendOptions::default();
    frontend_opts.debug = true;
    let module = waffle::Module::from_wasm_bytes(&raw_bytes[..], &frontend_opts)?;
    let mut im = image::build_image(&module, Some(&bytes[..]))?;
    let mut directives = directive::collect(&module, &mut im)?;
    log::debug!("Directives: {:?}", directives);

    // Keep only directives that correspond to one of the requested
    // collection sites.
    if site.len() > 0 {
        directives.retain(|d| site.contains(&d.user_id));
    }
    // Zero out pointer-to-specialize and original function pointer --
    // these will be filled in when the collection is later used.
    for d in &mut directives {
        d.func = waffle::Func::default();
        d.func_index_out_addr = 0;
    }

    // Bincode the result and dump it to a file.
    let dump = bincode::serialize(&directives)?;
    std::fs::write(&output_requests, dump)?;

    Ok(())
}

fn merge(output: PathBuf, inputs: Vec<PathBuf>) -> anyhow::Result<()> {
    // Deserialize all of the input files.
    let mut input_directives = BTreeSet::new();
    for input in &inputs {
        let directives: Vec<directive::Directive> = bincode::deserialize(&std::fs::read(input)?)?;
        input_directives.extend(directives.into_iter());
    }
    let output_directives = input_directives.into_iter().collect::<Vec<_>>();
    let dump = bincode::serialize(&output_directives)?;
    std::fs::write(&output, dump)?;

    Ok(())
}
