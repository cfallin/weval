#![allow(dead_code)]

use std::path::PathBuf;
use structopt::StructOpt;

mod directive;
mod escape;
mod eval;
mod filter;
mod image;
mod intrinsics;
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
    },

    /// Run a Wasm module normally, collecting all weval requests at
    /// the end.
    Collect {
        /// The input Wasm module.
        #[structopt(short = "i")]
        input_module: PathBuf,

        /// The output weval request collection.
        #[structopt(short = "o")]
        output_requests: PathBuf,

        /// Which weval sites to collect requests from, as a list of
        /// integers.
        #[structopt(short = "s")]
        site: Vec<u32>,
    },

    /// Strip weval intrinsics without performing any other processing.
    Strip {
        /// The input Wasm module.
        #[structopt(short = "i")]
        input_module: PathBuf,

        /// The output Wasm module.
        #[structopt(short = "o")]
        output_module: PathBuf,
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
        } => weval(input_module, output_module, wizen, corpus, show_stats),
        Command::Collect {
            input_module,
            output_requests,
            site,
        } => collect(input_module, output_requests, site),
        Command::Strip {
            input_module,
            output_module,
        } => strip(input_module, output_module),
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
    let mut im = image::build_image(&module)?;

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

    // Partially evaluate.
    let progress = indicatif::ProgressBar::new(0);
    let mut result = eval::partially_evaluate(
        module,
        &mut im,
        &directives[..],
        &corpus[..],
        Some(progress),
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
                "   specialized: {} blocks, {} insts",
                stats.specialized_blocks, stats.specialized_insts
            );
            let mut buckets = stats
                .blocks_and_insts_by_bucket
                .into_iter()
                .collect::<Vec<_>>();
            buckets.sort_by_key(|(_bucket, (_blocks, insts))| std::cmp::Reverse(*insts));
            for (bucket, (blocks, insts)) in buckets {
                eprintln!(" * bucket {:?}: {} blocks, {} insts", bucket, blocks, insts);
            }
        }
    }

    let bytes = result.module.to_wasm_bytes()?;

    let bytes = filter::filter(&bytes[..])?;

    std::fs::write(&output_module, &bytes[..])?;

    Ok(())
}

fn collect(input_module: PathBuf, output_requests: PathBuf, site: Vec<u32>) -> anyhow::Result<()> {
    let raw_bytes = std::fs::read(&input_module)?;
    let module_bytes = wizen(raw_bytes)?;
    let mut frontend_opts = waffle::FrontendOptions::default();
    frontend_opts.debug = true;
    let module = waffle::Module::from_wasm_bytes(&module_bytes[..], &frontend_opts)?;
    let mut im = image::build_image(&module)?;
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

fn strip(input_module: PathBuf, output_module: PathBuf) -> anyhow::Result<()> {
    let raw_bytes = std::fs::read(&input_module)?;
    let bytes = filter::filter(&raw_bytes[..])?;
    std::fs::write(&output_module, &bytes[..])?;
    Ok(())
}
