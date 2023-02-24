#![allow(dead_code)]

use std::path::PathBuf;
use structopt::StructOpt;

mod directive;
mod eval;
mod image;
mod intrinsics;
mod state;
mod value;

#[derive(Clone, Debug, StructOpt)]
pub struct Options {
    /// The input Wasm module.
    #[structopt(short = "i")]
    input_module: PathBuf,

    /// The output Wasm module.
    #[structopt(short = "o")]
    output_module: PathBuf,

    /// Add tracing to IR, for debugging.
    #[structopt(long = "tracing")]
    add_tracing: bool,

    /// Run IR in interpreter prior to weval'ing.
    #[structopt(long = "run-pre")]
    run_pre: bool,

    /// Run IR in interpreter after weval'ing.
    #[structopt(long = "run-post")]
    run_post: bool,
}

fn main() -> anyhow::Result<()> {
    let _ = env_logger::try_init();
    let opts = Options::from_args();

    // Load module.
    let bytes = std::fs::read(&opts.input_module)?;
    let mut frontend_opts = waffle::FrontendOptions::default();
    frontend_opts.debug = true;
    let mut module = waffle::Module::from_wasm_bytes(&bytes[..], &frontend_opts)?;

    // If we're going to run the interpreter, we need to expand all
    // functions.
    if opts.run_pre || opts.run_post {
        module.expand_all_funcs()?;
    }

    // Build module image.
    let mut im = image::build_image(&module)?;

    // Collect directives.
    let directives = directive::collect(&module, &mut im)?;
    log::debug!("Directives: {:?}", directives);

    // Partially evaluate.
    eval::partially_evaluate(&mut module, &mut im, &directives[..], &opts)?;

    // Run in interpreter, if requested. `run_pre` also causes about
    // `partially_evaluate` to not actually perform the transform,
    // only insert tracing.
    if opts.run_pre {
        run_interp(&module);
        return Ok(());
    }

    // Update memories in module.
    image::update(&mut module, &im);

    log::debug!("Final module:\n{}", module.display());

    if opts.run_post {
        run_interp(&module);
        return Ok(());
    }

    let bytes = module.to_wasm_bytes()?;
    std::fs::write(&opts.output_module, &bytes[..])?;

    Ok(())
}

fn run_interp(module: &waffle::Module<'_>) {
    let mut ctx = waffle::InterpContext::new(module);
    if let Some(start) = module.start_func {
        ctx.call(module, start, &[]).ok().unwrap();
    }
    if let Some(waffle::Export {
        kind: waffle::ExportKind::Func(func),
        ..
    }) = module.exports.iter().find(|e| &e.name == "_start")
    {
        ctx.call(&module, *func, &[]).ok().unwrap();
    }
}
