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
struct Options {
    /// The input Wasm module.
    #[structopt(short = "i")]
    input_module: PathBuf,

    /// The output Wasm module.
    #[structopt(short = "o")]
    output_module: PathBuf,
}

fn main() -> anyhow::Result<()> {
    let _ = env_logger::try_init();
    let opts = Options::from_args();

    // Load module.
    let bytes = std::fs::read(&opts.input_module)?;
    let mut module = waffle::Module::from_wasm_bytes(&bytes[..])?;

    // Build module image.
    let mut im = image::build_image(&module)?;

    // Collect directives.
    let directives = directive::collect(&module, &mut im)?;
    log::debug!("Directives: {:?}", directives);

    // Partially evaluate.
    eval::partially_evaluate(&mut module, &mut im, &directives[..])?;

    // Update memories in module.
    image::update(&mut module, &im);

    log::debug!("Final module:\n{}", module.display());

    let bytes = module.to_wasm_bytes()?;
    std::fs::write(&opts.output_module, &bytes[..])?;

    Ok(())
}
