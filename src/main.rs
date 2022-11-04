#![allow(dead_code)]

use std::path::PathBuf;
use structopt::StructOpt;
use walrus::Module;

mod directive;
mod eval;
mod image;
mod intrinsics;
mod stackify;
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
    let mut module = Module::from_file(&opts.input_module)?;

    let bytes = std::fs::read(&opts.input_module)?;
    let mut waffle_module = waffle::Module::from_wasm_bytes(&bytes[..])?;

    // Build module image.
    let mut im = image::build_image(&waffle_module)?;

    // Collect directives.
    let directives = directive::collect(&waffle_module, &mut im)?;
    log::debug!("Directives: {:?}", directives);

    // Partially evaluate.
    eval::partially_evaluate(&mut module, &mut im, &directives[..])?;

    // Update memories in module.
    image::update(&mut waffle_module, &im);

    let bytes = waffle_module.to_wasm_bytes()?;
    std::fs::write(&opts.output_module, &bytes[..])?;

    Ok(())
}
