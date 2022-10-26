#![allow(dead_code)]

use std::path::PathBuf;
use structopt::StructOpt;
use walrus::Module;

mod directive;
mod eval;
mod image;
mod intrinsics;
mod state;
mod value;
mod stackify;

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

    // Build module image.
    let mut im = image::build_image(&module)?;

    // Collect directives.
    let directives = directive::collect(&module, &mut im)?;
    println!("Directives: {:?}", directives);

    // Partially evaluate.
    eval::partially_evaluate(&mut module, &mut im, &directives[..])?;

    // Update memories in module.
    image::update(&mut module, &im);

    module.emit_wasm_file(&opts.output_module)?;

    Ok(())
}
