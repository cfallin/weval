use std::path::PathBuf;
use structopt::StructOpt;
use walrus::Module;

mod directive;
mod eval;
mod image;
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
    let opts = Options::from_args();
    let mut module = Module::from_file(&opts.input_module)?;

    // Build module image.
    let im = image::build_image(&module)?;

    // Collect directives.
    let directives = directive::collect(&module, &im)?;

    // Partially evaluate.
    let result = eval::partially_evaluate(&mut module, &im, &directives[..])?;

    module.emit_wasm_file(&opts.output_module)?;

    Ok(())
}
