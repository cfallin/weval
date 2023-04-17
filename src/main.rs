#![allow(dead_code)]

use std::collections::VecDeque;
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

    /// Run IR in interpreter differentially, before and after
    /// wevaling, comparing trace outputs.
    #[structopt(long = "run-diff")]
    run_diff: bool,

    /// Optimization fuel, for bisecting issues, or 0 for infinite.
    #[structopt(long = "opt-fuel", default_value = "0")]
    fuel: u64,
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
    if opts.run_diff {
        module.expand_all_funcs()?;
    }

    // Build module image.
    let mut im = image::build_image(&module)?;

    // Collect directives.
    let directives = directive::collect(&module, &mut im)?;
    log::debug!("Directives: {:?}", directives);

    // Partially evaluate.
    let progress = indicatif::ProgressBar::new(0);
    let mut result =
        eval::partially_evaluate(module, &mut im, &directives[..], &opts, Some(progress))?;

    // Update memories in module.
    image::update(&mut result.module, &im);

    log::debug!("Final module:\n{}", result.module.display());

    if opts.run_diff {
        image::update(result.orig_module.as_mut().unwrap(), &im);
        run_diff(result.orig_module.unwrap(), result.module);
        return Ok(());
    }

    let bytes = result.module.to_wasm_bytes()?;
    std::fs::write(&opts.output_module, &bytes[..])?;

    Ok(())
}

struct TraceIter {
    thread: std::thread::JoinHandle<()>,
    channel: std::sync::mpsc::Receiver<(usize, Vec<waffle::ConstVal>)>,
}

impl TraceIter {
    fn new(module: waffle::Module<'static>) -> TraceIter {
        let mut ctx = waffle::InterpContext::new(&module).unwrap();
        if let Some(start) = module.start_func {
            ctx.call(&module, start, &[]).ok().unwrap();
        }

        let entry = if let Some(waffle::Export {
            kind: waffle::ExportKind::Func(func),
            ..
        }) = module.exports.iter().find(|e| &e.name == "_start")
        {
            *func
        } else {
            panic!("No _start entrypoint");
        };

        let (sender, receiver) = std::sync::mpsc::sync_channel(1000);
        let thread = std::thread::spawn(move || {
            let count = std::sync::Arc::new(std::sync::atomic::AtomicU64::new(0));
            let count_cloned = count.clone();
            ctx.trace_handler = Some(Box::new(move |id, args| {
                count_cloned.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                sender.send((id, args)).is_ok()
            }));
            let result = ctx.call(&module, entry, &[]);
            if let Err(e) = result.ok() {
                eprintln!(
                    "Panic after {} steps: {:?}",
                    count.load(std::sync::atomic::Ordering::Relaxed),
                    e
                );
                let handler = ctx.trace_handler.unwrap();
                handler(0, vec![]);
            }
        });

        TraceIter {
            thread,
            channel: receiver,
        }
    }
}

impl Iterator for TraceIter {
    type Item = (usize, Vec<waffle::ConstVal>);
    fn next(&mut self) -> Option<Self::Item> {
        self.channel.recv().ok()
    }
}

fn run_diff(orig_module: waffle::Module<'_>, wevaled_module: waffle::Module<'_>) {
    let orig_text = format!("{}", orig_module.display());
    let wevaled_text = format!("{}", wevaled_module.display());
    let orig = TraceIter::new(orig_module.without_orig_bytes());
    let wevaled = TraceIter::new(wevaled_module.without_orig_bytes());

    let mut progress: u64 = 0;
    let mut last_n = VecDeque::new();
    for ((orig_id, orig_args), (wevaled_id, wevaled_args)) in orig.zip(wevaled) {
        progress += 1;
        if progress % 100000 == 0 {
            eprintln!("{} steps", progress);
        }

        last_n.push_back((orig_id, orig_args.clone()));
        if last_n.len() > 10 {
            last_n.pop_front();
        }

        if orig_id != wevaled_id || orig_args != wevaled_args {
            eprintln!("Original:\n{}\n", orig_text);
            eprintln!("wevaled:\n{}\n", wevaled_text);
            eprintln!("Recent tracepoints:");
            for (id, args) in last_n {
                eprintln!("* {}, {:?}", id, args);
            }
            panic!(
                "Mismatch: orig ({}, {:?}), wevaled ({}, {:?})",
                orig_id, orig_args, wevaled_id, wevaled_args
            );
        }
    }
}
