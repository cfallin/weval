## weval: the WebAssembly partial evaluator

`weval` partially evaluates WebAssembly snapshots to turn interpreters into
compilers (see [Futamura
projection](https://en.wikipedia.org/wiki/Partial_evaluation#Futamura_projections)
for more).

`weval` binaries are available via releases on this repo or via an [npm
package](https://www.npmjs.com/package/@cfallin/weval).

Usage of weval is like:

```
$ weval weval -w -i program.wasm -o wevaled.wasm
```

which runs Wizer on `program.wasm` to obtain a snapshot, then processes any
weval requests (function specialization requests) in the resulting heap image,
appending the specialized functions and filling in function pointers in
`wevaled.wasm`.

See the API in `include/weval.h` for more.

### Releasing Checklist

- Bump the version in `Cargo.toml` and `cargo check` to ensure `Cargo.lock` is
  updated as well.
- Bump the tag version (`TAG` constant) in `npm/weval/index.js`.
- Bump the npm package version in `npm/weval/package.json`.
- Run `npm i` in `npm/weval/` to ensure the `package-lock.json` file is
  updated.

- Commit all of this as a "version bump" PR.
- Push it to `main` and ensure CI completes successfully.
- Tag as `v0.x.y` and push that tag.
- `cargo publish` from the root.
- `npm publish` from `npm/weval/`.
