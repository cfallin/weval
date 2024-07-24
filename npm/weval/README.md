# weval

> Prebuilt weval binaries available via npm

See the [weval repository](https://github.com/cfallin/weval) for more details.

## API

```
$ npm install --save @cfallin/weval
```

```js
const execFile = require('child_process').execFile;
const weval = require('@cfallin/weval');

execFile(weval, ['-w', '-i', 'snapshot.wasm', '-o', 'wevaled.wasm'], (err, stdout) => {
	console.log(stdout);
});
```
