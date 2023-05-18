# wizer

> Prebuilt weval binaries available via npm

## API

```
$ npm install --save @cfallin/weval
```

```js
const execFile = require('child_process').execFile;
const weval = require('@cfallin/weval');

execFile(weval, ['-i', 'snapshot.wasm', '-o', 'wevaled.wasm'], (err, stdout) => {
	console.log(stdout);
});
```
