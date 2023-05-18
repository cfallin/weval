import { endianness } from "node:os";
import { fileURLToPath } from 'node:url';
import { dirname, join, parse } from 'node:path';
import { platform, arch } from "node:process";
import { mkdir } from "node:fs/promises";
import { existsSync } from "node:fs";
import decompress from 'decompress';
import decompressUnzip from 'decompress-unzip';
import decompressTar from 'decompress-tar';
import plzma from 'plzmasdk';
const __dirname = dirname(fileURLToPath(import.meta.url));

const knownPlatforms = {
  "win32 x64 LE": "x86_64-windows",
  "darwin arm64 LE": "aarch64-macos",
  "darwin x64 LE": "x86_64-macos",
  "linux x64 LE": "x86_64-linux",
  "linux arm64 LE": "aarch64-linux",
};

function getPlatformName() {
  let platformKey = `${platform} ${arch} ${endianness()}`;

  if (platformKey in knownPlatforms) {
    return knownPlatforms[platformKey];
  }
  throw new Error(`Unsupported platform: "${platformKey}". "weval does not have a precompiled binary for the platform/architecture you are using. You can open an issue on https://github.com/cfallin/weval/issues to request for your platform/architecture to be included."`);
}

async function getJSON(url) {
    let response = await fetch(url);
    if (!response.ok) {
        console.error(`Bad response from ${url}`);
        process.exit(1);
    }
    return response.json();
}

const platformName = getPlatformName();
const assetSuffix = (platform == 'win32') ? 'zip' : 'tar.xz';
const exeSuffix = (platform == 'win32') ? '.exe' : '';

const exeDir = join(__dirname, platformName);
const exe = join(exeDir, `weval${exeSuffix}`);

if (!existsSync(exe)) {
    await mkdir(exeDir, { recursive: true });

    let tag = "v0.1.0";
    let repoBaseURL = `https://api.github.com/repos/cfallin/weval`;
    let response = await getJSON(`${repoBaseURL}/releases/tags/${tag}`);
    let id = response.id;
    let assets = await getJSON(`${repoBaseURL}/releases/${id}/assets`);
    let releaseAsset = `weval-${tag}-${platformName}.${assetSuffix}`;
    let asset = assets.find(asset => asset.name === releaseAsset);
    if (!asset) {
        console.error(`Can't find an asset named ${releaseAsset}`);
        process.exit(1);
    }
    let data = await fetch(asset.browser_download_url);
    if (!data.ok) {
        console.error(`Error downloading ${asset.browser_download_url}`);
        process.exit(1);
    }
    let buf = await data.arrayBuffer();

    if (releaseAsset.endsWith('.xz')) {
        const archiveDataInStream = new plzma.InStream(buf);
        const decoder = new plzma.Decoder(archiveDataInStream, plzma.FileType.xz);
        decoder.open();

        // We know the xz archive only contains 1 file, the tarball
        // We extract the tarball in-memory, for later use in the `decompress` function
        const selectedItemsToStreams = new Map();
        selectedItemsToStreams.set(decoder.itemAt(0), plzma.OutStream());

        decoder.extract(selectedItemsToStreams);
        for (const value of selectedItemsToStreams.values()) {
            buf = value.copyContent();
        }
    }
    await decompress(Buffer.from(buf), exeDir, {
        // Remove the leading directory from the extracted file.
        strip: 1,
        plugins: [
            decompressUnzip(),
            decompressTar()
        ],
        // Only extract the binary file and nothing else
        filter: file => parse(file.path).base === `weval${exeSuffix}`,
    });
}

export default exe;
