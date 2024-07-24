import { endianness } from "node:os";
import { fileURLToPath } from 'node:url';
import { dirname, join, parse } from 'node:path';
import { platform, arch } from "node:process";
import { mkdir } from "node:fs/promises";
import { existsSync } from "node:fs";
import decompress from 'decompress';
import decompressUnzip from 'decompress-unzip';
import decompressTar from 'decompress-tar';
import xz from '@napi-rs/lzma/xz';
const __dirname = dirname(fileURLToPath(import.meta.url));

const TAG = "v0.2.1";

async function getWeval() {
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

        let repoBaseURL = `https://api.github.com/repos/cfallin/weval`;
        let response = await getJSON(`${repoBaseURL}/releases/tags/${TAG}`);
        let id = response.id;
        let assets = await getJSON(`${repoBaseURL}/releases/${id}/assets`);
        let releaseAsset = `weval-${TAG}-${platformName}.${assetSuffix}`;
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
            buf = await xz.decompress(new Uint8Array(buf));
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

    return exe;
}

export default getWeval;
