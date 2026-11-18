<!-- Original work Copyright 2026 Valentyn Karachevtsev (SpriteLoom)
     Licensed under the Apache License, Version 2.0
     Modifications Copyright 2026 Alessio Russo (Pixagon) -->

# Pixagon: an AI pixel-art assistant for Aseprite

Generate sprites from a text prompt, edit an existing sprite with a plain instruction, or redraw only a selected region. All of it runs inside Aseprite, on your own GPU or through an optional cloud provider if you don't have one.

Pixagon is an Aseprite extension. You stay in the editor the whole time. Results open in a side window, and a click drops one into your sprite as a new layer. Pixagon never edits your existing pixels.

Goblin with a dagger · Timber cottage · Gold coin with a crown · Adventurer in a leather coat
*Made with Pixagon, one generation each.*

## What it does

**Generate:** make a sprite from a text prompt ("wooden cottage, thatched roof").
**Edit with AI:** change a sprite by instruction ("make the sword glow blue"). Style and everything you did not mention stays put.
**Inpaint Selection:** the same, but it only touches the region you selected.
**Rotate + Instruct:** view the same subject from another angle, for turnarounds.

Tall man in a cloak, three-quarter view · The same man seen from the front
*Rotate + Instruct: the same character from two angles.*

Output goes through a pixel-art postprocess step: palette quantization, background removal, optional mirror symmetry, and fit-to-canvas, so results land clean at your sprite size.

## Two ways to run it

Pick one in Setup. You can change it later.

**Local GPU (private, free to run).** Pixagon runs the FLUX.2 Klein 4B diffusion model on your own computer. Nothing leaves your machine.

- **GPU:** NVIDIA with 12+ GB VRAM, or an Apple Silicon Mac (M1 or newer, macOS 14+). 8 GB NVIDIA cards work in a slower "Legacy 8 GB" mode.
- **RAM:** 16 GB minimum. On a Mac, 24 GB or more of unified memory is recommended.
- **Disk:** about 15 GB for the model, downloaded by Setup.

**OpenRouter (cloud, no GPU needed).** No NVIDIA GPU, no Apple Silicon Mac, or a card that is too weak? Use an OpenRouter image model instead. There is no 15 GB download. Paste your OpenRouter API key into Setup and go. The default model is google/gemini-2.5-flash-image, and you can type any OpenRouter model that returns images. Things to know:

- Requests are billed to your own OpenRouter account.
- Your prompts, and the sprite and selection you edit, are sent to OpenRouter and the model's provider.
- Cloud models are slower per picture and follow the seed less closely.
- Results still go through the same pixel-art cleanup as local ones.

There is no CPU fallback for the local model, and AMD cards and Intel Macs are not supported locally. OpenRouter works on any machine that runs Aseprite.

**Both:** Aseprite 1.3 or newer, on Windows, Linux (NVIDIA driver with CUDA 12.8 for local) or macOS.

## Install

1. Download the .aseprite-extension file and double-click it, or use Edit > Preferences > Extensions > Add Extension in Aseprite.
2. Open the Pixagon panel. Setup opens by itself if something is missing. You can also open it from the Pixagon Setup... command or the Setup... button on the panel.
3. Pick a provider: Local GPU or OpenRouter (type your API key and press Save OpenRouter settings). Press Install. It installs uv and a Python environment, plus the 15 GB model for Local GPU only. You do not need Python installed first.
4. The panel starts the server for you once everything is installed. Type a prompt and generate.

One file works on every OS. Everything Pixagon downloads lives in a per-user folder (%APPDATA%\Pixagon, ~/Library/Application Support/Pixagon or ~/.config/pixagon), not inside the extension, so updating the extension never wipes your model.

Pixagon is free, or pay what you want. The source stays open if you prefer to build it yourself.

## Status

The single-file extension is new. It has been run end to end on Linux with the OpenRouter provider. The local model on Windows and macOS still needs more real-hardware testing. If something misbehaves, please report what you see in the comments or on GitHub.

## Open source and licenses

Pixagon is open source under Apache 2.0, and is a derivative work of SpriteLoom by Valentyn Karachevtsev. The FLUX.2 Klein 4B model is licensed separately by Black Forest Labs under Apache 2.0 and is downloaded at runtime, not redistributed.

Source, samples and docs: github.com/blkdmr/pixagon and blkdmr.github.io/pixagon
