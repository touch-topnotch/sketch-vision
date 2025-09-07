# sketch-vision

A tiny Rust CLI that turns photos into sketch-like edge images using the Sobel operator.

## Install

Prerequisites: Rust toolchain (`rustup`)

```bash
cargo install --path .
```

Alternatively, build locally:

```bash
cargo build --release
./target/release/sketch-vision --help
```

## Usage

```bash
sketch-vision <INPUT> [-o <OUTPUT>] [--threshold <0-255>] [--invert]
```

- **INPUT**: path to the input image (PNG/JPEG/etc.)
- **-o, --output**: output path (default: `edges.png`)
- **--threshold**: optional 0..255 threshold to binarize edges
- **--invert**: invert output colors

Examples:

```bash
# Basic edge map
sketch-vision examples/cat.jpg -o out/cat_edges.png

# Binarized edges with threshold = 80
sketch-vision examples/cat.jpg -o out/cat_edges_bw.png --threshold 80

# Inverted edge map
sketch-vision examples/cat.jpg -o out/cat_edges_inv.png --invert
```

## What it does

- Converts the image to grayscale
- Applies 3x3 Sobel filters (Gx and Gy)
- Computes magnitude sqrt(Gx^2 + Gy^2)
- Normalizes to 0..255
- Optionally applies thresholding and inversion

## License

Dual-licensed under either:

- MIT License
- Apache License, Version 2.0

at your option.
