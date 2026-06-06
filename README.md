# GAME-rs

A Rust port of **GAME** (Generative Adaptive MIDI Extractor) — an inference engine that transcribes singing/vocal audio into note events. It loads a pre-trained [GGUF](https://github.com/ggml-org/ggml/blob/master/docs/gguf.md) model and converts a WAV file into MIDI, TXT, or CSV by running three model stages: an **encoder**, a D3PM-diffusion **segmenter**, and an **estimator**.

This is a from-scratch port of [openvpi/GAME](https://github.com/openvpi/GAME) with no Python or external ML-framework runtime dependency — just Rust, with hand-written CPU kernels and an optional WGPU backend.

---

## Features

- **Self-contained CLI** — one binary, no Python interpreter, no `libtorch` / ONNX runtime.
- **GGUF model loading** with safety-hardened parsing (bounds checks, alignment-safe decode).
- **Two backends**:
  - **CPU** — hand-optimized kernels (blocked attention, GEMM-backed linear/matmul, depthwise conv, RoPE).
  - **GPU** — WGPU compute shaders (Vulkan / Metal / DX12 / GL), enabled via the `gpu` feature.
- **Automatic device selection** — tries GPU, falls back to CPU on failure (including driver timeouts).
- **Chunk parallelism** — long audio is sliced on silence, split into bounded chunks, and inferred in parallel across CPU cores with deterministic per-chunk seeding.
- **Multiple output formats** — MIDI (`.mid`), tab-separated text (`.txt`), or CSV (`.csv`).
- **Structured progress + logging** — rich TTY progress bars, plus `RUST_LOG` integration for headless runs.
- **Production hardening** — panic isolation in workers, bounded allocations, memory back-pressure, and contextual error messages.

---

## Installation

Requires a **stable Rust toolchain (≥ 1.85, edition 2024)**. The repo pins this via `rust-toolchain.toml`.

```bash
git clone https://github.com/Jobsecond/GAME-rs.git
cd GAME-rs

# Build the CPU-only release binary (default)
cargo build --release --no-default-features
```

The binary is produced at `target/release/game-cli` (`game-cli.exe` on Windows).

> **Note:** Always pass `--no-default-features` for CPU-only builds. The `default` feature set is empty, but omitting the flag can pull in unintended dependencies on some configurations.

### Building with GPU support

```bash
cargo build --release --features gpu
```

The GPU backend uses WGPU and will pick a Vulkan / Metal / DX12 / GL adapter at runtime.

---

## Usage

The CLI has two subcommands: `extract` (audio → notes) and `inspect` (examine a GGUF model).

### `extract` — transcribe audio to notes

```bash
game-cli extract --model path/to/model.gguf --output out.mid input.wav
```

The output format is inferred from the `--output` extension (`.mid`/`.midi` → MIDI, `.txt` → TXT, `.csv` → CSV), or set explicitly with `--format`.

#### Common options

| Flag | Default | Description |
|---|---|---|
| `-m, --model <PATH>` | *(required)* | Path to the GGUF model file. |
| `-o, --output <PATH>` | *(required)* | Output file path; format inferred from extension. |
| `--format <midi\|txt\|csv>` | from extension | Force the output format. |
| `--device <cpu\|gpu>` | gpu if available, else cpu | Compute backend. |
| `--seed <U64>` | `0` | RNG seed; `0` means non-deterministic (random). |
| `--d3pm-nsteps <N>` | `1` | Number of D3PM diffusion refinement steps. Higher = better quality, slower. |
| `--d3pm-t0 <F>` | `0.0` | D3PM starting time. |
| `--boundary-threshold <F>` | `0.2` | Note-boundary detection threshold. |
| `--boundary-radius <N>` | `2` | Boundary smoothing radius. |
| `--note-threshold <F>` | `0.2` | Voicing/note presence threshold. |
| `--language <N>` | `0` | Language ID (for multi-language models). |
| `--chunk-parallelism <auto\|on\|off>` | `auto` | Parallelize inference across audio chunks. |
| `--max-chunk-seconds <N>` | `60` | Hard-split sliced chunks longer than this many seconds. |

#### GPU adapter selection

When multiple GPUs are present, pick a specific one:

| Flag | Description |
|---|---|
| `--gpu-name <SUBSTRING>` | Match adapter name (case-insensitive substring). |
| `--gpu-vendor-id <ID>` | Match PCI vendor ID (e.g. `0x10de` for NVIDIA). Accepts hex (`0x…`) or decimal. |
| `--gpu-device-id <ID>` | Match PCI device ID. |

#### Examples

```bash
# Higher-quality transcription with 8 diffusion steps, deterministic output
game-cli extract -m path/to/model.gguf -o vocals.mid --d3pm-nsteps 8 --seed 42 vocals.wav

# CSV output, CPU only, serial (no chunk parallelism)
game-cli extract -m path/to/model.gguf -o notes.csv --device cpu --chunk-parallelism off song.wav

# Force a specific NVIDIA GPU
game-cli extract -m path/to/model.gguf -o out.mid --device gpu --gpu-vendor-id 0x10de input.wav
```

### `inspect` — examine a GGUF model

```bash
game-cli inspect --model path/to/model.gguf
```

Prints the GGUF version, architecture, quantization, tensor/parameter counts, model config, and inference parameters (sample rate, hop size, mel-spectrogram setup, etc.).

| Flag | Default | Description |
|---|---|---|
| `-m, --model <PATH>` | *(required)* | Path to the GGUF model file. |
| `--show-tensors <N>` | `8` | Number of tensors to list. |
| `--tensor-prefix <PREFIX>` | — | Filter listed tensors by name prefix. |
| `--format <text\|json>` | `text` | Output format. Use `json` for machine parsing. |

```bash
# Machine-readable summary
game-cli inspect -m model.gguf --format json

# List all estimator tensors
game-cli inspect -m model.gguf --tensor-prefix estimator --show-tensors 100
```

---

## Output formats

- **MIDI** (`.mid`) — single-track SMF with note-on/note-off events at the configured tempo.
- **TXT** (`.txt`) — tab-separated: `offset<TAB>duration<TAB>pitch`.
- **CSV** (`.csv`) — comma-separated with header: `offset,duration,pitch`.

For text formats, timing is in **seconds** and pitch is in **MIDI numbers** (60 = C4, fractional values allowed for microtonal pitch). Unvoiced segments are emitted as `rest`.

---

## Architecture

The project is a Cargo workspace with four library crates plus the CLI binary:

| Crate | Path | Responsibility |
|---|---|---|
| `game-core` | `crates/core` | GGUF loading, model forward passes, tensor backends (CPU/GPU), mel spectrogram, RNG, profiler. |
| `game-audio` | `crates/audio` | WAV decode, resample, mono mixdown, silence-based slicing, long-chunk splitting. |
| `game-output` | `crates/output` | MIDI encoding (via `midly`), TXT/CSV output. |
| `game-service` | `crates/service` | Orchestration: request → audio prep → chunk parallelism → inference → output. Public API: `extract_with_notifier()`. |
| `game-cli` | `src/` | CLI with `inspect` and `extract` subcommands. |

### Inference pipeline

1. **Audio prep** — decode WAV, mix to mono, resample to the model's target rate.
2. **Slicing** — cut on silence boundaries, then hard-split chunks longer than `--max-chunk-seconds`.
3. **Per-chunk inference** (parallel on CPU):
   - **Encoder** — mel spectrogram → contextual embeddings.
   - **Segmenter** — iterative D3PM diffusion refinement (run `--d3pm-nsteps` times).
   - **Estimator** — final pitch/voicing logits → note events.
4. **Aggregation** — chunks sorted by index, note offsets shifted by chunk position, concatenated.
5. **Output** — encode to MIDI/TXT/CSV.

### Tensor backends

A `Tensor` trait with two implementations is dispatched at model-load time:

- **CPU** (`tensor/cpu/`) — `Arc<Vec<f32>>` storage with stride-based views and hand-written kernels (`attention.rs`, `matmul.rs`, `conv.rs`, `norm.rs`, `rope.rs`, …).
- **GPU** (`tensor/gpu/`) — WGPU compute with WGSL shaders in `tensor/gpu/shaders/`.

---

## Configuration via environment variables

| Variable | Default | Purpose |
|---|---|---|
| `GAME_ATTENTION_BLOCK_K` | `128` | K-dimension block size for blocked attention (`0` disables, uses the old path). |
| `GAME_MAX_ATTENTION_SCORE_ELEMENTS` | `32M` | Attention score allocation cap. |
| `GAME_MAX_CONCURRENT_CHUNKS` | num threads | Max chunks processed simultaneously (memory back-pressure limiter). |
| `GAME_LINEAR_TARGET_TASKS` | physical cores | Rayon tasks for the linear layer. |
| `GAME_LINEAR_MIN_OUTPUTS_PER_CHUNK` | `16384` | Min outputs per Rayon task chunk. |
| `GAME_DISABLE_CHUNK_PARALLELISM` | — | Disable chunk parallelism at runtime. |
| `GAME_CPU_PROFILE` | off | Enable hand-rolled scope-based CPU profiling. |
| `GAME_CPU_PROFILE_TOP` | `20` | Number of top profiling entries to show. |
| `RUST_LOG` | — | Standard `env_logger` filter (e.g. `RUST_LOG=info`) for headless logging. |

---

## Development

```bash
# Fast compile check (no codegen)
cargo check --no-default-features

# Run the full test suite (--workspace is REQUIRED — the repo root is itself a
# package, so a bare `cargo test` only runs the CLI's tests and skips the crates)
cargo test --workspace --no-default-features

# Run a single test with output
cargo test --workspace --no-default-features <test_name> -- --nocapture

# GPU compile check / tests
cargo check --features gpu
cargo test --features gpu tensor::gpu -- --nocapture

# Lint and format (advisory; matches CI)
cargo fmt --all --check
cargo clippy --workspace --all-targets --no-default-features
```

### Feature flags

- `gpu` — WGPU-based GPU inference.
- `cpu-attention-gemm-gemm` — use the `gemm` crate for attention matmul (default CPU path).
- `cpu-attention-gemm-matrixmultiply` — swap to `matrixmultiply` for A/B testing (mutually exclusive with the above).

### CI

GitHub Actions runs an enforcing build+test matrix across **Linux / macOS / Windows × two CPU attention backends**, a GPU compile-check on all three OSes, and an advisory `fmt` + `clippy` + `cargo-deny` pass. See `.github/workflows/ci.yml`.

---

## License

Licensed under the [MIT License](LICENSE).

This project is a Rust port of [GAME](https://github.com/openvpi/GAME) by **Team OpenVPI**, also distributed under the MIT License. The upstream copyright notice is preserved in the [`LICENSE`](LICENSE) file as required.
