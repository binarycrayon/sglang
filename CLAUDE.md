# CLAUDE.md - SGLang Codebase Guide for AI Assistants

> **Last Updated:** 2025-01-04
> **SGLang Version:** 0.5.3rc0
> **Purpose:** Comprehensive guide for AI assistants working with the SGLang codebase

---

## Table of Contents

1. [Repository Overview](#repository-overview)
2. [Codebase Structure](#codebase-structure)
3. [Technology Stack](#technology-stack)
4. [Development Environment Setup](#development-environment-setup)
5. [Code Quality & Standards](#code-quality--standards)
6. [Testing Strategy](#testing-strategy)
7. [Contribution Workflow](#contribution-workflow)
8. [Key Conventions & Patterns](#key-conventions--patterns)
9. [Important Files & Locations](#important-files--locations)
10. [Common Development Tasks](#common-development-tasks)
11. [Environment Variables](#environment-variables)
12. [Troubleshooting](#troubleshooting)

---

## Repository Overview

### What is SGLang?

SGLang is a **fast serving framework for large language models (LLMs) and vision language models (VLMs)**. It is designed for high-performance inference with advanced optimization techniques and an intuitive programming interface.

**Key Features:**
- **Fast Backend Runtime**: RadixAttention, zero-overhead CPU scheduler, prefill-decode disaggregation, speculative decoding, continuous batching, paged attention, tensor/pipeline/expert/data parallelism
- **Flexible Frontend Language**: Intuitive DSL for LLM programming with control flow, chaining, multi-modal support
- **Extensive Model Support**: 50+ model implementations (Llama, Qwen, DeepSeek, GPT, Gemma, Mistral, etc.)
- **Production Scale**: Powers trillions of tokens daily, deployed on 1,000,000+ GPUs worldwide

**Industry Adoption:**
- Used by: xAI, AMD, NVIDIA, Intel, LinkedIn, Cursor, Oracle Cloud, Google Cloud, Microsoft Azure, AWS, and major universities (MIT, Stanford, UC Berkeley, Tsinghua)
- Award recipient: a16z Open Source AI Grant

### Repository Metadata
- **License:** Apache 2.0
- **Homepage:** https://github.com/sgl-project/sglang
- **Documentation:** https://docs.sglang.ai/
- **Slack Community:** https://slack.sglang.ai/
- **Python Requirement:** >=3.10

---

## Codebase Structure

### Top-Level Directory Organization

```
sglang/
├── python/sglang/           # Main Python package (frontend + runtime)
│   ├── srt/                 # SGLang Runtime (SRT) - backend inference engine
│   ├── lang/                # Frontend DSL and compiler
│   ├── eval/                # Evaluation utilities
│   └── test/                # Unit test utilities
├── sgl-kernel/              # High-performance CUDA/GPU kernels (C++/CUDA)
├── sgl-router/              # Rust-based request routing and protocol handling
├── docs/                    # Sphinx documentation
├── examples/                # Usage examples and tutorials
├── benchmark/               # Benchmarking suite (38+ categories)
├── test/                    # Integration and system tests
├── scripts/                 # Utility and automation scripts
├── docker/                  # Docker containerization
├── 3rdparty/                # Third-party dependencies (AMD-specific)
├── .github/                 # CI/CD workflows and issue templates
└── .devcontainer/           # Dev container configuration
```

### Core Python Package: `python/sglang/`

#### 1. SGLang Runtime (SRT) - `python/sglang/srt/`

The core inference engine with 25+ specialized modules:

**Core Services:**
- `managers/` - Request, memory, and batch managers
- `model_executor/` - Model inference execution
- `model_loader/` - Weight loading & initialization
- `entrypoints/` - Server entry points (API, gRPC, HTTP)

**Optimization Layers:**
- `layers/attention/` - Attention mechanisms (RadixAttention, FlashInfer)
- `layers/moe/` - Mixture-of-Experts support
- `layers/quantization/` - FP4/FP8/INT4/AWQ/GPTQ quantization
- `sampling/` - Token sampling & generation
- `speculative/` - Speculative decoding
- `constrained/` - Structured output constraints

**Model Support:**
- `models/` - 50+ model implementations
- `multimodal/` - Vision-language model support
- `parser/` - Model-specific parsing (JSON, Regex, FSM)

**Advanced Features:**
- `distributed/` - Tensor, pipeline, expert, data parallelism
- `disaggregation/` - Prefill-decode disaggregation
- `eplb/` - Expert Parallel Load Balancing
- `lora/` - LoRA fine-tuning support
- `mem_cache/` - Memory & caching management
- `tokenizer/` - Tokenizer implementations
- `grpc/` - gRPC protocol support
- `metrics/` - Performance monitoring
- `tracing/` - OpenTelemetry integration

**Key Files:**
- `server_args.py` - Runtime server configuration
- `constants.py` - Constants and magic numbers
- `operations.py` - Operation definitions
- `custom_op.py` - Custom CUDA operator interface

#### 2. Frontend Language - `python/sglang/lang/`

DSL compiler and interpreter:
- `api.py` - Public API functions
- `compiler.py` - Program compilation
- `interpreter.py` - Program execution
- `ir.py` - Intermediate representation
- `tracer.py` - Program tracing
- `choices.py` - Selection/ranking functions
- `chat_template.py` - Chat format templates
- `backend/` - External provider backends (OpenAI, Anthropic, LiteLLM, VertexAI)

#### 3. Public API Entry Points (`python/sglang/__init__.py`)

**Frontend Language APIs:**
- `Engine`, `Runtime`, `function`, `gen`, `gen_string`, `gen_int`, `select`
- `assistant`, `system`, `user` (conversation roles)
- `image`, `video` (multimodal support)
- `separate_reasoning`, `flush_cache`

**Backend Support APIs:**
- `OpenAI`, `Anthropic`, `LiteLLM`, `VertexAI`
- `RuntimeEndpoint`, `ServerArgs`

**Selection Functions:**
- `greedy_token_selection`, `token_length_normalized`, `unconditional_likelihood_normalized`

### SGLang Kernel Library - `sgl-kernel/`

High-performance C++/CUDA kernels (separate package):

**Build System:**
- `CMakeLists.txt` - Main C++ build config
- `pyproject.toml` - Python binding config
- Language: C++17, CUDA, Python bindings

**Source Code (`csrc/`):**
- `attention/` - Attention kernels
- `moe/` - MoE kernels
- `gemm/` - Matrix multiplication
- `quantization/` - Quantization kernels
- `memory/` - Memory operations
- `speculative/` - Speculative decoding kernels
- `allreduce/` - Distributed collective ops
- `mamba/` - Mamba architecture
- `grammar/` - FSM grammar kernels

**Current Version:** 0.3.10

### SGLang Router - `sgl-router/`

Rust-based request routing and protocol handling:

**Rust Source (`src/`):**
- `core/` - Core routing logic
- `routers/` - Routing strategies
- `grpc/` - gRPC server implementation
- `protocols/` - Protocol implementations
- `tokenizer/` - Tokenizer bindings
- `tool_parser/` - Tool/function call parsing
- `mcp/` - Model Context Protocol
- `reasoning_parser/` - Reasoning token parsing

**Key Dependencies:**
- Tokio async runtime, Axum web framework, Tonic gRPC
- PyO3 for Python bindings

---

## Technology Stack

### Languages & Runtimes
- **Python 3.10+** - Main development language
- **C++17** - GPU kernel implementation
- **CUDA 11.8+** - NVIDIA GPU support
- **Rust** - Router and protocol handling (Tokio, Axum, Tonic)

### Core Dependencies

**Python Core:**
```python
torch==2.8.0
transformers==4.56.1
flashinfer_python==0.3.1
sgl-kernel==0.3.9.post2
openai==1.99.1
```

**Specialized Libraries:**
- `llguidance` (0.7.11+) - Constraint guidance
- `xgrammar` (0.1.24) - Grammar-based constraints
- `outlines` (0.1.11) - Structured generation
- `compressed-tensors` - Model compression
- `timm` (1.0.16) - Vision models
- `torchao` (0.9.0) - Quantization

**Serving Infrastructure:**
- FastAPI + Uvicorn - HTTP server
- Pydantic - Data validation
- `pyzmq` (25.1.2+) - ZeroMQ messaging
- `uvloop` - High-performance async

**Monitoring & Tracing:**
- `prometheus-client` - Metrics collection
- `opentelemetry` - Distributed tracing

### Build Systems
- **Python Package:** setuptools + wheel
- **Kernel Build:** scikit-build-core + CMake
- **Router Build:** Cargo (Rust)

---

## Development Environment Setup

### Fork and Clone Repository

**Important:** New contributors do NOT have write permission to the official repo. Fork first!

```bash
# Fork the repository on GitHub
# Then clone your fork
git clone https://github.com/<your_username>/sglang.git
cd sglang
```

### Install from Source

```bash
# Install main package
pip install -e "python[dev]"

# Install sgl-kernel from source
cd sgl-kernel
pip install -e .
cd ..

# Install sgl-router (requires Rust toolchain)
cd sgl-router
pip install -e .
cd ..
```

### Install Pre-commit Hooks

**CRITICAL:** Always run pre-commit checks before submitting PRs!

```bash
pip3 install pre-commit
pre-commit install
pre-commit run --all-files
```

If it fails the first time, re-run to ensure all fixes are applied.

---

## Code Quality & Standards

### Pre-commit Hooks Configuration

File: `.pre-commit-config.yaml`

| Tool | Version | Purpose | Scope |
|------|---------|---------|-------|
| **black** | 24.10.0 | Python code formatter | Jupyter notebooks + Python files |
| **isort** | 5.13.2 | Import sorter | Python files (excludes gRPC generated) |
| **ruff** | 0.11.7 | Fast linter | F401 (unused imports) for `benchmark/`, `docs/`, `examples/` |
| **clang-format** | 18.1.8 | C++/CUDA formatter | C++, CUDA files (Google style) |
| **codespell** | 2.4.1 | Spell checker | All files |
| **nbstripout** | 0.8.1 | Notebook cleaner | Removes metadata, keeps output |

**Standard Checks:**
- Trailing whitespace, EOF fixing, YAML/TOML validation
- Python AST checking, merge conflict detection
- Large file prevention, private key detection
- Debug statement detection, branch protection

### Code Formatting Style

**Python:**
- Black formatter (default settings)
- isort with `profile = "black"` and `known_first_party = ["sglang"]`

**C++/CUDA:**
- Google style via clang-format
- 2-space indentation, 120-column limit

**General:**
- UTF-8 encoding, LF line endings
- Python/JSON/YAML: 2-4 space indents

### Running Code Quality Checks

```bash
# Run all pre-commit checks
pre-commit run --all-files --show-diff-on-failure

# Format Python code
make format  # Formats only modified files

# Check dependencies for formatting tools
make check-deps

# Install formatting dependencies
make install-deps
```

---

## Testing Strategy

### Test Organization

```
test/
├── srt/                    # Backend/SRT tests
│   ├── run_suite.py        # Test suite orchestrator
│   ├── test_srt_endpoint.py
│   ├── test_bench_*.py     # Performance tests
│   ├── test_eval_accuracy_*.py  # Accuracy tests
│   ├── models/             # Model-specific tests
│   ├── lora/               # LoRA tests
│   └── openai_server/      # API server tests
├── lang/                   # Frontend language tests
│   ├── run_suite.py
│   └── test_srt_backend.py
└── pytest.ini              # pytest configuration

sgl-kernel/tests/           # Kernel unit tests
sgl-router/tests/           # Router unit tests
sgl-router/py_test/         # Router Python tests
```

### Test Suites

Defined in `test/srt/run_suite.py`:

| Suite | GPUs | Purpose | Time |
|-------|------|---------|------|
| **per-commit** | 1 | Fast tests on every PR | 60-437s per test |
| **per-commit-2-gpu** | 2 | Multi-GPU tests | ~2 partitions |
| **per-commit-4-gpu** | 4 | Tensor parallelism | ~2 partitions |
| **per-commit-8-gpu** | 8 | Pipeline parallelism | ~2 partitions |
| **per-commit-4-gpu-deepep** | 4 | DeepSeek Expert Parallelism | - |
| **per-commit-8-gpu-b200** | 8 | Blackwell GPU specific | - |
| **nightly** | Varies | Extended test suite | 24 hours |

### Running Tests

```bash
# Single test file
python3 test/srt/test_srt_endpoint.py

# Single test method
python3 -m unittest test.srt.test_srt_endpoint.TestSRTEndpoint.test_simple_decode

# Test suite
cd test/srt
python3 run_suite.py --suite per-commit

# With partitioning (for parallel execution)
python3 run_suite.py --suite per-commit --auto-partition-id 0 --auto-partition-size 10
```

### Accuracy Testing

**Quick Sanity Check (GSM8K):**
```bash
# Launch server
python3 -m sglang.launch_server --model Qwen/Qwen2-7B-Instruct

# Evaluate
python3 -m sglang.test.few_shot_gsm8k --num-questions 200
```

**Note:** This is a sanity check, not rigorous testing. Expect 1%-5% variance due to batching and non-determinism.

**Additional Accuracy Tests:**
- `test/srt/test_eval_accuracy_large.py`
- `test/srt/test_gpt_oss_1gpu.py`

### Framework
- **Primary:** Python's built-in `unittest` library
- **Test Runner:** Custom runners in `run_suite.py`
- **Additional:** `pytest`, `expecttest`

---

## Contribution Workflow

### Pull Request Process

#### 1. Create a Feature Branch

**NEVER commit directly to `main`!**

```bash
git checkout -b feature/my-new-feature
# Make your changes
git add .
git commit -m "Description of changes"
git push origin feature/my-new-feature
```

#### 2. PR Requirements Checklist

**From `.github/pull_request_template.md`:**

- [ ] **Motivation** - Describe purpose and goals
- [ ] **Modifications** - Detail changes made
- [ ] **Accuracy Tests** - If affecting model outputs
- [ ] **Benchmarking and Profiling** - If affecting inference speed
- [ ] Format code with pre-commit
- [ ] Add unit tests
- [ ] Update documentation
- [ ] Provide accuracy and speed benchmark results

#### 3. CI/CD Requirements

**CI Workflow Triggers:**
- Must have `run-ci` label (auto-fails without it)
- Cannot be a draft PR (auto-fails if draft)
- Path-based filtering:
  - Changes to `python/**`, `scripts/ci/**`, `test/**`, `.github/workflows/pr-test.yml` trigger main tests
  - Changes to `sgl-kernel/**` trigger kernel build and tests

**How to Get `run-ci` Label:**
- **Write access:** Auto-tagged by @sglang-bot
- **Triage access:** Manually add via PR page
- **No access:** Request review and ask maintainer to add

#### 4. Code Review

**CODEOWNERS (mandatory reviews):**
- `.github/` → @merrymercy @zhyncs
- `/docker/` → @zhyncs @HaiShaw @ByronHsu
- `/python/sglang/srt/` → @merrymercy @Ying1123 @zhyncs @hnyls2002
- `/sgl-kernel/` → @zhyncs @ispobock @HandH1998 @BBuf @yizhang2077 @merrymercy @FlamingoPg @HaiShaw
- `/sgl-router/` → @slin1237 @ByronHsu

**Area Reviewers (see `.github/REVIEWERS.md`):**
- Hardware, Kernels, Scheduler/Memory, Parallelism, Models, API Server

**Finding Reviewers:**
- Check CODEOWNERS and REVIEWERS.md
- Review file modification history
- Contact frequent contributors

### CI/CD Workflows

**Location:** `.github/workflows/` (38+ workflows)

**Primary Testing:**
- `pr-test.yml` - Main PR test suite
- `pr-test-amd.yml` - AMD GPU tests
- `pr-test-npu.yml` - Ascend NPU tests
- `pr-test-h20.yml` - Habana H20 tests
- `pr-test-rust.yml` - Rust component tests
- `nightly-test.yml` - Nightly extended tests

**Code Quality:**
- `lint.yml` - Pre-commit checks on every PR

**Release Workflows:**
- `release-pypi.yml` - PyPI publishing (triggered on version.py changes)
- `release-docker.yml` - Multi-CUDA Docker images (cu126, cu128, cu129)
- `release-whl-kernel.yml` - Kernel wheel releases
- `release-pypi-router.yml` - Router package release
- `release-docs.yml` - Documentation deployment

### Version Management

**Version File:** `python/sglang/version.py`

```python
__version__ = "0.5.3rc0"
```

**Update Versions:**
```bash
make update <new_version>
```

This synchronizes versions across:
- `python/sglang/version.py`
- `python/pyproject.toml`
- `python/pyproject_other.toml`
- `sgl-kernel/pyproject.toml`
- Docker build files

### Updating sgl-kernel

**Important:** sglang and sgl-kernel are separate packages. CI doesn't support updating and using a kernel in the same PR.

**Process:**
1. Submit PR to update sgl-kernel source without using it ([example #8884](https://github.com/sgl-project/sglang/pull/8884/files))
2. Bump sgl-kernel version ([example #9220](https://github.com/sgl-project/sglang/pull/9220/files))
   - Merging triggers automatic PyPI release
3. Update sgl-kernel version in `sglang/python/pyproject.toml` and use new kernel

---

## Key Conventions & Patterns

### General Code Style

From `docs/developer_guide/contribution_guide.md`:

**1. Avoid Code Duplication**
- If >5 lines appear multiple times, extract into a shared function

**2. Minimize Device Synchronization**
- Reduce expensive CPU-GPU synchronization (`tensor.item()`, `tensor.cpu()`)
- Use vectorized code

**3. Keep Files Concise**
- If file exceeds 2,000 lines, split into smaller files

**4. Prioritize Extreme Efficiency**
- SGLang is a runtime; most code runs on critical path for every request
- Optimize all minor overheads, especially in model forward code
- **Pattern:** Cache runtime checks in model forward pass as boolean values (see [example](https://github.com/sgl-project/sglang/blob/f1b0eda55c2c4838e8ab90a0fac7fb1e3d7064ab/python/sglang/srt/models/deepseek_v2.py#L486-L491))

**5. Strive for Pure Functions**
- Avoid in-place modification of arguments

**6. Hardware/Feature Support Guidelines**
- Don't drastically change existing code
- Prefer new files for hardware-specific components (e.g., `allocator_ascend.py`)
- In if/else blocks, common path (NVIDIA/existing) should be first branch

### Import Organization

**isort Configuration:**
```python
profile = "black"
known_first_party = ["sglang"]
```

**Pattern in `__init__.py`:**
```python
# Public APIs
from sglang.lang.api import function, gen, select, ...

# Lazy imports for heavy libraries
from sglang.utils import LazyImport
OpenAI = LazyImport("sglang.lang.backend.openai", "OpenAI")

# Define __all__ for explicit exports
__all__ = ["function", "gen", "select", ...]
```

### File Exclusions

**Excluded from Linting:**
- Auto-generated gRPC files: `*_pb2.py`, `*_pb2_grpc.py`
- gRPC Python module

**Codespell Exceptions:**
```python
ignore-words-list = "ans, als, hel, boostrap, childs, te, vas, hsa, ment"
skip = "*.json,*.jsonl,*.patch,*.txt"
```

---

## Important Files & Locations

### Configuration Files

| File | Purpose |
|------|---------|
| `python/pyproject.toml` | Main Python package configuration |
| `python/pyproject_other.toml` | Optional dependency groups |
| `sgl-kernel/pyproject.toml` | Kernel package configuration |
| `sgl-router/Cargo.toml` | Router Rust package configuration |
| `.pre-commit-config.yaml` | Pre-commit hooks configuration |
| `.editorconfig` | Editor formatting standards |
| `.isort.cfg` | Import sorting configuration |
| `Makefile` | Version management and formatting |
| `pytest.ini` | Pytest configuration |

### Documentation

| Location | Content |
|----------|---------|
| `README.md` | Project overview and getting started |
| `docs/` | Full Sphinx documentation |
| `docs/get_started/install.md` | Installation instructions |
| `docs/developer_guide/contribution_guide.md` | Contribution guidelines |
| `docs/developer_guide/benchmark_and_profiling.md` | Performance testing |
| `docs/references/environment_variables.md` | Environment variable reference |
| `.github/pull_request_template.md` | PR template |
| `.github/CODEOWNERS` | Code ownership and mandatory reviewers |
| `.github/REVIEWERS.md` | Recommended reviewers by area |

### Key Source Files

| File | Purpose |
|------|---------|
| `python/sglang/__init__.py` | Public API entry points |
| `python/sglang/version.py` | Version definition |
| `python/sglang/srt/server_args.py` | Runtime server configuration |
| `python/sglang/srt/constants.py` | System constants |
| `python/sglang/lang/api.py` | Frontend API functions |
| `test/srt/run_suite.py` | Test suite orchestrator |

---

## Common Development Tasks

### 1. Running the Server

```bash
# Launch server with a model
python3 -m sglang.launch_server --model Qwen/Qwen2-7B-Instruct

# With specific GPU configuration
python3 -m sglang.launch_server \
    --model meta-llama/Llama-3-8B-Instruct \
    --tp 2  # Tensor parallelism

# With quantization
python3 -m sglang.launch_server \
    --model deepseek-ai/DeepSeek-V3 \
    --quantization fp8
```

### 2. Adding a New Model

1. **Create model implementation:** `python/sglang/srt/models/my_model.py`
2. **Register in model loader:** Update `python/sglang/srt/model_loader/loader.py`
3. **Add tests:** Create `test/srt/models/test_my_model.py`
4. **Update documentation:** Add to `docs/supported_models/generative_models.md`

### 3. Benchmarking Performance

```bash
# See detailed instructions
# docs/developer_guide/benchmark_and_profiling.md

# Quick benchmark
cd benchmark
python3 benchmark_batch.py --model <model_name>
```

### 4. Building Documentation

```bash
cd docs
pip install -r requirements.txt
make html
# Open _build/html/index.html
```

### 5. Running Specific Test Suites

```bash
# SRT tests
cd test/srt
python3 run_suite.py --suite per-commit

# Lang tests
cd test/lang
python3 run_suite.py --suite per-commit

# Single test
python3 -m unittest test.srt.test_srt_endpoint.TestSRTEndpoint.test_simple_decode
```

### 6. Debugging with Environment Variables

```bash
# Enable memory debugging
export SGLANG_DEBUG_MEMORY_POOL=true

# Enable torch compile
export SGLANG_ENABLE_TORCH_COMPILE=true

# Disable request logging
export SGLANG_DISABLE_REQUEST_LOGGING=true

# Run with profiling
export SGLANG_TORCH_PROFILER_DIR=/tmp/profiler
export SGLANG_RECORD_STEP_TIME=true
```

---

## Environment Variables

### General Configuration

| Variable | Description | Default |
|----------|-------------|---------|
| `SGLANG_USE_MODELSCOPE` | Enable ModelScope models | `false` |
| `SGLANG_HOST_IP` | Server host IP | `0.0.0.0` |
| `SGLANG_PORT` | Server port | auto-detected |
| `SGLANG_DISABLE_REQUEST_LOGGING` | Disable request logging | `false` |
| `SGLANG_HEALTH_CHECK_TIMEOUT` | Health check timeout (seconds) | `20` |

### Performance Tuning

| Variable | Description | Default |
|----------|-------------|---------|
| `SGLANG_ENABLE_TORCH_COMPILE` | Enable torch.compile | `true` |
| `SGLANG_ENABLE_TORCH_INFERENCE_MODE` | Use torch.inference_mode | `false` |
| `SGLANG_IS_FLASHINFER_AVAILABLE` | FlashInfer availability check | `true` |
| `SGLANG_SKIP_P2P_CHECK` | Skip P2P access check | `false` |
| `SGL_CHUNKED_PREFIX_CACHE_THRESHOLD` | Chunked prefix cache threshold | `8192` |

### Memory Management

| Variable | Description | Default |
|----------|-------------|---------|
| `SGLANG_DEBUG_MEMORY_POOL` | Enable memory pool debugging | `false` |
| `SGLANG_CLIP_MAX_NEW_TOKENS_ESTIMATION` | Clip max tokens for memory planning | `4096` |
| `SGL_DISABLE_TP_MEMORY_INBALANCE_CHECK` | Disable TP memory imbalance check | Not set |

### Model-Specific

| Variable | Description | Default |
|----------|-------------|---------|
| `SGLANG_USE_AITER` | Use AITER optimization | `false` |
| `SGLANG_INT4_WEIGHT` | Enable INT4 weight quantization | `false` |
| `SGLANG_MOE_PADDING` | Enable MoE padding | `0` |
| `SGLANG_FORCE_FP8_MARLIN` | Force FP8 MARLIN kernels | `false` |

### DeepGEMM Configuration

| Variable | Description | Default |
|----------|-------------|---------|
| `SGL_ENABLE_JIT_DEEPGEMM` | Enable JIT DeepGEMM compilation | `"true"` |
| `SGL_JIT_DEEPGEMM_PRECOMPILE` | Enable precompilation | `"true"` |
| `SGL_JIT_DEEPGEMM_COMPILE_WORKERS` | Parallel compilation workers | `4` |
| `SGL_DG_CACHE_DIR` | DeepGEMM cache directory | `~/.cache/deep_gemm` |

### Distributed Computing

| Variable | Description | Default |
|----------|-------------|---------|
| `SGLANG_BLOCK_NONZERO_RANK_CHILDREN` | Block non-zero rank children | `1` |
| `SGL_IS_FIRST_RANK_ON_NODE` | First rank on node indicator | `"true"` |
| `SGLANG_PP_LAYER_PARTITION` | Pipeline parallel layer partition | Not set |

### Testing & Debugging

| Variable | Description | Default |
|----------|-------------|---------|
| `SGLANG_IS_IN_CI` | Running in CI environment | `false` |
| `SGLANG_AMD_CI` | Running in AMD CI | `0` |
| `SGLANG_RECORD_STEP_TIME` | Record step time profiling | `false` |
| `SGLANG_TORCH_PROFILER_DIR` | PyTorch profiler output dir | `/tmp` |

**Full reference:** `docs/references/environment_variables.md`

---

## Troubleshooting

### Common Issues

#### 1. Pre-commit Hook Failures

**Problem:** Pre-commit checks fail

**Solution:**
```bash
# Re-run to apply fixes
pre-commit run --all-files

# If still failing, check specific tool
pre-commit run black --all-files
pre-commit run isort --all-files
```

#### 2. Import Errors After Installation

**Problem:** `ModuleNotFoundError: No module named 'sglang'`

**Solution:**
```bash
# Ensure editable install
pip install -e "python[dev]"

# Check installation
python3 -c "import sglang; print(sglang.__version__)"
```

#### 3. CUDA/Kernel Compilation Errors

**Problem:** sgl-kernel fails to build

**Solution:**
```bash
# Check CUDA version
nvcc --version

# Ensure PyTorch CUDA version matches
python3 -c "import torch; print(torch.version.cuda)"

# Build with verbose output
cd sgl-kernel
pip install -e . -v
```

#### 4. Test Failures

**Problem:** Tests fail locally but should pass

**Solution:**
```bash
# Clean torch compilation cache
rm -rf ~/.cache/torch_extensions

# Run single test with verbose output
python3 -m unittest test.srt.test_srt_endpoint.TestSRTEndpoint.test_simple_decode -v

# Check environment variables
env | grep SGLANG
```

#### 5. CI Workflow Not Triggering

**Problem:** PR doesn't run CI

**Solution:**
- Ensure PR has `run-ci` label
- Ensure PR is not in draft mode
- Check that modified files match path filters in `.github/workflows/pr-test.yml`

### Getting Help

- **Slack Community:** https://slack.sglang.ai/
- **GitHub Issues:** https://github.com/sgl-project/sglang/issues
- **Documentation:** https://docs.sglang.ai/
- **Bi-Weekly Dev Meeting:** https://meeting.sglang.ai/
- **Contribution Guide:** `docs/developer_guide/contribution_guide.md`
- **Code Walk-through:** https://github.com/zhaochenyang20/Awesome-ML-SYS-Tutorial/tree/main/sglang/code-walk-through

---

## Quick Reference Commands

```bash
# Setup
git clone https://github.com/<your_username>/sglang.git
cd sglang
pip install -e "python[dev]"
pre-commit install

# Code Quality
pre-commit run --all-files
make format

# Testing
cd test/srt && python3 run_suite.py --suite per-commit
python3 -m sglang.test.few_shot_gsm8k --num-questions 200

# Server
python3 -m sglang.launch_server --model Qwen/Qwen2-7B-Instruct

# Version Management
make update <new_version>

# Documentation
cd docs && pip install -r requirements.txt && make html
```

---

## Additional Resources

- **Roadmap:** https://github.com/sgl-project/sglang/issues/7736
- **Blog Posts:** https://lmsys.org/blog (v0.2, v0.3, v0.4 releases)
- **Learning Materials:** https://github.com/sgl-project/sgl-learning-materials
- **Supported Models:** `docs/supported_models/`
- **Benchmark Examples:** `benchmark/` (38+ categories)

---

**Last Updated:** 2025-01-04
**Maintained by:** SGLang Community
**For updates:** Check git history or submit PR to update this document
