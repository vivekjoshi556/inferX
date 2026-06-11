# InferX

Building a LLM inference engine from scratch in C++ and CUDA.

---

# Vision

InferX is a systems engineering project focused on understanding and implementing the complete LLM inference stack.

The primary target is:

* Llama-family models
* Single NVIDIA T4 GPU
* Native C++ runtime
* CUDA kernels
* Python bindings

InferX is not intended to be a general-purpose deep learning framework.

The objective is to understand and optimize the components that make modern inference engines such as vLLM, TensorRT-LLM and llama.cpp successful.

---

# Long-Term Goals

## Stage 1

Build a complete Llama inference stack.

Capabilities:

* Load model checkpoints
* Execute forward pass
* Generate tokens
* Python API

---

## Stage 2

Build a production-style runtime.

Capabilities:

* KV cache management
* Continuous batching
* Runtime scheduling
* Memory pooling

---

## Stage 3

Build specialized optimizations.

Capabilities:

* Custom GEMM kernels
* Fused kernels
* Optimized attention
* Quantized execution

---

## Stage 4

Demonstrate measurable advantages on at least one benchmark.

Target competitors:

* vLLM
* llama.cpp

Target hardware:

* NVIDIA T4

---

# Development Roadmap

---

## Milestone 0 — Foundation

Repository setup.

### Setup

* [x] CMake build system
* [x] Python package scaffold
* [x] PyBind11 integration

### Optional TODOs:

* [ ] CI pipeline
* [ ] Formatting

---

## Milestone 1 — Model Loader

Load and inspect Llama checkpoints.

### Config Loader

* [ ] Parse config.json
* [ ] Validate architecture
* [ ] Extract model metadata

### Weight Loader

* [ ] Load checkpoint tensors
* [ ] Tensor mapping
* [ ] Device upload

### Validation

* [ ] Shape verification
* [ ] Tensor inspection tools
* [ ] Checkpoint consistency checks

### Success Criteria

InferX can load a Llama checkpoint and print all model parameters and tensor shapes.

---

## Milestone 2 — Transformer Components

Implement transformer building blocks.

### RMSNorm

* [ ] CPU implementation
* [ ] CUDA implementation

### RoPE

* [ ] CPU implementation
* [ ] CUDA implementation

### MLP

* [ ] Gate projection
* [ ] Up projection
* [ ] Down projection

### Attention

* [ ] Q projection
* [ ] K projection
* [ ] V projection
* [ ] Output projection

### Validation

* [ ] Match HuggingFace reference outputs

---

## Milestone 3 — Single Layer Execution

Execute one complete transformer block.

### Tasks

* [ ] Layer forward pass
* [ ] Intermediate buffer management
* [ ] CPU backend
* [ ] CUDA backend

### Validation

* [ ] Match HuggingFace hidden states

---

## Milestone 4 — Full Model Inference

Execute the complete model.

### Tasks

* [ ] Token embedding
* [ ] Layer execution
* [ ] Final RMSNorm
* [ ] LM head

### Validation

* [ ] Match HuggingFace logits

### Success Criteria

InferX produces logits matching the reference implementation.

---

## Milestone 5 — Token Generation

Build decoding support.

### Tasks

* [ ] Greedy decoding
* [ ] Sampling
* [ ] Generation loop

### Validation

* [ ] Reproduce expected outputs

---

## Milestone 6 — KV Cache System

Build decoding memory infrastructure.

### Version 1

* [ ] Contiguous KV cache

### Version 2

* [ ] Block allocator
* [ ] Cache reuse

### Version 3

* [ ] Paged KV cache
* [ ] Fragmentation analysis

---

## Milestone 7 — Runtime Engine

Build request execution infrastructure.

### Request Management

* [ ] Request lifecycle
* [ ] Token tracking
* [ ] Completion handling

### Execution

* [ ] Decode loop
* [ ] Batch execution
* [ ] Memory coordination

---

## Milestone 8 — Scheduler

Build vLLM-style scheduling primitives.

### Static Batching

* [ ] Fixed-size batching

### Dynamic Batching

* [ ] Request merging

### Continuous Batching

* [ ] Token-level scheduling

---

## Milestone 9 — Backend Abstraction

Support multiple execution backends.

### CPU Backend

* [ ] Reference kernels
* [ ] Correctness testing

### CUDA Backend

* [ ] Production kernels
* [ ] Memory management

### Validation

* [ ] Backend parity tests

---

## Milestone 10 — Python API

Expose InferX to Python.

### Bindings

* [ ] pybind11 integration
* [ ] Tensor exchange

### User API

* [ ] Model class
* [ ] Generate API
* [ ] Streaming API

---

## Milestone 11 — Benchmark Suite

Every optimization must be measurable.

### Metrics

* [ ] Prefill latency
* [ ] Decode latency
* [ ] Tokens/sec
* [ ] Memory usage
* [ ] GPU utilization

### Comparisons

* [ ] InferX vs vLLM
* [ ] InferX vs llama.cpp
* [ ] InferX vs TensorRT-LLM

---

## Milestone 12 — Optimization Phase

Begin only after correctness is established.

### Compute

* [ ] GEMM optimization
* [ ] Kernel fusion
* [ ] Occupancy tuning

### Attention

* [ ] FlashAttention
* [ ] Fused attention kernels

### Runtime

* [ ] Memory pools
* [ ] Allocation reduction
* [ ] Scheduler optimization

---

# Success Definition

InferX is successful when:

* A Llama-family model can be loaded entirely through InferX.
* Tokens can be generated through a Python API.
* The runtime supports continuous batching.
* The engine contains custom CUDA kernels.
* CPU and CUDA backends produce equivalent outputs.
* Every subsystem is benchmarked.
* At least one benchmark demonstrates measurable advantages over an existing inference engine on a T4 GPU.

---

Measure everything.

Optimize only after proving correctness.
