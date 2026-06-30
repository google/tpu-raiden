# Raiden Examples

## Overview
This `examples/` folder contains scripts, microbenchmarks, and end-to-end serving demos demonstrating how to interact with and evaluate TPU Raiden.

## Available Examples

*   [Microbenchmarks](microbenchmarks/README.md): Microbenchmarks designed to test the raw DMA bandwidth (Host-to-Device and Device-to-Host) of the Raiden engine without Python or framework-level overhead.
*   [Single-host Disaggregated Serving](single_host_disagg/README.md): An end-to-end demo of disaggregated LLM serving on a single TPU VM, using `tpu-raiden` to transfer KV-cache between two local TPU chips (prefill on chip 0, decode on chip 1).
*   [Multi-host Disaggregated Serving](multihost_disagg/README.md): An end-to-end demo of disaggregated LLM serving across two TPU VMs, using `tpu-raiden` to transfer KV-cache over the network from the prefill VM to the decode VM.
*   [Single-host KV Host Offloading](kv_host_offloading/README.md): An end-to-end demo of prefix-cache offloading from TPU HBM to host RAM on a single TPU VM, using `tpu-raiden` to dynamically save and load KV-cache blocks.

## How to Run
For specific execution details, prerequisites, and instructions on how to interpret outputs, please refer to the README inside each specific directory.
