# SM100 sparse forward runtime

`interface.py` launches BF16 sparse attention and K2 combine.
`sparse_index_utils.py` validates q2k inputs and calls the GPU CSR builder.

- `src/sm100/fwd/`: attention producer, configuration, and combine.
- `src/sm100/prepare_scheduler.py`: split, flattened/LPT, and TMA tile scheduling.
- `src/sm100/prepare_k2q_csr.py`: allocation and lazy CSR-extension wrapper.
- `src/sm100/build_k2q_csr/`: CUDA metadata kernels.
- `src/common/`: required shared CuTe-DSL/TMA/math helpers.

The Maxwell Block Sparse Attention attention pipeline is documented in the root project.
