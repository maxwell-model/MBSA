# Supplemental upstream licenses

The retained Fixed BSA backend includes FlashAttention helpers and an interface
adapted from Block-Sparse-Attention. Their original source copyright headers
remain in `python/mbsa/cute/src/sm100/fixed/`.

These license texts supplement the retained FlashInfer LICENSE and NOTICE;
they do not replace either document or relicense third-party code.

| Project | License | Upstream license blob |
| --- | --- | --- |
| FlashAttention | BSD-3-Clause | `5860e4b33f3d9d85fc636137c559331d51783a5b` |
| Block-Sparse-Attention | BSD-3-Clause | `5ce2ef099d2f700af4bc54fe75e2619c8ccde474` |

Sources: https://github.com/Dao-AILab/flash-attention and
https://github.com/mit-han-lab/Block-Sparse-Attention.
The FlashAttention AUTHORS file is also retained (blob
`e35a781665eafa7421c30241962ef8e49588bffc`). These hashes identify the supplied
notice texts, not the revisions from which the kernel code was imported.
The backend import revision is recorded in `../PROVENANCE.md`.
