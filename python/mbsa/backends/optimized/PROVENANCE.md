<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Optimized backend source and licensing scope

This record covers `mbsa.backends.optimized`. The default backend's FA4,
Fireworks and MiniMax helper records describe different component scopes;
their public-source statements are not a clean-room claim for this backend.

## Historical delivery snapshot

[UPSTREAM.json](UPSTREAM.json) retains the historical delivery repository,
revision, source paths, Git blob IDs, derivation relationships and local
SHA-256 values. That historical repository may not be anonymously accessible.
Recording its identity does not assert public availability, nor does a local
hash match independently establish the upstream author's permissions.
The distributed source and notices are self-contained; the snapshot's
internal commit history is not required to import this package.

The manifest's `license` expression describes the aggregate component terms
of the distributed files, not a repository-wide license grant for the
historical snapshot. In particular, the original NVIDIA delivery files
carried All rights reserved headers. Current individual file declarations and
retained [license texts](licenses/) state the distributed component terms.
Source identity, licensing declarations, approval and numerical validation
are distinct; this provenance document does not issue an approval.

## Customer-base host and router preimages

Five records identify source preimages in customer MBSA base
`07ebee24d561190437bae8740fabeea0f506d0b7`, rather than presenting unavailable
intermediate integration revisions as fetchable customer-repository commits:

- `index.py` and `index_base.py` refactor `python/mbsa/cute/prefill.py`.
  They preserve Maxwell Team attribution alongside NVIDIA's adaptations.
- `src/common/pipeline.py` retains the base's corresponding source blob.
- `src/sm100/router.py` and `mx_router.py` adapt the base's corresponding
  routers, including namespace, dispatch and architecture-specific changes.

For derived files, a preimage is not a claim of byte identity. The manifest
retains the previously recorded intermediate revision/path/blob separately,
marked unverified and without assigning it an unconfirmed repository.
The current local hashes describe the actual shipped files. The manifest
itself is not self-hashed, which would introduce a circular digest.

## Remaining earlier-origin limits

The historical snapshot is an intermediate source, not proof of every
earlier origin. The exact Block-Sparse-Attention API revision and exact
CUTLASS FMHA example revision for the two recorded BSA source regions remain
unconfirmed. A dependency revision must not be substituted for the example's
source revision. These limits are preserved in `earlier_origin_gaps`; they
are not resolved merely by compacting the manifest or matching local hashes.

Original third-party notices and license texts remain authoritative for their
code. No original author is replaced by the package's root license.

## Feature scope

The optimized backend is selected explicitly; default backend selection is
unchanged. The low-level NVFP4 KV attention and optional Transformer Engine
quantization helpers are retained as experimental source, not an FP4 Indexer
or a default prepared/serving option. Their presence and source hashes are
not production-quality or safety qualification.
