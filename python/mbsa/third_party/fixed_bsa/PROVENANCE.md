# Anthony SM100 block-sparse attention source

This directory is a focused source import from FlashInfer commit
`22a829c3fd2e5fe9aecab41c0e6b9b1118e13dd2`, reached through Anthony Chang's
`anchengc/block-sparse-gqa-kernel-opt` work.

Imported paths are limited to the block-128 SM100/SM103 CuTe DSL backend and
its local helpers, plus the upstream `LICENSE` and `NOTICE`.  The package is
not wired into production by this commit.

The imported commit adds block-aligned bottom-right causal coordinates but
retains a divergent TMEM-allocation/register-lifecycle program structure under
Compute Sanitizer synccheck.  The independently validated convergent repair is
applied in a separate commit before any backend dispatch is enabled.

Original-Commit: 22a829c3fd2e5fe9aecab41c0e6b9b1118e13dd2
Original-Author: Anthony Chang <27950904+rosenrodt@users.noreply.github.com>

