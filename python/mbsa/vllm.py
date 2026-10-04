"""Optional vLLM paged-prefill adapter; importing it does not import vLLM."""

from collections import OrderedDict

import torch

from .cutedsl_backend import prepare_cutedsl_pipeline


class PagedPrefill:
    """Bounded eager plans shared serially across model layers.

    Cache keys include exact request geometry and tensor strides. Preparation
    happens on a cache miss; hot calls refresh routing maps and input pointers.
    CUDA-graph serving requires a separately preallocated, stable plan and is
    rejected here instead of allocating or evicting during capture.
    """

    def __init__(self, max_cached_plans=4, *, use_graphs=False, max_cached_graphs=64):
        if max_cached_plans < 1:
            raise ValueError("max_cached_plans must be positive")
        self.max_cached_plans = max_cached_plans
        if max_cached_graphs < 1:
            raise ValueError("max_cached_graphs must be positive")
        self.use_graphs = use_graphs
        self.max_cached_graphs = max_cached_graphs
        self.plans = OrderedDict()
        self.hits = 0
        self.misses = 0

    def __call__(
        self,
        q,
        key_cache,
        value_cache,
        *,
        index_q,
        index_k_cache,
        index_page_indices,
        page_table,
        request_lens,
        out,
        partial_dtype,
        index_q_scales=None,
        **unused,
    ):
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError(
                "MBSA eager adapter cannot run during CUDA graph capture"
            )
        if index_q_scales is not None:
            raise ValueError("MBSA prefill requires BF16 Index-Q/K")
        if any(
            t.dtype != torch.bfloat16
            for t in (q, key_cache, value_cache, index_q, index_k_cache)
        ):
            raise ValueError("MBSA prefill requires BF16 Q/K/V and Index-Q/K")
        lengths = tuple((int(qn), int(kn)) for qn, kn in request_lens if qn > 0)
        if not lengths or sum(qn for qn, _ in lengths) != q.shape[0]:
            raise ValueError("request_lens must cover all query tokens")
        # vLLM uses [pages,128,Hkv,D] slices of an interleaved K/V pool.
        k, v = (tensor.permute(0, 2, 1, 3) for tensor in (key_cache, value_cache))
        ik = index_k_cache.view(-1, 1, 128, 128)
        batch = len(lengths)
        table = page_table[:batch]
        pages = index_page_indices
        if pages.ndim == 2:
            pages = pages[:batch]
        operands = (index_q, ik, q, k, v, pages, table, out)
        key = (
            lengths,
            partial_dtype,
            tuple((tuple(t.shape), t.stride(), t.dtype, t.device) for t in operands),
        )
        pipeline = self.plans.pop(key, None)
        if pipeline is None:
            self.misses += 1
            # Evict before allocating a new workspace, bounding peak memory.
            if len(self.plans) >= self.max_cached_plans:
                self.plans.popitem(last=False)
            common = dict(
                index_page_indices=pages,
                page_table=table,
                partial_dtype=partial_dtype,
                out=out,
            )
            if self.use_graphs:
                staged = (
                    index_q.clone(),
                    q.clone(),
                    pages.clone(),
                    table.clone(),
                    torch.empty_like(out),
                )
                common.update(
                    index_page_indices=staged[2], page_table=staged[3], out=staged[4]
                )
                prepared_iq, prepared_q = staged[:2]
            else:
                prepared_iq, prepared_q = index_q, q
            if batch == 1:
                common["logical_k_tokens"] = lengths[0][1]
            else:
                q_prefix, k_prefix = [0], [0]
                for qn, kn in lengths:
                    q_prefix.append(q_prefix[-1] + qn)
                    k_prefix.append(k_prefix[-1] + kn)
                common["cu_seqlens_q"] = torch.tensor(
                    q_prefix, dtype=torch.int32, device=q.device
                )
                common["cu_seqlens_k"] = torch.tensor(
                    k_prefix, dtype=torch.int32, device=q.device
                )
            pipeline = prepare_cutedsl_pipeline(
                prepared_iq, ik, prepared_q, k, v, **common
            )
            if self.use_graphs:
                pipeline._vllm_staging = staged
                pipeline._vllm_graphs = OrderedDict()
        else:
            self.hits += 1
            if not self.use_graphs:
                pipeline.rebind(
                    index_q,
                    ik,
                    q,
                    k,
                    v,
                    index_page_indices=pages,
                    page_table=table,
                    out=out,
                )
        self.plans[key] = pipeline
        if self.use_graphs:
            return self._run_graph(pipeline, index_q, ik, q, k, v, pages, table, out)
        return pipeline.run()[0]

    def _run_graph(self, pipeline, index_q, ik, q, k, v, pages, table, out):
        staged_iq, staged_q, staged_pages, staged_table, staged_out = (
            pipeline._vllm_staging
        )
        for destination, source in zip(
            (staged_iq, staged_q, staged_pages, staged_table),
            (index_q, q, pages, table),
        ):
            destination.copy_(source)
        pipeline.rebind(
            staged_iq,
            ik,
            staged_q,
            k,
            v,
            index_page_indices=staged_pages,
            page_table=staged_table,
            out=staged_out,
        )
        # Main/Index-K cache pointers identify a model layer. Queries, metadata
        # and output use one stable staging set shared by all layer graphs.
        key = (ik.data_ptr(), k.data_ptr(), v.data_ptr())
        graphs = pipeline._vllm_graphs
        entry = graphs.pop(key, None)
        if entry is None:
            if len(graphs) >= self.max_cached_graphs:
                graphs.popitem(last=False)
            stream = torch.cuda.Stream(device=q.device)
            stream.wait_stream(torch.cuda.current_stream(q.device))
            with torch.cuda.stream(stream):
                for _ in range(2):
                    pipeline.run()
            stream.synchronize()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, stream=stream):
                pipeline.run()
            # Keep caller-owned cache storage alive for the captured pointers.
            entry = graph, (ik, k, v)
        graphs[key] = entry
        entry[0].replay()
        out.copy_(staged_out)
        return out


_ADAPTERS = {}


def prefill(q, key_cache, value_cache, **kwargs):
    """Run a device-local, serially shared eager vLLM prefill adapter."""
    import os

    graph_mode = os.environ.get("MBSA_VLLM_PREFILL_GRAPH", "0")
    if graph_mode not in ("0", "1"):
        raise ValueError("MBSA_VLLM_PREFILL_GRAPH must be 0 or 1")
    key = q.device, graph_mode
    adapter = _ADAPTERS.setdefault(key, PagedPrefill(use_graphs=graph_mode == "1"))
    return adapter(q, key_cache, value_cache, **kwargs)
