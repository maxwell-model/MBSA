# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT

"""Compile-time configurations for the SM100 sparse-attention forward kernel."""

from dataclasses import dataclass, replace
from enum import Enum


MSA_OUTPUT_TILE_ROWS = 128
MSA_CLC_MIN_KV_TOKENS = 16 * 1024


class QLoad(str, Enum):
    """Global-to-shared Q transfer used by the load warp group."""

    CP_ASYNC = "cpasync"
    TMA = "cpasync_bulk"


class OutputStore(str, Enum):
    """Final O/LSE global-store mechanism."""

    STG = "stg"
    TMA = "utmastg"


class OutputLayout(str, Enum):
    """Partial-output layout consumed by combine."""

    SCATTERED = "scattered"
    SHUFFLED = "shuffled"


class Scheduler(str, Enum):
    """CTA work-assignment policy."""

    STATIC = "static"
    CLC = "clc"


class WorkOrder(str, Enum):
    """Host-side descriptor order within a schedule."""

    NATIVE = "native"
    LPT = "lpt"


@dataclass(frozen=True)
class Cfg:
    """Compile-time choices shared by four- and five-WG MSA kernels."""

    regs: tuple[int, ...]
    q_stages: int
    kv_stages: int
    full_kv_blocks: bool = False
    q_load: QLoad = QLoad.TMA
    output_store: OutputStore = OutputStore.STG
    output_layout: OutputLayout = OutputLayout.SCATTERED
    scheduler: Scheduler = Scheduler.STATIC
    ragged_tma_store: bool | None = None

    def __post_init__(self) -> None:
        choices = (
            ("q_load", self.q_load, QLoad),
            ("output_store", self.output_store, OutputStore),
            ("output_layout", self.output_layout, OutputLayout),
            ("scheduler", self.scheduler, Scheduler),
        )
        for name, value, choice_type in choices:
            if not isinstance(value, choice_type):
                raise TypeError(f"{name} must be {choice_type.__name__}, got {value!r}")
        if self.ragged_tma_store is None:
            object.__setattr__(
                self,
                "ragged_tma_store",
                self.output_store is OutputStore.TMA,
            )
        if len(self.regs) not in (4, 5):
            raise ValueError("MSA requires four or five warp groups")
        if any(reg < 24 or reg > 256 or reg % 8 for reg in self.regs):
            raise ValueError(
                "each warp-group register target must be 24..256 and divisible by 8"
            )
        if 128 * sum(self.regs) > 65536:
            raise ValueError(
                "warp-group register targets exceed the 65,536-register CTA limit"
            )
        if self.q_stages < 1 or self.kv_stages < 1:
            raise ValueError("pipeline stage counts must be positive")
        if (
            self.use_clc
            and self.kv_stages != 2
            and self.output_store is not OutputStore.TMA
        ):
            raise ValueError("CLC requires a two-stage K/V pipeline")
        if self.output_layout is OutputLayout.SHUFFLED and not self.full_kv_blocks:
            raise ValueError("shuffled output requires full K/V blocks")
        if (
            self.output_store is OutputStore.TMA
            and (
                self.output_layout is not OutputLayout.SHUFFLED
                or self.kv_stages != 1
            )
        ):
            raise ValueError(
                "UTMASTG requires shuffled partial-output rows and one K/V stage"
            )
        if self.ragged_tma_store and self.output_store is not OutputStore.TMA:
            raise ValueError("ragged TMA output requires UTMASTG")

    @property
    def num_wgs(self) -> int:
        return len(self.regs)

    @property
    def use_clc(self) -> bool:
        return self.scheduler is Scheduler.CLC

    @property
    def symbol_suffix(self) -> str:
        regs = "_".join(str(reg) for reg in self.regs)
        data = "full_ldtm" if self.full_kv_blocks else "generic"
        return (
            f"{self.num_wgs}wg_{self.q_load.value}_{self.output_store.value}_"
            f"{self.output_layout.value}_{self.scheduler.value}_"
            f"{data}_q{self.q_stages}_kv{self.kv_stages}_r{regs}"
            f"{'_ragged' if self.ragged_tma_store else ''}"
        )


@dataclass(frozen=True)
class MsaVariant:
    """One benchmark cell: a device kernel plus its host work order."""

    cfg: Cfg
    work_order: WorkOrder

    def __post_init__(self) -> None:
        if not isinstance(self.cfg, Cfg):
            raise TypeError(f"cfg must be Cfg, got {self.cfg!r}")
        if not isinstance(self.work_order, WorkOrder):
            raise TypeError(
                f"work_order must be WorkOrder, got {self.work_order!r}"
            )

    @property
    def use_lpt(self) -> bool:
        return self.work_order is WorkOrder.LPT

    @property
    def name(self) -> str:
        return f"{self.cfg.symbol_suffix}_{self.work_order.value}"


MSA_4WG_STATIC = Cfg(
    regs=(192, 192, 80, 48),
    q_stages=2,
    kv_stages=1,
    full_kv_blocks=True,
)

MSA_4WG_STATIC_CAUSAL = Cfg(
    regs=(176, 176, 112, 48),
    q_stages=2,
    kv_stages=1,
)

MSA_4WG_STATIC_GENERIC = Cfg(
    regs=MSA_4WG_STATIC.regs,
    q_stages=2,
    kv_stages=1,
)

MSA_4WG_CLC = Cfg(
    regs=(192, 192, 32, 96),
    q_stages=2,
    kv_stages=2,
    full_kv_blocks=True,
    scheduler=Scheduler.CLC,
)

MSA_5WG_STATIC = Cfg(
    # A 640-thread launch reserves at most REG96 per thread. Role-specific
    # setmaxregister operations redistribute that 61,440-register pool after
    # all five warp groups have diverged.
    regs=(184, 104, 104, 24, 64),
    q_stages=2,
    kv_stages=2,
    full_kv_blocks=True,
)

MSA_PRODUCTION_CLC_MIN_KV_TOKENS = 32 * 1024


def make_msa_variant(
    *,
    q_load: QLoad,
    output_store: OutputStore,
    output_layout: OutputLayout,
    num_wgs: int,
    scheduler: Scheduler,
    work_order: WorkOrder,
    ragged_tma_store: bool | None = None,
) -> MsaVariant:
    """Build one legal benchmark variant with its owned resource policy."""
    if num_wgs == 4:
        if scheduler is Scheduler.CLC and output_store is OutputStore.TMA:
            # UTMASTG keeps the store pipeline live in WG3.  A symmetric
            # consumer/store split uses the full 65,536-register budget and
            # avoids local-memory traffic in the persistent kernel.
            regs = (160, 160, 32, 160)
        else:
            regs = (
                MSA_4WG_CLC.regs
                if scheduler is Scheduler.CLC
                else MSA_4WG_STATIC.regs
            )
        kv_stages = (
            1
            if output_store is OutputStore.TMA
            else 2 if scheduler is Scheduler.CLC else 1
        )
    elif num_wgs == 5:
        if q_load is QLoad.CP_ASYNC:
            # LDGSTS needs one more allocation quantum in the Q-load group.
            # The matching donations also remove the residual epilogue/MMA
            # local traffic while retaining the 61,440-register launch pool.
            regs = (184, 96, 96, 32, 72)
        elif output_layout is OutputLayout.SHUFFLED:
            # Shuffled output keeps four score fragments live through the
            # max/exp path. Donate one post-divergence quantum from WG4 to WG0;
            # the launch still reserves the same 61,440 registers. The same
            # split is stable for persistent UTMASTG when every validation and
            # timing launch uses its owning CUDA stream.
            regs = (184, 96, 96, 24, 80)
        else:
            regs = MSA_5WG_STATIC.regs
        kv_stages = (
            1
            if output_store is OutputStore.TMA
            else 2 if scheduler is Scheduler.CLC else 1
        )
    else:
        raise ValueError(f"num_wgs must be 4 or 5, got {num_wgs}")
    return MsaVariant(
        cfg=Cfg(
            regs=regs,
            q_stages=2,
            kv_stages=kv_stages,
            full_kv_blocks=True,
            q_load=q_load,
            output_store=output_store,
            output_layout=output_layout,
            scheduler=scheduler,
            ragged_tma_store=ragged_tma_store,
        ),
        work_order=work_order,
    )


MSA_5WG_STATIC_LDGSTS_SHUFFLED = make_msa_variant(
    q_load=QLoad.CP_ASYNC,
    output_store=OutputStore.STG,
    output_layout=OutputLayout.SHUFFLED,
    num_wgs=5,
    scheduler=Scheduler.STATIC,
    work_order=WorkOrder.LPT,
).cfg

MSA_5WG_STATIC_UTMALDG_SCATTERED = make_msa_variant(
    q_load=QLoad.TMA,
    output_store=OutputStore.STG,
    output_layout=OutputLayout.SCATTERED,
    num_wgs=5,
    scheduler=Scheduler.STATIC,
    work_order=WorkOrder.LPT,
).cfg

MSA_5WG_STATIC_MASKED = replace(
    MSA_5WG_STATIC_UTMALDG_SCATTERED,
    full_kv_blocks=False,
)

MSA_5WG_CLC_UTMALDG_SHUFFLED = make_msa_variant(
    q_load=QLoad.TMA,
    output_store=OutputStore.STG,
    output_layout=OutputLayout.SHUFFLED,
    num_wgs=5,
    scheduler=Scheduler.CLC,
    work_order=WorkOrder.NATIVE,
).cfg

MSA_5WG_CLC_UTMALDG_SCATTERED = make_msa_variant(
    q_load=QLoad.TMA,
    output_store=OutputStore.STG,
    output_layout=OutputLayout.SCATTERED,
    num_wgs=5,
    scheduler=Scheduler.CLC,
    work_order=WorkOrder.NATIVE,
).cfg

MSA_5WG_CLC_MASKED = replace(
    MSA_5WG_CLC_UTMALDG_SCATTERED,
    # Mask-before-max keeps additional logits live in the softmax group.
    # Donate eight registers from each store group without changing the
    # 61,440-register launch pool or the full-block configuration.
    regs=(200, 96, 96, 24, 64),
    full_kv_blocks=False,
)


def _validate_msa_operand_roles(cfg: Cfg | None, requires_kv_staging: bool) -> None:
    """Reject explicit schedules without the required K/V conversion roles."""
    if cfg is not None and requires_kv_staging and (cfg.num_wgs == 5 or cfg.use_clc):
        raise ValueError(
            "K/V storage conversion requires the static four-WG dequantization roles"
        )


def select_msa_cfg(
    *,
    causal: bool,
    full_kv_blocks: bool,
    sm: tuple[int, int],
    max_seqlen_k: int,
    paged_kv: bool = False,
    qhead_per_kv: int = 8,
    requires_kv_staging: bool = False,
    causal_registers: tuple[int, int] = (176, 112),
) -> Cfg:
    """Select an SM100/SM103 schedule from host-visible shape properties.

    Args:
        causal: Whether Dynamic attention still requires causal masking.
        full_kv_blocks: Whether preparation proved every consumed K/V block
            fully visible and block-aligned for the single request.
        sm: CUDA device capability as a (major, minor) pair.
        max_seqlen_k: Maximum logical K/V length in tokens, used to select
            static versus persistent scheduling for eligible full blocks.
        paged_kv: Whether K/V uses a page table. Eligibility must already
            account for valid page lengths; this does not prove full blocks.
        qhead_per_kv: Query heads sharing each K/V head. Ratios 1, 2 and 4
            use hardware gather4 Q transport; the default ratio 8 and ratio
            16 use ordinary Q TMA. Both transports support five warpgroups.
        requires_kv_staging: Whether K or V storage needs conversion before
            MMA. Such requests retain the four-warpgroup dequantization roles.
        causal_registers: Softmax/store register targets for the causal path;
            defaults to (176, 112).

    Returns:
        A compile-time configuration. Instruction capabilities are selected
        separately; full-block geometry does not imply TMEM load-reduce support.
    """

    if causal:
        softmax_regs, store_regs = causal_registers
        if causal_registers not in ((176, 112), (192, 80)):
            raise ValueError(
                f"unsupported causal register policy: {causal_registers}"
            )
        return Cfg(
            regs=(
                softmax_regs,
                softmax_regs,
                store_regs,
                512 - softmax_regs * 2 - store_regs,
            ),
            q_stages=2,
            kv_stages=1,
        )
    if requires_kv_staging:
        return MSA_4WG_STATIC_GENERIC
    five_wg_q_tma = qhead_per_kv in (1, 2, 4, 8, 16)
    if full_kv_blocks and five_wg_q_tma and sm in ((10, 0), (10, 3)):
        if max_seqlen_k < MSA_PRODUCTION_CLC_MIN_KV_TOKENS:
            return MSA_5WG_STATIC_UTMALDG_SCATTERED
        return MSA_5WG_CLC_UTMALDG_SCATTERED
    if five_wg_q_tma and sm in ((10, 0), (10, 3)):
        if max_seqlen_k < MSA_PRODUCTION_CLC_MIN_KV_TOKENS:
            return MSA_5WG_STATIC_MASKED
        return MSA_5WG_CLC_MASKED
    return MSA_4WG_STATIC_GENERIC
