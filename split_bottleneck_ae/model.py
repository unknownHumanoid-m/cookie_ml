"""
Split-bottleneck autoencoder with a joint count classifier and two phase
regressors (single-pulse phi0 and two-pulse arccos(cos Δφ)).

Architecture
------------
Encoder and decoder are the SAME 2D-conv stacks used by the working
raw-input denoiser in src/denoising (`Ximg_to_Ypdf_Autoencoder`). The
pretrained raw-AE weights load directly into this model — the key names
`encoder.{0,2,4}.weight` and `decoder.{0,2,4}.weight` match the raw AE's
state dict.

The encoder emits a (N, 64, 18, 514) spatial feature map (see
`project-conv-ae-shape` memory). No recon-side bottleneck is inserted —
``feat`` feeds the decoder directly, so the pretrained raw-AE inverse
holds from step zero. Two task-side bottlenecks branch off ``feat``:

  * ``bottleneck_count``          — plain global-avg-pool path.
                                    feat -> mean over (H, W) -> (N, 64) ->
                                    Linear(64 -> B_count). Zero pooling
                                    params. Kept off the phase adapter
                                    because the count head doesn't need
                                    time-axis structure and paying the
                                    C*L flatten dim on FPGA for it is waste.
  * ``bottleneck_phase``          — feat -> TimeAdaptivePoolAdapter ->
                                    Linear(C*L -> B_phase); feeds BOTH
                                    phase heads. Keeping both heads on the
                                    same phase bottleneck lets the trunk
                                    learn one phase-relevant repr, and
                                    mirrors the on-FPGA gating pattern
                                    where the count classifier picks which
                                    head to trust.

The adapter pools the encoder feature to (N, C, L) via
adaptive_avg_pool1d over the time axis (L defaults to 16), then flattens
to (N, C*L). This keeps coarse "where along the pulse train" info alive
for the phase heads while staying tiny compared to the encoder.

At export time the decoder is stripped; only encoder ->
{GAP -> bottleneck_count, adapter -> bottleneck_phase} -> heads ships.
"""

from typing import List, Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


# --------------------------------------------------------------------------
# Encoder / decoder layer stacks — mirror
# src/denoising/ximg_to_ypdf_autoencoder_straight_training.py exactly so
# the pretrained raw-AE state_dict loads with matching key names.
# --------------------------------------------------------------------------
def _build_encoder() -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(1, 16, kernel_size=3, padding=2),
        nn.ReLU(),
        nn.Conv2d(16, 32, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.Conv2d(32, 64, kernel_size=3, padding=1),
        nn.ReLU(),
    )


def _build_decoder(output_activation: Optional[nn.Module]) -> nn.Sequential:
    layers: List[nn.Module] = [
        nn.ConvTranspose2d(64, 32, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.ConvTranspose2d(32, 16, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.ConvTranspose2d(16, 1, kernel_size=3, padding=2),
    ]
    if output_activation is not None:
        layers.append(output_activation)
    return nn.Sequential(*layers)


def _resolve_output_activation(name: str) -> Optional[nn.Module]:
    name = (name or "none").lower()
    if name == "sigmoid":
        return nn.Sigmoid()
    if name == "tanh":
        return nn.Tanh()
    if name in ("none", "", "linear", "identity"):
        return None
    raise ValueError(f"Unknown decoder_output_activation: {name!r}")


def _make_mlp(in_dim: int, hidden: Sequence[int], out_dim: int) -> nn.Sequential:
    layers: List[nn.Module] = []
    prev = int(in_dim)
    for h in hidden:
        layers.append(nn.Linear(prev, int(h)))
        layers.append(nn.ReLU(inplace=True))
        prev = int(h)
    layers.append(nn.Linear(prev, int(out_dim)))
    return nn.Sequential(*layers)


class TimeAdaptivePoolAdapter(nn.Module):
    """(N, C, H, W) -> (N, C * pool_len) preserving coarse W-axis info.

    Rationale: full GAP collapses H and W to a single number per channel,
    which throws away *where along the pulse train* each pulse landed —
    exactly the signal the phase heads need. Instead:

        feat (N, C, H, W)
          -> mean over rows              -> (N, C, W)
          -> adaptive_avg_pool1d(W -> L) -> (N, C, L)
          -> flatten                     -> (N, C * L)

    L is small (default 16) so C*L (= 64*16 = 1024 for the default encoder)
    stays tiny compared to the encoder itself, but the phase / count
    Linears now see a spatial signature instead of a scalar.
    """

    def __init__(self, in_channels: int, pool_len: int):
        super().__init__()
        self.in_channels = int(in_channels)
        self.pool_len = int(pool_len)
        self.output_dim = int(in_channels) * int(pool_len)

    def forward(self, feat: torch.Tensor) -> torch.Tensor:
        # feat: (N, C, H, W). Collapse rows, then adaptive-pool along W.
        row_mean = feat.mean(dim=2)                              # (N, C, W)
        pooled = F.adaptive_avg_pool1d(row_mean, self.pool_len)  # (N, C, L)
        return pooled.flatten(start_dim=1)                       # (N, C*L)


class SplitBottleneckAE(nn.Module):
    """2D-conv encoder + three-way bottleneck fanout. See module docstring.

    forward(x) returns a dict with:
        * ``recon``            — (N, rows, cols), decoder output (only when
                                 run_recon=True)
        * ``count_logits``     — (N, num_count_classes)
        * ``phase_single_out`` — (N, 2), (sin, cos) of phi0
        * ``phase_two_out``    — (N, 1), scalar arccos(cos Δφ) prediction
        * ``z_count``, ``z_phase`` — task bottleneck outputs (debugging)
        * ``feat``             — full encoder feature map (debugging)
    """

    ENCODER_OUT_CHANNELS = 64

    def __init__(
        self,
        input_shape,                # (rows, cols)
        bottleneck_count,
        bottleneck_phase,
        count_head_hidden,
        phase_head_single_hidden,
        phase_head_two_hidden,
        decoder_output_activation,
        num_count_classes,
        phase_single_output_dim,
        phase_two_output_dim,
        adapter_pool_len,           # coarse time-axis length after pool
    ):
        super().__init__()
        rows, cols = int(input_shape[0]), int(input_shape[1])
        self.input_shape = (rows, cols)

        # Top-level names `encoder` / `decoder` match the pretrained raw-AE
        # state_dict layout so weights load without prefix surgery. No recon
        # bottleneck is inserted between them — feat goes straight into the
        # decoder so the pretrained raw-AE inverse holds from step zero.
        self.encoder = _build_encoder()
        self.decoder = _build_decoder(
            _resolve_output_activation(decoder_output_activation)
        )

        # Phase branch: pool the encoder feature along the time axis to a
        # (64 * pool_len)-D vector so the phase heads see coarse "where
        # along the pulse train this happened" info instead of a scalar.
        self.adapter = TimeAdaptivePoolAdapter(
            in_channels=self.ENCODER_OUT_CHANNELS,
            pool_len=int(adapter_pool_len),
        )
        tapped_dim = self.adapter.output_dim
        self.bottleneck_phase = nn.Linear(tapped_dim, int(bottleneck_phase))

        # Count branch: plain global average pool, no time-axis structure.
        # feat -> mean over (H, W) -> (N, 64) -> Linear(64 -> B_count).
        self.bottleneck_count = nn.Linear(
            self.ENCODER_OUT_CHANNELS, int(bottleneck_count)
        )

        self.count_head = _make_mlp(
            in_dim=int(bottleneck_count),
            hidden=count_head_hidden,
            out_dim=int(num_count_classes),
        )
        self.phase_head_single = _make_mlp(
            in_dim=int(bottleneck_phase),
            hidden=phase_head_single_hidden,
            out_dim=int(phase_single_output_dim),
        )
        self.phase_head_two = _make_mlp(
            in_dim=int(bottleneck_phase),
            hidden=phase_head_two_hidden,
            out_dim=int(phase_two_output_dim),
        )

        self.num_count_classes = int(num_count_classes)
        self.phase_single_output_dim = int(phase_single_output_dim)
        self.phase_two_output_dim = int(phase_two_output_dim)
        self.bottleneck_count_dim = int(bottleneck_count)
        self.bottleneck_phase_dim = int(bottleneck_phase)

    # ------------------------------------------------------------------
    # Convenience: load pretrained raw-AE weights into encoder + decoder.
    # ------------------------------------------------------------------
    def load_pretrained_denoiser(self, weights_path: str, map_location=None):
        """Copy encoder / decoder weights from a raw-AE checkpoint.

        The raw AE state_dict has keys ``encoder.{0,2,4}.{weight,bias}`` and
        ``decoder.{0,2,4}.{weight,bias}``. This loader restricts to those
        keys — task-branch params keep their fresh init.
        """
        sd = torch.load(weights_path, map_location=map_location)
        if isinstance(sd, dict) and "state_dict" in sd and not any(
            k.startswith(("encoder.", "decoder.")) for k in sd.keys()
        ):
            sd = sd["state_dict"]
        filtered = {
            k: v for k, v in sd.items()
            if k.startswith("encoder.") or k.startswith("decoder.")
        }
        missing = [k for k in self.state_dict()
                   if (k.startswith("encoder.") or k.startswith("decoder."))
                   and k not in filtered]
        if missing:
            raise RuntimeError(
                f"load_pretrained_denoiser: missing keys {missing[:5]}... "
                f"in {weights_path}"
            )
        self.load_state_dict(filtered, strict=False)
        print(
            f"[model] loaded pretrained encoder/decoder from {weights_path} "
            f"({len(filtered)} tensors)"
        )

    # ------------------------------------------------------------------
    # Forward passes
    # ------------------------------------------------------------------
    def _ensure_channel_dim(self, x: torch.Tensor) -> torch.Tensor:
        # Loader delivers (N, rows, cols); the conv encoder wants (N, 1, r, c).
        if x.dim() == 3:
            return x.unsqueeze(1)
        return x

    def forward(
        self,
        x: torch.Tensor,
        run_recon: bool = True,
        run_task: bool = True,
    ):
        rows, cols = self.input_shape
        x = self._ensure_channel_dim(x)
        feat = self.encoder(x)  # (N, 64, 18, 514)

        recon = None
        if run_recon:
            recon_img = self.decoder(feat)                 # (N, 1, rows, cols)
            recon = recon_img.squeeze(1).view(-1, rows, cols)

        z_count = None
        z_phase = None
        count_logits = None
        phase_single_out = None
        phase_two_out = None
        if run_task:
            gap = feat.mean(dim=(2, 3))                    # (N, 64)
            z_count = self.bottleneck_count(gap)
            tapped = self.adapter(feat)                    # (N, 64*L)
            z_phase = self.bottleneck_phase(tapped)
            count_logits = self.count_head(z_count)
            phase_single_out = self.phase_head_single(z_phase)
            phase_two_out = self.phase_head_two(z_phase)

        return {
            "recon": recon,
            "count_logits": count_logits,
            "phase_single_out": phase_single_out,
            "phase_two_out": phase_two_out,
            "feat": feat,
            "z_count": z_count,
            "z_phase": z_phase,
        }


class InferenceSubgraph(nn.Module):
    """Deployed subgraph: encoder -> {GAP -> bottleneck_count -> count_head,
    adapter -> bottleneck_phase -> phase heads}.

    This is the FPGA target. Structurally identical to
    ``SplitBottleneckAE.forward(run_recon=False)`` but references only the
    parameters that ship on FPGA, so it traces / scripts cleanly.
    """

    def __init__(self, full_model: SplitBottleneckAE):
        super().__init__()
        self.input_shape = full_model.input_shape
        self.encoder = full_model.encoder
        self.adapter = full_model.adapter
        self.bottleneck_count = full_model.bottleneck_count
        self.bottleneck_phase = full_model.bottleneck_phase
        self.count_head = full_model.count_head
        self.phase_head_single = full_model.phase_head_single
        self.phase_head_two = full_model.phase_head_two
        self.num_count_classes = full_model.num_count_classes
        self.phase_single_output_dim = full_model.phase_single_output_dim
        self.phase_two_output_dim = full_model.phase_two_output_dim
        self.bottleneck_count_dim = full_model.bottleneck_count_dim
        self.bottleneck_phase_dim = full_model.bottleneck_phase_dim

    def forward(self, x: torch.Tensor):
        if x.dim() == 3:
            x = x.unsqueeze(1)
        feat = self.encoder(x)
        gap = feat.mean(dim=(2, 3))
        tapped = self.adapter(feat)
        z_count = self.bottleneck_count(gap)
        z_phase = self.bottleneck_phase(tapped)
        return (
            self.count_head(z_count),
            self.phase_head_single(z_phase),
            self.phase_head_two(z_phase),
        )


def build_model_from_config(config_module) -> SplitBottleneckAE:
    """Convenience factory used by train / eval / export scripts."""
    m = config_module.MODEL
    d = config_module.DATA
    return SplitBottleneckAE(
        input_shape=d["image_shape"],
        bottleneck_count=m["bottleneck_count"],
        bottleneck_phase=m["bottleneck_phase"],
        count_head_hidden=m["count_head_hidden"],
        phase_head_single_hidden=m["phase_head_single_hidden"],
        phase_head_two_hidden=m["phase_head_two_hidden"],
        decoder_output_activation=m["decoder_output_activation"],
        num_count_classes=config_module.num_count_classes(),
        phase_single_output_dim=config_module.PHASE_SINGLE_OUTPUT_DIM,
        phase_two_output_dim=config_module.PHASE_TWO_OUTPUT_DIM,
        adapter_pool_len=m["adapter_pool_len"],
    )
