"""
Strip the trained model down to the FPGA-deployed subgraph and export it.

At inference time we only need:
    encoder -> {GAP -> bottleneck_count -> count_head,
                adapter -> bottleneck_phase -> {phase_head_single,
                                                phase_head_two}}

The decoder is a training-time artifact and is discarded here (there is
no separate recon bottleneck any more — feat feeds the decoder directly).
The exported artifacts are:

    * ``inference_subgraph.pt``     — TorchScript trace of ``InferenceSubgraph``.
                                       Traced module returns
                                       (count_logits, phase_single_out,
                                        phase_two_out).
    * ``inference_subgraph_state.pth`` — dict with the state_dict of the
                                       stripped subgraph plus the metadata
                                       needed to rebuild it (input_shape,
                                       bottleneck widths, num classes,
                                       phase head output dims).

Also prints a parameter-count comparison (full model vs deployed subgraph)
so it's obvious what was saved by dropping the recon branch.

Run:
    python3 export_inference.py [--ckpt /path/to/model.pth] [--out_dir ...]
"""

import argparse
import json
import os

import torch

import config as cfg
from model import InferenceSubgraph, build_model_from_config


def count_params(module: torch.nn.Module) -> int:
    return sum(p.numel() for p in module.parameters())


def default_ckpt_path():
    return os.path.join(cfg.IO["save_dir"], cfg.IO["run_name"], "model.pth")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, default=None,
                        help="Path to the trained .pth checkpoint. Defaults "
                             "to IO['save_dir']/IO['run_name']/model.pth.")
    parser.add_argument("--out_dir", type=str, default=None,
                        help="Where to write the exported subgraph. Defaults "
                             "to <ckpt_dir>/inference/.")
    args = parser.parse_args()

    device = torch.device("cpu")  # export on CPU — traces are portable.
    ckpt_path = args.ckpt or default_ckpt_path()
    print(f"[export] loading {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)

    model = build_model_from_config(cfg)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    full_params = count_params(model)
    subgraph = InferenceSubgraph(model).eval()
    sub_params = count_params(subgraph)
    dropped = full_params - sub_params
    print(f"[export] full model params      = {full_params:,}")
    print(f"[export] deployed subgraph      = {sub_params:,}")
    print(f"[export] dropped (decoder)      = {dropped:,} "
          f"({100.0 * dropped / max(full_params, 1):.1f}% of the full model)")

    rows, cols = model.input_shape
    # New conv encoder consumes (N, rows, cols) (auto-adds a channel dim);
    # the flat vector shape used by the previous FC trunk is gone.
    example = torch.zeros(1, rows, cols)
    with torch.no_grad():
        count_logits, phase_single_out, phase_two_out = subgraph(example)
    print(f"[export] traced count_logits.shape     = {tuple(count_logits.shape)}")
    print(f"[export] traced phase_single_out.shape = {tuple(phase_single_out.shape)}")
    print(f"[export] traced phase_two_out.shape    = {tuple(phase_two_out.shape)}")

    out_dir = args.out_dir or os.path.join(
        os.path.dirname(os.path.abspath(ckpt_path)), "inference"
    )
    os.makedirs(out_dir, exist_ok=True)

    # TorchScript trace of just the deployed path.
    traced = torch.jit.trace(subgraph, example)
    ts_path = os.path.join(out_dir, "inference_subgraph.pt")
    traced.save(ts_path)
    print(f"[export] TorchScript trace -> {ts_path}")

    # Plain state_dict + metadata (useful for hls4ml which prefers PyTorch
    # nn.Module state to a traced module).
    state_path = os.path.join(out_dir, "inference_subgraph_state.pth")
    torch.save({
        "state_dict": subgraph.state_dict(),
        "input_shape": model.input_shape,
        "num_count_classes": model.num_count_classes,
        "phase_single_output_dim": model.phase_single_output_dim,
        "phase_two_output_dim": model.phase_two_output_dim,
        "bottleneck_count_dim": model.bottleneck_count_dim,
        "bottleneck_phase_dim": model.bottleneck_phase_dim,
        "min_pulses": cfg.DATA["min_pulses"],
        "max_pulses": cfg.DATA["max_pulses"],
        "arch": {
            "encoder": "conv2d_1_16_32_64_padding=[2,1,1]",
            "adapter": f"time_adaptive_pool_len={cfg.MODEL['adapter_pool_len']}",
            "count_head_hidden": cfg.MODEL["count_head_hidden"],
            "phase_head_single_hidden": cfg.MODEL["phase_head_single_hidden"],
            "phase_head_two_hidden": cfg.MODEL["phase_head_two_hidden"],
        },
    }, state_path)
    print(f"[export] state_dict + meta -> {state_path}")

    meta_path = os.path.join(out_dir, "inference_meta.json")
    with open(meta_path, "w") as f:
        json.dump({
            "source_checkpoint": os.path.abspath(ckpt_path),
            "full_model_params": full_params,
            "deployed_subgraph_params": sub_params,
            "dropped_params": dropped,
            "input_shape": list(model.input_shape),
            "num_count_classes": model.num_count_classes,
            "phase_single_output_dim": model.phase_single_output_dim,
            "phase_two_output_dim": model.phase_two_output_dim,
            "bottleneck_count_dim": model.bottleneck_count_dim,
            "bottleneck_phase_dim": model.bottleneck_phase_dim,
            "min_pulses": cfg.DATA["min_pulses"],
            "max_pulses": cfg.DATA["max_pulses"],
        }, f, indent=2)
    print(f"[export] meta               -> {meta_path}")


if __name__ == "__main__":
    main()
