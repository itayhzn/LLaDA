#!/usr/bin/env python3
"""
Sweep w_hallucinate and w_summary over any OpenCompass base config.

Usage:
    # Sweep on a single benchmark:
    python sweep_hallucinate_summary.py examples/llada_instruct_gen_gsm8k_length512_block512_confidence.py

    # Sweep on multiple benchmarks:
    python sweep_hallucinate_summary.py \
        examples/llada_instruct_gen_gsm8k_length512_block512_confidence.py \
        examples/llada_instruct_gen_math_length512_block512_confidence.py

    # Baseline only (no sweep, just run the original config):
    python sweep_hallucinate_summary.py --baseline-only examples/llada_instruct_gen_gsm8k_length512_block512_confidence.py

    # Skip baseline:
    python sweep_hallucinate_summary.py --no-baseline examples/llada_instruct_gen_gsm8k_length512_block512_confidence.py

    # Custom sweep values:
    python sweep_hallucinate_summary.py --w-values 0.1 0.3 0.5 --t-values 1 examples/llada_instruct_gen_gsm8k_length512_block512_confidence.py

All configs run from inside opencompass/. Temporary sweep configs are
generated in opencompass/examples/_sweep/ and cleaned up after the run.
"""

import argparse
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent
OPENCOMPASS_DIR = REPO_ROOT / "opencompass"
PYTHONPATH_PREFIX = f"{REPO_ROOT}:{os.environ.get('PYTHONPATH', '')}"


def derive_name(config_path: str) -> str:
    """Extract a short benchmark name from the config filename."""
    stem = Path(config_path).stem
    # e.g. llada_instruct_gen_gsm8k_length512_block512_confidence -> gsm8k_length512_block512_confidence
    m = re.match(r"llada_(?:instruct|base|1p5)_gen_(.+)", stem)
    return m.group(1) if m else stem


def make_sweep_config(base_config: str, extra_params: dict) -> str:
    """Read a base config, inject extra params into eval_cfg, return modified content."""
    with open(base_config) as f:
        content = f.read()

    # Find the eval_cfg dict and inject params before the closing }
    additions = ", ".join(f"'{k}': {v}" for k, v in extra_params.items())
    content = re.sub(
        r"(eval_cfg\s*=\s*\{.*?)(})",
        rf"\1, {additions}\2",
        content,
        count=1,
    )
    return content


def run_config(config_path: str, output_dir: str):
    """Run a single OpenCompass config."""
    cmd = [
        sys.executable, "run.py",
        config_path,
        "-w", output_dir,
    ]
    env = os.environ.copy()
    env["PYTHONPATH"] = PYTHONPATH_PREFIX
    print(f"\n{'='*60}")
    print(f"Running: {config_path}")
    print(f"Output:  {output_dir}")
    print(f"{'='*60}\n")
    subprocess.run(cmd, cwd=str(OPENCOMPASS_DIR), env=env, check=True)


def main():
    parser = argparse.ArgumentParser(description="Sweep hallucinate/summary guidance over OpenCompass configs")
    parser.add_argument("configs", nargs="+",
                        help="Base config file(s) relative to opencompass/, e.g. examples/llada_instruct_gen_gsm8k_length512_block512_confidence.py")
    parser.add_argument("--w-values", nargs="+", type=float, default=[0.2, 0.4, 0.6, 0.8],
                        help="w_hallucinate / w_summary values to sweep (default: 0.2 0.4 0.6 0.8)")
    parser.add_argument("--t-values", nargs="+", type=float, default=[1., 2.],
                        help="hallucinate_t / summary_t values to sweep (default: 1 2)")
    parser.add_argument("--no-baseline", action="store_true",
                        help="Skip the baseline (no guidance) run")
    parser.add_argument("--baseline-only", action="store_true",
                        help="Only run baseline, no sweep")
    parser.add_argument("--hal-only", action="store_true",
                        help="Only sweep hallucinate, skip summary")
    parser.add_argument("--sum-only", action="store_true",
                        help="Only sweep summary, skip hallucinate")
    args = parser.parse_args()

    sweep_dir = OPENCOMPASS_DIR / "examples" / "_sweep"
    sweep_dir.mkdir(exist_ok=True)

    try:
        for base_config in args.configs:
            base_path = OPENCOMPASS_DIR / base_config
            if not base_path.exists():
                print(f"ERROR: {base_path} not found", file=sys.stderr)
                sys.exit(1)

            name = derive_name(base_config)

            # Baseline
            if not args.no_baseline:
                run_config(base_config, f"outputs/{name}_baseline")

            if args.baseline_only:
                continue

            # Hallucinate sweep
            if not args.sum_only:
                for w in args.w_values:
                    for t in args.t_values:
                        w_str = str(w).replace(".", "p")
                        t_str = str(int(t)) if t == int(t) else str(t).replace(".", "p")
                        tag = f"hal_w{w_str}_t{t_str}"
                        content = make_sweep_config(str(base_path), {
                            "w_hallucinate": w,
                            "hallucinate_t": t,
                        })
                        cfg_path = sweep_dir / f"{Path(base_config).stem}_{tag}.py"
                        cfg_path.write_text(content)
                        run_config(
                            f"examples/_sweep/{cfg_path.name}",
                            f"outputs/{name}_{tag}",
                        )

            # Summary sweep
            if not args.hal_only:
                for w in args.w_values:
                    for t in args.t_values:
                        w_str = str(w).replace(".", "p")
                        t_str = str(int(t)) if t == int(t) else str(t).replace(".", "p")
                        tag = f"sum_w{w_str}_t{t_str}"
                        content = make_sweep_config(str(base_path), {
                            "w_summary": w,
                            "summary_t": t,
                        })
                        cfg_path = sweep_dir / f"{Path(base_config).stem}_{tag}.py"
                        cfg_path.write_text(content)
                        run_config(
                            f"examples/_sweep/{cfg_path.name}",
                            f"outputs/{name}_{tag}",
                        )
    finally:
        # Clean up generated configs
        if sweep_dir.exists():
            for f in sweep_dir.iterdir():
                f.unlink()
            sweep_dir.rmdir()


if __name__ == "__main__":
    main()
