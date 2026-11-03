"""Check an exported policy before moving it to another simulator: I/O names, widths, and one forward pass.

Isaac Lab's rsl_rl ``play.py`` writes ``exported/policy.onnx`` (and ``policy.pt``) next to the checkpoint it loads
(verify: Isaac Lab v2.2 ``export_policy_as_onnx``). Newton's ANYmal example performs the same check with
``validate_policy_io_shapes(obs_width=48, action_width=12)`` before it runs the PhysX-trained policy.

    python scripts/check_onnx_policy.py artifacts/policy.onnx --obs 48 --act 12      # ANYmal flat velocity policy
    python scripts/check_onnx_policy.py artifacts/push_policy.onnx --obs 28 --act 5  # push task, 5 arm DoF
    # push-task widths: arm pos n + arm vel n + wrist 3 + object 3 + target pose 7 + last action n = 3n + 13

Needs ``onnx`` and, for the forward pass, ``onnxruntime`` (both run on CPU). Not executed in this repo: neither package
is installed in the cloud container, and no policy was trained here.
"""

from __future__ import annotations

import argparse
import sys


def _width(value_info) -> int | None:
    dims = value_info.type.tensor_type.shape.dim
    return dims[-1].dim_value if dims and dims[-1].dim_value > 0 else None


def check(path: str, obs: int, act: int) -> list[str]:
    """Return a list of problems (empty when the policy matches the expected widths)."""
    import onnx

    model = onnx.load(path)
    onnx.checker.check_model(model)
    problems: list[str] = []
    inputs = [i for i in model.graph.input if i.name not in {init.name for init in model.graph.initializer}]
    if len(inputs) != 1 or len(model.graph.output) != 1:
        problems.append(f"expected 1 input and 1 output, got {len(inputs)} and {len(model.graph.output)}")
        return problems
    w_in, w_out = _width(inputs[0]), _width(model.graph.output[0])
    print(f"input  {inputs[0].name!r} width {w_in}\noutput {model.graph.output[0].name!r} width {w_out}")
    if w_in != obs:
        problems.append(f"observation width {w_in} != {obs}")
    if w_out != act:
        problems.append(f"action width {w_out} != {act}")
    try:
        import numpy as np
        import onnxruntime as ort
    except ImportError:
        print("onnxruntime not installed: skipped the forward pass")
        return problems
    sess = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
    out = sess.run(None, {inputs[0].name: np.zeros((1, obs), dtype=np.float32)})[0]
    print(f"zero-observation action: {np.round(out, 3)}")
    if not np.all(np.isfinite(out)):
        problems.append("non-finite action for a zero observation")
    return problems


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("onnx_path")
    ap.add_argument("--obs", type=int, required=True, help="expected observation width")
    ap.add_argument("--act", type=int, required=True, help="expected action width")
    args = ap.parse_args(argv)
    problems = check(args.onnx_path, args.obs, args.act)
    for p in problems:
        print("PROBLEM", p)
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
