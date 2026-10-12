#!/usr/bin/env python3
"""Patch AIDE so head-only adaptation keeps the frozen backbone in eval mode."""

import argparse
import py_compile
import shutil
from pathlib import Path


OLD = """    model.train(True)\n    metric_logger = utils.MetricLogger(delimiter=\"  \")\n"""

NEW = """    model.train(True)\n\n    # During head-only adaptation, requires_grad=False prevents gradient\n    # updates but does not stop BatchNorm statistics, Dropout, or DropPath\n    # behavior from changing. Keep the complete frozen backbone in eval mode\n    # and explicitly return only the classifier head to training mode.\n    if args is not None and getattr(args, \"trainable_modules\", \"all\") == \"fc\":\n        model_for_mode = model.module if hasattr(model, \"module\") else model\n\n        for module_name, module in model_for_mode.named_modules():\n            is_classifier = (\n                module_name == \"fc\" or module_name.startswith(\"fc.\")\n            )\n            if module_name and not is_classifier:\n                module.eval()\n\n        model_for_mode.fc.train(True)\n\n        unexpected_training_modules = [\n            module_name\n            for module_name, module in model_for_mode.named_modules()\n            if module_name\n            and not (module_name == \"fc\" or module_name.startswith(\"fc.\"))\n            and module.training\n        ]\n        if unexpected_training_modules:\n            raise RuntimeError(\n                \"Frozen backbone modules remained in training mode: \"\n                + \", \".join(unexpected_training_modules[:10])\n            )\n\n        print(\"Head-only mode: frozen backbone is in eval mode; fc is in train mode\")\n\n    metric_logger = utils.MetricLogger(delimiter=\"  \")\n"""


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--aide-root",
        type=Path,
        default=Path.home() / "projects" / "AIDE",
    )
    args = parser.parse_args()

    target = args.aide_root.expanduser().resolve() / "engine_finetune.py"
    backup = target.with_name(target.name + ".before-frozen-backbone-eval")

    if not target.is_file():
        raise SystemExit(f"AIDE engine file was not found: {target}")

    source = target.read_text()

    if "Head-only mode: frozen backbone is in eval mode" in source:
        print(f"Already patched: {target}")
        return

    if OLD not in source:
        raise SystemExit(
            "Expected model.train(True) block was not found; no file was changed."
        )

    if backup.exists():
        raise SystemExit(
            f"Backup already exists: {backup}\n"
            "Inspect it before applying another patch."
        )

    patched = source.replace(OLD, NEW, 1)
    compile(patched, str(target), "exec")

    shutil.copy2(target, backup)
    target.write_text(patched)
    py_compile.compile(str(target), doraise=True)

    print(f"Patched: {target}")
    print(f"Backup:  {backup}")
    print("Head-only training now keeps the frozen backbone in eval mode.")


if __name__ == "__main__":
    main()
