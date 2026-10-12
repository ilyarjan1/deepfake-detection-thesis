#!/usr/bin/env python3
"""Add explicit pretrained initialization and head-only adaptation to AIDE.

The patch is deliberately narrow and reversible. It creates
main_finetune.py.before-head-adaptation before changing main_finetune.py.
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path


PARSER_ANCHOR = '''    parser.add_argument('--use_amp', type=str2bool, default=False, 
                        help="Use apex AMP (Automatic Mixed Precision) or not")
    return parser
'''

PARSER_REPLACEMENT = '''    parser.add_argument(
        '--init_checkpoint', default='', type=str,
        help='Load model weights only and start a fresh optimizer at epoch 0')
    parser.add_argument(
        '--trainable_modules', default='all', choices=['all', 'fc'],
        help='Choose all parameters or only the final AIDE classifier')
    parser.add_argument('--use_amp', type=str2bool, default=False, 
                        help="Use apex AMP (Automatic Mixed Precision) or not")
    return parser
'''

MODEL_ANCHOR = '''    model = AIDE.__dict__[args.model](
        resnet_path=args.resnet_path, 
        convnext_path=args.convnext_path
    )
        
    model.to(device)
'''

MODEL_REPLACEMENT = '''    model = AIDE.__dict__[args.model](
        resnet_path=args.resnet_path, 
        convnext_path=args.convnext_path
    )

    if args.init_checkpoint:
        checkpoint = torch.load(args.init_checkpoint, map_location='cpu')
        checkpoint_model = checkpoint['model'] if 'model' in checkpoint else checkpoint
        model.load_state_dict(checkpoint_model, strict=True)
        print(f"Initialized model weights from: {args.init_checkpoint}")
        print("Optimizer, scaler, and epoch state were not loaded")
        del checkpoint_model
        del checkpoint

    if args.trainable_modules == 'fc':
        for parameter in model.parameters():
            parameter.requires_grad = False
        for parameter in model.fc.parameters():
            parameter.requires_grad = True

    trainable_names = [
        name for name, parameter in model.named_parameters()
        if parameter.requires_grad
    ]
    trainable_count = sum(
        parameter.numel() for parameter in model.parameters()
        if parameter.requires_grad
    )
    total_count = sum(parameter.numel() for parameter in model.parameters())
    if args.trainable_modules == 'fc':
        if not trainable_names or any(
            not name.startswith('fc.') for name in trainable_names
        ):
            raise RuntimeError(
                f"Head-only freeze verification failed: {trainable_names}"
            )
    print(f"Trainable mode: {args.trainable_modules}")
    print(f"Trainable parameters: {trainable_count:,} / {total_count:,}")
    print("Trainable parameter names:")
    for name in trainable_names:
        print(f"  {name}")

    model.to(device)
'''


def replace_once(text: str, anchor: str, replacement: str, description: str) -> str:
    count = text.count(anchor)
    if count != 1:
        raise RuntimeError(
            f"Expected exactly one {description} anchor, found {count}. "
            "The file was not modified."
        )
    return text.replace(anchor, replacement, 1)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--aide-root",
        type=Path,
        default=Path.home() / "projects/AIDE",
    )
    args = parser.parse_args()

    target = args.aide_root.expanduser().resolve() / "main_finetune.py"
    backup = target.with_name("main_finetune.py.before-head-adaptation")

    if not target.is_file():
        raise FileNotFoundError(target)

    original = target.read_text(encoding="utf-8")
    if "--init_checkpoint" in original or "Trainable mode:" in original:
        raise RuntimeError(
            "The adaptation patch appears to be present already; nothing was changed."
        )

    patched = replace_once(original, PARSER_ANCHOR, PARSER_REPLACEMENT, "parser")
    patched = replace_once(patched, MODEL_ANCHOR, MODEL_REPLACEMENT, "model")

    # Validate Python syntax before changing the repository file.
    compile(patched, str(target), "exec")

    if backup.exists():
        raise FileExistsError(
            f"Backup already exists: {backup}\n"
            "Nothing was changed. Inspect or rename that backup before retrying."
        )

    shutil.copy2(target, backup)
    target.write_text(patched, encoding="utf-8")

    print(f"Patched: {target}")
    print(f"Backup:  {backup}")
    print("Added options: --init_checkpoint and --trainable_modules {all,fc}")


if __name__ == "__main__":
    main()
