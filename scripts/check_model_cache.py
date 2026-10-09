"""诊断并修复 bge-m3 本地缓存：找出悬空/缺失文件并重新下载真实副本。

背景：沙箱会拒绝非提权进程创建符号链接，huggingface_hub 的 snapshot 里可能
留下指向不存在 blob 的悬空链接，表现为"文件在但读不到"，进而报出误导性的
`Pooling.__init__() missing 1 required positional argument: 'embedding_dimension'`。
"""

from __future__ import annotations

import os
from pathlib import Path

REPO = "BAAI/bge-m3"
CACHE_ROOT = Path(__file__).resolve().parent.parent / ".cache" / "huggingface" / "hub"
SNAP_DIR = CACHE_ROOT / f"models--{REPO.replace('/', '--')}" / "snapshots"

FILES = [
    "config.json",
    "config_sentence_transformers.json",
    "modules.json",
    "sentence_bert_config.json",
    "special_tokens_map.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "sentencepiece.bpe.model",
    "pytorch_model.bin",
    "1_Pooling/config.json",
    # 注意：BAAI/bge-m3 仓库中不存在 2_Normalize/config.json（Normalize 模块无需配置），
    # 若把不存在的文件放进清单，修复阶段会拿到 404 而中断。
]


def audit() -> list[str]:
    """检查 snapshot 内每个文件是否真实可读。

    Returns:
        损坏或缺失的相对路径列表。
    """
    snaps = [p for p in SNAP_DIR.glob("*") if p.is_dir()]
    if not snaps:
        print(f"[audit] 未找到 snapshot 目录: {SNAP_DIR}")
        return FILES
    snap = snaps[0]
    print(f"[audit] snapshot = {snap}")
    broken: list[str] = []
    for rel in FILES:
        p = snap / rel
        if not p.exists():
            print(f"  [缺失] {rel}")
            broken.append(rel)
            continue
        try:
            size = p.stat().st_size
        except OSError as exc:
            print(f"  [不可读] {rel} -> {type(exc).__name__}")
            broken.append(rel)
            continue
        try:
            with p.open("rb") as fh:
                fh.read(64)
            print(f"  [OK  ] {rel} ({size} bytes)")
        except OSError as exc:
            print(f"  [悬空] {rel} -> {type(exc).__name__}: {exc.strerror}")
            broken.append(rel)
    return broken


def repair(broken: list[str]) -> None:
    """重新下载损坏/缺失的文件（不建符号链接，直接写真实文件）。

    Args:
        broken: :func:`audit` 返回的相对路径列表。
    """
    from huggingface_hub import hf_hub_download

    for rel in broken:
        # local_dir 模式下 huggingface_hub 直接落真实文件，不依赖符号链接
        path = hf_hub_download(
            repo_id=REPO,
            filename=rel,
            local_dir=str(CACHE_ROOT / "local_bge_m3"),
            force_download=True,
        )
        print(f"  [已修复] {rel} -> {path} ({Path(path).stat().st_size} bytes)")


if __name__ == "__main__":
    print(f"HF_ENDPOINT = {os.environ.get('HF_ENDPOINT', '(未设置)')}")
    print(f"HF_HUB_DISABLE_SYMLINKS = {os.environ.get('HF_HUB_DISABLE_SYMLINKS', '(未设置)')}")
    bad = audit()
    print(f"\n[audit] 损坏/缺失 {len(bad)} 项: {bad}")
    if bad:
        print("[repair] 开始重新下载…")
        repair(bad)
        print("\n[re-audit] 二次检查:")
        audit()
    else:
        print("[audit] 全部文件真实可读，无需修复")
