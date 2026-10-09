"""给 PowerShell 脚本加 UTF-8 BOM。

背景：Windows PowerShell 5.1 读取 .ps1 时，若文件无 BOM，会按系统 ANSI
代码页（中文 Windows 为 GBK/936）解码，导致脚本内的中文字符串变成乱码、
引号失配，报出误导性的语法错误：
    Array index expression is missing or not valid.
    The string is missing the terminator: ".
PowerShell 7 默认按 UTF-8 读取，不受影响。为了让脚本同时兼容 5.1 与 7，
统一带 BOM 保存。

用法：
    .venv\\Scripts\\python.exe scripts\\ensure_ps1_bom.py
    .venv\\Scripts\\python.exe scripts\\ensure_ps1_bom.py --check   # 只检查不修改
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

BOM = b"\xef\xbb\xbf"
SCRIPTS_DIR = Path(__file__).resolve().parent


def process(path: Path, check_only: bool) -> str:
    """确保单个 .ps1 文件带 UTF-8 BOM。

    Args:
        path: 目标脚本路径。
        check_only: 为 True 时仅检查，不写盘。

    Returns:
        状态字符串：``ok`` / ``fixed`` / ``need-fix``。
    """
    raw = path.read_bytes()
    if raw.startswith(BOM):
        return "ok"
    if check_only:
        return "need-fix"
    # 先按 utf-8-sig 解码（兼容原本已有 BOM 的情况），再带 BOM 写回
    text = raw.decode("utf-8")
    path.write_bytes(BOM + text.encode("utf-8"))
    return "fixed"


def main(argv: list[str] | None = None) -> int:
    """批量为 scripts 目录下的 .ps1 文件补齐 BOM。

    Args:
        argv: 命令行参数。

    Returns:
        进程退出码：0 全部合规，1 存在需要修复的文件（--check 模式）。
    """
    parser = argparse.ArgumentParser(description="确保 .ps1 文件带 UTF-8 BOM（PS 5.1 兼容）")
    parser.add_argument("--check", action="store_true", help="只检查，不修改文件")
    parser.add_argument(
        "--dir", default=str(SCRIPTS_DIR), help="扫描目录（默认脚本自身所在目录）"
    )
    args = parser.parse_args(argv)

    target = Path(args.dir)
    files = sorted(target.glob("*.ps1"))
    if not files:
        print(f"[warn] {target} 下没有 .ps1 文件")
        return 0

    need_fix = 0
    for path in files:
        status = process(path, args.check)
        print(f"  [{status:8s}] {path.name}")
        if status != "ok":
            need_fix += 1

    if args.check:
        print(f"\n[check] {need_fix}/{len(files)} 个文件缺少 BOM")
        return 1 if need_fix else 0

    print(f"\n[done] 共处理 {len(files)} 个文件，修复 {need_fix} 个")
    return 0


if __name__ == "__main__":
    sys.exit(main())
