#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""查找并打印 Push-T 数据集的实际路径"""

import sys
from pathlib import Path

# 添加 src 到路径
sys.path.insert(0, str(Path(__file__).parent / "src"))

from hw1_imitation.data import download_pusht

# 尝试常见路径
possible_dirs = [
    Path.home() / ".cache" / "hw1_data",          # 用户缓存
    Path(__file__).parent / "data",               # 项目 data 目录
    Path(__file__).parent / "datasets",           # 项目 datasets 目录
    Path.cwd() / "data",                          # 当前工作目录
]

print("🔍 正在查找数据集...\n")

for dataset_dir in possible_dirs:
    zarr_path = dataset_dir / "pusht" / "pusht_cchi_v7_replay.zarr"
    print(f"检查: {zarr_path}")
    if zarr_path.exists():
        print(f"✅ 找到数据集！\n📁 路径: {zarr_path.resolve()}\n")
        # 加载验证
        import zarr
        root = zarr.open(zarr_path, mode="r")
        states = root["data"]["state"][:]
        actions = root["data"]["action"][:]
        print(f"📊 数据验证: states={states.shape}, actions={actions.shape}")
        break
else:
    print("\n❌ 未找到现有数据集，将尝试下载...")
    # 选择默认目录下载
    dataset_dir = Path.home() / ".cache" / "hw1_data"
    print(f"📥 下载目标: {dataset_dir}")
    try:
        zarr_path = download_pusht(dataset_dir)
        print(f"✅ 下载完成！数据集位置: {zarr_path.resolve()}")
    except Exception as e:
        print(f"❌ 下载失败: {e}")
        print("\n💡 请手动下载: https://diffusion-policy.cs.columbia.edu/data/training/pusht.zip")
        print(f"   解压到: {dataset_dir / 'pusht'}")