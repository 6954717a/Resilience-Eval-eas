#!/usr/bin/env python3
# isort: skip_file
"""
简单的多进程调度器：先按 episode 划分区间，再启动子进程执行 planner_demo_mp 单进程运行。

使用方式示例：
python -m habitat_llm.examples.planner_demo_chunked \
  --config-name baselines/qwen_centralized_zero_shot_react_summary_vllm.yaml \
  habitat.dataset.data_path="data/datasets/partnr_episodes/v0_0/val_mini.json.gz" \
  +evaluation.planner.plan_config.llm.generation_params.batch_size=32 \
  num_proc=4

说明：
- 本脚本只负责“分块 + 启子进程”，真正的规划/仿真仍由 planner_demo_mp 完成。
- 我们强制子进程 num_proc=1，并追加 start_index/end_index 覆盖，确保每个进程跑互斥的 episode 区间。
- 其它 Hydra 覆盖（模型/端口/参数等）会被原样传递给子进程。
"""

import math
import subprocess
import sys
from typing import List

import hydra
from omegaconf import OmegaConf

from habitat_llm.agent.env.dataset import CollaborationDatasetV0
from habitat_llm.utils import cprint, fix_config, setup_config


def _filter_base_args(argv: List[str]) -> List[str]:
    """
    过滤掉会与子进程覆盖冲突的参数（num_proc/start_index/end_index），其余原样透传。
    """
    filtered = []
    for arg in argv:
        if any(
            arg.lstrip("+").startswith(prefix)
            for prefix in ("num_proc", "start_index", "end_index")
        ):
            continue
        filtered.append(arg)
    return filtered


@hydra.main(config_path="../conf")
def main(config) -> None:
    # 让配置完整化（路径、种子等），与 planner_demo_mp 保持一致
    fix_config(config)
    config = setup_config(config)

    # 加载数据集，获取 episode 数量
    dataset = CollaborationDatasetV0(config.habitat.dataset)
    episodes = dataset.episodes
    total = len(episodes)
    if total == 0:
        raise ValueError("数据集中没有 episode，无法分块。")

    num_proc = config.get("num_proc", 1)
    if num_proc < 1:
        num_proc = 1
    # 计算区间
    chunk_size = math.ceil(total / num_proc)
    chunks = []
    start = 0
    while start < total:
        end = min(start + chunk_size, total)
        chunks.append((start, end))
        start = end

    cprint(
        f"共 {total} 个 episodes，将启动 {len(chunks)} 个子进程，分块: {chunks}",
        "blue",
    )

    base_args = _filter_base_args(sys.argv[1:])
    procs: List[subprocess.Popen] = []

    for i, (s, e) in enumerate(chunks):
        cmd = [
            sys.executable,
            "-m",
            "habitat_llm.examples.planner_demo_mp",
            *base_args,
            "num_proc=1",  # 子进程内禁用多进程，避免递归
            f"+start_index={s}",  # 以 Hydra append 方式新增键
            f"+end_index={e}",
        ]
        cprint(f"[子进程 {i}] 处理 [{s}, {e})，命令: {' '.join(cmd)}", "green")
        procs.append(subprocess.Popen(cmd))

    # 等待所有子进程
    retcodes = []
    for i, p in enumerate(procs):
        rc = p.wait()
        retcodes.append(rc)
        if rc != 0:
            cprint(f"[子进程 {i}] 退出码 {rc}", "red")
        else:
            cprint(f"[子进程 {i}] 完成", "green")

    if any(rc != 0 for rc in retcodes):
        raise SystemExit(f"至少一个子进程失败: {retcodes}")


if __name__ == "__main__":
    main()

