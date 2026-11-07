import json
from pathlib import Path

import yaml


# 脚本放在仓库根目录，因此它的父目录就是项目目录
ROOT = Path(__file__).resolve().parent
RUNS = ROOT / "runs"

# 每组实验应该使用的参数
EXPECTED = {
    "baseline": {"match_thresh": 0.8, "track_buffer": 30},
    "match07": {"match_thresh": 0.7, "track_buffer": 30},
    "match09": {"match_thresh": 0.9, "track_buffer": 30},
    "buffer60": {"match_thresh": 0.8, "track_buffer": 60},
}


def check_equal(actual, expected, description):
    """不相等时停止检查，并指出具体问题。"""
    if actual != expected:
        raise ValueError(
            f"{description}：期望 {expected!r}，实际 {actual!r}"
        )


# 先读取 baseline，后面用它检查共同配置
baseline_dir = RUNS / "A_baseline"

baseline_run = json.loads(
    (baseline_dir / "run_config.json").read_text(encoding="utf-8")
)

baseline_tracker = yaml.safe_load(
    (baseline_dir / baseline_run["tracker_snapshot"]).read_text(
        encoding="utf-8"
    )
)

common_fields = [
    "source", "model", "conf", "imgsz", "device", "classes"
]

# 去掉实验中允许改变的两个字段，留下其余配置
baseline_other = {
    key: value
    for key, value in baseline_tracker.items()
    if key not in {"match_thresh", "track_buffer"}
}


for name, expected_parameters in EXPECTED.items():
    run_dir = RUNS / f"A_{name}"
    print(f"正在检查：{run_dir.name}")

    run = json.loads(
        (run_dir / "run_config.json").read_text(encoding="utf-8")
    )

    tracker = yaml.safe_load(
        (run_dir / run["tracker_snapshot"]).read_text(
            encoding="utf-8"
        )
    )

    # 1. 视频、权重和推理参数必须一致
    for field in common_fields:
        check_equal(
            run[field],
            baseline_run[field],
            f"{name} 的 {field}",
        )

    # 2. 所有实验都要完整处理同一个视频
    check_equal(
        run["video"]["processed_frames"], 525,
        f"{name} 的处理帧数",
    )
    check_equal(
        run["stop_reason"], "read_finished",
        f"{name} 的结束原因",
    )

    for field in ["width", "height", "fps"]:
        check_equal(
            run["video"][field],
            baseline_run["video"][field],
            f"{name} 的视频 {field}",
        )

    # 3. 实验参数要符合预先设计
    for field, expected_value in expected_parameters.items():
        check_equal(
            tracker[field],
            expected_value,
            f"{name} 的 {field}",
        )

    # 4. 除实验参数外，其余跟踪配置必须一致
    tracker_other = {
        key: value
        for key, value in tracker.items()
        if key not in {"match_thresh", "track_buffer"}
    }

    check_equal(
        tracker_other,
        baseline_other,
        f"{name} 的其他跟踪配置",
    )

    with open(run_dir / "tracks.txt", 'r', encoding='utf-8') as f:
        rows = sum(1 for line in f if line.strip())
        check_equal(
            rows,
            run["track_rows"],
            f"{name} 的目标轨迹记录",
        )


    print(f"{name}：通过\n")


print("四组实验配置全部检查通过")
