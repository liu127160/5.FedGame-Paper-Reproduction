"""Plot and summarize the local FedGame reproduction experiments.

This script does not change training. It reads scalar summaries under ``runs``
and writes figures plus a CSV summary under ``plots`` by default.
"""

import argparse
import csv
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


METHOD_ORDER = [
    "FedAvg Clean",
    "FedAvg Attack",
    "FLTrust (lr=0.01)",
    "FLTrust (lr=0.001)",
    "FedGame Out",
]


def compact(text):
    return "".join(ch for ch in text.lower() if ch.isalnum())


def classify_run(run_dir):
    name = run_dir.name.lower()
    if "debug" in name:
        return None

    # The directory prefix is generated from the actual task class. Prefer it
    # over the user-provided --name suffix, which can accidentally be misleading.
    if "mnist_fltrust" in name:
        return "FLTrust (lr=0.001)" if "lr001" in name else "FLTrust (lr=0.01)"
    if "mnist_ours" in name:
        return "FedGame Out"
    if "mnist_fedavg" in name:
        if "fltrust" in name:
            print("Warning: run name says FLTrust but actual task is MNIST_FedAvg: {}".format(run_dir))
        return "FedAvg Clean" if "clean" in name else "FedAvg Attack"

    # Fallback for non-standard directory names.
    if "clean" in name and ("fedavg" in name or "repro" in name):
        return "FedAvg Clean"
    if "fltrust" in name:
        return "FLTrust (lr=0.001)" if "lr001" in name else "FLTrust (lr=0.01)"
    if "fedgame" in name or "ours" in name:
        return "FedGame Out"
    if "fedavg" in name:
        return "FedAvg Attack"
    return None


def choose_metric_tag(tags, backdoor, event_dir):
    target = "testbackdoortrue" if backdoor else "testbackdoorfalse"
    candidates = [tag for tag in tags if target in compact(tag)]
    if not candidates:
        return None

    # SummaryWriter.add_scalars stores Accuracy and Loss in separate child
    # directories while keeping the same scalar tag. Exclude the Loss child.
    directory_name = compact(str(event_dir))
    if "loss" in directory_name and "accuracy" not in directory_name:
        return None

    def priority(tag):
        name = compact(tag)
        if "loss" in name:
            return 3
        if "accuracy" in name or "top1" in name or name.endswith("acc"):
            return 0
        return 1

    return sorted(candidates, key=lambda tag: (priority(tag), len(tag)))[0]


def read_event_directory(event_dir, run_root):
    accumulator = EventAccumulator(str(event_dir), size_guidance={"scalars": 0})
    accumulator.Reload()
    scalar_tags = accumulator.Tags().get("scalars", [])
    ta_tag = choose_metric_tag(scalar_tags, backdoor=False, event_dir=event_dir)
    asr_tag = choose_metric_tag(scalar_tags, backdoor=True, event_dir=event_dir)

    def series(tag):
        if tag is None:
            return []
        return [(event.step, float(event.value), event.wall_time)
                for event in accumulator.Scalars(tag)]

    return {
        "directory": event_dir,
        "run_root": run_root,
        "ta_tag": ta_tag,
        "asr_tag": asr_tag,
        "ta": series(ta_tag),
        "asr": series(asr_tag),
        "all_tags": scalar_tags,
    }


def merge_series(records, metric):
    by_step = {}
    for record in records:
        for step, value, wall_time in record[metric]:
            previous = by_step.get(step)
            if previous is None or wall_time >= previous[1]:
                by_step[step] = (value, wall_time)
    return [(step, by_step[step][0]) for step in sorted(by_step)]


def discover_results(runs_dir):
    event_dirs = sorted({path.parent for path in runs_dir.rglob("events.out.tfevents.*")})
    grouped = {method: [] for method in METHOD_ORDER}

    for event_dir in event_dirs:
        relative_parts = event_dir.relative_to(runs_dir).parts
        run_root = runs_dir / relative_parts[0] if relative_parts else runs_dir
        method = classify_run(run_root)
        if method is None:
            continue
        try:
            grouped[method].append(read_event_directory(event_dir, run_root))
        except Exception as error:
            print("Skipping {}: {}".format(event_dir, error))

    results = {}
    for method in METHOD_ORDER:
        records = grouped[method]
        if not records:
            continue
        ta = merge_series(records, "ta")
        asr = merge_series(records, "asr")
        if not ta and not asr:
            print("No TA/ASR tag found for {}. Available tags:".format(method))
            for record in records:
                print("  {}: {}".format(record["directory"], record["all_tags"]))
            continue
        results[method] = {
            "ta": ta,
            "asr": asr,
            "directories": sorted({str(record["run_root"]) for record in records}),
        }
    return results


def last_value(series):
    return series[-1][1] if series else math.nan


def add_value_labels(axis, bars):
    for bar in bars:
        height = bar.get_height()
        if not math.isnan(height):
            axis.text(bar.get_x() + bar.get_width() / 2, height + 1,
                      "{:.2f}".format(height), ha="center", va="bottom", fontsize=8)


def plot_curves(results, output_dir):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for method in METHOD_ORDER:
        result = results.get(method)
        if result is None:
            continue
        label = "FedGame Out (nc_steps=10)" if method == "FedGame Out" else method
        if result["ta"]:
            axes[0].plot([point[0] for point in result["ta"]],
                         [point[1] for point in result["ta"]], label=label)
        if result["asr"]:
            axes[1].plot([point[0] for point in result["asr"]],
                         [point[1] for point in result["asr"]], label=label)

    axes[0].set_title("Clean Test Accuracy (TA)")
    axes[1].set_title("Attack Success Rate (ASR)")
    for axis in axes:
        axis.set_xlabel("Communication round")
        axis.set_ylabel("Percent (%)")
        axis.set_ylim(0, 105)
        axis.grid(alpha=0.3)
        axis.legend(fontsize=8)
    fig.suptitle("MNIST IID Scaling Attack - Training Curves")
    fig.tight_layout()
    path = output_dir / "training_curves.png"
    fig.savefig(str(path), dpi=200, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_experiment_comparison(results, output_dir):
    methods = [method for method in METHOD_ORDER if method in results]
    if not methods:
        return None

    labels = ["FedGame Out\n(nc_steps=10)" if method == "FedGame Out" else method
              for method in methods]
    positions = list(range(len(methods)))
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for axis, metric, title in zip(
            axes, ("ta", "asr"), ("Clean Test Accuracy (TA)", "Attack Success Rate (ASR)")):
        experiment_values = [last_value(results[method][metric]) for method in methods]
        experiment_bars = axis.bar(positions, experiment_values, width=0.62)
        axis.set_title(title)
        axis.set_ylabel("Percent (%)")
        axis.set_xticks(positions)
        axis.set_xticklabels(labels, rotation=12, ha="right")
        axis.set_ylim(0, 110)
        axis.grid(axis="y", alpha=0.3)
        add_value_labels(axis, experiment_bars)

    fig.suptitle("MNIST IID Scaling Attack - Local Experiment Comparison")
    fig.text(0.5, 0.01,
             "FLTrust learning rates are shown separately; FedGame uses nc_steps=10.",
             ha="center", fontsize=9)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    path = output_dir / "experiment_comparison.png"
    fig.savefig(str(path), dpi=200, bbox_inches="tight")
    plt.close(fig)
    return path


def write_summary(results, output_dir):
    path = output_dir / "results_summary.csv"
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.writer(handle)
        writer.writerow([
            "method", "last_round", "experiment_TA", "experiment_ASR",
            "special_setting", "tensorboard_directories"
        ])
        for method in METHOD_ORDER:
            if method not in results:
                continue
            result = results[method]
            all_steps = [step for step, _ in result["ta"] + result["asr"]]
            writer.writerow([
                "FedGame Out (nc_steps=10)" if method == "FedGame Out" else method,
                max(all_steps) if all_steps else "",
                "{:.4f}".format(last_value(result["ta"])),
                "{:.4f}".format(last_value(result["asr"])),
                ("learning rate is shown in method name" if method.startswith("FLTrust")
                 else ("nc_steps=10" if method == "FedGame Out" else "")),
                " | ".join(result["directories"]),
            ])
    return path


def plot_summary_table(results, output_dir):
    rows = []
    for method in METHOD_ORDER:
        if method not in results:
            continue
        result = results[method]
        all_steps = [step for step, _ in result["ta"] + result["asr"]]
        if method == "FedGame Out":
            method_name = "FedGame"
            setting = "nc_steps=10"
        elif method.startswith("FLTrust"):
            method_name = "FLTrust"
            setting = method[method.find("(") + 1:method.rfind(")")]
        else:
            method_name = method
            setting = "default"
        rows.append([
            method_name,
            setting,
            str(max(all_steps)) if all_steps else "-",
            "{:.2f}%".format(last_value(result["ta"])),
            "{:.2f}%".format(last_value(result["asr"])),
        ])

    if not rows:
        return None

    fig_height = max(2.6, 0.55 * len(rows) + 1.5)
    fig, axis = plt.subplots(figsize=(10, fig_height))
    axis.axis("off")
    table = axis.table(
        cellText=rows,
        colLabels=["Method", "Special setting", "Last round", "TA", "ASR"],
        cellLoc="center",
        colLoc="center",
        loc="center",
        colWidths=[0.24, 0.24, 0.16, 0.16, 0.16],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1, 1.65)

    for (row, column), cell in table.get_celld().items():
        cell.set_edgecolor("#c7c7c7")
        if row == 0:
            cell.set_facecolor("#376a9e")
            cell.set_text_props(color="white", weight="bold")
        elif row % 2 == 0:
            cell.set_facecolor("#eef4fa")
        else:
            cell.set_facecolor("white")

    axis.set_title("MNIST IID Scaling Attack - Experiment Summary",
                   fontsize=14, weight="bold", pad=14)
    fig.tight_layout()
    path = output_dir / "results_summary_table.png"
    fig.savefig(str(path), dpi=200, bbox_inches="tight")
    plt.close(fig)
    return path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", default="runs", help="TensorBoard log directory")
    parser.add_argument("--output", default="plots", help="Directory for figures and CSV")
    args = parser.parse_args()

    runs_dir = Path(args.runs)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not runs_dir.exists():
        raise SystemExit("TensorBoard directory does not exist: {}".format(runs_dir))

    results = discover_results(runs_dir)
    if not results:
        raise SystemExit(
            "No usable TA/ASR scalars were found under {}. "
            "Run the experiments with tb: true first.".format(runs_dir)
        )

    created = [plot_curves(results, output_dir),
               plot_experiment_comparison(results, output_dir),
               plot_summary_table(results, output_dir),
               write_summary(results, output_dir)]
    print("Created:")
    for path in created:
        if path is not None:
            print("  {}".format(path))


if __name__ == "__main__":
    main()
