import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

PATTERN = "data/evaluation_results/detailed_predictions/ginic_full_dataset_train_{}_wav2vec2-large-xlsr-53-buckeye-ipa_detailed_predictions.csv"

DUR = "duration"
PER = "phone_error_rates"

MIN_DUR = 0.0
MAX_DUR = 12.0

BIN_1S = 1.0
BIN_01S = 0.1
BIN_001S = 0.01

MOVING_STEP = 0.1

OUTPUT_DIR = Path(__file__).parent / "output"


def load_best_train():
    dfs = {}
    mean_pers = {}

    for i in range(1, 6):
        df = pd.read_csv(PATTERN.format(i), usecols=[DUR, PER]).dropna()
        df = df[(df[DUR] >= MIN_DUR) & (df[DUR] <= MAX_DUR)]

        dfs[i] = df
        mean_pers[i] = df[PER].mean()

    best_train = min(mean_pers, key=mean_pers.get)
    best_df = dfs[best_train].sort_values(DUR).reset_index(drop=True)

    print(f"Best train: train_{best_train}")
    print(f"Mean PER: {mean_pers[best_train]:.6f}")
    print(f"Samples: {len(best_df):,}")

    return best_df


def make_binned_summary(df, bin_width):

    bins = np.arange(MIN_DUR, MAX_DUR + bin_width, bin_width)

    summary = (
        df.groupby(pd.cut(df[DUR], bins, include_lowest=True), observed=True)
        .agg(
            bin_start=(DUR, "min"),
            bin_end=(DUR, "max"),
            mean_duration=(DUR, "mean"),
            mean_phone_error_rate=(PER, "mean"),
            sample_count=(PER, "count"),
        )
        .reset_index(drop=True)
    )

    return summary[summary["sample_count"] > 0]


def save_csv(df, name):
    path = OUTPUT_DIR / name
    df.to_csv(path, index=False)
    print("Saved", path)


def plot_per_and_count(summary, xmin, xmax, name, title):

    fig, ax1 = plt.subplots(figsize=(8, 5))

    # PER
    ax1.plot(
        summary["mean_duration"], summary["mean_phone_error_rate"], color="tab:blue", marker="o", linewidth=2, label="PER"
    )

    ax1.set_xlabel("Duration (seconds)")
    ax1.set_ylabel("Phone Error Rate", color="tab:blue")
    ax1.tick_params(axis="y", labelcolor="tab:blue")
    ax1.set_xlim(xmin, xmax)

    # Sample count
    ax2 = ax1.twinx()

    ax2.plot(
        summary["mean_duration"], summary["sample_count"], color="tab:orange", linewidth=2, alpha=0.7, label="Sample Count"
    )

    ax2.fill_between(summary["mean_duration"], summary["sample_count"], color="tab:orange", alpha=0.2)

    ax2.set_ylabel("Sample Count", color="tab:orange")
    ax2.tick_params(axis="y", labelcolor="tab:orange")
    ax2.set_xlim(xmin, xmax)

    plt.title(title)
    plt.tight_layout()

    path = OUTPUT_DIR / name
    plt.savefig(path, dpi=200)
    plt.close()

    print("Saved", path)


def make_moving_average(df):

    thresholds = np.arange(0.1, MAX_DUR + 1e-9, MOVING_STEP)

    rows = []

    for t in thresholds:
        ge = df[df[DUR] >= t]
        le = df[df[DUR] <= t]

        rows.append({"threshold_sec": t, "mean_per_ge": ge[PER].mean(), "mean_per_le": le[PER].mean()})

    return pd.DataFrame(rows)


def plot_moving(summary, column, name, title, xlabel):

    plt.figure(figsize=(7, 5))

    plt.plot(summary["threshold_sec"], summary[column], linewidth=2)

    plt.xlabel(xlabel)
    plt.ylabel("Mean Phone Error Rate")
    plt.title(title)
    plt.xlim(0, 12)
    plt.ylim(0.22, 0.40)

    plt.grid(alpha=0.3)

    path = OUTPUT_DIR / name
    plt.savefig(path, dpi=200)
    plt.close()

    print("Saved", path)


def main():

    OUTPUT_DIR.mkdir(exist_ok=True)

    df = load_best_train()

    # CSV summaries
    summary_1s = make_binned_summary(df, BIN_1S)
    summary_01s = make_binned_summary(df, BIN_01S)

    save_csv(summary_1s, "0_12.csv")
    save_csv(summary_01s, "0_12_finer.csv")

    # 0–1 plots
    plot_per_and_count(
        make_binned_summary(df[df[DUR] <= 1], BIN_01S), 0, 1, "0_1.png", "PER and Sample Count vs Duration (0–1s, 0.1s bins)"
    )

    plot_per_and_count(
        make_binned_summary(df[df[DUR] <= 1], BIN_001S),
        0,
        1,
        "0_1_finer.png",
        "PER and Sample Count vs Duration (0–1s, 0.01s bins)",
    )

    # 0–12 plots
    plot_per_and_count(summary_01s, 0, 12, "0_12_finer.png", "PER and Sample Count vs Duration (0–12s, 0.1s bins)")

    plot_per_and_count(summary_1s, 0, 12, "0_12.png", "PER and Sample Count vs Duration (0–12s, 1s bins)")

    # moving averages
    moving = make_moving_average(df)

    plot_moving(
        moving,
        "mean_per_ge",
        "moving_average_increasing.png",
        "PER of samples with duration ≥ threshold",
        "Minimum duration threshold (seconds)",
    )

    plot_moving(
        moving,
        "mean_per_le",
        "moving_average_decreasing.png",
        "PER of samples with duration ≤ threshold",
        "Maximum duration threshold (seconds)",
    )

    print("\nAll outputs saved in ./output")


if __name__ == "__main__":
    main()
