import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
import glob

ITER_RANGE = [500, 1000, 2000, 5000]

# =============================================================================
# 1. LOAD AND AGGREGATE DATA
# =============================================================================


def load_standard_results(base_path):
    all_data = []
    # Pattern: results/500it/mixed/results_raw.csv
    files = glob.glob(os.path.join(base_path, "*it", "*", "*_raw.csv"))

    for f in files:
        try:
            parts = f.split(os.sep)
            iterations_str = next(
                p for p in parts if "it" in p and p[:-2].isdigit()
            )
            iterations = int(iterations_str.replace("it", ""))
            config_name = parts[-2]

            df = pd.read_csv(f)
            df["Iterations"] = iterations
            df["Config"] = config_name
            all_data.append(df)
        except Exception as e:
            print(f"Skipping standard file {f}: {e}")

    return (
        pd.concat(all_data, ignore_index=True) if all_data else pd.DataFrame()
    )


def load_llm_results(base_path):
    all_data = []
    # Pattern: results/LLM/mixed/results_raw.csv
    files = glob.glob(os.path.join(base_path, "LLM", "*", "*_raw.csv"))

    for f in files:
        try:
            parts = f.split(os.sep)
            config_name = parts[-2]

            df = pd.read_csv(f)

            # --- FILTER: Skip standard baselines in LLM folder ---
            # We assume these are already covered in the main iteration loops
            if "Strategy" in df.columns:
                # Keep rows that are NOT Random OR Baseline
                mask = ~df["Strategy"].str.contains(
                    "Random|Baseline", case=False, regex=True
                )
                df = df[mask]

            if df.empty:
                continue

            # Mark LLM results with a special Iteration flag (-1)
            df["Iterations"] = -1
            df["Config"] = config_name

            # Standardize LLM name if needed
            if "Strategy" in df.columns:
                df["Strategy"] = df["Strategy"].replace(
                    {"GPT-4o-mini": "LLM (GPT-5.1)", "LLM": "LLM Baseline"}
                )

            all_data.append(df)
        except Exception as e:
            print(f"Skipping LLM file {f}: {e}")

    return (
        pd.concat(all_data, ignore_index=True) if all_data else pd.DataFrame()
    )


def load_all_results(base_path):
    df_std = load_standard_results(base_path)
    df_llm = load_llm_results(base_path)

    if df_std.empty and df_llm.empty:
        print("No data found!")
        return pd.DataFrame()

    return pd.concat([df_std, df_llm], ignore_index=True)


# =============================================================================
# 2. PLOTTING FUNCTIONS
# =============================================================================


import matplotlib.pyplot as plt
import seaborn as sns


def get_strategy_palette(df):
    """
    Consistent palette:
    - known strategies use fixed colorblind colors from strategy_colors
    - unknown strategies get deterministic fallback colors
    """

    strategy_order = [
        "Baseline MLE",
        "Synthetic EM",
        "Iterative Proportional Fitting",
        "Copula Optimization",
        "Constraint Optimization",
        "MCMC Optimization",
        "Random Guess",
        "LLM (GPT-5.1)",
    ]

    # --- OPTION 2: Seaborn Set2 (pastel, publication-ready) ---
    palette = sns.color_palette("Set2", n_colors=len(strategy_order))

    # --- OPTION 4: Tableau (professional, print-friendly) ---
    # palette = sns.color_palette("viridis", n_colors=len(strategy_order))

    strategy_colors = {
        name: palette[i] for i, name in enumerate(strategy_order)
    }

    palette = dict(strategy_colors)  # exact mapping you specified

    for s in df["Strategy"].dropna().unique():
        if s not in palette:
            palette[s] = "gray"

    return palette


# ...existing code...


# --- 2. Updated Convergence Plot ---
def plot_convergence(df, metric, title, save_path, palette):
    """
    Shows how a metric improves with Iterations.
    LLM Baselines are plotted as horizontal dashed lines.
    """
    plt.figure(figsize=(10, 6))

    dynamic_df = df[df["Iterations"] > 0]
    baseline_df = df[df["Iterations"] == -1]

    # Plot Dynamic Strategies (Pass the palette dict here)
    sns.lineplot(
        data=dynamic_df,
        x="Iterations",
        y=metric,
        hue="Strategy",
        style="Strategy",
        palette=palette,
        markers=True,
        dashes=False,
        err_style="band",
        errorbar=("ci", 95),
        err_kws={"alpha": 0.05},
    )

    # Plot LLM Baselines (Lookup color from the same palette dict)
    if not baseline_df.empty:
        llm_strategies = baseline_df["Strategy"].unique()

        for strat in llm_strategies:
            strat_data = baseline_df[baseline_df["Strategy"] == strat]
            mean_val = strat_data[metric].mean()

            # Get color from the unified palette (fallback to black if missing)
            c = palette.get(strat, "black")

            plt.axhline(
                y=mean_val,
                color=c,
                linestyle="--",
                linewidth=2,
                label=f"{strat} (Avg)",
            )

    plt.title(title)
    plt.ylabel(metric.replace("_", " "))
    plt.xlabel("Iterations")
    plt.xscale("log")
    plt.xticks(ITER_RANGE, [str(x) for x in ITER_RANGE])
    plt.minorticks_off()
    plt.grid(True, which="both", ls="-", alpha=0.2)

    # Legend below plot
    plt.legend(loc="upper center", bbox_to_anchor=(0.5, -0.1), ncol=4)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


# --- 3. Updated Sensitivity Plot ---
def plot_sensitivity(df, metric, title, fixed_iters, save_path, palette):
    """
    Shows performance across different Data Types (configs).
    Includes LLM baseline (Iterations=-1).
    """
    subset = df[(df["Iterations"] == fixed_iters) | (df["Iterations"] == -1)]

    if subset.empty:
        print(f"No data for {fixed_iters} iterations.")
        return

    plt.figure(figsize=(10, 6))

    # Pass the same palette dict here
    sns.barplot(
        data=subset,
        x="Config",
        y=metric,
        hue="Strategy",
        palette=palette,
        edgecolor="black",
        errorbar="sd",
    )

    plt.title(title)
    plt.ylabel(metric.replace("_", " "))
    plt.xlabel("Data Type Configuration")

    # Legend below plot
    plt.legend(loc="upper center", bbox_to_anchor=(0.5, -0.1), ncol=4)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


# =============================================================================
# 3. EXECUTION
# =============================================================================
def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    base_path = os.path.join(script_dir, "..", "paper", "results")

    # Create directory structure
    plot_base = os.path.join(base_path, "plots")
    os.makedirs(os.path.join(plot_base, "convergence"), exist_ok=True)
    os.makedirs(os.path.join(plot_base, "sensitivity"), exist_ok=True)

    # Define plot configurations
    convergence_plots = {
        "Dec_Regret": " Decision Regret",
        "Avg_Weighted_KL": " Weighted KL Divergence",
        "Avg_Weighted_Hellinger": " Weighted Hellinger Distance",
        "Inf_Accuracy": " Inference Accuracy",
    }

    sensitivity_plots = {
        "Avg_Weighted_Hellinger": "Data Type Sensitivity: Weighted Hellinger Distance",
        "Inf_Accuracy": "Data Type Sensitivity: Inference Accuracy",
        "Dec_Regret": "Data Type Sensitivity: Decision Regret",
        "Avg_Weighted_KL": "Data Type Sensitivity: Weighted KL Divergence",
    }

    # Load data
    df = load_all_results(base_path)

    if not df.empty:
        print(f"Loaded {len(df)} rows of data.")
        my_palette = get_strategy_palette(df)

        # 1. Global Convergence Plots (Aggregated across Configs)
        print("Generating Global Convergence Plots...")
        os.makedirs(os.path.join(plot_base, "convergence"), exist_ok=True)
        for metric, title in convergence_plots.items():
            save_path = os.path.join(plot_base, "convergence", f"{metric}.pdf")
            plot_convergence(
                df,
                metric,
                f"Global Convergence: {title}",
                save_path,
                my_palette,
            )
            print(f"  Saved: {metric}.pdf")

        # 2. Per-Config Convergence Plots (Separated)
        print("Generating Per-Config Convergence Plots...")
        unique_configs = df["Config"].unique()
        for config in unique_configs:
            save_dir = os.path.join(plot_base, "convergence", config)
            os.makedirs(save_dir, exist_ok=True)

            # Filter Data for this Config
            # Note: We must include the LLM baseline for this specific config too!
            config_df = df[df["Config"] == config]

            if config_df.empty:
                continue

            for metric, title in convergence_plots.items():
                save_path = os.path.join(save_dir, f"{metric}.pdf")
                plot_convergence(
                    config_df,
                    metric,
                    f"Convergence ({config}): {title}",
                    save_path,
                    my_palette,
                )
                print(f"  Saved: {config}/{metric}.pdf")

        # 3. Generate sensitivity plots
        print("Generating Sensitivity Plots...")
        for iter in ITER_RANGE:
            for metric, title in sensitivity_plots.items():
                save_path = os.path.join(
                    plot_base, "sensitivity", f"{metric}_{iter}it.pdf"
                )
                plot_sensitivity(
                    df,
                    metric,
                    f"{title} (at {iter} iters)",
                    iter,
                    save_path,
                    my_palette,
                )
                print(f"  Saved: {metric}_{iter}it.pdf")

        print("Done!")
    else:
        print("Skipping plots (No data).")


if __name__ == "__main__":
    main()
