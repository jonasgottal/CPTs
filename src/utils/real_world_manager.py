import os
import time
import json
import pandas as pd
import numpy as np
import networkx as nx
from pgmpy.estimators import HillClimbSearch, BIC
from pgmpy.models import DiscreteBayesianNetwork
from pgmpy.factors.discrete import TabularCPD
from pgmpy.inference import VariableElimination

from extractor.extractor import StructuredExtractor
from utils.converter import PDFToPairwiseConverter
from utils.eval import calculate_metrics_robust


class RealWorldExperimentManager:
    def __init__(
        self,
        pdf_path,
        csv_path,
        type_mapping,
        target_columns=None,
        cache_file="extracted_data.json",
    ):
        self.pdf_path = pdf_path
        self.csv_path = csv_path
        self.type_mapping = type_mapping
        self.target_columns = target_columns  # List of columns to keep
        self.cache_file = cache_file

        # Internal State
        self.df_raw = None
        self.df_discrete = None
        self.state_names = None
        self.gt_bn = None
        self.skeleton = None
        self.pairwise_reports = None
        self.continuous_map = {}
        self.results_log = []

    def prepare_data(self, api_key):
        print(f"[Manager] Loading raw data from {self.csv_path}...")
        self.df_raw = pd.read_csv(self.csv_path)

        # --- 1. Filter Columns ---
        if self.target_columns:
            print(
                f"[Manager] Filtering data to {len(self.target_columns)} columns..."
            )
            valid_cols = [
                c for c in self.target_columns if c in self.df_raw.columns
            ]
            self.df_raw = self.df_raw[valid_cols]

        # --- 2. Prepare Ground Truth ---
        print("[Manager] Preparing Ground Truth Data...")
        self.df_discrete, self.state_names = self._prepare_ground_truth_data(
            self.df_raw
        )

        # --- 3. Learn Structure ---
        est = HillClimbSearch(self.df_discrete)
        try:
            dag_model = est.estimate(scoring_method=BIC(self.df_discrete))
        except ValueError:
            dag_model = nx.DiGraph()
            dag_model.add_nodes_from(self.df_discrete.columns)

        gt_edges = list(dag_model.edges())
        dag_nx = nx.DiGraph(gt_edges)

        # --- 4. Extract Data (Cached) ---
        if os.path.exists(self.cache_file):
            print(f"[Manager] [CACHE HIT] Loading extracted data...")
            with open(self.cache_file, "r") as f:
                data_dict = json.load(f)
        else:
            print(f"[Manager] Extracting data from PDF...")
            extractor = StructuredExtractor(
                api_key=api_key, model="gpt-4o-mini", dag=dag_nx
            )
            result = extractor.extract(self.pdf_path)
            data_dict = result.model_dump(exclude_none=True)
            with open(self.cache_file, "w") as f:
                json.dump(data_dict, f, indent=2)

        # --- 5. Convert to Pairwise ---
        converter = PDFToPairwiseConverter(data_dict, gt_edges)
        pairwise_reports_raw = converter.convert()

        # --- 6. Align Reports ---
        print("[Manager] Aligning Extracted Reports...")
        self.pairwise_reports = self._align_json_reports(pairwise_reports_raw)

        # --- 7. Fit GT Parameters ---
        print("[Manager] Fitting Ground Truth BN...")
        self.gt_bn = DiscreteBayesianNetwork(gt_edges)
        self.gt_bn.add_nodes_from(dag_model.nodes())
        self.gt_bn.fit(self.df_discrete)

        self.skeleton = DiscreteBayesianNetwork(gt_edges)
        self.skeleton.add_nodes_from(self.gt_bn.nodes())

        # --- 8. Populate continuous_map  ---
        # Copula strategy needs 'source_col' to map continuous variable -> discrete node
        for col, ctype in self.type_mapping.items():
            if self.target_columns and col not in self.target_columns:
                continue

            if ctype == "continuous":
                self.continuous_map[col] = {
                    "type": "continuous",
                    "source_col": col,  
                    "map": {},
                }

    def _evaluate_json_distribution_diff(
        self,
        learned_model,
        json_reports,
        n_eval_samples=5000,
        seed=0,
    ):
        """
        Compare JSON-provided marginals (per variable) to marginals estimated from
        data sampled from the learned model.

        Returns:
            {
              "JSON_AvgAbsDiff": avg |p_json - p_sample| per probability entry,
              "JSON_AvgTVD": avg TVD per variable distribution
            }
        """
        try:
            df_samp = learned_model.simulate(
                n_samples=n_eval_samples,
                show_progress=False,
                seed=seed + 9999,
            )
        except Exception:
            return {}

        abs_diffs = []
        tvds = []

        for report in json_reports or []:
            vars_block = report.get("variables", {}) or {}
            for var, info in vars_block.items():
                dist_json = info.get("distribution", None)
                if not isinstance(dist_json, dict) or len(dist_json) == 0:
                    continue

                if var not in df_samp.columns:
                    continue

                # Categories/states to compare on
                cats = info.get("categories", None)
                if not isinstance(cats, (list, tuple)) or len(cats) == 0:
                    cats = list(dist_json.keys())

                # Normalize JSON distribution (robust)
                z = float(sum(dist_json.values()))
                if z <= 0:
                    continue
                p_json = {c: float(dist_json.get(c, 0.0)) / z for c in cats}

                # Empirical distribution from sampled data
                vc = (
                    df_samp[var]
                    .value_counts(normalize=True, dropna=False)
                    .to_dict()
                )
                p_samp = {c: float(vc.get(c, 0.0)) for c in cats}

                # Element-wise abs diff
                diffs = [abs(p_json[c] - p_samp[c]) for c in cats]
                abs_diffs.extend(diffs)

                # TVD per variable
                tvds.append(0.5 * float(np.sum(diffs)))

        if not abs_diffs:
            return {}

        return {
            "JSON_AvgAbsDiff": float(np.mean(abs_diffs)),
            "JSON_AvgTVD": float(np.mean(tvds)) if tvds else float("nan"),
        }

    def run_experiment(
        self,
        strategies,
        n_seeds=5,
        iterations=2000,
        n_samples=10000,
        n_test_queries=500,
    ):
        print(
            f"Starting Real-World Experiment ({len(self.gt_bn.nodes())} nodes)"
        )
        row_frequencies = self._calculate_row_frequencies(self.df_discrete)
        utility_model = self._create_utility_model()

        for seed in range(n_seeds):
            print(f"  > Experiment Run {seed+1}/{n_seeds}")

            # Generate Test Queries
            np.random.seed(seed + 1000)
            df_test = self.gt_bn.simulate(
                n_samples=n_test_queries, show_progress=False
            )
            test_queries = self._generate_test_queries(
                df_test, n_queries_per_sample=1
            )

            for strategy_name, strategy_func in strategies.items():
                print(f"    Running strategy: {strategy_name}")
                try:
                    start_time = time.time()

                    learned_model = strategy_func(
                        json_reports=self.pairwise_reports,
                        skeleton=self.skeleton,
                        raw_data=self.df_discrete,
                        state_names=self.state_names,
                        seed=seed,
                        iterations=iterations,
                        n_samples=n_samples,
                    )

                    learned_model = self._ensure_complete_model(
                        learned_model, self.state_names
                    )
                    elapsed = time.time() - start_time

                    # Eval
                    cpt_m = self._evaluate_cpt_metrics(
                        learned_model, row_frequencies
                    )
                    inf_m = self._evaluate_inference(
                        learned_model, test_queries
                    )
                    dec_m = self._evaluate_decisions(
                        learned_model, test_queries, utility_model
                    )
                    json_m = self._evaluate_json_distribution_diff(
                        learned_model,
                        self.pairwise_reports,
                        n_eval_samples=min(5000, n_samples),
                        seed=seed,
                    )

                    all_metrics = {"Strategy": strategy_name, "Seed": seed}
                    all_metrics.update(cpt_m)
                    all_metrics.update(inf_m)
                    all_metrics.update(dec_m)
                    all_metrics.update(json_m)
                    all_metrics.update({"Avg_Time_Sec": elapsed})

                    self.results_log.append(all_metrics)
                    print(
                        f"      KL={all_metrics.get('Avg_KL_Divergence', 'N/A'):.3f} | Acc={all_metrics.get('Inf_Accuracy', 'N/A'):.3f}"
                    )

                except Exception as e:
                    print(f"    ! Strategy '{strategy_name}' failed: {e}")
                    import traceback

                    traceback.print_exc()

        return pd.DataFrame(self.results_log)

    # --- HELPERS (Same as before) ---
    def _generate_test_queries(self, df_test, n_queries_per_sample=1):
        dag = nx.DiGraph(self.gt_bn.edges())
        dag.add_nodes_from(self.gt_bn.nodes())
        leaf_nodes = [n for n in dag.nodes() if dag.out_degree(n) == 0]
        if not leaf_nodes:
            leaf_nodes = list(self.gt_bn.nodes())
        test_queries = []
        rng = np.random.default_rng()
        for idx, row in df_test.iterrows():
            n_targets = min(n_queries_per_sample, len(leaf_nodes))
            if n_targets == 0:
                continue
            targets = rng.choice(
                leaf_nodes, size=n_targets, replace=False
            ).tolist()
            evidence = {v: row[v] for v in df_test.columns if v not in targets}
            true_states = {t: row[t] for t in targets}
            test_queries.append(
                {
                    "targets": targets,
                    "evidence": evidence,
                    "true_states": true_states,
                }
            )
        return test_queries

    def _create_utility_model(self):
        dag = nx.DiGraph(self.gt_bn.edges())
        dag.add_nodes_from(self.gt_bn.nodes())
        leaf_nodes = [n for n in dag.nodes() if dag.out_degree(n) == 0]
        utility_model = {}
        rng = np.random.default_rng(42)
        for leaf in leaf_nodes:
            cpd = self.gt_bn.get_cpds(leaf)
            states = cpd.state_names[leaf]
            n_states = len(states)
            actions = (
                ["no_action", "action"]
                if n_states == 2
                else [f"action_{i}" for i in range(n_states)]
            )
            U = {}
            for a_idx, action in enumerate(actions):
                U[action] = {}
                for s_idx, state in enumerate(states):
                    U[action][state] = (
                        10.0 if a_idx == s_idx else -5.0
                    ) + rng.uniform(-1.0, 1.0)
            utility_model[leaf] = {"actions": actions, "U": U}
        return utility_model

    def _calculate_row_frequencies(self, df_sample):
        frequencies = {}
        for node in self.gt_bn.nodes():
            parents = list(self.gt_bn.get_parents(node))
            if not parents:
                continue
            parent_configs = df_sample[parents].value_counts(sort=False)
            frequencies[node] = parent_configs.values.astype(float)
        return frequencies

    def _ensure_complete_model(self, learned_model, gt_state_names):
        for node in learned_model.nodes():
            if learned_model.get_cpds(node) is None:
                parents = list(learned_model.get_parents(node))
                states = gt_state_names[node]
                card = len(states)
                if not parents:
                    values = np.ones((card, 1)) / card
                    cpd = TabularCPD(
                        variable=node,
                        variable_card=card,
                        values=values,
                        state_names={node: states},
                    )
                else:
                    parent_cards = [len(gt_state_names[p]) for p in parents]
                    n_cols = int(np.prod(parent_cards))
                    values = np.ones((card, n_cols)) / card
                    full_state_names = {node: states}
                    for p in parents:
                        full_state_names[p] = gt_state_names[p]
                    cpd = TabularCPD(
                        variable=node,
                        variable_card=card,
                        values=values,
                        evidence=parents,
                        evidence_card=parent_cards,
                        state_names=full_state_names,
                    )
                learned_model.add_cpds(cpd)
        return learned_model

    def _evaluate_cpt_metrics(self, learned_model, row_frequencies):
        node_metrics = []
        for node in self.gt_bn.nodes():
            if not (
                self.gt_bn.get_cpds(node) and learned_model.get_cpds(node)
            ):
                continue
            freq = row_frequencies.get(node, None)
            m = calculate_metrics_robust(
                self.gt_bn.get_cpds(node),
                learned_model.get_cpds(node),
                row_frequencies=freq,
            )
            if "error" not in m:
                node_metrics.append(m)
        if not node_metrics:
            return {}
        result = {}
        for key in [
            "KL_Divergence",
            "TVD",
            "MAE",
            "Hellinger",
            "RMSE",
            "Weighted_KL",
            "Weighted_Hellinger",
            "Weighted_TVD",
        ]:
            values = [m[key] for m in node_metrics if key in m]
            if values:
                result[f"Avg_{key}"] = float(np.mean(values))
        return result

    def _evaluate_inference(self, learned_model, test_queries):
        try:
            ve_gt = VariableElimination(self.gt_bn)
            ve_learned = VariableElimination(learned_model)
        except:
            return {}
        log_losses, hellinger_dists, kl_dists, accuracies = [], [], [], []
        for q in test_queries:
            targets, evidence, true_states = (
                q["targets"],
                q["evidence"],
                q["true_states"],
            )
            for target in targets:
                try:
                    q_true = ve_gt.query(
                        [target], evidence=evidence, show_progress=False
                    )
                    p_true = q_true.values
                    true_idx = list(q_true.state_names[target]).index(
                        true_states[target]
                    )
                    q_learned = ve_learned.query(
                        [target], evidence=evidence, show_progress=False
                    )
                    p_learned = q_learned.values
                    p_safe = np.clip(p_learned[true_idx], 1e-10, 1.0)
                    log_losses.append(-np.log(p_safe))
                    h = np.sqrt(
                        0.5
                        * np.sum((np.sqrt(p_true) - np.sqrt(p_learned)) ** 2)
                    )
                    hellinger_dists.append(h)
                    p_true_safe = np.clip(p_true, 1e-10, 1.0)
                    p_learned_safe = np.clip(p_learned, 1e-10, 1.0)
                    kl = np.sum(
                        p_true_safe * np.log(p_true_safe / p_learned_safe)
                    )
                    kl_dists.append(kl)
                    accuracies.append(
                        1.0 if np.argmax(p_learned) == true_idx else 0.0
                    )
                except:
                    pass
        if not log_losses:
            return {}
        return {
            "Inf_LogLoss": float(np.mean(log_losses)),
            "Inf_Hellinger": float(np.mean(hellinger_dists)),
            "Inf_KL": float(np.mean(kl_dists)),
            "Inf_Accuracy": float(np.mean(accuracies)),
        }

    def _evaluate_decisions(self, learned_model, test_queries, utility_model):
        try:
            ve_gt = VariableElimination(self.gt_bn)
            ve_learned = VariableElimination(learned_model)
        except:
            return {}
        regrets, accs = [], []
        for q in test_queries:
            targets, evidence = q["targets"], q["evidence"]
            for target in targets:
                if target not in utility_model:
                    continue
                try:
                    q_true = ve_gt.query(
                        [target], evidence=evidence, show_progress=False
                    )
                    q_learned = ve_learned.query(
                        [target], evidence=evidence, show_progress=False
                    )
                    p_true, p_learned = q_true.values, q_learned.values
                    U, actions = (
                        utility_model[target]["U"],
                        utility_model[target]["actions"],
                    )
                    eu_true = {
                        a: sum(
                            p_true[i] * U[a][s]
                            for i, s in enumerate(q_true.state_names[target])
                        )
                        for a in actions
                    }
                    a_true = max(eu_true, key=eu_true.get)
                    eu_learned = {
                        a: sum(
                            p_learned[i] * U[a][s]
                            for i, s in enumerate(q_true.state_names[target])
                        )
                        for a in actions
                    }
                    a_learned = max(eu_learned, key=eu_learned.get)
                    regrets.append(eu_true[a_true] - eu_true[a_learned])
                    accs.append(1.0 if a_true == a_learned else 0.0)
                except:
                    pass
        if not regrets:
            return {}
        return {
            "Dec_Regret": float(np.mean(regrets)),
            "Dec_Acc": float(np.mean(accs)),
        }

    def _prepare_ground_truth_data(self, df):
        df_discrete = df.copy()
        state_names = {}
        for col in df_discrete.columns:
            ctype = self.type_mapping.get(col, "categorical")
            if ctype == "continuous":
                valid_vals = (
                    pd.to_numeric(df_discrete[col], errors="coerce")
                    .dropna()
                    .values
                )
                if len(valid_vals) == 0:
                    df_discrete[col] = "low"
                    state_names[col] = ["high", "low", "medium"]
                    continue
                rng = np.random.default_rng(42)
                noise = rng.uniform(-1e-5, 1e-5, size=len(valid_vals))
                q33, q66 = np.quantile(valid_vals + noise, [0.3333, 0.6667])
                if q66 <= q33:
                    q66 = q33 + 1e-6

                def bin_mapper(x):
                    if pd.isna(x):
                        return np.nan
                    if x <= q33:
                        return "low"
                    if x <= q66:
                        return "medium"
                    return "high"

                binned = pd.to_numeric(
                    df_discrete[col], errors="coerce"
                ).apply(bin_mapper)
                if binned.isnull().any():
                    binned = binned.fillna(
                        binned.mode()[0] if len(binned.mode()) > 0 else "low"
                    )
                df_discrete[col] = binned.astype(str)
                state_names[col] = ["high", "low", "medium"]
            elif ctype == "binary":
                vals = pd.to_numeric(df_discrete[col], errors="coerce")

                def binary_map(x):
                    return np.nan if pd.isna(x) else ("yes" if x > 0 else "no")

                binned = vals.apply(binary_map)
                if binned.isnull().any():
                    binned = binned.fillna(
                        binned.mode()[0] if len(binned.mode()) > 0 else "no"
                    )
                df_discrete[col] = binned.astype(str)
                state_names[col] = ["no", "yes"]
            else:
                s = df_discrete[col].astype(str).replace("nan", np.nan)
                if s.isnull().any():
                    s = s.fillna(
                        s.mode()[0] if len(s.mode()) > 0 else "Unknown"
                    )
                df_discrete[col] = s
                state_names[col] = sorted(df_discrete[col].unique().tolist())
        return df_discrete, state_names

    def _align_json_reports(self, json_reports, seed=42):
        rng = np.random.default_rng(seed)
        aligned = []
        for report in json_reports:
            new_report = report.copy()
            new_report["variables"] = {
                k: v.copy() for k, v in report["variables"].items()
            }
            for var, info in new_report["variables"].items():
                ctype = self.type_mapping.get(var, "categorical")
                if var in self.state_names:
                    target_keys = self.state_names[var]
                else:
                    target_keys = (
                        ["high", "medium", "low"]
                        if ctype == "continuous"
                        else (
                            ["no", "yes"] if ctype == "binary" else ["unknown"]
                        )
                    )

                n_keys = len(target_keys)
                is_unknown = info.get("type") == "unknown" or (
                    "distribution" not in info and "mean" not in info
                )
                final_dist = {}

                if is_unknown:
                    final_dist = {k: 1.0 / n_keys for k in target_keys}
                elif ctype == "continuous":
                    mu = info.get("mean", info.get("median"))
                    sigma = info.get("std", info.get("sd"))
                    samples = []
                    if mu is not None:
                        if sigma is None:
                            sigma = 1.0
                        samples = rng.normal(mu, sigma, 1000)
                    if len(samples) > 0:
                        try:
                            binned = pd.qcut(
                                samples, q=3, labels=["low", "medium", "high"]
                            )
                            counts = binned.value_counts(normalize=True)
                            final_dist = counts.to_dict()
                        except:
                            final_dist = {k: 1.0 / 3 for k in target_keys}
                    else:
                        final_dist = {k: 1.0 / 3 for k in target_keys}
                elif ctype == "binary":
                    curr_dist = info.get("distribution", {}) or {}
                    sorted_keys = sorted(curr_dist.keys())
                    if len(sorted_keys) >= 2:
                        final_dist = {
                            "no": curr_dist[sorted_keys[0]],
                            "yes": curr_dist[sorted_keys[1]],
                        }
                    elif len(sorted_keys) == 1:
                        final_dist = (
                            {"no": 0.0, "yes": 1.0}
                            if str(sorted_keys[0]).lower()
                            in ["yes", "1", "true"]
                            else {"no": 1.0, "yes": 0.0}
                        )
                    else:
                        final_dist = {"no": 0.5, "yes": 0.5}
                else:
                    curr_dist = info.get("distribution", {}) or {}
                    for k in target_keys:
                        final_dist[k] = curr_dist.get(k, 0.0)

                for k in target_keys:
                    if k not in final_dist:
                        final_dist[k] = 0.0
                tot = sum(final_dist.values())
                if tot > 0:
                    final_dist = {k: v / tot for k, v in final_dist.items()}
                else:
                    final_dist = {k: 1.0 / n_keys for k in target_keys}

                info["type"] = "ordinal"
                info["categories"] = target_keys
                info["distribution"] = final_dist
            aligned.append(new_report)
        return aligned

    def get_summary_df(self):
        df = pd.DataFrame(self.results_log)
        if df.empty:
            return df
        metric_cols = [c for c in df.columns if c not in ["Strategy", "Seed"]]
        return df.groupby("Strategy")[metric_cols].agg(["mean", "std"])

    def export_latex(self, filename="results.tex"):
        raw_df = pd.DataFrame(self.results_log)
        if not raw_df.empty:
            raw_df.to_csv(filename.replace(".tex", "_raw.csv"), index=False)
        df = self.get_summary_df()
        if df.empty:
            return

        columns_map = [
            ("Parameter Fidelity", "W-Hell", "Avg_Weighted_Hellinger"),
            ("Parameter Fidelity", "W-KL", "Avg_Weighted_KL"),
            ("Parameter Fidelity", "TVD", "Avg_TVD"),
            ("Parameter Fidelity", "JSON", "JSON_AvgAbsDiff"),
            ("Inference Power", "LogLoss", "Inf_LogLoss"),
            ("Inference Power", "Hell", "Inf_Hellinger"),
            ("Inference Power", "KL", "Inf_KL"),
            ("Inference Power", "Acc", "Inf_Accuracy"),
            ("Decision Utility", "Regret", "Dec_Regret"),
            ("Decision Utility", "Acc", "Dec_Acc"),
            ("Resource Usage", "Time (s)", "Avg_Time_Sec"),
        ]

        baselines = [s for s in df.index if "Baseline" in s]
        randoms = [s for s in df.index if "Random" in s]
        others = [
            s for s in df.index if s not in baselines and s not in randoms
        ]

        if others:
            rank_df = pd.DataFrame(index=others)
            for _, _, metric in columns_map:
                if metric in df.columns.levels[0]:
                    vals = df.loc[others, (metric, "mean")]
                    rank_df[metric] = (
                        vals.rank(ascending=False)
                        if "Acc" in metric
                        else vals.rank(ascending=True)
                    )
            rank_df["avg_rank"] = rank_df.mean(axis=1)
            sorted_others = rank_df.sort_values("avg_rank").index.tolist()
        else:
            sorted_others = []

        new_index = baselines + sorted_others + randoms
        df = df.reindex(new_index)

        lines = []
        lines.append(r"\begin{table*}[ht]")
        lines.append(r"    \centering")
        lines.append(
            r"    \caption{Real-World Benchmark Results (Mean $\pm$ Std).}"
        )
        lines.append(r"    \label{tab:real_world_results}")
        lines.append(r"    \resizebox{\textwidth}{!}{%")
        col_def = "l " + "c" * 4 + " " + "c" * 4 + " " + "c" * 2 + " c"
        lines.append(f"        \\begin{{tabular}}{{{col_def}}}")
        lines.append(r"            \toprule")
        lines.append(
            r"                                 & \multicolumn{4}{c}{\textbf{Parameter Fidelity}} & \multicolumn{4}{c}{\textbf{Inference Power}} & \multicolumn{2}{c}{\textbf{Decision Utility}} & \textbf{Resource Usage} \\"
        )
        lines.append(
            r"            \cmidrule(lr){2-5} \cmidrule(lr){6-9} \cmidrule(lr){10-11} \cmidrule(lr){12-12}"
        )

        header_row = [r"\textbf{Strategy}"]
        for _, sub, _ in columns_map:
            header_row.append(r"\textbf{" + sub + "}")
        lines.append(r"            " + " & ".join(header_row) + r" \\")
        lines.append(r"            \midrule")

        for strategy in df.index:
            row_mean = [f"{strategy}"]
            row_std = [""]
            for _, _, internal in columns_map:
                if internal in df.columns.levels[0]:
                    val_mean = df.loc[strategy, (internal, "mean")]
                    val_std = df.loc[strategy, (internal, "std")]
                    row_mean.append(f"{val_mean:.3f}")
                    row_std.append(f"({val_std:.2f})")
                else:
                    row_mean.append("-")
                    row_std.append("")
            lines.append(r"            " + " & ".join(row_mean) + r" \\")
            lines.append(r"            " + " & ".join(row_std) + r" \\")
            if strategy != df.index[-1]:
                lines.append(r"            \addlinespace[0.2em]")

        lines.append(r"            \bottomrule")
        lines.append(r"        \end{tabular}%")
        lines.append(r"    }")
        lines.append(r"\end{table*}")

        with open(filename, "w") as f:
            f.write("\n".join(lines))
        print(f"Saved LaTeX to {filename}")
