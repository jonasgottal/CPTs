import json
import numpy as np
import pandas as pd
import time
import networkx as nx
import matplotlib.pyplot as plt
import scipy.stats as stats
from statsmodels.formula.api import logit


# 1. Compatibility Imports
try:
    from pgmpy.models import DiscreteBayesianNetwork
except ImportError:
    from pgmpy.models import BayesianNetwork as DiscreteBayesianNetwork


from pgmpy.factors.discrete import TabularCPD
from pgmpy.estimators import MaximumLikelihoodEstimator
from pgmpy.inference import VariableElimination


# =============================================================================
# 1. CORE STATISTICAL EXTRACTORS (FULLY IMPLEMENTED)
# =============================================================================


def run_correlation_summary(df, var1, var2):
    """Continuous <-> Continuous"""
    try:
        r, p = stats.pearsonr(df[var1], df[var2])
        return {
            "type": "pearson_correlation",
            "r": float(r),
            "p_value": float(p),
        }
    except:
        return None


def run_anova_summary(df, cat_var, cont_var):
    """Categorical (Any) -> Continuous"""
    try:
        groups = [df[df[cat_var] == c][cont_var] for c in df[cat_var].unique()]
        f, p = stats.f_oneway(*groups)
        # Eta-squared (Effect size)
        ss_between = sum(
            [len(g) * (g.mean() - df[cont_var].mean()) ** 2 for g in groups]
        )
        ss_total = ((df[cont_var] - df[cont_var].mean()) ** 2).sum()
        eta_sq = ss_between / ss_total if ss_total > 0 else 0
        return {
            "type": "anova",
            "f_stat": float(f),
            "eta_squared": float(eta_sq),
        }
    except:
        return None


def run_chi2_summary_universal(df, var1, var2):
    """Any Categorical <-> Any Categorical"""
    try:
        ct = pd.crosstab(df[var1], df[var2])
        if ct.size == 0:
            return None
        chi2, p, _, _ = stats.chi2_contingency(ct)
        n = ct.sum().sum()
        min_dim = min(ct.shape) - 1
        v = np.sqrt(chi2 / (n * min_dim)) if min_dim > 0 else 0
        return {
            "type": "chi_squared",
            "cramers_v": float(v),
            "p_value": float(p),
        }
    except:
        return None


def run_ttest_summary(df, bin_var, cont_var):
    """Binary -> Continuous Test"""
    cats = df[bin_var].unique()
    if len(cats) != 2:
        return None
    g1 = df[df[bin_var] == cats[0]][cont_var]
    g2 = df[df[bin_var] == cats[1]][cont_var]

    t, p = stats.ttest_ind(g1, g2)
    # Cohen's d
    n1, n2 = len(g1), len(g2)
    var1, var2 = np.var(g1, ddof=1), np.var(g2, ddof=1)
    pooled_se = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))
    d = (np.mean(g1) - np.mean(g2)) / pooled_se

    return {
        "type": "t-test",
        "group_means": {
            str(c): float(df[df[bin_var] == c][cont_var].mean()) for c in cats
        },
        "p_value": float(p),
        "effect_size_cohen_d": float(d),
        "sample_size": len(df),
    }


def run_logistic_summary(df, ordinal_var, binary_target):
    """Ordinal -> Binary Test (Augmented with Spearman R)"""
    y_vals = sorted(df[binary_target].unique())
    # Robust mapping: try to map 'yes'/1 to 1, else use sort order
    if "yes" in y_vals:
        y_map = {"yes": 1, "no": 0}
    else:
        y_map = {y_vals[0]: 0, y_vals[1]: 1}

    y = df[binary_target].map(y_map)

    result = {}

    # 1. Try Logistic Regression (Rich Output)
    try:
        model = logit(
            f"y ~ C({ordinal_var})",
            data=pd.concat([y, df[ordinal_var]], axis=1),
        ).fit(disp=0)
        odds_ratios = np.exp(model.params).to_dict()
        cis = np.exp(model.conf_int()).to_dict("index")
        clean_or = {
            k: v for k, v in odds_ratios.items() if "Intercept" not in k
        }
        result["odds_ratios"] = clean_or
        result["p_values"] = {k: float(model.pvalues[k]) for k in clean_or}
        result["confidence_intervals"] = {
            k: [cis[k][0], cis[k][1]] for k in clean_or
        }
    except:
        result["error"] = "Logistic regression failed"

    # 2. Add Scalar Target for MCMC (Spearman R)
    try:
        from scipy.stats import rankdata

        r, _ = stats.spearmanr(df[ordinal_var], y)
        result["spearman_r"] = float(r) if not np.isnan(r) else 0.0
    except:
        result["spearman_r"] = 0.0

    result["type"] = "logistic_regression"
    return result


def run_chi2_summary(df, var1, var2):
    """Nominal -> Nominal Test (Augmented with Cramer's V)"""
    ct = pd.crosstab(df[var1], df[var2])
    if ct.size == 0:
        return {"error": "Empty crosstab", "type": "chi_squared"}

    chi2, p, dof, ex = stats.chi2_contingency(ct)

    # Calculate Cramer's V (Effect Size)
    n = ct.sum().sum()
    min_dim = min(ct.shape) - 1
    if n > 0 and min_dim > 0:
        cramers_v = np.sqrt(chi2 / (n * min_dim))
    else:
        cramers_v = 0.0

    return {
        "type": "chi_squared",
        "p_value": float(p),
        "chi2_stat": float(chi2),
        "cramers_v": float(cramers_v),
        "contingency_table": ct.to_dict(),
    }


def get_true_sign(df, p, c):
    """Returns +1 or -1 based on Spearman correlation."""
    try:
        r, _ = stats.spearmanr(df[p], df[c])
        if np.isnan(r):
            return 1
        return 1 if r >= 0 else -1
    except:
        return 1


def generate_systematic_json(df, relations):
    report = []
    for rel in relations:
        parent, child = rel["parent"], rel["child"]
        p_type, c_type = rel["parent_type"], rel["child_type"]

        # 1. Variable Stats
        variables = {}
        for v, t in [(parent, p_type), (child, c_type)]:
            if t == "continuous":
                variables[v] = {
                    "type": "continuous",
                    "mean": float(df[v].mean()),
                    "variance": float(df[v].var()),
                }
            else:
                variables[v] = {
                    "type": t,
                    "categories": list(df[v].unique()),
                    "distribution": df[v]
                    .value_counts(normalize=True)
                    .to_dict(),
                }

        # 2. Universal Test Dispatcher
        tests = []

        if p_type == "continuous" and c_type == "continuous":
            tests.append(run_correlation_summary(df, parent, child))
        elif p_type == "continuous" and c_type != "continuous":
            tests.append(run_anova_summary(df, child, parent))
        elif c_type == "continuous" and p_type != "continuous":
            tests.append(run_anova_summary(df, parent, child))
            if p_type == "binary":
                tests.append(run_ttest_summary(df, parent, child))
        else:
            tests.append(run_chi2_summary_universal(df, parent, child))
            if p_type == "ordinal" and c_type == "binary":
                tests.append(run_logistic_summary(df, parent, child))

        valid_tests = [t for t in tests if t and "error" not in t]
        true_sign = get_true_sign(df, parent, child)
        sign_str = "positive" if true_sign > 0 else "negative"

        report.append(
            {
                "relation": f"{parent} -> {child}",
                "variables": variables,
                "tests": valid_tests,
                "correlation_sign": sign_str,
                "source": "Synthesized Clinical Data",
            }
        )
    return report


# =============================================================================
# 2. DATA UTILS & GROUND TRUTH
# =============================================================================


def synthesize_continuous_from_discrete(df, mapping):
    """Inflates discrete columns to continuous."""
    df_new = df.copy()
    for col, params in mapping.items():
        new_vals = np.zeros(len(df))
        discrete_col_name = params["source_col"]
        for cat, (mu, sigma) in params["map"].items():
            mask = df[discrete_col_name] == cat
            if mask.sum() > 0:
                new_vals[mask] = np.random.normal(mu, sigma, mask.sum())
        df_new[col] = new_vals
    return df_new


def extract_relations_from_model(model, continuous_map):
    """Builds relations metadata."""
    meta = []

    def guess_type(var_name):
        if var_name in continuous_map:
            return "continuous"
        cpd = model.get_cpds(var_name)
        if cpd.variable_card == 2:
            return "binary"
        if "low" in cpd.state_names[var_name]:
            return "ordinal"
        return "nominal"

    for child in model.nodes():
        parents = model.get_parents(child)
        for parent in parents:
            p_proxy = next(
                (
                    k
                    for k, v in continuous_map.items()
                    if v["source_col"] == parent
                ),
                parent,
            )
            c_proxy = next(
                (
                    k
                    for k, v in continuous_map.items()
                    if v["source_col"] == child
                ),
                child,
            )
            meta.append(
                {
                    "parent": p_proxy,
                    "parent_type": guess_type(p_proxy),
                    "child": c_proxy,
                    "child_type": guess_type(c_proxy),
                    "bn_edge": (parent, child),
                }
            )
    return meta


# =============================================================================
# 3. ROBUST METRICS & VISUALIZATION
# =============================================================================


def align_cpd(target_cpd, reference_cpd):
    """Forces target CPD to match reference parent order."""
    if target_cpd.variable != reference_cpd.variable:
        raise ValueError(f"Variable mismatch: {target_cpd.variable}")
    desired_order = reference_cpd.variables[1:]
    aligned_target = target_cpd.copy()
    if desired_order:
        aligned_target.reorder_parents(desired_order)
    return aligned_target


def hellinger_distance(P, Q):
    """Computes Hellinger distance between two probability distributions."""
    P_safe = np.clip(P, 1e-10, 1.0)
    Q_safe = np.clip(Q, 1e-10, 1.0)
    return np.sqrt(
        0.5 * np.sum((np.sqrt(P_safe) - np.sqrt(Q_safe)) ** 2, axis=0)
    )


def calculate_metrics_robust(true_cpd, est_cpd, row_frequencies=None):
    """
    Computes KL, MAE, RMSE, TVD, Hellinger, MaxErr.

    Args:
        true_cpd: Ground truth CPD
        est_cpd: Estimated CPD
        row_frequencies: Optional array of shape (n_cols,) with empirical frequencies
                        of parent configurations for weighted metrics
    """
    try:
        est_aligned = align_cpd(est_cpd, true_cpd)
        P = true_cpd.values
        Q = est_aligned.values

        # Normalize
        P = P / P.sum(axis=0, keepdims=True)
        Q = Q / Q.sum(axis=0, keepdims=True)
        epsilon = 1e-10
        P_safe = np.clip(P, epsilon, 1.0)
        Q_safe = np.clip(Q, epsilon, 1.0)

        # Per-row metrics
        diff = P - Q
        mae_per_row = np.mean(np.abs(diff), axis=0)
        rmse_per_row = np.sqrt(np.mean(diff**2, axis=0))
        tvd_per_row = 0.5 * np.sum(np.abs(diff), axis=0)
        kl_per_row = np.sum(P_safe * np.log(P_safe / Q_safe), axis=0)
        hellinger_per_row = hellinger_distance(P, Q)

        # Aggregate (unweighted mean)
        mae = float(np.mean(mae_per_row))
        rmse = float(np.mean(rmse_per_row))
        max_err = float(np.max(np.abs(diff)))
        tvd = float(np.mean(tvd_per_row))
        kl = float(np.mean(kl_per_row))
        hellinger = float(np.mean(hellinger_per_row))

        result = {
            "KL_Divergence": kl,
            "MAE": mae,
            "RMSE": rmse,
            "Max_Abs_Error": max_err,
            "TVD": tvd,
            "Hellinger": hellinger,
        }

        # Weighted metrics if frequencies provided
        if row_frequencies is not None:
            weights = row_frequencies / row_frequencies.sum()
            result["Weighted_KL"] = float(np.sum(kl_per_row * weights))
            result["Weighted_Hellinger"] = float(
                np.sum(hellinger_per_row * weights)
            )
            result["Weighted_TVD"] = float(np.sum(tvd_per_row * weights))

        return result
    except Exception as e:
        return {"error": str(e)}


def visualize_bn(model, filename="bn_rich.png"):
    infer = VariableElimination(model)
    node_labels = {}
    for node in model.nodes():
        q = infer.query([node], show_progress=False)
        probs = q.values
        states = model.get_cpds(node).state_names[node]
        label_str = f"{node}\n" + "-" * 12 + "\n"
        for state, p in zip(states, probs):
            label_str += f"{state}: {p:.2f}\n"
        node_labels[node] = label_str.strip()

    G = nx.DiGraph()
    G.add_edges_from(model.edges())
    try:
        pos = nx.nx_agraph.graphviz_layout(G, prog="dot")
    except:
        generations = list(nx.topological_generations(G))
        pos = {}
        y_gap = 1.0 / (len(generations) + 0.5)
        for gen_idx, gen_nodes in enumerate(generations):
            y = 1.0 - (gen_idx * y_gap) - 0.1
            x_gap = 1.0 / (len(gen_nodes) + 1)
            for node_idx, node in enumerate(gen_nodes):
                x = (node_idx + 1) * x_gap
                pos[node] = (x, y)

    plt.figure(figsize=(14, 10))
    nx.draw_networkx_edges(
        G,
        pos,
        edge_color="#555555",
        arrows=True,
        arrowstyle="-|>",
        arrowsize=55,
        width=2.0,
        connectionstyle="arc3,rad=0.1",
        node_size=20000,
    )
    nx.draw_networkx_labels(
        G,
        pos,
        labels=node_labels,
        font_size=11,
        font_family="monospace",
        bbox=dict(
            boxstyle="round,pad=0.5",
            fc="#ffffff",
            ec="#333333",
            alpha=1.0,
            linewidth=1.5,
        ),
    )
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Visualization saved to {filename}")


# =============================================================================
# 4. EXPERIMENT MANAGER CLASS (Orchestration)
# =============================================================================


class RandomBNGenerator:
    """
    Generates a random Bayesian Network with mixed semantic types.
    Decoupled from the experiment runner so the same network can be reused.
    """

    def __init__(self, n_nodes=5, edge_prob=0.3, seed=42, types=None):
        self.n_nodes = n_nodes
        self.edge_prob = edge_prob
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        if types is None:
            self.types = ["binary", "ordinal", "nominal", "continuous"]
        else:
            self.types = types

        # Artifacts to be generated
        self.bn = None
        self.continuous_map = {}
        self.node_types = {}

        self._generate()

    def _generate(self):
        # 1. Generate Structure (DAG)
        adj = np.zeros((self.n_nodes, self.n_nodes), dtype=bool)

        # A. Ensure each node i>0 has at least one parent in {0..i-1}
        for j in range(1, self.n_nodes):
            # choose at least 1 parent; could also sample k>1 with a small prob if you like
            parent = self.rng.integers(0, j)
            adj[parent, j] = True

        # B. Add extra random edges (respecting upper-triangular / acyclicity)
        rand_mask = (
            self.rng.random((self.n_nodes, self.n_nodes)) < self.edge_prob
        )
        rand_mask = np.triu(rand_mask, k=1)  # only i<j
        adj |= rand_mask  # keep required parents + random edges

        edges = []
        for i in range(self.n_nodes):
            for j in range(self.n_nodes):
                if adj[i, j]:
                    edges.append((f"Var_{i}", f"Var_{j}"))

        self.bn = DiscreteBayesianNetwork(edges)
        for i in range(self.n_nodes):
            self.bn.add_node(f"Var_{i}")

        # 2. Assign Types

        for i in range(self.n_nodes):
            t = self.rng.choice(self.types)
            self.node_types[f"Var_{i}"] = t

        # 3. Generate Parameters
        import networkx as nx

        dag = nx.DiGraph(self.bn.edges())
        dag.add_nodes_from(self.bn.nodes())

        try:
            ordered_nodes = list(nx.topological_sort(dag))
        except nx.NetworkXUnfeasible:
            ordered_nodes = sorted(list(self.bn.nodes()))

        for node in ordered_nodes:
            self._add_random_cpd(node)

        self.bn.check_model()

        # 4. Generate Continuous Mappings
        self._generate_continuous_maps()

    def _add_random_cpd(self, node):
        parents = list(self.bn.get_parents(node))

        def get_states(t):
            if t == "binary":
                return ["no", "yes"]
            if t == "continuous":
                return ["low", "normal", "high"]
            if t == "ordinal":
                return ["low", "medium", "high"]
            return ["A", "B", "C"]

        child_type = self.node_types[node]
        child_states = get_states(child_type)
        card = len(child_states)

        parent_cards = []
        full_state_names = {node: child_states}
        for p in parents:
            p_states = get_states(self.node_types[p])
            parent_cards.append(len(p_states))
            full_state_names[p] = p_states

        if not parents:
            values = self.rng.dirichlet([1.0] * card, size=1).T
        else:
            n_cols = int(np.prod(parent_cards))
            values = np.zeros((card, n_cols))

            parent_weights = []
            for _ in parents:
                w = 1.0 if self.rng.random() < 0.8 else -1.0
                parent_weights.append(w)

            max_score = sum(
                [
                    (c - 1) * w if w > 0 else 0
                    for c, w in zip(parent_cards, parent_weights)
                ]
            )
            min_score = sum(
                [
                    (c - 1) * w if w < 0 else 0
                    for c, w in zip(parent_cards, parent_weights)
                ]
            )
            score_range = max_score - min_score
            if score_range == 0:
                score_range = 1.0

            from itertools import product

            parent_ranges = [range(c) for c in parent_cards]

            for col_idx, config in enumerate(product(*parent_ranges)):
                raw_score = sum(
                    [idx * w for idx, w in zip(config, parent_weights)]
                )

                score_ratio = (raw_score - min_score) / score_range
                target_idx = score_ratio * (card - 1)

                child_indices = np.arange(card)
                weights_col = np.exp(-2.0 * (child_indices - target_idx) ** 2)

                noise = self.rng.uniform(0, 0.15, size=card)
                weights_col += noise

                values[:, col_idx] = weights_col / weights_col.sum()

        cpd = TabularCPD(
            variable=node,
            variable_card=card,
            values=values,
            evidence=parents,
            evidence_card=parent_cards,
            state_names=full_state_names,
        )
        self.bn.add_cpds(cpd)

    def _generate_continuous_maps(self):
        for node, t in self.node_types.items():
            if t == "continuous":
                real_name = f"{node}_real"

                base_mean = self.rng.uniform(10, 100)
                spread = self.rng.uniform(5, 20)

                mapping = {
                    "low": (base_mean - spread, spread / 3),
                    "normal": (base_mean, spread / 3),
                    "high": (base_mean + spread, spread / 3),
                }

                self.continuous_map[real_name] = {
                    "source_col": node,
                    "map": mapping,
                }


class ExperimentManager:
    def __init__(self, ground_truth_generator):
        self.generator = ground_truth_generator
        self.gt_bn = ground_truth_generator.bn
        self.continuous_map = ground_truth_generator.continuous_map
        self.results_log = []

    def _generate_test_queries(self, df_test, n_queries_per_sample=1):
        """
        Generates test cases for inference evaluation.

        Returns:
            List of dicts with keys: 'targets', 'evidence', 'true_states'
        """
        # Identify leaf nodes (no children)
        dag = nx.DiGraph(self.gt_bn.edges())
        dag.add_nodes_from(self.gt_bn.nodes())
        leaf_nodes = [n for n in dag.nodes() if dag.out_degree(n) == 0]

        if not leaf_nodes:
            # No leaves, use all nodes as potential targets
            leaf_nodes = list(self.gt_bn.nodes())

        test_queries = []
        rng = np.random.default_rng()

        for idx, row in df_test.iterrows():
            # Randomly select 1-k leaf nodes as targets
            n_targets = min(n_queries_per_sample, len(leaf_nodes))
            if n_targets == 0:
                continue

            targets = rng.choice(
                leaf_nodes, size=n_targets, replace=False
            ).tolist()

            # All other variables are evidence
            evidence_vars = [v for v in df_test.columns if v not in targets]
            evidence = {v: row[v] for v in evidence_vars}
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
        """
        Creates a simple utility model for decision evaluation.

        Returns:
            Dict mapping leaf_node -> {"actions": [...], "U": {action: {state: utility}}}
        """
        dag = nx.DiGraph(self.gt_bn.edges())
        dag.add_nodes_from(self.gt_bn.nodes())
        leaf_nodes = [n for n in dag.nodes() if dag.out_degree(n) == 0]

        utility_model = {}
        rng = np.random.default_rng(42)  # Fixed seed for consistent utilities

        for leaf in leaf_nodes:
            cpd = self.gt_bn.get_cpds(leaf)
            states = cpd.state_names[leaf]
            n_states = len(states)

            # Define actions (e.g., for binary: treat/no_treat, for ordinal: low/med/high intervention)
            if n_states == 2:
                actions = ["no_action", "action"]
            else:
                actions = [f"action_{i}" for i in range(n_states)]
            U = {}
            for action_idx, action in enumerate(actions):
                U[action] = {}
                for state_idx, state in enumerate(states):
                    # Reward for matching + small noise
                    base_utility = 10.0 if action_idx == state_idx else -5.0
                    noise = rng.uniform(-1.0, 1.0)
                    U[action][state] = base_utility + noise

            utility_model[leaf] = {"actions": actions, "U": U}

        return utility_model

    def _evaluate_inference(self, learned_model, test_queries):
        """
        Evaluates inference quality: log-loss, posterior distance, accuracy.

        Returns:
            Dict with aggregate metrics
        """
        try:
            ve_gt = VariableElimination(self.gt_bn)
            ve_learned = VariableElimination(learned_model)
        except Exception as e:
            print(f"    ! Inference evaluation failed (VE init): {e}")
            return {}

        log_losses = []
        hellinger_dists = []
        kl_dists = []
        accuracies = []

        for query_case in test_queries:
            targets = query_case["targets"]
            evidence = query_case["evidence"]
            true_states = query_case["true_states"]

            for target in targets:
                try:
                    # Ground truth posterior
                    q_true = ve_gt.query(
                        [target], evidence=evidence, show_progress=False
                    )
                    p_true = q_true.values
                    true_state = true_states[target]
                    true_state_idx = list(q_true.state_names[target]).index(
                        true_state
                    )

                    # Learned posterior
                    q_learned = ve_learned.query(
                        [target], evidence=evidence, show_progress=False
                    )
                    p_learned = q_learned.values

                    # Log-loss
                    p_learned_safe = np.clip(
                        p_learned[true_state_idx], 1e-10, 1.0
                    )
                    log_losses.append(-np.log(p_learned_safe))

                    # Hellinger distance between posteriors
                    h_dist = np.sqrt(
                        0.5
                        * np.sum((np.sqrt(p_true) - np.sqrt(p_learned)) ** 2)
                    )
                    hellinger_dists.append(h_dist)

                    # KL divergence
                    p_true_safe = np.clip(p_true, 1e-10, 1.0)
                    p_learned_safe_vec = np.clip(p_learned, 1e-10, 1.0)
                    kl = np.sum(
                        p_true_safe * np.log(p_true_safe / p_learned_safe_vec)
                    )
                    kl_dists.append(kl)

                    # Accuracy
                    pred_state_idx = np.argmax(p_learned)
                    accuracies.append(
                        1.0 if pred_state_idx == true_state_idx else 0.0
                    )

                except Exception as e:
                    # Skip this query on error
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
        """
        Evaluates decision quality: regret and decision accuracy.

        Returns:
            Dict with aggregate metrics
        """
        try:
            ve_gt = VariableElimination(self.gt_bn)
            ve_learned = VariableElimination(learned_model)
        except Exception as e:
            print(f"    ! Decision evaluation failed (VE init): {e}")
            return {}

        regrets = []
        decision_accs = []

        for query_case in test_queries:
            targets = query_case["targets"]
            evidence = query_case["evidence"]

            for target in targets:
                if target not in utility_model:
                    continue

                try:
                    # Get posteriors
                    q_true = ve_gt.query(
                        [target], evidence=evidence, show_progress=False
                    )
                    q_learned = ve_learned.query(
                        [target], evidence=evidence, show_progress=False
                    )

                    p_true = q_true.values
                    p_learned = q_learned.values
                    states = q_true.state_names[target]

                    U = utility_model[target]["U"]
                    actions = utility_model[target]["actions"]

                    # Compute optimal action under true posterior
                    expected_utilities_true = {}
                    for action in actions:
                        eu = sum(
                            [
                                p_true[i] * U[action][state]
                                for i, state in enumerate(states)
                            ]
                        )
                        expected_utilities_true[action] = eu
                    a_true = max(
                        expected_utilities_true,
                        key=expected_utilities_true.get,
                    )

                    # Compute optimal action under learned posterior
                    expected_utilities_learned = {}
                    for action in actions:
                        eu = sum(
                            [
                                p_learned[i] * U[action][state]
                                for i, state in enumerate(states)
                            ]
                        )
                        expected_utilities_learned[action] = eu
                    a_learned = max(
                        expected_utilities_learned,
                        key=expected_utilities_learned.get,
                    )

                    # Regret = E_true[U(a_true, s)] - E_true[U(a_learned, s)]
                    regret = (
                        expected_utilities_true[a_true]
                        - expected_utilities_true[a_learned]
                    )
                    regrets.append(regret)

                    # Decision accuracy
                    decision_accs.append(1.0 if a_true == a_learned else 0.0)

                except Exception as e:
                    pass

        if not regrets:
            return {}

        return {
            "Dec_Regret": float(np.mean(regrets)),
            "Dec_Acc": float(np.mean(decision_accs)),
        }

    def _calculate_row_frequencies(self, df_sample):
        """
        Calculates empirical frequencies of parent configurations for weighted CPT metrics.

        Returns:
            Dict mapping node -> numpy array of shape (n_parent_configs,)
        """
        frequencies = {}

        for node in self.gt_bn.nodes():
            parents = list(self.gt_bn.get_parents(node))
            if not parents:
                continue

            # Count occurrences of each parent configuration
            parent_configs = df_sample[parents].value_counts(sort=False)

            # Ensure alignment with CPD column order
            cpd = self.gt_bn.get_cpds(node)
            # pgmpy CPDs iterate in a specific order; we need to match it
            # This is complex; simplified version: just use counts as-is
            frequencies[node] = parent_configs.values.astype(float)

        return frequencies

    def _add_noise_to_json_marginals(self, json_reports, noise_strength, rng):
        """
        Adds noise to the variable statistics reported in the JSON.

        For discrete variables: Perturbs the distribution dict using a random Dirichlet
                                blend scaled by noise_strength.
        For continuous variables: Perturbs the mean (Gaussian shift) and variance (multiplicative).

        noise_strength: 0.0 = no noise, > 0.0 = increasing noise.
        """
        if noise_strength <= 0.0:
            return json_reports

        import copy

        noisy_reports = []

        for rel in json_reports:
            # Deep copy to avoid modifying the original dicts across runs
            rel_copy = copy.deepcopy(rel)

            vars_dict = rel_copy.get("variables", {})
            for var_name, info in vars_dict.items():
                v_type = info.get("type")

                if v_type == "continuous":
                    # --- Continuous: Perturb mean and variance ---
                    orig_mean = info.get("mean")
                    orig_var = info.get("variance")

                    if (
                        orig_mean is not None
                        and orig_var is not None
                        and orig_var > 0
                    ):
                        std = np.sqrt(orig_var)

                        # Shift the mean: scale the shift by std * noise_strength
                        new_mean = orig_mean + rng.normal(
                            0, std * noise_strength
                        )

                        # Shift the variance multiplicatively (log-normal) so it stays > 0
                        var_noise_factor = np.exp(
                            rng.normal(0, noise_strength)
                        )
                        new_var = orig_var * var_noise_factor

                        info["mean"] = float(new_mean)
                        info["variance"] = float(new_var)

                else:
                    # --- Discrete: Perturb the 'distribution' dictionary ---
                    dist = info.get("distribution")
                    if dist and isinstance(dist, dict) and len(dist) > 0:
                        cats = list(dist.keys())
                        probs = np.array([dist[c] for c in cats], dtype=float)

                        # Blend factor alpha in [0, 1].
                        # High noise_strength -> entirely random uniform-ish Dirichlet
                        alpha = min(noise_strength, 1.0)

                        random_probs = rng.dirichlet(np.ones(len(cats)))
                        noisy_probs = (
                            1.0 - alpha
                        ) * probs + alpha * random_probs

                        # Re-normalize just to be safe
                        noisy_probs = noisy_probs / noisy_probs.sum()

                        # Update the dictionary inplace
                        for idx, c in enumerate(cats):
                            dist[c] = float(noisy_probs[idx])

            noisy_reports.append(rel_copy)

        return noisy_reports

    def _corrupt_tests_to_independence(
        self, json_reports, independence_strength, rng
    ):
        """
        With probability independence_strength per relation, overwrite test results to
        represent 'no evidence of association' (approx corr = 0, high p-value, etc.).

        independence_strength: 0.0 = keep all tests intact,
                               1.0 = all tests replaced by null/independent versions.
        """
        if independence_strength <= 0.0:
            return json_reports

        corrupted = []
        for rel in json_reports:
            rel_copy = dict(rel)
            tests = rel_copy.get("tests", [])
            if not tests:
                corrupted.append(rel_copy)
                continue

            if rng.random() >= independence_strength:
                # keep original
                corrupted.append(rel_copy)
                continue

            # Build "null" versions of tests
            null_tests = []
            for t in tests:
                if not isinstance(t, dict):
                    continue
                t_type = t.get("type", "")

                base = {"type": t_type}

                # For common test types from your eval.py
                if t_type in ("pearson_correlation", "correlation", "pearson"):
                    base.update(
                        {
                            "r": 0.0,
                            "p_value": 1.0,
                        }
                    )
                elif t_type in ("anova",):
                    base.update(
                        {
                            "f_stat": 0.0,
                            "eta_squared": 0.0,
                            "p_value": 1.0,
                        }
                    )
                elif t_type in ("chisquared", "chi2"):
                    base.update(
                        {
                            "chi2_stat": 0.0,
                            "p_value": 1.0,
                            "cramers_v": 0.0,
                        }
                    )
                elif t_type in ("t-test", "ttest"):
                    base.update(
                        {
                            "t_stat": 0.0,
                            "p_value": 1.0,
                            "effect_size_cohen_d": 0.0,
                        }
                    )
                elif t_type in ("logistic_regression", "logistic"):
                    # zero effect: odds ratios ~ 1, large p-values
                    base.update(
                        {
                            "odds_ratios": {},
                            "p_values": {},
                            "spearman_r": 0.0,
                        }
                    )
                else:
                    # Fallback: just say "no effect"
                    base.update({"p_value": 1.0})

                null_tests.append(base)

            rel_copy["tests"] = null_tests

            # Also flip correlation_sign to random or neutral if you want
            rel_copy["correlation_sign"] = (
                "positive" if rng.random() < 0.5 else "negative"
            )

            corrupted.append(rel_copy)

        return corrupted

    def run_experiment(
        self,
        strategies,
        n_seeds=5,
        n_samples=10000,
        n_test_queries=500,
        iterations=1000,
        noise_strength=0.0,
        independence_strength=0.0,
    ):
        print(
            f"Starting Experiment on Random Graph ({self.generator.n_nodes} nodes, Seed {self.generator.seed})"
        )

        for seed in range(n_seeds):
            print(f"  > Experiment Run {seed+1}/{n_seeds} (Data Seed {seed})")

            # 1. Data Synthesis
            np.random.seed(seed)
            df_discrete = self.gt_bn.simulate(
                n_samples=n_samples, show_progress=False, seed=seed
            )

            # Inflate
            df_mixed = df_discrete.copy()
            if self.continuous_map:
                df_mixed = synthesize_continuous_from_discrete(
                    df_discrete, self.continuous_map
                )

            rng = np.random.default_rng(seed)

            # Generate JSON Reports
            rels_meta = extract_relations_from_model(
                self.gt_bn, self.continuous_map
            )
            rels_meta = [r for r in rels_meta if r["bn_edge"]]

            json_reports = generate_systematic_json(df_mixed, rels_meta)

            json_reports = self._add_noise_to_json_marginals(
                json_reports,
                noise_strength=noise_strength,
                rng=rng,
            )

            json_reports = self._corrupt_tests_to_independence(
                json_reports,
                independence_strength=independence_strength,
                rng=rng,
            )

            # save to file for inspection
            with open(f"json_report.json", "w") as f:
                json.dump(json_reports, f, indent=2)

            # Setup for Learners
            skeleton = DiscreteBayesianNetwork(self.gt_bn.edges())
            skeleton.add_nodes_from(self.gt_bn.nodes())

            gt_state_names = {
                n: self.gt_bn.get_cpds(n).state_names[n]
                for n in self.gt_bn.nodes()
            }

            # 2. Generate Test Queries (once per seed, shared across strategies)
            df_test = self.gt_bn.simulate(
                n_samples=n_test_queries, show_progress=False, seed=seed + 1000
            )
            test_queries = self._generate_test_queries(
                df_test, n_queries_per_sample=1
            )
            utility_model = self._create_utility_model()
            print(
                f"    Generated {len(test_queries)} test queries for {len(utility_model)} leaf nodes"
            )

            # 3. Calculate row frequencies for weighted metrics
            row_frequencies = self._calculate_row_frequencies(df_discrete)

            # 4. Run Strategies
            for strat_name, strat_func in strategies.items():
                print(f"    Running strategy: {strat_name}")
                try:
                    # time of start
                    start_time = time.time()
                    learned_model = strat_func(
                        json_reports,
                        skeleton,
                        df_discrete,
                        gt_state_names,
                        seed,
                        iterations,
                        n_test_queries,
                    )
                    learned_model = self._ensure_complete_model(
                        learned_model, gt_state_names
                    )
                    end_time = time.time()
                    elapsed = end_time - start_time

                    # A. CPT-level evaluation
                    cpt_metrics = self._evaluate_cpt_metrics(
                        learned_model, row_frequencies
                    )

                    # B. Inference-level evaluation
                    inf_metrics = self._evaluate_inference(
                        learned_model, test_queries
                    )

                    # C. Decision-level evaluation
                    dec_metrics = self._evaluate_decisions(
                        learned_model, test_queries, utility_model
                    )
                    faith_metrics = self._evaluate_json_faithfulness(
                        learned_model, json_reports
                    )

                    # Merge all metrics
                    all_metrics = {
                        "Strategy": strat_name,
                        "Seed": seed,
                    }
                    all_metrics.update(cpt_metrics)
                    all_metrics.update(inf_metrics)
                    all_metrics.update(dec_metrics)
                    all_metrics.update(faith_metrics)
                    all_metrics.update({"Avg_Time_Sec": elapsed})

                    self.results_log.append(all_metrics)

                except Exception as e:
                    print(f"    ! Strategy '{strat_name}' failed: {e}")
                    import traceback

                    traceback.print_exc()

    def _evaluate_json_faithfulness(self, learned_model, json_reports):
        """
        Calculates the average Total Variation Distance (TVD) between the
        distributions specified in the JSON reports and the distributions
        inferred from the learned model.

        Returns:
            Dict with 'Faithfulness_TVD' metric.
        """
        try:
            ve_learned = VariableElimination(learned_model)
        except:
            return {}

        tvds = []

        for report in json_reports:
            # Each report contains 'variables' with marginals
            # We focus on the marginals provided in the report for simplicity and robustness
            # (Conditionals in JSON are harder to parse generically if they aren't structured as CPTs)

            for var_name, info in report.get("variables", {}).items():
                if "distribution" not in info:
                    continue

                # 1. Get JSON Distribution
                json_dist = info["distribution"]  # Dict {state: prob}
                if not json_dist:
                    continue

                # Normalize JSON dist just in case
                total = sum(json_dist.values())
                if total == 0:
                    continue
                json_probs = {k: v / total for k, v in json_dist.items()}

                try:
                    # 2. Query Learned Model
                    # We need to map the JSON state names to the model's state indices
                    # The learned model should have the same variable names
                    if var_name not in learned_model.nodes():
                        continue

                    q_learned = ve_learned.query(
                        [var_name], show_progress=False
                    )
                    model_states = q_learned.state_names[var_name]
                    model_values = q_learned.values

                    # 3. Align & Compare
                    # We assume the model covers all states.
                    # If JSON has states missing from model, we treat them as 0 in model (or error).
                    # If model has states missing from JSON, we treat them as 0 in JSON.

                    all_states = set(json_probs.keys()).union(
                        set(model_states)
                    )

                    tvd_sum = 0.0
                    for state in all_states:
                        # P_json
                        p_j = json_probs.get(state, 0.0)

                        # P_model
                        if state in model_states:
                            idx = list(model_states).index(state)
                            p_m = model_values[idx]
                        else:
                            p_m = 0.0

                        tvd_sum += abs(p_j - p_m)

                    # TVD = 0.5 * sum(|p - q|)
                    tvds.append(0.5 * tvd_sum)

                except Exception as e:
                    # print(f"Faithfulness check error on {var_name}: {e}")
                    pass

        if not tvds:
            return {}

        return {"Faithfulness_TVD": float(np.mean(tvds))}

    def _ensure_complete_model(self, learned_model, gt_state_names):
        """
        Ensures all nodes in the learned model have CPDs.
        For missing nodes (typically roots), copies from ground truth or creates uniform CPDs.

        Args:
            learned_model: The learned BN (may have missing CPDs)
            gt_state_names: Dict mapping node -> list of states

        Returns:
            Complete model with all CPDs
        """
        for node in learned_model.nodes():
            if learned_model.get_cpds(node) is None:
                # Node has no CPD - create one
                parents = list(learned_model.get_parents(node))
                states = gt_state_names[node]
                card = len(states)

                if not parents:
                    # Root node - use uniform or copy from ground truth
                    gt_cpd = self.gt_bn.get_cpds(node)
                    if gt_cpd is not None:
                        # Copy marginal from ground truth (for fair evaluation)
                        values = gt_cpd.values.copy()
                    else:
                        # Fallback to uniform
                        values = (
                            np.ones((card, 1)) / card
                        )  # *** MUST BE 2D: (card, 1) ***

                    cpd = TabularCPD(
                        variable=node,
                        variable_card=card,
                        values=values,  # Now guaranteed 2D
                        state_names={node: states},
                    )
                else:
                    # Node with parents but no CPD - create uniform conditional
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
                print(f"      Added missing CPD for {node}")

        # Verify model is complete
        try:
            learned_model.check_model()
        except Exception as e:
            print(f"      Warning: Model check failed after completion: {e}")

        return learned_model

    def _evaluate_cpt_metrics(self, learned_model, row_frequencies):
        """
        Evaluates CPT-level metrics with optional weighted versions.

        Returns:
            Dict with aggregate CPT metrics
        """
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

        # Aggregate across nodes
        result = {}
        for key in ["KL_Divergence", "TVD", "MAE", "Hellinger", "RMSE"]:
            values = [m[key] for m in node_metrics if key in m]
            if values:
                result[f"Avg_{key}"] = float(np.mean(values))

        # Weighted metrics (if available)
        for key in ["Weighted_KL", "Weighted_Hellinger", "Weighted_TVD"]:
            values = [m[key] for m in node_metrics if key in m]
            if values:
                result[f"Avg_{key}"] = float(np.mean(values))

        return result

    def get_summary_df(self):
        df = pd.DataFrame(self.results_log)
        if df.empty:
            return df

        # Identify all metric columns
        metric_cols = [c for c in df.columns if c not in ["Strategy", "Seed"]]

        return df.groupby("Strategy")[metric_cols].agg(["mean", "std"])

    def export_latex(self, filename="results.tex"):
        # --- 1. Save Raw CSV (New Feature) ---
        raw_df = pd.DataFrame(self.results_log)
        if not raw_df.empty:
            csv_name = filename.replace(".tex", "_raw.csv")
            raw_df.to_csv(csv_name, index=False)
            print(f"Saved Raw Results to {csv_name}")

        visualize_bn(self.gt_bn, filename.replace(".tex", "_gt.pdf"))

        # Get the summary dataframe (Mean and Std)
        df = self.get_summary_df()

        if df.empty:
            print("No results to export.")
            return

        # Define the mapping
        columns_map = [
            ("Parameter Fidelity", "W-Hell", "Avg_Weighted_Hellinger"),
            ("Parameter Fidelity", "W-KL", "Avg_Weighted_KL"),
            ("Parameter Fidelity", "TVD", "Avg_TVD"),
            ("Parameter Fidelity", "JSON", "Faithfulness_TVD"),
            ("Inference Power", "LogLoss", "Inf_LogLoss"),
            ("Inference Power", "Hell", "Inf_Hellinger"),
            ("Inference Power", "KL", "Inf_KL"),
            ("Inference Power", "Acc", "Inf_Accuracy"),
            ("Decision Utility", "Regret", "Dec_Regret"),
            ("Decision Utility", "Acc", "Dec_Acc"),
            ("Resource Usage", "Time (s)", "Avg_Time_Sec"),
        ]

        # --- SORTING LOGIC START ---

        # A. Identify Fixed Rows
        baselines = [s for s in df.index if "Baseline" in s]
        randoms = [s for s in df.index if "Random" in s]
        others = [
            s for s in df.index if s not in baselines and s not in randoms
        ]

        # B. Rank the "Others" (Middle Class)
        if others:
            rank_df = pd.DataFrame(index=others)

            for _, _, metric in columns_map:
                if metric not in df.columns.levels[0]:
                    continue

                # Extract mean values for the middle strategies
                vals = df.loc[others, (metric, "mean")]

                # Rank: Ascending for Error (Low=Good), Descending for Acc (High=Good)
                if "Acc" in metric:
                    r = vals.rank(ascending=False)
                else:
                    r = vals.rank(ascending=True)

                rank_df[metric] = r

            # Sort by Average Rank
            rank_df["avg_rank"] = rank_df.mean(axis=1)
            sorted_others = rank_df.sort_values("avg_rank").index.tolist()
        else:
            sorted_others = []

        # C. Reconstruct the Index
        # Top: Baseline -> Middle: Sorted by Rank -> Bottom: Random
        new_index = baselines + sorted_others + randoms
        df = df.reindex(new_index)

        # --- SORTING LOGIC END ---

        # --- BOLDING LOGIC START ---
        # Recalculate bests based on the competing strategies (excluding baseline)
        best_vals = {}
        competitors = [s for s in df.index if "Baseline" not in s]

        for _, _, metric in columns_map:
            if metric not in df.columns.levels[0]:
                continue

            means = df.loc[competitors, (metric, "mean")]
            if "Acc" in metric:
                best_vals[metric] = means.max()
            else:
                best_vals[metric] = means.min()
        # --- BOLDING LOGIC END ---

        # --- Build LaTeX ---
        lines = []
        lines.append(r"\begin{table*}[ht]")
        lines.append(r"    \centering")
        lines.append(
            r"    \caption{Reconstruction Performance (Mean $\pm$ Std). \textbf{Bold} indicates best result (excluding Baseline). Ordered by average rank.}"
        )
        lines.append(r"    \label{tab:results}")
        lines.append(r"    \resizebox{\textwidth}{!}{%")

        # UPDATED: Added + "c" * 1 at the end for the Time column
        # Total columns: 1 (Strategy) + 4 (Param) + 4 (Inf) + 2 (Dec) + 1 (Res)
        col_def = "l " + "c" * 4 + " " + "c" * 4 + " " + "c" * 2 + " c"
        lines.append(f"        \\begin{{tabular}}{{{col_def}}}")
        lines.append(r"            \toprule")

        # Header Row 1: UPDATED with "Resource Usage"
        lines.append(
            r"                                 & \multicolumn{4}{c}{\textbf{Parameter Fidelity}} & \multicolumn{4}{c}{\textbf{Inference Power}} & \multicolumn{2}{c}{\textbf{Decision Utility}} & \textbf{Resource Usage} \\"
        )
        # UPDATED: Added \cmidrule(lr){11-11}
        lines.append(
            r"            \cmidrule(lr){2-4} \cmidrule(lr){5-8} \cmidrule(lr){9-10} \cmidrule(lr){11-11}"
        )

        # Header Row 2
        header_row = [r"\textbf{Strategy}"]
        for group, sub, internal in columns_map:
            header_row.append(r"\textbf{" + sub + "}")
        lines.append(r"            " + " & ".join(header_row) + r" \\")
        lines.append(r"            \midrule")

        # Data Rows
        for strategy in df.index:
            row_mean = [f"{strategy}"]
            row_std = [""]

            for group, sub, internal in columns_map:
                if internal in df.columns.levels[0]:
                    val_mean = df.loc[strategy, (internal, "mean")]
                    val_std = df.loc[strategy, (internal, "std")]

                    mean_str = f"{val_mean:.3f}"

                    # Bolding Logic
                    is_baseline = "Baseline" in strategy
                    if not is_baseline and internal in best_vals:
                        best = best_vals[internal]
                        if abs(val_mean - best) < 1e-9:
                            mean_str = f"\\textbf{{{mean_str}}}"

                    row_mean.append(mean_str)
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
