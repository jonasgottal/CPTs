import numpy as np
import pandas as pd
import scipy.stats as stats
from pgmpy.models import DiscreteBayesianNetwork
from pgmpy.factors.discrete import TabularCPD
from pgmpy.estimators import MaximumLikelihoodEstimator


# -----------------------------------------------------------------------------
# 1. HELPER: Gaussian Copula Sampling
# -----------------------------------------------------------------------------
def sample_gaussian_copula(n_samples, correlation, margin_1_fn, margin_2_fn):
    """
    Generates paired samples (x, y) with rank correlation ~ 'correlation'
    and marginals defined by margin_1_fn, margin_2_fn.
    """
    mean = [0, 0]
    cov = [[1, correlation], [correlation, 1]]

    # 1. Sample from Bivariate Normal
    xy_normal = np.random.multivariate_normal(mean, cov, size=n_samples)

    # 2. Probability Integral Transform (Normal CDF -> Uniform [0,1])
    u = stats.norm.cdf(xy_normal)

    # 3. Inverse CDF of target marginals
    x = margin_1_fn(u[:, 0])
    y = margin_2_fn(u[:, 1])

    return pd.DataFrame({"Parent": x, "Child": y})


class CopulaOptimization:
    def __init__(
        self,
        json_reports,
        skeleton,
        state_names,
        continuous_map=None,
    ):
        self.reports = json_reports
        self.skeleton = skeleton
        self.state_names = state_names
        self.continuous_map = continuous_map or {}

        # 1. Initialize Params
        self.global_params = self._init_global_params()
        self.edge_params = self._init_edge_params()

        # 2. Smart Init (Independent BN)
        self.best_bn = self._create_independent_init_bn()
        self.best_loss = float("inf")

    def _init_global_params(self):
        """Parses JSONs to get means, variances, and category probabilities."""
        params = {}
        for report in self.reports:
            for var_name, info in report["variables"].items():
                if var_name in params:
                    continue

                # Robust Type Access
                var_type = info.get("type", "nominal")

                if var_type == "continuous":
                    params[var_name] = {
                        "mu": info.get("mean", 0),
                        "sigma": np.sqrt(info.get("variance", 1)),
                    }
                elif var_type == "binary":
                    cats = info.get("categories", ["0", "1"])
                    dist = info.get("distribution", {})
                    # Assume last cat is "positive"
                    target = cats[-1] if cats else "1"
                    p = dist.get(target, 0.5)
                    params[var_name] = {"p": p, "cats": cats}
                else:  # ordinal/nominal
                    cats = info.get("categories", ["A", "B"])
                    dist = info.get("distribution", {})
                    probs = [
                        dist.get(c, 1.0 / max(1, len(cats))) for c in cats
                    ]
                    # Normalize
                    s = np.sum(probs)
                    if s == 0:
                        probs = [1.0 / len(cats)] * len(cats)
                    else:
                        probs = np.array(probs) / s
                    params[var_name] = {"probs": probs, "cats": cats}
        return params

    def _init_edge_params(self):
        params = {}
        for report in self.reports:
            rel = report["relation"]
            sign = (
                1.0
                if report.get("correlation_sign", "positive") == "positive"
                else -1.0
            )
            params[rel] = 0.1 * sign  # Start slightly in the correct direction
        return params

    def _create_independent_init_bn(self):
        model = DiscreteBayesianNetwork(self.skeleton.edges())
        model.add_nodes_from(self.skeleton.nodes())

        for node in model.nodes():
            parents = list(model.get_parents(node))
            if node not in self.state_names:
                continue

            card = len(self.state_names[node])
            parent_cards = [len(self.state_names[p]) for p in parents]

            # Get Marginal
            if node in self.global_params:
                p = self.global_params[node]
                if "probs" in p:
                    marginal = p["probs"]
                elif "p" in p:
                    marginal = [1 - p["p"], p["p"]]
                else:
                    marginal = [1.0 / card] * card
            else:
                marginal = [1.0 / card] * card

            # Broadcast
            n_cols = int(np.prod(parent_cards)) if parent_cards else 1
            # Reshape marginal to (card, 1) and repeat
            vals = np.tile(np.array(marginal).reshape(-1, 1), n_cols)

            cpd = TabularCPD(
                node,
                card,
                vals,
                evidence=parents,
                evidence_card=parent_cards,
                state_names=self.state_names,
            )
            model.add_cpds(cpd)

        # Robust check (might fail if graph cycles exist in skeleton, but here assumed DAG)
        try:
            model.check_model()
        except:
            pass
        return model

    def _simulation_step(self, current_globals, current_edges):
        n = 1000
        vars_list = list(current_globals.keys())
        n_vars = len(vars_list)
        if n_vars == 0:
            return pd.DataFrame()

        corr_matrix = np.eye(n_vars)
        var_idx = {v: i for i, v in enumerate(vars_list)}

        for rel_str, rho in current_edges.items():
            try:
                p, c = rel_str.split(" -> ")
                if p in var_idx and c in var_idx:
                    i, j = var_idx[p], var_idx[c]
                    corr_matrix[i, j] = rho
                    corr_matrix[j, i] = rho
            except:
                pass

        # Force PSD
        evals, evecs = np.linalg.eigh(corr_matrix)
        if np.min(evals) < 1e-6:
            evals = np.maximum(evals, 1e-6)
            corr_matrix = evecs @ np.diag(evals) @ evecs.T

        Z = np.random.multivariate_normal(
            np.zeros(n_vars), corr_matrix, size=n
        )
        U = stats.norm.cdf(Z)

        df_dict = {}
        for i, var in enumerate(vars_list):
            u_vec = U[:, i]
            param = current_globals[var]

            if "mu" in param:
                df_dict[var] = stats.norm.ppf(
                    u_vec, loc=param["mu"], scale=param["sigma"]
                )
            elif "p" in param:
                cats = param["cats"]
                # Inverse transform for binary: u < (1-p) -> 0
                df_dict[var] = np.where(
                    u_vec < (1 - param["p"]), cats[0], cats[1]
                )
            elif "probs" in param:
                cats = param["cats"]
                probs = np.cumsum(param["probs"])
                mapped = np.searchsorted(probs, u_vec)
                mapped = np.clip(mapped, 0, len(cats) - 1)
                df_dict[var] = [cats[k] for k in mapped]

        return pd.DataFrame(df_dict)

    def _discretize_and_learn(self, df_synth):
        df_disc = df_synth.copy()

        # Reverse Map Continuous
        for cont_col, info in self.continuous_map.items():
            discrete_target = info["source_col"]
            if cont_col in df_disc.columns:
                try:
                    labels = self.state_names.get(discrete_target, [])
                    if len(labels) > 0:
                        df_disc[discrete_target] = pd.qcut(
                            df_disc[cont_col], len(labels), labels=labels
                        )
                except:
                    pass

        model = DiscreteBayesianNetwork(self.skeleton.edges())
        model.add_nodes_from(self.skeleton.nodes())

        cols = [c for c in model.nodes() if c in df_disc.columns]
        if len(cols) < len(model.nodes()):
            return self.best_bn  # Fail safe

        try:
            model.fit(
                df_disc[cols],
                estimator=MaximumLikelihoodEstimator,
                state_names=self.state_names,
            )
            return model
        except:
            return self.best_bn

    def _calculate_loss(self, df_sample):
        """
        Calculates loss by comparing synthetic sample statistics against JSON targets.
        Crucially uses 'correlation_sign' to enforce directionality for unsigned tests.
        """
        total_loss = 0.0

        for report in self.reports:
            rel = report["relation"]
            try:
                p_col, c_col = rel.split(" -> ")
            except:
                continue

            # Check for column existence (including proxies if logic added)
            if (
                p_col not in df_sample.columns
                or c_col not in df_sample.columns
            ):
                continue

            # 1. Determine Target Direction
            # Default to positive if missing
            sign_str = report.get("correlation_sign", "positive")
            target_sign = 1.0 if sign_str == "positive" else -1.0

            for test in report.get("tests", []):
                # Skip failed tests
                if "error" in test:
                    continue

                test_type = test.get("type", "")

                # CASE A: T-TEST (Directional Difference)
                if test_type == "t-test":
                    means = test.get("group_means", {})
                    if len(means) < 2:
                        continue
                    cats = list(means.keys())

                    # Target Gap
                    target_diff = means[cats[1]] - means[cats[0]]

                    try:
                        g1 = df_sample[df_sample[p_col] == cats[0]][
                            c_col
                        ].mean()
                        g2 = df_sample[df_sample[p_col] == cats[1]][
                            c_col
                        ].mean()
                        curr_diff = g2 - g1
                        total_loss += (curr_diff - target_diff) ** 2
                    except:
                        total_loss += 1.0

                # CASE B: CORRELATION (Already Signed)
                elif test_type in [
                    "pearson_correlation",
                    "spearman_correlation",
                ]:
                    # JSON usually has 'r' or 'spearman_r'
                    target_r = test.get("r", test.get("spearman_r", 0.0))

                    try:
                        # Calculate matching sample metric
                        if test_type == "pearson_correlation":
                            curr_r = df_sample[p_col].corr(df_sample[c_col])
                        else:
                            curr_r, _ = stats.spearmanr(
                                df_sample[p_col], df_sample[c_col]
                            )

                        total_loss += (curr_r - target_r) ** 2
                    except:
                        total_loss += 0.5

                # CASE C: UNSIGNED ASSOCIATION (Chi2 / ANOVA)
                # We use the 'target_sign' to turn the Magnitude into a Signed Target
                elif test_type in ["chi_squared", "anova"]:

                    # Extract Magnitude
                    if test_type == "chi_squared":
                        mag = test.get("cramers_v", 0.0)
                    else:  # anova
                        # eta_squared is variance explained (0-1). Sqrt it to get linear-ish correlation scale
                        eta = test.get("eta_squared", 0.0)
                        mag = np.sqrt(eta)

                    # Construct Signed Target
                    target_r_proxy = target_sign * mag
                    if (
                        df_sample[p_col].nunique() <= 1
                        or df_sample[c_col].nunique() <= 1
                    ):
                        curr_r = 0.0  # Undefined correlation is effectively 0 for this purpose
                        # Optional: Add a small penalty
                        total_loss += 0.5
                    else:
                        try:
                            # Optimization Proxy: Sample Spearman R
                            # We try to make the Sample Correlation match (Sign * CramersV)
                            curr_r, _ = stats.spearmanr(
                                df_sample[p_col], df_sample[c_col]
                            )

                            if np.isnan(curr_r):
                                curr_r = 0.0

                            total_loss += (curr_r - target_r_proxy) ** 2
                        except:
                            curr_r = 0.0
                            total_loss += 0.5

        return total_loss

    def fit(self, steps=500, n_samples=500):
        """
        Simple greedy MCMC optimization.
        Only accepts improvements. No early stopping.
        """
        current_loss = float("inf")
        improvements = 0

        # Initial evaluation
        try:
            df_init = self._simulation_step(
                self.global_params, self.edge_params
            )
            bn_init = self._discretize_and_learn(df_init)
            df_val_init = bn_init.simulate(n_samples=300, show_progress=False)

            df_val_inf = df_val_init.copy()
            for c_col, info in self.continuous_map.items():
                src = info["source_col"]
                if src in df_val_inf:

                    def mapper(x):
                        return info["map"].get(x, (0, 0))[0]

                    df_val_inf[c_col] = df_val_inf[src].apply(mapper)

            current_loss = self._calculate_loss(df_val_inf)
            self.best_bn = bn_init
            # print(f"Copula MCMC: Initial Loss = {current_loss:.4f}")
        except Exception as e:
            print(f"DEBUG: Initial eval failed: {e}")

        for i in range(steps):
            # Perturb one random edge
            new_edges = self.edge_params.copy()
            if not new_edges:
                break

            k = np.random.choice(list(new_edges.keys()))
            new_edges[k] += np.random.normal(0, 0.2)
            new_edges[k] = np.clip(new_edges[k], -0.95, 0.95)

            try:
                # Generate synthetic data
                df_synth = self._simulation_step(self.global_params, new_edges)

                # Learn BN from synthetic data
                bn = self._discretize_and_learn(df_synth)

                # Sample from learned BN
                df_valid = bn.simulate(
                    n_samples=n_samples, show_progress=False
                )

                # Inflate continuous variables
                df_val_inf = df_valid.copy()
                for c_col, info in self.continuous_map.items():
                    src = info["source_col"]
                    if src in df_val_inf:

                        def mapper(x):
                            return info["map"].get(x, (0, 0))[0]

                        df_val_inf[c_col] = df_val_inf[src].apply(mapper)

                # Calculate loss
                loss = self._calculate_loss(df_val_inf)

                # Greedy acceptance (only accept if better)
                if loss < current_loss:
                    current_loss = loss
                    self.best_bn = bn
                    self.edge_params = new_edges
                    improvements += 1

                    # if improvements % 5 == 0:
                    #     print(
                    #         f"  Step {i+1}/{steps}: Loss = {current_loss:.4f} (Improved {improvements} times)"
                    #     )

            except Exception as e:
                pass

        # print(f"Copula MCMC: Final Loss = {current_loss:.4f}. Total improvements = {improvements}/{steps}")
        return self.best_bn


# Strategy Wrapper (Clean signature)
def strategy_copula_optimization(
    json_reports,
    skeleton,
    raw_data_placeholder,
    state_names,
    continuous_map=None,
    iterations=None,
    n_samples=None,
):
    # raw_data_placeholder is IGNORED
    strat = CopulaOptimization(
        json_reports, skeleton, state_names, continuous_map
    )
    return strat.fit(steps=iterations, n_samples=n_samples)
