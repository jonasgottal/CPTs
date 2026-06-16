from utils.patches import apply_em_missing_values_patch

apply_em_missing_values_patch()
import numpy as np
import pandas as pd
import scipy.stats as stats
from pgmpy.models import DiscreteBayesianNetwork

from pgmpy.estimators import ExpectationMaximization
from approaches.copula_optimization import (
    CopulaOptimization,
    sample_gaussian_copula,
)


class SyntheticEMStrategy(CopulaOptimization):
    """
    1. Local MCMC: Optimize Copula correlation 'r' per edge until pairwise data satisfies the JSON report.
    2. Stacking: Fuse these optimized partial datasets (filling other parents with NaN).
    3. Global Learning: Run EM once on the stacked data to recover the CPTs.
    """

    def fit(self, mcmc_steps_per_edge=200, n_samples_per_edge=1000):
        partial_dfs = []
        all_nodes = set(self.skeleton.nodes())

        # print(f"Starting EM Stacking (Local Optimization -> Global EM)...")

        # 1. LOCAL OPTIMIZATION LOOP (Per Edge)
        for report in self.reports:
            rel = report.get("relation")
            if not rel:
                continue

            try:
                p_name, c_name = rel.split(" -> ")
            except ValueError:
                continue

            # Skip if variables are not in our scope
            if (
                p_name not in self.global_params
                or c_name not in self.global_params
            ):
                continue

            # A. Run Mini-MCMC to find best data for this specific pair
            # print(f"  > Optimizing pair: {rel}")
            df_best_pair = self._optimize_local_pair(
                p_name,
                c_name,
                report,
                steps=mcmc_steps_per_edge,
                n_samples=n_samples_per_edge,
            )

            # B. Prepare Partial DataFrame for Stacking
            # We need to handle Variable Name Mapping (Continuous -> Discrete BN Nodes)
            p_node = self._get_discrete_name(p_name)
            c_node = self._get_discrete_name(c_name)

            # Skip if mapped nodes aren't in the skeleton
            if p_node not in all_nodes or c_node not in all_nodes:
                continue

            # Identify ALL parents of the child (to create NaN columns for them)
            parents_of_child = list(self.skeleton.predecessors(c_node))

            # Create Schema: Child + All Parents
            cols_needed = list(set([c_node] + parents_of_child))

            # Initialize DF with NaNs
            df_partial = pd.DataFrame(
                np.nan, index=range(len(df_best_pair)), columns=cols_needed
            )

            # Fill the known values from our optimized pairwise data
            # (Note: df_best_pair comes out already discretized/mapped from _optimize_local_pair)
            df_partial[c_node] = df_best_pair[c_node].values
            df_partial[p_node] = df_best_pair[p_node].values

            partial_dfs.append(df_partial)

        if not partial_dfs:
            print("Warning: No partial data generated. Returning init BN.")
            return self.best_bn

        # 2. STACKING
        # print(f"Stacking {len(partial_dfs)} partial datasets...")
        master_df = pd.concat(partial_dfs, ignore_index=True)

        # 3. GLOBAL LEARNING (Expectation Maximization)
        # print("Running Expectation Maximization on stacked data...")

        model = DiscreteBayesianNetwork(self.skeleton.edges())
        model.add_nodes_from(self.skeleton.nodes())

        # We explicitly pass state_names so EM knows the domain of the NaN values
        estimator = ExpectationMaximization(
            model, data=master_df, state_names=self.state_names
        )

        try:
            # Run EM to find parameters
            cpds = estimator.get_parameters(max_iter=mcmc_steps_per_edge)
            model.add_cpds(*cpds)

            if model.check_model():
                # print("EM Learning successful.")
                return model
            else:
                raise ValueError("Model check failed.")

        except Exception as e:
            print(f"EM Learning failed: {e}. Fallback to Independent BN.")
            return self.best_bn

    # =============================================================================
    # =============================================================================

    def _optimize_local_pair(self, p_name, c_name, report, steps, n_samples):
        sign_str = report.get("correlation_sign", "positive")
        current_r = 0.3 if sign_str == "positive" else -0.3

        best_r = current_r
        best_loss = float("inf")

        for i in range(steps):
            df_pair = self._generate_pairwise_data(
                p_name, c_name, current_r, n_samples
            )
            loss = self._calculate_single_report_loss(
                df_pair, report, p_name, c_name
            )

            if loss < best_loss:
                best_loss = loss
                best_r = current_r

            noise = np.random.normal(0, 0.1)
            current_r = best_r + noise
            current_r = np.clip(current_r, -0.99, 0.99)

            if best_loss < 0.001:
                break

        # print(f"    Best r={best_r:.3f} (Loss={best_loss:.4f})")

        df_final = self._generate_pairwise_data(
            p_name, c_name, best_r, n_samples
        )
        df_disc = self._discretize_pairwise(df_final)

        p_node = self._get_discrete_name(p_name)
        c_node = self._get_discrete_name(c_name)
        df_disc.rename(columns={p_name: p_node, c_name: c_node}, inplace=True)

        return df_disc

    def _calculate_single_report_loss(self, df, report, p_col, c_col):
        loss = 0.0
        sign_str = report.get("correlation_sign", "positive")
        target_sign = 1.0 if sign_str == "positive" else -1.0

        for test in report.get("tests", []):
            if "error" in test:
                continue
            test_type = test.get("type", "")

            try:
                if test_type == "t-test":
                    means = test.get("group_means", {})
                    if len(means) < 2:
                        continue
                    cats = list(means.keys())
                    target_diff = means[cats[1]] - means[cats[0]]

                    g1 = df[df[p_col] == cats[0]][c_col].mean()
                    g2 = df[df[p_col] == cats[1]][c_col].mean()

                    if pd.isna(g1) or pd.isna(g2):
                        raise ValueError("Empty group")

                    curr_diff = g2 - g1
                    loss += (curr_diff - target_diff) ** 2

                elif test_type in [
                    "pearson_correlation",
                    "spearman_correlation",
                ]:
                    target_r = test.get("r", test.get("spearman_r", 0.0))

                    if test_type == "pearson_correlation":
                        curr_r = df[p_col].corr(df[c_col])
                    else:
                        curr_r, _ = stats.spearmanr(df[p_col], df[c_col])

                    if pd.isna(curr_r):
                        curr_r = 0.0
                    loss += (curr_r - target_r) ** 2

                elif test_type in ["chi_squared", "anova"]:
                    if test_type == "chi_squared":
                        mag = test.get("cramers_v", 0.0)
                    else:
                        mag = np.sqrt(test.get("eta_squared", 0.0))

                    target_proxy = target_sign * mag
                    curr_r, _ = stats.spearmanr(df[p_col], df[c_col])

                    if pd.isna(curr_r):
                        curr_r = 0.0
                    loss += (curr_r - target_proxy) ** 2

            except:
                loss += 1.0

        return loss

    def _generate_pairwise_data(self, p_name, c_name, r, n_samples):
        p_param = self.global_params[p_name]
        c_param = self.global_params[c_name]

        p_margin = self._get_marginal_fn(p_param)
        c_margin = self._get_marginal_fn(c_param)

        df = sample_gaussian_copula(n_samples, r, p_margin, c_margin)
        df.columns = [p_name, c_name]
        return df

    def _get_marginal_fn(self, param):
        if "mu" in param:
            return lambda u: stats.norm.ppf(
                u, loc=param["mu"], scale=param["sigma"]
            )
        elif "p" in param:
            cats = param["cats"]
            return lambda u: np.where(u < (1 - param["p"]), cats[0], cats[1])
        elif "probs" in param:
            cats = param["cats"]
            probs = np.cumsum(param["probs"])

            def map_cat(u_vec):
                mapped = np.searchsorted(probs, u_vec)
                mapped = np.clip(mapped, 0, len(cats) - 1)
                return np.array([cats[k] for k in mapped])

            return map_cat
        return lambda u: u

    def _discretize_pairwise(self, df):
        df_disc = df.copy()

        # Check if continuous_map exists
        if self.continuous_map is None or not isinstance(
            self.continuous_map, dict
        ):
            # If no continuous_map, discretize using state_names directly
            for col in df.columns:
                if col in self.state_names:
                    labels = self.state_names[col]
                    try:
                        # Ensure labels are strings
                        labels = [str(l) for l in labels]
                        df_disc[col] = pd.qcut(
                            df_disc[col],
                            len(labels),
                            labels=labels,
                            duplicates="drop",
                        )
                    except Exception as e:
                        # Fallback: use cut instead of qcut
                        try:
                            df_disc[col] = pd.cut(
                                df_disc[col], len(labels), labels=labels
                            )
                        except:
                            pass
            return df_disc

        # Original logic with continuous_map
        for col in df.columns:
            if col in self.continuous_map:
                info = self.continuous_map[col]
                target_col = info["source_col"]
                labels = self.state_names.get(target_col, [])
                if labels:
                    # Ensure labels are strings
                    labels = [str(l) for l in labels]
                    try:
                        df_disc[col] = pd.qcut(
                            df_disc[col],
                            len(labels),
                            labels=labels,
                            duplicates="drop",
                        )
                    except:
                        try:
                            df_disc[col] = pd.cut(
                                df_disc[col], len(labels), labels=labels
                            )
                        except:
                            pass
        return df_disc

    def _get_discrete_name(self, var_name):
        if var_name in self.continuous_map:
            return self.continuous_map[var_name]["source_col"]
        return var_name


def strategy_synthetic_em(
    json_reports,
    skeleton,
    raw_data,
    state_names,
    continuous_map=None,
    seed=None,
    iterations=None,
    n_samples=None,
):
    if seed is not None:
        np.random.seed(seed)
    strategy = SyntheticEMStrategy(
        json_reports, skeleton, state_names, continuous_map
    )
    return strategy.fit(
        mcmc_steps_per_edge=iterations, n_samples_per_edge=n_samples
    )
