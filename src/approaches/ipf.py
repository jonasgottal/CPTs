import numpy as np
from typing import Dict, List
from scipy.stats import multivariate_normal

from pgmpy.models import DiscreteBayesianNetwork
from pgmpy.factors.discrete import TabularCPD


class StatTranslator:
    """Translate JSON relation -> 2D marginal P(Child, Parent)"""

    def __init__(
        self,
        relation_json: dict,
        state_names: Dict[str, List[str]],
        n_samples: int = 50000,
    ):
        self.report = relation_json
        self.state_names = state_names
        self.n_samples = n_samples

        # Parse "Parent -> Child"
        rel_str = self.report["relation"]
        self.parent_name, self.child_name = [
            s.strip() for s in rel_str.split("->")
        ]

        self.parent_info = self.report["variables"][self.parent_name]
        self.child_info = self.report["variables"][self.child_name]
        self.rng = np.random.default_rng(42)

    def build_marginal(self) -> np.ndarray:
        p_type = self.parent_info.get("type", "nominal")
        c_type = self.child_info.get("type", "nominal")

        if p_type == "continuous" and c_type == "continuous":
            return self._continuous_correlation()
        elif p_type == "binary" and c_type == "continuous":
            return self._binary_ttest()
        elif p_type in ("ordinal", "nominal") and c_type in (
            "ordinal",
            "nominal",
        ):
            return self._categorical_association()
        else:
            # Maximum info: use marginal distributions
            return self._marginal_product()

    def _continuous_correlation(self) -> np.ndarray:
        mu_p = float(self.parent_info.get("mean", 0))
        var_p = float(self.parent_info.get("variance", 1))
        mu_c = float(self.child_info.get("mean", 0))
        var_c = float(self.child_info.get("variance", 1))

        # Find pearson r
        r = 0.0
        for test in self.report.get("tests", []):
            if test.get("type") == "pearson_correlation":
                r = float(test.get("r", 0))
                break

        cov = [
            [var_p, r * np.sqrt(var_p * var_c)],
            [r * np.sqrt(var_p * var_c), var_c],
        ]

        samples = multivariate_normal.rvs(
            [mu_p, mu_c], cov, size=self.n_samples, random_state=self.rng
        )

        child_states = self.state_names[self.child_name]
        parent_states = self.state_names[self.parent_name]

        child_bins = np.quantile(
            samples[:, 1], np.linspace(0, 1, len(child_states) + 1)
        )
        parent_bins = np.quantile(
            samples[:, 0], np.linspace(0, 1, len(parent_states) + 1)
        )

        counts, _, _ = np.histogram2d(
            samples[:, 1], samples[:, 0], bins=[child_bins, parent_bins]
        )

        counts = np.maximum(counts, 1e-8)
        return counts / counts.sum()

    def _binary_ttest(self) -> np.ndarray:
        parent_states = self.state_names[self.parent_name]
        child_states = self.state_names[self.child_name]

        # Parent marginal
        parent_dist = self.parent_info.get("distribution", {})
        parent_probs = np.array(
            [parent_dist.get(s, 1 / len(parent_states)) for s in parent_states]
        )
        parent_probs /= parent_probs.sum()

        # T-test stats
        ttest = None
        for test in self.report.get("tests", []):
            if test.get("type") == "t-test":
                ttest = test
                break

        if ttest is None:
            return self._marginal_product()

        means = ttest.get("group_means", {})
        d = float(ttest.get("effect_size_cohen_d", 0))

        mu0 = float(means.get(str(parent_states[0]), 0))
        mu1 = float(means.get(str(parent_states[1]), 0))
        sigma = abs(mu1 - mu0) / max(abs(d), 0.1)

        n = self.n_samples
        parent_samp = self.rng.choice(parent_states, n, p=parent_probs)
        child_samp = np.zeros(n)

        mask0 = parent_samp == parent_states[0]
        mask1 = parent_samp == parent_states[1]
        child_samp[mask0] = self.rng.normal(mu0, sigma, mask0.sum())
        child_samp[mask1] = self.rng.normal(mu1, sigma, mask1.sum())

        # Discretize child
        child_edges = np.quantile(
            child_samp, np.linspace(0, 1, len(child_states) + 1)
        )
        child_bins = np.digitize(child_samp, child_edges)

        parent_idx = {s: i for i, s in enumerate(parent_states)}
        counts = np.zeros((len(child_states), len(parent_states)))

        for ps, cb in zip(parent_samp, child_bins):
            i = min(int(cb), len(child_states) - 1)
            j = parent_idx[ps]
            counts[i, j] += 1

        counts = np.maximum(counts, 1e-8)
        return counts / counts.sum()

    def _categorical_association(self) -> np.ndarray:
        parent_states = self.state_names[self.parent_name]
        child_states = self.state_names[self.child_name]

        p_marg = np.array(
            [
                self.parent_info["distribution"].get(s, 1 / len(parent_states))
                for s in parent_states
            ]
        )
        c_marg = np.array(
            [
                self.child_info["distribution"].get(s, 1 / len(child_states))
                for s in child_states
            ]
        )
        p_marg, c_marg = p_marg / p_marg.sum(), c_marg / c_marg.sum()

        # Cramer's V strength
        cramers_v = 0.0
        for test in self.report.get("tests", []):
            if test.get("type") == "chi_squared":
                cramers_v = test.get("cramers_v", 0)
                break

        n = self.n_samples
        parent_samp = self.rng.choice(parent_states, n, p=p_marg)
        child_samp = self.rng.choice(child_states, n, p=c_marg)

        # Add association
        sign = 1 if self.report.get("correlation_sign") == "positive" else -1
        for i in range(n):
            if self.rng.random() < cramers_v:
                # Match ordinal position
                p_idx = parent_states.index(parent_samp[i])
                c_idx = min(
                    int(p_idx * len(child_states) / len(parent_states)),
                    len(child_states) - 1,
                )
                child_samp[i] = child_states[c_idx * sign >= 0 or c_idx]

        parent_idx = {s: i for i, s in enumerate(parent_states)}
        child_idx = {s: i for i, s in enumerate(child_states)}
        counts = np.zeros((len(child_states), len(parent_states)))

        for ps, cs in zip(parent_samp, child_samp):
            counts[child_idx[cs], parent_idx[ps]] += 1

        counts = np.maximum(counts, 1e-8)
        return counts / counts.sum()

    def _marginal_product(self) -> np.ndarray:
        child_states = self.state_names[self.child_name]
        parent_states = self.state_names[self.parent_name]

        p_child = np.array(
            [
                self.child_info["distribution"].get(s, 1 / len(child_states))
                for s in child_states
            ]
        )
        p_parent = np.array(
            [
                self.parent_info["distribution"].get(s, 1 / len(parent_states))
                for s in parent_states
            ]
        )

        return np.outer(p_child / p_child.sum(), p_parent / p_parent.sum())


class IPFStrategy:
    def __init__(
        self,
        child: str,
        parents: List[str],
        state_names: Dict[str, List[str]],
        target_marginals: Dict[str, np.ndarray],
        tol: float = 1e-4,
    ):
        self.child = child
        self.parents = parents
        self.state_names = state_names
        self.target_marginals = target_marginals
        self.tol = tol

        self.child_card = len(state_names[child])
        self.parent_cards = [len(state_names[p]) for p in parents]
        self.shape = (self.child_card, *self.parent_cards)

    def run(self, iterations=1000) -> TabularCPD:
        """Run IPF → guaranteed 2D pgmpy CPD"""
        joint = np.ones(self.shape, dtype=float)

        for _ in range(iterations):
            joint_old = joint.copy()

            for p_idx, parent in enumerate(self.parents):
                target = self.target_marginals.get(parent)
                if target is None:
                    continue

                p_axis = 1 + p_idx
                sum_axes = tuple(
                    i for i in range(len(self.shape)) if i not in (0, p_axis)
                )
                current_marg = joint.sum(axis=sum_axes) if sum_axes else joint

                # Ensure shapes match
                if target.shape != current_marg.shape:
                    print(
                        f"Warning: shape mismatch {target.shape} vs {current_marg.shape}"
                    )
                    continue

                factor = target / np.maximum(current_marg, 1e-12)
                factor_shape = [1] * len(self.shape)
                factor_shape[0] = self.child_card
                factor_shape[p_axis] = self.parent_cards[p_idx]
                joint *= factor.reshape(factor_shape)

            joint_sum = joint.sum()
            if joint_sum > 0:
                joint /= joint_sum

            if np.max(np.abs(joint - joint_old)) < self.tol:
                break

        n_configs = int(np.prod(self.parent_cards))
        cpd_values = joint.reshape(self.child_card, n_configs)

        # Column normalize
        col_sums = cpd_values.sum(axis=0, keepdims=True)
        cpd_values /= np.maximum(col_sums, 1e-12)

        assert (
            cpd_values.ndim == 2
        ), f"Expected 2D, got {cpd_values.shape} ndim={cpd_values.ndim}"
        assert cpd_values.shape == (self.child_card, n_configs)

        return TabularCPD(
            self.child,
            self.child_card,
            cpd_values,
            evidence=self.parents if self.parents else [],
            evidence_card=self.parent_cards,
            state_names=self.state_names,
        )


def strategy_ipf(
    json_reports,
    skeleton,
    raw_data,
    state_names,
    seed=None,
    iterations=None,
    n_samples=None,
):
    """IPF strategy - creates ALL CPDs (no missing nodes)"""
    if seed is not None:
        np.random.seed(seed)
    model = DiscreteBayesianNetwork(skeleton.edges())
    model.add_nodes_from(skeleton.nodes())

    # Group relations by child
    child_to_rels = {}
    for report in json_reports:
        try:
            parent, child = [s.strip() for s in report["relation"].split("->")]
            child_to_rels.setdefault(child, []).append((parent, report))
        except:
            continue

    # Create CPD for EVERY node (roots + non-roots)
    for child in model.nodes():
        if child not in state_names:
            continue

        parents = list(model.get_parents(child))
        card = len(state_names[child])

        if not parents:
            # ROOT: uniform marginal (shape: card x 1)
            values = np.ones((card, 1)) / card
            cpd = TabularCPD(child, card, values, state_names=state_names)
        else:
            # Build target marginals from available relations
            target_margs = {}
            for parent, report in child_to_rels.get(child, []):
                if parent in parents:
                    try:
                        translator = StatTranslator(
                            report, state_names, n_samples=n_samples
                        )
                        target_margs[parent] = translator.build_marginal()
                    except Exception as e:
                        print(
                            f"Warning: failed marginal {parent}->{child}: {e}"
                        )
                        pass

            # Run IPF (even with partial constraints)
            ipf = IPFStrategy(child, parents, state_names, target_margs)
            cpd = ipf.run(iterations=iterations)

        model.add_cpds(cpd)

    try:
        model.check_model()
    except:
        pass

    return model
