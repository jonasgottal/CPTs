import numpy as np
import scipy.optimize as opt
from scipy.optimize import LinearConstraint, Bounds
from pgmpy.models import DiscreteBayesianNetwork
from pgmpy.factors.discrete import TabularCPD
from typing import Dict, Any, List, Tuple, Optional
from itertools import product


def _x_to_joint_matrix(
    x: np.ndarray, child_card: int, n_cols: int
) -> np.ndarray:
    """Reshape flattened x into joint matrix J of shape (child_card, n_cols) using Fortran order."""
    return x.reshape(child_card, n_cols, order="F")


def _joint_matrix_to_x(J: np.ndarray) -> np.ndarray:
    """Flatten joint matrix J back to x using Fortran order."""
    return J.reshape(-1, order="F")


def _conditional_from_joint_matrix(
    J: np.ndarray, eps: float = 1e-12
) -> np.ndarray:
    """Convert joint matrix P(C,Pa) to conditional CPT matrix P(C|Pa)."""
    parent_marg = J.sum(axis=0, keepdims=True)
    parent_marg = np.clip(parent_marg, eps, None)
    return J / parent_marg


def _build_parent_config_index(
    parent_cards: List[int],
) -> List[Tuple[int, ...]]:
    """Parent configurations in itertools.product order (last parent varies fastest)."""
    return list(product(*[range(c) for c in parent_cards]))


def _marginalize_to_parent(
    J_full: np.ndarray,
    parent_cards: List[int],
    parent_index: int,
    configs: List[Tuple[int, ...]],
) -> np.ndarray:
    """Marginalize joint matrix P(C,Pa) to P(C, P_i) for a given parent index."""
    child_card, _ = J_full.shape
    card_i = parent_cards[parent_index]

    cols_by_state = [[] for _ in range(card_i)]
    for col_idx, cfg in enumerate(configs):
        cols_by_state[cfg[parent_index]].append(col_idx)

    J_marg = np.zeros((child_card, card_i), dtype=float)
    for s, cols in enumerate(cols_by_state):
        if cols:
            J_marg[:, s] = J_full[:, cols].sum(axis=1)
    return J_marg


# =============================================================================
# 1. HELPER FUNCTIONS
# =============================================================================


def _get_dims(
    child_states: List[str], parent_states: List[str]
) -> Tuple[int, int]:
    """Returns (child_card, parent_card)."""
    return len(child_states), len(parent_states)


def _joint_to_conditional(
    x: np.ndarray, child_card: int, parent_card: int, eps: float = 1e-12
) -> np.ndarray:
    """
    Converts flattened joint P(C, P) -> Conditional P(C | P).
    Output shape: (child_card, parent_card)
    """
    joint = x.reshape(child_card, parent_card, order="F")
    marginal_parent = joint.sum(axis=0, keepdims=True)
    marginal_parent = np.clip(marginal_parent, eps, None)
    return joint / marginal_parent


def _get_numeric_map(
    node: str,
    state_names: Dict[str, List[str]],
    continuous_map: Optional[Dict[str, Any]] = None,
) -> Optional[np.ndarray]:
    """
    Retrieves numeric values (bin centers) for a node if available.
    Returns np.array of shape (card,) or None.
    """
    states = state_names[node]

    # Invert the continuous_map to key by DiscreteVar
    discrete_to_cont = {}
    if isinstance(continuous_map, dict):
        for real_var, info in continuous_map.items():
            if not isinstance(info, dict):
                continue
            source = info.get("source_col")
            if source:
                discrete_to_cont[source] = info.get("map", {})

    if node in discrete_to_cont:
        mapping = discrete_to_cont[node]
        # Extract mean (mu) for each state
        values = []
        for s in states:
            # default to 0.0 if state not found, though it should be
            val_params = mapping.get(s, (0.0, 1.0))
            # val_params is typically (mu, sigma) or just mu
            val = (
                val_params[0]
                if isinstance(val_params, (list, tuple))
                else val_params
            )
            values.append(float(val))
        return np.array(values)

    # If node is naturally ordinal/numeric (optional heuristic)
    try:
        return np.array([float(s) for s in states])
    except ValueError:
        return None


# =============================================================================
# 2. OBJECTIVE FUNCTION
# =============================================================================


def _neg_entropy_objective(x: np.ndarray, eps: float = 1e-12) -> float:
    """Minimize Sum(x * log(x)) -> Maximize Entropy."""
    # Clip to avoid log(0)
    x_safe = np.clip(x, eps, 1.0)
    return np.sum(x_safe * np.log(x_safe))


# =============================================================================
# 3. STATISTICAL CONSTRAINT FUNCTIONS
# =============================================================================


def _constr_ttest(
    x: np.ndarray,
    child_vals: np.ndarray,
    child_card: int,
    parent_card: int,
    p_idx0: int,
    p_idx1: int,
    target_diff: float,
    eps: float = 1e-12,
) -> float:
    """
    Constraint: (Mean(Child|P0) - Mean(Child|P1)) - target = 0
    Note: The sign depends on the order defined in group_means.
    """
    joint = x.reshape(child_card, parent_card, order="F")

    # Marginals P(P)
    p_p = joint.sum(axis=0)
    p0 = max(p_p[p_idx0], eps)
    p1 = max(p_p[p_idx1], eps)

    # Conditional Means E[C|P] = Sum(C_val * P(C,P)) / P(P)
    mu0 = np.sum(joint[:, p_idx0] * child_vals) / p0
    mu1 = np.sum(joint[:, p_idx1] * child_vals) / p1

    return (mu1 - mu0) - target_diff


def _constr_pearson(
    x: np.ndarray,
    child_vals: np.ndarray,
    parent_vals: np.ndarray,
    child_card: int,
    parent_card: int,
    target_r: float,
    eps: float = 1e-12,
) -> float:
    """
    Constraint: Pearson_r(x) - target_r = 0
    """
    joint = x.reshape(child_card, parent_card, order="F")

    # Marginals
    p_c = joint.sum(axis=1)
    p_p = joint.sum(axis=0)

    # Means
    mu_c = np.sum(p_c * child_vals)
    mu_p = np.sum(p_p * parent_vals)

    # Covariance E[CP] - E[C]E[P]
    # E[CP] = Sum_ij (Joint_ij * C_i * P_j)
    # Use broadcasting: (C_vals column) * (P_vals row)
    val_grid = np.outer(child_vals, parent_vals)  # (child, parent)
    ex_cp = np.sum(joint * val_grid)
    covariance = ex_cp - (mu_c * mu_p)

    # Standard Deviations
    var_c = np.sum(p_c * (child_vals - mu_c) ** 2)
    var_p = np.sum(p_p * (parent_vals - mu_p) ** 2)
    std_prod = np.sqrt(var_c * var_p) + eps

    return (covariance / std_prod) - target_r


def _constr_cramers_v(
    x: np.ndarray,
    child_card: int,
    parent_card: int,
    target_v: float,
    eps: float = 1e-12,
) -> float:
    """
    Constraint: CramersV(x) - target_v = 0
    V = sqrt(chi2 / (N * min_dim))
    Since N=1 (probabilities), V = sqrt(chi2_stat / min_dim)
    """
    joint = x.reshape(child_card, parent_card, order="F")

    p_c = joint.sum(axis=1, keepdims=True)  # (child, 1)
    p_p = joint.sum(axis=0, keepdims=True)  # (1, parent)

    expected = np.dot(p_c, p_p)  # Independence hypothesis
    expected = np.clip(expected, eps, 1.0)

    # Chi2 statistic on probabilities
    chi2_stat = np.sum((joint - expected) ** 2 / expected)

    min_dim = min(child_card - 1, parent_card - 1)
    if min_dim < 1:
        min_dim = 1

    calculated_v = np.sqrt(chi2_stat / min_dim)

    return calculated_v - target_v


def _constr_anova_eta(
    x: np.ndarray,
    child_vals: np.ndarray,
    child_card: int,
    parent_card: int,
    target_eta_sq: float,
    eps: float = 1e-12,
) -> float:
    """
    Constraint: Eta_Squared(x) - target_eta_sq = 0
    Eta^2 = SS_between / SS_total
    """
    joint = x.reshape(child_card, parent_card, order="F")

    # Global Mean E[C]
    p_c = joint.sum(axis=1)
    mu_total = np.sum(p_c * child_vals)

    # Total Variance (SS_total proxy since N=1) = Var(C)
    ss_total = np.sum(p_c * (child_vals - mu_total) ** 2)

    # SS Between = Sum_groups P(g) * (mu_g - mu_total)^2
    p_p = joint.sum(axis=0)  # group probabilities

    ss_between = 0.0
    for j in range(parent_card):
        p_group = max(p_p[j], eps)
        # E[C|group]
        mu_group = np.sum(joint[:, j] * child_vals) / p_group
        ss_between += p_group * (mu_group - mu_total) ** 2

    calculated_eta_sq = ss_between / (ss_total + eps)

    return calculated_eta_sq - target_eta_sq


# =============================================================================
# 4. OPTIMIZER CORE
# =============================================================================


def optimize_cpd(
    parent: str,
    child: str,
    state_names: Dict[str, List[str]],
    json_test: Dict[str, Any],
    continuous_map: Optional[Dict[str, Any]] = None,
    iterations: int = 1000,
) -> TabularCPD:
    if not isinstance(json_test, dict):
        raise ValueError(
            f"json_test must be dict, got {type(json_test)}: {json_test}"
        )

    test_type = json_test.get("type", "")
    if not test_type:
        raise ValueError(f"No 'type' in json_test: {json_test}")

    parent_states = state_names[parent]
    child_states = state_names[child]
    child_card, parent_card = _get_dims(child_states, parent_states)

    # Initial Guess: Uniform Independent
    # x represents P(C, P). Uniform P(P)=1/Np, Uniform P(C)=1/Nc
    x0 = np.ones(child_card * parent_card) / (child_card * parent_card)

    # Constraints list
    constraints = []

    # 1. Axiom: Sum(x) = 1
    constraints.append(LinearConstraint(np.ones(x0.shape), lb=1.0, ub=1.0))

    # 2. Axiom: 0 <= x <= 1
    bounds = Bounds(np.zeros_like(x0), np.ones_like(x0))

    # 3. Statistical Constraint
    test_type = json_test.get("type")

    # Pre-fetch numeric values if needed
    child_vals = _get_numeric_map(child, state_names, continuous_map)
    parent_vals = _get_numeric_map(parent, state_names, continuous_map)

    constraint_fun = None

    if test_type == "t-test":

        group_means = json_test.get("group_means", {})
        

        if len(group_means) == 2 and child_vals is not None:
            cats = list(group_means.keys())
            try:
                # Find indices of the two groups in the parent state list
                p_idx0 = parent_states.index(cats[0])
                p_idx1 = parent_states.index(cats[1])
                target_diff = group_means[cats[1]] - group_means[cats[0]]

                constraint_fun = lambda x: _constr_ttest(
                    x,
                    child_vals,
                    child_card,
                    parent_card,
                    p_idx0,
                    p_idx1,
                    target_diff,
                )
            except ValueError:
                pass  # Parent states didn't match

    elif test_type in ["pearson_correlation", "spearman_correlation"]:
        # Note: Spearman is treated as Pearson on rank-mapped values if raw values unavailable
        # But here we strictly use the numeric map (bin centers).
        target_r = json_test.get("r", json_test.get("spearman_r", 0.0))
        if child_vals is not None and parent_vals is not None:
            constraint_fun = lambda x: _constr_pearson(
                x, child_vals, parent_vals, child_card, parent_card, target_r
            )

    elif test_type == "chi_squared":
        # Cramers V
        target_v = json_test.get("cramers_v", 0.0)
        constraint_fun = lambda x: _constr_cramers_v(
            x, child_card, parent_card, target_v
        )

    elif test_type == "anova":
        target_eta = json_test.get("eta_squared", 0.0)
        if child_vals is not None:
            constraint_fun = lambda x: _constr_anova_eta(
                x, child_vals, child_card, parent_card, target_eta
            )

    if constraint_fun:
        constraints.append({"type": "eq", "fun": constraint_fun})

    # Run Optimization
    # We use SLSQP as it handles equality constraints + bounds well
    res = opt.minimize(
        _neg_entropy_objective,
        x0,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
        options={"maxiter": iterations, "ftol": 1e-6},
    )

    # If optimization fails, we return the result anyway (likely close enough or stuck at init)
    # But usually, we want the optimal x
    x_final = res.x

    # Convert to CPT
    cond_values = _joint_to_conditional(x_final, child_card, parent_card)

    return TabularCPD(
        variable=child,
        variable_card=child_card,
        values=cond_values,
        evidence=[parent],
        evidence_card=[parent_card],
        state_names=state_names,
    )


def optimize_cpd_multi(
    parents: List[str],
    child: str,
    state_names: Dict[str, List[str]],
    parent_tests: Dict[str, Dict[str, Any]],
    continuous_map: Optional[Dict[str, Any]] = None,
    iterations: int = 1000,
) -> TabularCPD:
    """Max-entropy optimization for P(C|Pa(C)) using per-parent constraints on marginals P(C,P_i)."""

    if not parents:
        raise ValueError("optimize_cpd_multi requires at least one parent")

    child_states = state_names[child]
    child_card = len(child_states)

    parent_cards = [len(state_names[p]) for p in parents]
    n_cols = int(np.prod(parent_cards))

    # x represents the joint matrix J = P(C, Pa(C)) flattened in Fortran order.
    x0 = np.ones(child_card * n_cols) / (child_card * n_cols)

    constraints = [LinearConstraint(np.ones(x0.shape), lb=1.0, ub=1.0)]
    bounds = Bounds(np.zeros_like(x0), np.ones_like(x0))

    configs = _build_parent_config_index(parent_cards)

    child_vals = _get_numeric_map(child, state_names, continuous_map)

    for parent_index, parent in enumerate(parents):
        json_test = parent_tests.get(parent)
        if not isinstance(json_test, dict):
            continue

        test_type = json_test.get("type", "")
        if not test_type:
            continue

        parent_states = state_names[parent]
        parent_card = len(parent_states)
        parent_vals = _get_numeric_map(parent, state_names, continuous_map)

        def _make_marginal_x(x_full, p_idx=parent_index):
            J_full = _x_to_joint_matrix(x_full, child_card, n_cols)
            J_marg = _marginalize_to_parent(
                J_full, parent_cards, p_idx, configs
            )
            return _joint_matrix_to_x(J_marg)

        constraint_fun = None

        if test_type == "t-test":
            group_means = json_test.get("group_means", {})
            if (
                isinstance(group_means, dict)
                and len(group_means) == 2
                and child_vals is not None
            ):
                cats = list(group_means.keys())
                try:
                    p_idx0 = parent_states.index(cats[0])
                    p_idx1 = parent_states.index(cats[1])
                    target_diff = group_means[cats[1]] - group_means[cats[0]]

                    def constraint_fun(
                        x_full, p0=p_idx0, p1=p_idx1, td=target_diff
                    ):
                        x_marg = _make_marginal_x(x_full)
                        return _constr_ttest(
                            x_marg,
                            child_vals,
                            child_card,
                            parent_card,
                            p0,
                            p1,
                            td,
                        )

                except ValueError:
                    pass

        elif test_type in ["pearson_correlation", "spearman_correlation"]:
            target_r = json_test.get("r", json_test.get("spearman_r", 0.0))
            if child_vals is not None and parent_vals is not None:

                def constraint_fun(x_full, tr=target_r):
                    x_marg = _make_marginal_x(x_full)
                    return _constr_pearson(
                        x_marg,
                        child_vals,
                        parent_vals,
                        child_card,
                        parent_card,
                        tr,
                    )

        elif test_type == "chi_squared":
            target_v = json_test.get("cramers_v", 0.0)

            def constraint_fun(x_full, tv=target_v):
                x_marg = _make_marginal_x(x_full)
                return _constr_cramers_v(x_marg, child_card, parent_card, tv)

        elif test_type == "anova":
            target_eta = json_test.get("eta_squared", 0.0)
            if child_vals is not None:

                def constraint_fun(x_full, te=target_eta):
                    x_marg = _make_marginal_x(x_full)
                    return _constr_anova_eta(
                        x_marg, child_vals, child_card, parent_card, te
                    )

        if constraint_fun is not None:
            constraints.append({"type": "eq", "fun": constraint_fun})

    res = opt.minimize(
        _neg_entropy_objective,
        x0,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
        options={"maxiter": iterations, "ftol": 1e-6},
    )

    x_final = res.x
    J_final = _x_to_joint_matrix(x_final, child_card, n_cols)
    cond_values = _conditional_from_joint_matrix(J_final)

    return TabularCPD(
        variable=child,
        variable_card=child_card,
        values=cond_values,
        evidence=parents,
        evidence_card=parent_cards,
        state_names=state_names,
    )


# =============================================================================
# 5. STRATEGY WRAPPER
# =============================================================================


def strategy_constraint_optimization(
    json_reports: List[Dict],
    skeleton: Any,
    raw_data_placeholder: Any,
    state_names: Dict[str, List[str]],
    continuous_map: Dict[str, Any] = None,
    iterations: int = 1000,
) -> DiscreteBayesianNetwork:
    """
    Direct Optimization Strategy.
    Iterates over the skeleton structure. If a relation has a JSON statistical target,
    it optimizes the CPT to satisfy that equation with maximum entropy.
    Otherwise, it defaults to a uniform distribution.
    """

    model = DiscreteBayesianNetwork(skeleton.edges())
    model.add_nodes_from(skeleton.nodes())

    # Index reports by relation "Parent -> Child"
    report_map = {}
    for r in json_reports:
        if "relation" in r:
            report_map[r["relation"]] = r

    for node in model.nodes():
        parents = list(model.get_parents(node))
        child_card = len(state_names[node])

        # CASE 1: Root Node (No parents) -> Uniform (or could use global marginals if added)
        if not parents:
            values = np.ones((child_card, 1)) / child_card
            cpd = TabularCPD(node, child_card, values, state_names=state_names)
            model.add_cpds(cpd)
            continue

        if parents:
            parent_tests = {}
            for parent in parents:
                rel_key = f"{parent} -> {node}"
                if rel_key not in report_map:
                    continue

                report = report_map[rel_key]
                tests = report.get("tests", [])
                target_test = None

                for tType in [
                    "t-test",
                    "pearson_correlation",
                    "anova",
                    "chi_squared",
                ]:
                    found = next(
                        (t for t in tests if t.get("type") == tType), None
                    )
                    if found:
                        target_test = found
                        break

                if target_test is not None:
                    parent_tests[parent] = target_test

            # If no constraints exist for any parent edge, use a uniform CPT.
            if not parent_tests:
                parent_cards = [len(state_names[p]) for p in parents]
                n_cols = int(np.prod(parent_cards))
                values = np.ones((child_card, n_cols)) / child_card
                cpd = TabularCPD(
                    variable=node,
                    variable_card=child_card,
                    values=values,
                    evidence=parents,
                    evidence_card=parent_cards,
                    state_names=state_names,
                )
                model.add_cpds(cpd)
                continue

            try:
                cpd = optimize_cpd_multi(
                    parents=parents,
                    child=node,
                    state_names=state_names,
                    parent_tests=parent_tests,
                    continuous_map=continuous_map,
                    iterations=iterations,
                )
                model.add_cpds(cpd)
                continue
            except Exception as e:
                print(
                    f"Optimization failed for {node} with parents {parents}: {e}. Falling back to uniform."
                )

        # CASE 3: Fallback (Multi-parent or Optimization Failed or No Report)
        # Create Uniform CPT
        parent_cards = [len(state_names[p]) for p in parents]
        n_cols = np.prod(parent_cards)
        values = np.ones((child_card, n_cols)) / child_card

        cpd = TabularCPD(
            variable=node,
            variable_card=child_card,
            values=values,
            evidence=parents,
            evidence_card=parent_cards,
            state_names=state_names,
        )
        model.add_cpds(cpd)

    model.check_model()
    return model
