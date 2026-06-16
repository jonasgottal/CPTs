import os
import warnings
import logging
import numpy as np
import pandas as pd

from approaches.copula_optimization import strategy_copula_optimization
from utils.baselines import strategy_random_dirichlet, strategy_mle
from utils.eval_sensitivity import ExperimentManager, RandomBNGenerator
from approaches.synthetic_em import strategy_synthetic_em
from approaches.constraint_optimization import strategy_constraint_optimization
from approaches.ipf import strategy_ipf
from approaches.mcmc import strategy_mcmc

# ---------------------------------------------------------------------
# Global config (baseline)
# ---------------------------------------------------------------------
BASE_N_NODES = 5
EDGE_PROB = 0.3
BN_SEED = 123

BASE_N_SEEDS = 5
BASE_N_SAMPLES = 10000
BASE_N_TEST_QUERIES = 500
BASE_ITERATIONS = 2000

TYPES = ["binary", "ordinal", "nominal", "continuous"]

# Sensitivity grids
NOISE_LEVELS = [0.0, 0.1, 0.25, 0.5, 1.0]
INDEP_LEVELS = [0.0, 0.25, 0.5, 0.75, 1.0]
NODE_LEVELS = [2, 4, 6, 8, 10, 12]
NQ_LEVELS = [100, 250, 500, 1000, 2000]

# ---------------------------------------------------------------------
# Warnings/logging hygiene
# ---------------------------------------------------------------------
warnings.filterwarnings(
    "ignore", category=UserWarning, module="pgmpy.factors.discrete.CPD"
)
warnings.filterwarnings(
    "ignore", message="Probability values don't exactly sum to 1.*"
)
logging.getLogger("pgmpy").setLevel(logging.ERROR)


def make_strategies(manager):
    # Wrappers that match ExperimentManager API
    def strategy_direct_wrapper(
        json_reports,
        skeleton,
        raw_data,
        state_names,
        seed=None,
        iterations=None,
        n_samples=None,
    ):
        if seed is not None:
            np.random.seed(seed)
        return strategy_mcmc(
            json_reports,
            skeleton,
            None,
            state_names,
            continuous_map=manager.continuous_map,
            iterations=iterations,
            n_samples=n_samples,
        )

    def strategy_copula_wrapper(
        json_reports,
        skeleton,
        raw_data,
        state_names,
        seed=None,
        iterations=None,
        n_samples=None,
    ):
        if seed is not None:
            np.random.seed(seed)
        return strategy_copula_optimization(
            json_reports,
            skeleton,
            None,
            state_names,
            continuous_map=manager.continuous_map,
            iterations=iterations,
            n_samples=n_samples,
        )

    def strategy_em_wrapper(
        json_reports,
        skeleton,
        raw_data,
        state_names,
        seed=None,
        iterations=None,
        n_samples=None,
    ):
        if seed is not None:
            np.random.seed(seed)
        return strategy_synthetic_em(
            json_reports,
            skeleton,
            None,
            state_names,
            continuous_map=manager.continuous_map,
            seed=seed,
            iterations=iterations,
            n_samples=n_samples,
        )

    def strategy_opt_wrapper(
        json_reports,
        skeleton,
        raw_data,
        state_names,
        seed=None,
        iterations=None,
        n_samples=None,
    ):
        if seed is not None:
            np.random.seed(seed)
        return strategy_constraint_optimization(
            json_reports,
            skeleton,
            None,
            state_names,
            continuous_map=manager.continuous_map,
            iterations=iterations,
        )

    def strategy_ipf_wrapper(
        json_reports,
        skeleton,
        raw_data,
        state_names,
        seed=None,
        iterations=None,
        n_samples=None,
    ):
        return strategy_ipf(
            json_reports,
            skeleton,
            None,
            state_names,
            iterations=iterations,
            n_samples=n_samples,
        )

    def strategy_random_wrapper(
        json_reports,
        skeleton,
        raw_data,
        state_names,
        seed=None,
        iterations=None,
        n_samples=None,
    ):
        if seed is not None:
            np.random.seed(seed)
        return strategy_random_dirichlet(
            json_reports, skeleton, None, state_names, seed=seed
        )

    strategies = {
        "Baseline MLE": strategy_mle,
        "Random Guess": strategy_random_wrapper,
        "Synthetic EM": strategy_em_wrapper,
        "Constraint Optimization": strategy_opt_wrapper,
        "Iterative Proportional Fitting": strategy_ipf_wrapper,
        "MCMC Optimization": strategy_direct_wrapper,
        "Copula Optimization": strategy_copula_wrapper,
    }
    return strategies


def run_sensitivity_suite():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    base_results_path = os.path.join(script_dir, "..", "paper", "sensitivity")
    os.makedirs(base_results_path, exist_ok=True)

    # -----------------------------------------------------------------
    # 1. Noise strength
    # -----------------------------------------------------------------
    for noise in NOISE_LEVELS:
        print(f"\n=== Sensitivity: noise={noise} ===")
        gen = RandomBNGenerator(
            n_nodes=BASE_N_NODES,
            edge_prob=EDGE_PROB,
            seed=BN_SEED,
            types=TYPES,
        )
        manager = ExperimentManager(gen)
        strategies = make_strategies(manager)

        manager.run_experiment(
            strategies,
            n_seeds=BASE_N_SEEDS,
            n_samples=BASE_N_SAMPLES,
            n_test_queries=BASE_N_TEST_QUERIES,
            iterations=BASE_ITERATIONS,
            noise_strength=noise,
            independence_strength=0.0,
        )
        df_summary = manager.get_summary_df()
        df_summary.to_csv(
            os.path.join(base_results_path, f"noise_{noise}.csv")
        )

    # # -----------------------------------------------------------------
    # # 2. Independence / data-loss in tests
    # # -----------------------------------------------------------------
    for indep in INDEP_LEVELS:
        print(f"\n=== Sensitivity: independence_strength={indep} ===")
        gen = RandomBNGenerator(
            n_nodes=BASE_N_NODES,
            edge_prob=EDGE_PROB,
            seed=BN_SEED,
            types=TYPES,
        )
        manager = ExperimentManager(gen)
        strategies = make_strategies(manager)

        manager.run_experiment(
            strategies,
            n_seeds=BASE_N_SEEDS,
            n_samples=BASE_N_SAMPLES,
            n_test_queries=BASE_N_TEST_QUERIES,
            iterations=BASE_ITERATIONS,
            noise_strength=0.0,
            independence_strength=indep,
        )
        df_summary = manager.get_summary_df()
        df_summary.to_csv(
            os.path.join(base_results_path, f"independence_{indep}.csv")
        )

    # -----------------------------------------------------------------
    # 3. Number of nodes
    # -----------------------------------------------------------------
    for n_nodes in NODE_LEVELS:
        print(f"\n=== Sensitivity: n_nodes={n_nodes} ===")
        gen = RandomBNGenerator(
            n_nodes=n_nodes,
            edge_prob=EDGE_PROB,
            seed=BN_SEED,
            types=TYPES,
        )
        manager = ExperimentManager(gen)
        strategies = make_strategies(manager)

        manager.run_experiment(
            strategies,
            n_seeds=BASE_N_SEEDS,
            n_samples=BASE_N_SAMPLES,
            n_test_queries=BASE_N_TEST_QUERIES,
            iterations=BASE_ITERATIONS,
            noise_strength=0.0,
            independence_strength=0.0,
        )
        df_summary = manager.get_summary_df()
        df_summary.to_csv(
            os.path.join(base_results_path, f"nodes_{n_nodes}.csv")
        )

    # -----------------------------------------------------------------
    # 4. Number of test queries
    # -----------------------------------------------------------------
    for n_q in NQ_LEVELS:
        print(f"\n=== Sensitivity: n_test_queries={n_q} ===")
        gen = RandomBNGenerator(
            n_nodes=BASE_N_NODES,
            edge_prob=EDGE_PROB,
            seed=BN_SEED,
            types=TYPES,
        )
        manager = ExperimentManager(gen)
        strategies = make_strategies(manager)

        manager.run_experiment(
            strategies,
            n_seeds=BASE_N_SEEDS,
            n_samples=BASE_N_SAMPLES,
            n_test_queries=n_q,
            iterations=BASE_ITERATIONS,
            noise_strength=0.0,
            independence_strength=0.0,
        )
        df_summary = manager.get_summary_df()
        df_summary.to_csv(os.path.join(base_results_path, f"nq_{n_q}.csv"))


def main():
    run_sensitivity_suite()


if __name__ == "__main__":
    main()
