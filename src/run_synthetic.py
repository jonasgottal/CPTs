from approaches.copula_optimization import strategy_copula_optimization
from utils.baselines import strategy_random_dirichlet, strategy_mle
from utils.eval import ExperimentManager, RandomBNGenerator
from approaches.synthetic_em import strategy_synthetic_em
from approaches.constraint_optimization import (
    strategy_constraint_optimization,
)
from approaches.ipf import strategy_ipf
from approaches.mcmc import strategy_mcmc
import numpy as np
import warnings
import logging
import os

# =============================================================================

# Suppress "Same ordering provided as current" from pgmpy's CPD.py
warnings.filterwarnings(
    "ignore", category=UserWarning, module="pgmpy.factors.discrete.CPD"
)

# Suppress probability sum warnings
warnings.filterwarnings(
    "ignore", message="Probability values don't exactly sum to 1.*"
)

# Suppress pgmpy logging messages
logging.getLogger("pgmpy").setLevel(logging.ERROR)

# =============================================================================
N_NODES = 5
EDGE_PROB = 0.3
BN_SEED = 123

N_SEEDS = 5
N_SAMPLES = 10000
N_TEST_QUERIES = 500
ITERATIONS = 2000

TYPES = ["binary", "ordinal", "nominal", "continuous"]

ITERATION_LEVELS = [500, 1000, 2000, 5000]

TYPE_CONFIGS = {
    "mixed": ["binary", "ordinal", "nominal", "continuous"],
    "discrete_only": ["binary", "ordinal", "nominal"],
    "binary_only": ["binary"],
}


def main():
    # Base path for results (relative to this script)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    base_results_path = os.path.join(script_dir, "..", "paper", "results")

    # --- OUTER LOOP: ITERATIONS ---
    for n_iters in ITERATION_LEVELS:

        # --- INNER LOOP: DATA TYPES ---
        for config_name, types in TYPE_CONFIGS.items():

            print(f"\n" + "=" * 60)
            print(
                f"STARTING RUN: {n_iters} Iterations | Config: {config_name}"
            )
            print(f"Types: {types}")
            print("=" * 60 + "\n")

            # 1. Prepare Output Directory: results/{iters}it/{config_name}/
            output_dir = os.path.join(
                base_results_path, f"{n_iters}it", config_name
            )
            os.makedirs(output_dir, exist_ok=True)
            output_file = os.path.join(output_dir, "results.tex")

            # 2. Initialize Generator with current TYPE settings
            # We re-init this every loop so the Ground Truth respects the type constraints
            gen = RandomBNGenerator(
                n_nodes=N_NODES, edge_prob=EDGE_PROB, seed=BN_SEED, types=types
            )

            # 3. Setup Manager
            manager = ExperimentManager(gen)

            # 4. Define Wrappers

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
                        None,  # raw_data is ignored
                        state_names,
                        continuous_map=manager.continuous_map,
                        iterations=iterations,
                        n_samples=n_samples,
                    )

            # Strategy: NOT allowed to see raw_data (receives it but ignores it)
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
                # We explicitly do NOT pass raw_data to the constructor
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
                    None,  # raw_data is ignored
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
                    None,  # raw_data ignored
                    state_names,
                    continuous_map=manager.continuous_map,
                    iterations=iterations,
                )

            strategies = {
                "Baseline MLE": strategy_mle,
                "Random Guess": strategy_random_dirichlet,
                "Synthetic EM": strategy_em_wrapper,
                "Constraint Optimization": strategy_opt_wrapper,
                "Iterative Proportional Fitting": strategy_ipf,
                "MCMC Optimization": strategy_direct_wrapper,
                "Copula Optimization": strategy_copula_wrapper,
            }

            manager.run_experiment(
                strategies,
                n_seeds=N_SEEDS,
                n_samples=N_SAMPLES,
                n_test_queries=N_TEST_QUERIES,
                iterations=n_iters,
            )

            # 7. Export Results
            print(f"Exporting results to: {output_file}")
            manager.export_latex(output_file)


if __name__ == "__main__":
    main()
