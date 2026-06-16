import os
import numpy as np
import warnings
import logging
from utils.eval import ExperimentManager, RandomBNGenerator
from utils.baselines import strategy_mle, strategy_random_dirichlet
from utils.llm_baseline import strategy_llm_baseline

# Configuration matching run_synthetic.py
N_NODES = 5
EDGE_PROB = 0.3
BN_SEED = 123
N_SEEDS = 5
N_SAMPLES = 10000
N_TEST_QUERIES = 500

TYPE_CONFIGS = {
    "mixed": ["binary", "ordinal", "nominal", "continuous"],
    # "discrete_only": ["binary", "ordinal", "nominal"],
    # "binary_only": ["binary"],
}


def strategy_llm_wrapper(
    json_reports,
    skeleton,
    raw_data,
    state_names,
    seed=None,
    iterations=None,
    n_samples=None,
):
    return strategy_llm_baseline(
        json_reports,
        skeleton,
        state_names,
        seed=seed,
        model_name="gpt-5.1",  # gpt-4o-mini, gpt-5.1 and o4-mini
    )


def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    # Output to results/LLM/...
    base_results_path = os.path.join(
        script_dir, "..", "paper", "results", "LLM (gpt-5.1)"
    )

    for config_name, types in TYPE_CONFIGS.items():
        print(f"\n{'='*60}")
        print(f"STARTING LLM RUN: Config: {config_name}")
        print(f"Types: {types}")
        print(f"{'='*60}\n")

        output_dir = os.path.join(base_results_path, config_name)
        os.makedirs(output_dir, exist_ok=True)
        output_file = os.path.join(output_dir, "results.tex")

        # Reuse exact same generator logic
        gen = RandomBNGenerator(
            n_nodes=N_NODES, edge_prob=EDGE_PROB, seed=BN_SEED, types=types
        )
        manager = ExperimentManager(gen)

        strategies = {
            "Random Guess": strategy_random_dirichlet,  # Good context
            "Baseline MLE": strategy_mle,  # Good context
            "gpt-5.1": strategy_llm_wrapper,  # The star
        }

        # Run (iterations arg is ignored by LLM wrapper)
        manager.run_experiment(
            strategies,
            n_seeds=N_SEEDS,
            n_samples=N_SAMPLES,
            n_test_queries=N_TEST_QUERIES,
            iterations=1,  # Dummy value
        )

        manager.export_latex(output_file)


if __name__ == "__main__":
    main()
