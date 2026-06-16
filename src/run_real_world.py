import logging
import os
import pandas as pd
import numpy as np
from dotenv import load_dotenv
import warnings

# Import the new Manager
from utils.real_world_manager import RealWorldExperimentManager

# Import Strategies
from approaches.synthetic_em import strategy_synthetic_em
from approaches.copula_optimization import strategy_copula_optimization
from approaches.constraint_optimization import strategy_constraint_optimization
from approaches.ipf import strategy_ipf
from approaches.mcmc import strategy_mcmc
from utils.baselines import strategy_random_dirichlet, strategy_mle
from utils.llm_baseline import strategy_llm_baseline

# ================= CONFIGURATION =================
USE_CACHE = True

N_SEEDS = 5
N_SAMPLES = 1000
N_TEST_QUERIES = 500
ITERATIONS = 2000
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REAL_WORLD_ROOT = os.path.join(SCRIPT_DIR, "extractor", "real_world")

base_results_path = os.path.join(SCRIPT_DIR, "..", "paper", "results", "real")

DATASET_CONFIGS = {
    "10.1186_s13104-019-4632-2": {
        "type_mapping": {
            "Mother's age": "continuous",
            "Age at admission": "continuous",
            "Estimated gestational weeks": "continuous",
            "TSB at admission": "continuous",
            "Change in bilirubin": "continuous",
            "Rate of change in bilirubin": "continuous",
            "TSB after phototherapy": "continuous",
            "Haemolytic causes": "binary",
            "Non-haemolytic causes": "binary",
            "Repeat phototherapy (Yes)": "binary",
            "Exchange transfusion (Yes)": "binary",
            "Mortality (Died)": "binary",
            "group": "categorical",
            "Type of discharge": "categorical",
        },
        "target_cols": [
            "TSB at admission",
            "Mortality (Died)",
            "Type of discharge",
            "Exchange transfusion (Yes)",
            "Repeat phototherapy (Yes)",
            "Rate of change in bilirubin",
            "TSB after phototherapy",
        ],
    },
    "10.1161_JAHA.118.011771": {
        "type_mapping": {
            "a+thalassemia": "binary",
            "thal_status": "categorical",
            "group": "categorical",
            "Smoker": "binary",
            "Women": "binary",
            "Hypertension": "binary",
            "Previously diagnosed with hypertension": "binary",
            "Taking antihypertensive medication": "binary",
            "Age": "continuous",
            "Hemoglobin": "continuous",
        },
        "target_cols": [
            "Hypertension",
            "Previously diagnosed with hypertension",
            "Taking antihypertensive medication",
            "Hemoglobin",
            "a+thalassemia",
            "thal_status",
            "Age",
            "Smoker",
        ],
    },
    "10.1371_journal.pmed.1002785": {
        "type_mapping": {
            "Patient gender (Female)": "binary",
            "Literate": "binary",
            "CGI–Improvement": "continuous",
            "CGI–Severity": "continuous",
            "cgi1_n.endline": "continuous",
            "cgi2_n.endline": "continuous",
            "Death for any reason": "binary",
            "Wandering": "binary",
            "Re-hospitalization due to schizophrenia": "binary",
            "group": "categorical",
            "Violence against others": "binary",
            "Damaging goods": "binary",
        },
        "target_cols": [
            "Re-hospitalization due to schizophrenia",
            "Death for any reason",
            "Violence against others",
            "Damaging goods",
            "Wandering",
            "CGI–Improvement",
            "CGI–Severity",
            "cgi1_n.endline",
            "cgi2_n.endline",
        ],
    },
}


def build_dataset_paths(folder_name: str):
    dataset_dir = os.path.join(REAL_WORLD_ROOT, folder_name)
    return {
        "pdf": os.path.join(dataset_dir, f"{folder_name}.pdf"),
        "csv": os.path.join(dataset_dir, f"{folder_name}_clean.csv"),
        "cache": os.path.join(dataset_dir, f"{folder_name}_extracted.json"),
    }


def list_dataset_folders(root: str):
    if not os.path.isdir(root):
        return []
    return sorted(
        [
            name
            for name in os.listdir(root)
            if os.path.isdir(os.path.join(root, name))
        ]
    )


def main():
    load_dotenv()
    warnings.filterwarnings("ignore")  # suppress all Python warnings
    pgmpy_logger = logging.getLogger("pgmpy")
    pgmpy_logger.setLevel(logging.ERROR)

    api_key = os.getenv("OPENAI_API_KEY")
    np.random.seed(0)

    folders = list_dataset_folders(REAL_WORLD_ROOT)
    if not folders:
        print(f"No dataset folders found under: {REAL_WORLD_ROOT}")
        return

    for folder_name in folders:
        config = DATASET_CONFIGS.get(folder_name)
        if config is None:
            print(
                f"[SKIP] No TYPE_MAPPING/TARGET_COLS config for: {folder_name}"
            )
            continue

        paths = build_dataset_paths(folder_name)
        if not os.path.exists(paths["pdf"]) or not os.path.exists(
            paths["csv"]
        ):
            print(f"[SKIP] Missing input files for: {folder_name}")
            print(f"       PDF: {paths['pdf']}")
            print(f"       CSV: {paths['csv']}")
            continue

        manager = RealWorldExperimentManager(
            pdf_path=paths["pdf"],
            csv_path=paths["csv"],
            type_mapping=config["type_mapping"],
            target_columns=config["target_cols"],
            cache_file=paths["cache"] if USE_CACHE else None,
        )

        manager.prepare_data(api_key=api_key)

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
                model_name="gpt-4o-mini",
            )

        def strategy_em_wrapper(
            json_reports,
            skeleton,
            raw_data,
            state_names,
            seed,
            iterations,
            n_samples,
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

        def strategy_copula_wrapper(
            json_reports,
            skeleton,
            raw_data,
            state_names,
            seed,
            iterations,
            n_samples,
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

        def strategy_opt_wrapper(
            json_reports,
            skeleton,
            raw_data,
            state_names,
            seed,
            iterations,
            n_samples,
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
            seed,
            iterations,
            n_samples,
        ):
            return strategy_ipf(json_reports, skeleton, None, state_names)

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

        def strategy_random_wrapper(
            json_reports,
            skeleton,
            raw_data,
            state_names,
            seed,
            iterations,
            n_samples,
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
            "LLM Baseline": strategy_llm_wrapper,
        }

        print(f"\nStarting Real-World Benchmark for: {folder_name}")
        manager.run_experiment(
            strategies,
            n_seeds=N_SEEDS,
            iterations=ITERATIONS,
            n_samples=N_SAMPLES,
            n_test_queries=N_TEST_QUERIES,
        )
        output_path = os.path.join(
            base_results_path, f"real_world_results_{folder_name}.tex"
        )

        manager.export_latex(output_path)


if __name__ == "__main__":
    main()
