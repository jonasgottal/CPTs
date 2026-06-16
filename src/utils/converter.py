import json
import numpy as np
from typing import List, Dict, Any
import pandas as pd


# --- 1. Robust Manual Binning ---
def prepare_ground_truth_data(df, type_mapping):
    """
    Manually bins continuous cols to ['low', 'medium', 'high']
    and binary cols to ['no', 'yes'].
    Imputes NaNs after binning.
    """
    df_discrete = df.copy()
    state_names = {}

    print(f"  [GT Prep] Manual Binning to High/Med/Low...")

    for col in df_discrete.columns:
        ctype = type_mapping.get(col, "categorical")

        # --- CONTINUOUS ---
        if ctype == "continuous":
            # 1. View valid distribution
            valid_vals = (
                pd.to_numeric(df_discrete[col], errors="coerce")
                .dropna()
                .values
            )

            if len(valid_vals) == 0:
                # Empty -> Default to 'low'
                df_discrete[col] = "low"
                state_names[col] = [
                    "high",
                    "low",
                    "medium",
                ]  # pgmpy sorts alphabetically usually, but we define order
                continue

            # 2. Calculate Quantiles (3 bins)
            # Add tiny jitter to valid_vals to prevent unique-value collapse
            rng = np.random.default_rng(42)
            noise = rng.uniform(-1e-5, 1e-5, size=len(valid_vals))

            # 33%, 66%
            q33 = np.quantile(valid_vals + noise, 0.3333)
            q66 = np.quantile(valid_vals + noise, 0.6667)

            # Ensure strict separation
            if q66 <= q33:
                q66 = q33 + 1e-6

            # 3. Replace values with bins
            def bin_mapper(x):
                if pd.isna(x):
                    return np.nan
                if x <= q33:
                    return "low"
                if x <= q66:
                    return "medium"
                return "high"

            # Apply to original column (handling NaNs naturally)
            # We map the numeric values
            numeric_col = pd.to_numeric(df_discrete[col], errors="coerce")
            binned = numeric_col.apply(bin_mapper)

            # 4. Impute NaNs with Mode
            if binned.isnull().any():
                mode_val = binned.mode()
                fill_val = mode_val[0] if len(mode_val) > 0 else "low"
                binned = binned.fillna(fill_val)

            df_discrete[col] = binned.astype(str)
            # Define vocabulary explicitly
            state_names[col] = ["high", "low", "medium"]

        # --- BINARY ---
        elif ctype == "binary":
            # 1. View valid
            vals = pd.to_numeric(df_discrete[col], errors="coerce")

            # 2. Map
            def binary_map(x):
                if pd.isna(x):
                    return np.nan
                return "yes" if x > 0 else "no"

            binned = vals.apply(binary_map)

            # 3. Impute
            if binned.isnull().any():
                mode_val = binned.mode()
                fill_val = mode_val[0] if len(mode_val) > 0 else "no"
                binned = binned.fillna(fill_val)

            df_discrete[col] = binned.astype(str)
            state_names[col] = ["no", "yes"]

        # --- CATEGORICAL ---
        else:
            s = df_discrete[col].astype(str).replace("nan", np.nan)
            if s.isnull().any():
                fill_val = s.mode()[0] if len(s.mode()) > 0 else "Unknown"
                s = s.fillna(fill_val)
            df_discrete[col] = s
            state_names[col] = sorted(df_discrete[col].unique().tolist())

    return df_discrete, state_names


# --- 2. JSON Alignment (Matching Vocabulary) ---
# ... (imports and TYPE_MAPPING remain the same) ...


def align_json_reports(json_reports, type_mapping, state_names, seed=42):
    """
    Aligns JSON reports to the Target Schema (from type_mapping & state_names).
    Handles "unknown" types by falling back to Uniform distribution over Target States.
    """
    rng = np.random.default_rng(seed)
    aligned = []

    for report in json_reports:
        new_report = report.copy()
        new_report["variables"] = {
            k: v.copy() for k, v in report["variables"].items()
        }

        for var, info in new_report["variables"].items():

            # 1. Determine Target Schema
            ctype = type_mapping.get(var, "categorical")
            target_keys = []

            # Get explicit keys from GT if available, otherwise defaults
            if var in state_names:
                target_keys = state_names[var]
            else:
                # Default Fallbacks if var not in GT (unlikely)
                if ctype == "continuous":
                    target_keys = ["high", "medium", "low"]
                elif ctype == "binary":
                    target_keys = ["no", "yes"]
                else:
                    target_keys = [
                        "unknown"
                    ]  # Should not happen if GT prep worked

            n_keys = len(target_keys)

            # 2. Check if Info is Usable
            is_unknown = info.get("type") == "unknown" or (
                "distribution" not in info and "mean" not in info
            )

            # 3. Logic Branching
            final_dist = {}

            # --- CASE A: UNKNOWN / MISSING INFO ---
            if is_unknown:
                # Fallback to Uniform over Target Keys
                # print(f"  Warning: {var} is unknown. Using Uniform fallback over {target_keys}")
                prob = 1.0 / n_keys
                final_dist = {k: prob for k in target_keys}

                info["type"] = "ordinal"  # Standardize
                info["categories"] = target_keys
                info["distribution"] = final_dist
                continue  # Done with this var

            # --- CASE B: CONTINUOUS EXTRACTED ---
            if ctype == "continuous":
                # Synthesis logic
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

            # --- CASE C: BINARY EXTRACTED ---
            elif ctype == "binary":
                curr_dist = info.get("distribution", {})
                if not isinstance(curr_dist, dict):
                    curr_dist = {}

                # Heuristic mapping
                sorted_keys = sorted(curr_dist.keys())
                new_dist = {}

                if len(sorted_keys) >= 2:
                    # Map sorted keys to no/yes
                    # e.g. '0'->no, '1'->yes OR 'no'->no, 'yes'->yes
                    # We assume alphabetical sort aligns with magnitude/logic often enough
                    new_dist["no"] = curr_dist[sorted_keys[0]]
                    new_dist["yes"] = curr_dist[sorted_keys[1]]
                elif len(sorted_keys) == 1:
                    # One key? Implies 100%?
                    k = sorted_keys[0]
                    if k.lower() in ["yes", "1", "true"]:
                        new_dist = {"no": 0.0, "yes": 1.0}
                    else:
                        new_dist = {"no": 1.0, "yes": 0.0}
                else:
                    new_dist = {"no": 0.5, "yes": 0.5}

                final_dist = new_dist

            # --- CASE D: CATEGORICAL ---
            else:
                # Pass through, but verify keys exist in Target
                curr_dist = info.get("distribution", {})
                final_dist = {}
                for k in target_keys:
                    # Soft match?
                    final_dist[k] = curr_dist.get(k, 0.0)

                # If explicit match failed, Uniform
                if sum(final_dist.values()) == 0:
                    final_dist = {k: 1.0 / n_keys for k in target_keys}

            # 4. Final Cleanup & Normalization
            # Ensure all target keys exist
            for k in target_keys:
                if k not in final_dist:
                    final_dist[k] = 0.0

            # Normalize
            total = sum(final_dist.values())
            if total > 0:
                final_dist = {k: v / total for k, v in final_dist.items()}
            else:
                final_dist = {k: 1.0 / n_keys for k in target_keys}

            # Update Report
            info["type"] = "ordinal"
            info["categories"] = target_keys
            info["distribution"] = final_dist

        aligned.append(new_report)
    return aligned


class PDFToPairwiseConverter:
    """
    Converts PDF extraction JSON + DAG into the 'pairwise_json' format
    expected by SyntheticEM and other strategies.
    """

    def __init__(self, pdf_json: Dict, dag_edges: List[List[str]]):
        self.pdf = pdf_json
        self.edges = dag_edges
        self.pairwise_reports = []

    def convert(self) -> List[Dict]:
        var_stats = self._index_variables()

        for parent, child in self.edges:
            # We need to find if these specific variables exist in our index
            # Variable names in PDF might differ slightly from DAG names
            # Simple exact match for now, can add fuzzy matching later
            p_stats = var_stats.get(parent, {"type": "unknown"})
            c_stats = var_stats.get(child, {"type": "unknown"})

            report = {
                "relation": f"{parent} -> {child}",
                "variables": {parent: p_stats, child: c_stats},
                "tests": [],
                "correlation_sign": "positive",  # Default, updated by _infer_sign
                "source": self.pdf.get("study_name", "PDF Extraction"),
            }

            # Find relevant tests
            tests = self._find_tests(parent, child)
            report["tests"] = tests

            if tests:
                report["correlation_sign"] = self._infer_sign(
                    tests, p_stats, c_stats
                )

            self.pairwise_reports.append(report)

        return self.pairwise_reports

    def _index_variables(self):
        index = {}

        # 1. Continuous Variables
        continuous = self.pdf.get("variables", {}).get("continuous", {})
        for name, groups in continuous.items():
            # Get the first group's data to determine available fields
            first_group_data = list(groups.values())[0]

            # Robustly extract central tendency (mean or median)
            if "mean" in first_group_data:
                val = first_group_data["mean"]
                stat_type = "mean"
            elif "median" in first_group_data:
                val = first_group_data["median"]
                stat_type = "median"
            else:
                val = 0
                stat_type = "unknown"

            index[name] = {
                "type": "continuous",
                "mean": val,  # Storing as 'mean' for compatibility
                "variance": 1.0, 
                "original_stat": stat_type,
                "distribution": groups,
            }

        # 2. Ordinal Variables (Treating as continuous/ordinal for now)
        ordinal = self.pdf.get("variables", {}).get("ordinal", {})
        for name, groups in ordinal.items():
            first_group_data = list(groups.values())[0]
            val = first_group_data.get("median", 0)

            index[name] = {
                "type": "ordinal",
                "categories": [
                    "low",
                    "medium",
                    "high",
                ],  # Placeholder categories
                "mean": val,
                "distribution": groups,
            }

        # 3. Binary Variables
        binary = self.pdf.get("variables", {}).get("binary", {})
        for name, groups in binary.items():
            total_yes = 0
            total_n = 0
            for g_data in groups.values():
                total_yes += g_data.get("count", 0)
                total_n += g_data.get("denominator", 0)

            p = total_yes / total_n if total_n > 0 else 0.5

            # Clean name (remove " (Yes)")
            clean_name = name.replace(" (Yes)", "").replace(" (No)", "")

            index[clean_name] = {
                "type": "binary",
                "categories": ["no", "yes"],
                "distribution": {"no": 1 - p, "yes": p},
            }
            # Also index the original name just in case DAG uses it
            if clean_name != name:
                index[name] = index[clean_name]

        study_groups = self.pdf.get("groups", [])
        group_sizes = self.pdf.get("group_sizes", {})

        if study_groups:
            # Calculate distribution from group_sizes if available, otherwise uniform
            if group_sizes:
                total = sum(group_sizes.values())
                group_dist = {
                    g: group_sizes.get(g, 0) / total for g in study_groups
                }
            else:
                n_groups = len(study_groups)
                group_dist = {g: 1.0 / n_groups for g in study_groups}

            # "group" is always nominal (categorical)
            index["group"] = {
                "type": "nominal",
                "categories": study_groups,
                "distribution": group_dist,
            }

        return index

    def _find_tests(self, p, c):
        found = []
        for test in self.pdf.get("statistical_tests", []):
            vars_in_test = test.get("variables", [])

            # Match Logic:
            # 1. Exact match of pair
            if p in vars_in_test and c in vars_in_test:
                found.append(self._format_test(test))
                continue

            # 2. Group comparison logic (common in medical papers)
            # If the test is about variable 'C' differing between groups defined by 'P'
            # Check if test has 'groups' and one variable 'C'
            if len(vars_in_test) == 1:
                var = vars_in_test[0]
                # Check if 'var' matches child 'c' (fuzzy match or exact)
                if self._match_name(var, c):
                    # And check if groups match 'p' (e.g. p="Treatment")
                    # Since we don't have P's levels explicitly mapped to groups often,
                    # we assume if P is the only binary parent, it defines the groups.
                    # For now, we assume ALL group comparison tests are valid if variable matches C
                    found.append(self._format_test(test))

        return found

    def _match_name(self, n1, n2):
        """Simple normalization for matching"""
        return n1.lower().replace(" (yes)", "") == n2.lower().replace(
            " (yes)", ""
        )

    def _format_test(self, test):
        formatted = test.copy()

        # Map types
        t_map = {
            "wilcoxon_mann_whitney": "t-test",  # Treat rank-sum as t-test for synthesis purposes
            "unpaired_t_test": "t-test",
            "chi_square": "chi_squared",
        }

        raw_type = formatted.get("test_type", "")
        formatted["type"] = t_map.get(raw_type, raw_type)

        # Format Group Means for t-test
        if "group_means" in formatted:
            # SyntheticEM expects dict {"GroupA": 1.0, "GroupB": 2.0}
            # PDF has list [1.0, 2.0] and groups ["GroupA", "GroupB"]
            if (
                isinstance(formatted["group_means"], list)
                and "groups" in formatted
            ):
                formatted["group_means"] = dict(
                    zip(formatted["groups"], formatted["group_means"])
                )

        return formatted

    def _infer_sign(self, tests, p_stats, c_stats):
        # Infer correlation sign from group means if available
        for t in tests:
            if t["type"] == "t-test" and "group_means" in t:
                # If group[0] < group[1], and we assume 0->1 is "positive" direction
                means = list(t["group_means"].values())
                if len(means) >= 2:
                    return "positive" if means[1] > means[0] else "negative"
        return "positive"
