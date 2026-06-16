import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from itertools import product

import warnings
from packaging import version

try:
    import pgmpy
    from pgmpy.estimators import ExpectationMaximization as EM
except ImportError as e:
    warnings.warn(f"pgmpy not installed, EM patch skipped: {e}")
    EM = None

# Minimum version that already includes the upstream fix.
# Adjust this once the patch is released in pgmpy.
# Curerntly, the fix is pending PR: https://github.com/pgmpy/pgmpy/pull/2173
_MIN_FIXED_VERSION = "1.1.0"


def _patched_parallel_compute_weights(
    self, data_unique, latent_card, n_counts, offset, batch_size
):
    cache = []

    for i in range(offset, min(offset + batch_size, data_unique.shape[0])):
        row = data_unique.iloc[i]
        observed = {col: val for col, val in row.items() if not pd.isnull(val)}
        missing = [col for col, val in row.items() if pd.isnull(val)]
        # If no missing, just use the row as is
        if not missing:
            v = list(product(*[range(card) for card in latent_card.values()]))
            latent_combinations = np.array(v, dtype=int)
            df = data_unique.iloc[
                [i] * latent_combinations.shape[0]
            ].reset_index(drop=True)
            for index, latent_var in enumerate(latent_card.keys()):
                df[latent_var] = latent_combinations[:, index]
            weights = np.e ** (
                df.apply(lambda t: self._get_log_likelihood(dict(t)), axis=1)
            )
            df["_weight"] = (weights / weights.sum()) * n_counts[
                tuple(data_unique.iloc[i])
            ]
            cache.append(df)
        else:
            # For each possible assignment to missing variables
            assignments = list(
                product(*[self.state_names[var] for var in missing])
            )
            dfs = []
            weights = []
            for assignment in assignments:
                full_row = observed.copy()
                full_row.update(dict(zip(missing, assignment)))
                # For latent variables, enumerate all possible states
                v = list(
                    product(*[range(card) for card in latent_card.values()])
                )
                latent_combinations = np.array(v, dtype=int)
                df = pd.DataFrame([full_row] * latent_combinations.shape[0])
                for index, latent_var in enumerate(latent_card.keys()):
                    df[latent_var] = latent_combinations[:, index]
                w = np.e ** (
                    df.apply(
                        lambda t: self._get_log_likelihood(dict(t)), axis=1
                    )
                )
                dfs.append(df)
                weights.append(w)
            all_df = pd.concat(dfs, ignore_index=True)
            all_weights = np.concatenate(weights)
            # Normalize weights for this row

            all_df["_weight"] = (
                all_weights
                / all_weights.sum()
                * n_counts.get(tuple(row.astype(object).fillna(-1)), 1)
            )
            cache.append(all_df)

    return pd.concat(cache, copy=False)


def _patched_compute_weights(self, n_jobs, latent_card, batch_size):
    """
    For each data point, creates extra data points for each possible combination
    of states of latent variables and assigns weights to each of them.
    Handles random missing values by marginalizing over missing variables.
    """
    data_unique = self.data.drop_duplicates()
    n_counts = (
        self.data.groupby(list(self.data.columns), observed=True)
        .size()
        .to_dict()
    )

    cache = Parallel(n_jobs=n_jobs)(
        delayed(self._parallel_compute_weights)(
            data_unique, latent_card, n_counts, i, batch_size
        )
        for i in range(0, data_unique.shape[0], batch_size)
    )

    return pd.concat(cache, copy=False)


def apply_em_missing_values_patch():
    """
    Conditionally monkey-patches pgmpy.estimators.ExpectationMaximization
    to support random missing values, using the upstream GitHub patch.

    Safe to call multiple times; does nothing if:
      - pgmpy is not installed
      - pgmpy.__version__ >= _MIN_FIXED_VERSION
    """
    if EM is None:
        return

    pgmpy_version = getattr(pgmpy, "__version__", "0.0.0")
    if version.parse(pgmpy_version) >= version.parse(_MIN_FIXED_VERSION):
        # Newer pgmpy already has the fix.
        return

    # Bind the patched methods to the EM class
    EM._parallel_compute_weights = _patched_parallel_compute_weights
    EM._compute_weights = _patched_compute_weights

    warnings.warn(
        f"Applied custom EM missing-values patch to pgmpy "
        f"(pgmpy {pgmpy_version} < fixed {_MIN_FIXED_VERSION})."
    )
