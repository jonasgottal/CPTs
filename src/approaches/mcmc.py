import numpy as np
from approaches.copula_optimization import CopulaOptimization
from utils.baselines import strategy_random_dirichlet


class MCMCStrategy(CopulaOptimization):
    """
    Optimizes CPT probability masses directly using Metropolis-Hastings.
    Inherits _calculate_loss and report handling from CopulaOptimization.
    """

    def __init__(
        self,
        json_reports,
        skeleton,
        state_names,
        continuous_map=None,
    ):
        super().__init__(json_reports, skeleton, state_names, continuous_map)

    def fit(self, steps=1000, n_samples=500):
        # 1. State: Initialize with strategy_random_dirichlet (Zero Knowledge)

        current_bn = strategy_random_dirichlet(
            None, self.skeleton, None, self.state_names
        )

        # Initial Likelihood/Loss
        current_loss = self._evaluate_bn(current_bn, n_samples)
        # print(f"Direct MCMC: Initial Loss = {current_loss:.4f}")

        history = []

        for i in range(steps):
            # 3. Proposal: Perturb one column in one CPT
            proposal_bn = self._perturb_model(current_bn)

            # 2. Likelihood Function
            proposal_loss = self._evaluate_bn(proposal_bn, n_samples)

            # 4. Acceptance (Metropolis-Hastings)
            
            delta_loss = proposal_loss - current_loss

            if delta_loss < 0 or np.random.rand() < np.exp(-delta_loss):
                current_bn = proposal_bn
                current_loss = proposal_loss

            
            if i >= (steps - 50):
            
                history.append(current_bn)

        # 5. Result: Bayesian Model Averaging
        final_bn = self._average_models(history)
        # print(f"Direct MCMC: Final Loss = {current_loss:.4f}")
        return final_bn

    def _evaluate_bn(self, bn, n_samples):
        """Generates data from the BN and calculates loss against JSON stats."""
        try:
            df_sample = bn.simulate(n_samples=n_samples, show_progress=False)

            
            df_val_inf = df_sample.copy()
            for c_col, info in self.continuous_map.items():
                src = info["source_col"]
                if src in df_val_inf:
                    mapper = lambda x: info["map"].get(x, (0, 0))[0]
                    df_val_inf[c_col] = df_val_inf[src].apply(mapper)

            return self._calculate_loss(df_val_inf)
        except Exception:

            return float("inf")

    def _perturb_model(self, bn):
        """
        Proposal Function:
        1. Pick one random CPT (Node).
        2. Pick one random column (Parent Context).
        3. Add Gaussian noise and re-normalize.
        """
        
        new_bn = bn.copy()

        nodes = list(new_bn.nodes())
        target_node = np.random.choice(nodes)

        cpd = new_bn.get_cpds(target_node)

        vals = cpd.values.copy()

        if vals.ndim == 1:
            # No parents
            noise = np.random.normal(0, 0.05, size=vals.shape)
            new_vals = np.abs(vals + noise)
            new_vals /= new_vals.sum()
            cpd.values = new_vals
        else:
            # Has parents
            n_cols = vals.shape[1]
            col_idx = np.random.randint(n_cols)

            # Extract column
            col = vals[:, col_idx]

            # Add small Gaussian noise
            noise = np.random.normal(0, 0.05, size=col.shape)

            # Re-normalize (ensure positivity with abs)
            new_col = np.abs(col + noise)
            new_col /= new_col.sum()

            # Update column
            vals[:, col_idx] = new_col
            cpd.values = vals

        return new_bn

    def _average_models(self, history):
        """Averages the CPT parameters of the models in history."""
        if not history:
            return None

        base_model = history[0]
        nodes = base_model.nodes()

        # Dictionary to accumulate values: {node: sum_of_cpt_values}
        acc_values = {}
        count = len(history)

        # Sum up all CPTs
        for bn in history:
            for node in nodes:
                vals = bn.get_cpds(node).values
                if node not in acc_values:
                    acc_values[node] = np.zeros_like(vals)
                acc_values[node] += vals

        # Create final model
        final_bn = base_model.copy()

        # Average and set
        for node in nodes:
            avg_vals = acc_values[node] / count

            # Renormalize to fix any floating point drift
            if avg_vals.ndim == 1:
                avg_vals /= avg_vals.sum()
            else:
                # Normalize along the variable axis (axis 0)
                sums = avg_vals.sum(axis=0)
                # Avoid division by zero
                sums[sums == 0] = 1
                avg_vals /= sums

            final_bn.get_cpds(node).values = avg_vals

        return final_bn


def strategy_mcmc(
    json_reports,
    skeleton,
    raw_data_placeholder,
    state_names,
    continuous_map=None,
    iterations=None,
    n_samples=None,
):
    """Wrapper function to be compatible with run_synthetic.py execution strategies."""
    strat = MCMCStrategy(json_reports, skeleton, state_names, continuous_map)
    return strat.fit(steps=iterations, n_samples=n_samples)
