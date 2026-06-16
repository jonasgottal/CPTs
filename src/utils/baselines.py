import numpy as np
from pgmpy.models import DiscreteBayesianNetwork
from pgmpy.factors.discrete import TabularCPD
from pgmpy.estimators import MaximumLikelihoodEstimator


def strategy_random_dirichlet(
    json_reports,
    skeleton,
    raw_data_discrete,
    state_names,
    seed=None,
    iterations=None,
    n_samples=None,
):
    """
    Constructs a BN with the correct structure but parameters sampled
    from a uniform Dirichlet distribution (alpha=1).
    This represents 'Zero Knowledge' / 'Pure Guessing'.
    """
    if seed is not None:
        np.random.seed(seed)
    model = DiscreteBayesianNetwork(skeleton.edges())

    # Iterate over every node to create a random TabularCPD
    for node in skeleton.nodes():
        # Get cardinality from state_names
        states = state_names[node]
        card = len(states)

        # Get parents to determine CPT shape
        parents = list(model.get_parents(node))
        parent_cards = [len(state_names[p]) for p in parents]

        # Total number of columns in the CPT (product of parent cardinalities)
        n_columns = int(np.prod(parent_cards)) if parent_cards else 1

        # Generate random probabilities
        # Shape: (variable_card, evidence_card_product)
        # We sample n_columns independent Dirichlet distributions
        # alpha=1.0 -> Uniform over the simplex (unbiased random)
        rand_values = np.random.dirichlet([1.0] * card, size=n_columns).T

        # Create the CPD
        cpd = TabularCPD(
            variable=node,
            variable_card=card,
            values=rand_values,
            evidence=parents,
            evidence_card=parent_cards,
            state_names=state_names,  
        )
        model.add_cpds(cpd)

    model.check_model()
    return model


def strategy_mle(
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
    model = DiscreteBayesianNetwork(skeleton.edges())
    model.add_nodes_from(skeleton.nodes())
    model.fit(
        raw_data, estimator=MaximumLikelihoodEstimator, state_names=state_names
    )
    return model
