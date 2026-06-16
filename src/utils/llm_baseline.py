import numpy as np
import itertools
from pydantic import BaseModel, Field
from typing import List, Dict, Optional
from pgmpy.models import DiscreteBayesianNetwork
from pgmpy.factors.discrete import TabularCPD
from openai import OpenAI
from dotenv import load_dotenv
import os
from pathlib import Path


# Load .env from project root (two levels up from current file)
env_path = Path(__file__).parent.parent.parent / ".env"
load_dotenv(dotenv_path=env_path)

api_key = os.environ.get("OPENAI_API_KEY")
client = OpenAI(api_key=api_key)

# --- Pydantic Models for Structured Output ---


class KeyValue(BaseModel):
    key: str
    value: str


class ProbEntry(BaseModel):
    state: str
    probability: float


class CPTRow(BaseModel):
    # List of parent assignments, e.g. [{"key": "Treatment", "value": "drug"}]
    parent_states: List[KeyValue]
    # List of probabilities, e.g. [{"state": "good", "probability": 0.8}]
    distributions: List[ProbEntry]


class NodeCPT(BaseModel):
    node_name: str
    rows: List[CPTRow]


# --- Few-Shot Examples ---


def _get_few_shot_example(node_type: str, parent_type: str) -> str:
    """Returns a text block demonstrating the expected output format."""
    if node_type == "binary" and parent_type == "binary":
        return """
Example for Binary Node 'Outcome' with Binary Parent 'Treatment':
Node: Outcome (states: ['bad', 'good'])
Parents: ['Treatment'] (states: ['drug', 'placebo'])
Report: "Positive correlation between Treatment(drug) and Outcome(good)."

Output CPT:
[
  {"condition": {"Treatment": "drug"},    "probabilities": {"bad": 0.2, "good": 0.8}},
  {"condition": {"Treatment": "placebo"}, "probabilities": {"bad": 0.6, "good": 0.4}}
]
"""
    elif node_type == "ordinal" and parent_type == "continuous":
        return """
Example for Ordinal Node 'Pain' (low, med, high) with Continuous Parent 'Dose':
Report: "High Dose significantly reduces Pain."

Output CPT (Continuous parent discretized to bins: low, normal, high):
[
  {"condition": {"Dose": "high"},   "probabilities": {"low": 0.8, "med": 0.15, "high": 0.05}},
  {"condition": {"Dose": "normal"}, "probabilities": {"low": 0.3, "med": 0.4, "high": 0.3}},
  {"condition": {"Dose": "low"},    "probabilities": {"low": 0.1, "med": 0.3, "high": 0.6}}
]
"""
    else:
        # Generic fallback
        return """
Example:
[
  {"condition": {"ParentA": "state1"}, "probabilities": {"ChildState1": 0.7, "ChildState2": 0.3}},
  ...
]
"""


# --- Main Strategy ---


def strategy_llm_baseline(
    json_reports, skeleton, state_names, seed=None, model_name="gpt-4o-mini"
):
    if seed is not None:
        np.random.seed(seed)

    bn_model = DiscreteBayesianNetwork(skeleton.edges())

    # Pre-compute parent map for easier processing
    parent_map = {n: list(bn_model.get_parents(n)) for n in skeleton.nodes()}

    print(f"Starting LLM Baseline ({model_name})...")

    for node in skeleton.nodes():
        parents = parent_map[node]
        node_states = state_names[node]
        parent_states = {p: state_names[p] for p in parents}

        # Determine types for few-shot selection (simple heuristic)

        n_type = "binary" if set(node_states) == {"yes", "no"} else "ordinal"
        p_type = (
            "binary"
            if parents and set(parent_states[parents[0]]) == {"yes", "no"}
            else "continuous"
        )

        few_shot_text = _get_few_shot_example(n_type, p_type)

        # Filter relevant reports
        relevant_reports = [
            r
            for r in json_reports
            if r.get("relation") and (node in r["relation"])
        ]

        prompt = f"""
        You are an expert Bayesian Statistician. Your task is to estimate the Conditional Probability Table (CPT) for a node in a Bayesian Network based on statistical summary reports.

        ### CONTEXT
        Target Node: '{node}'
        States: {node_states}
        
        Parents: {parents}
        Parent States: {parent_states}

        ### STATISTICAL REPORTS
        {relevant_reports}

        ### INSTRUCTIONS
        1. Analyze the reports to determine correlations (positive, negative, or neutral).
        2. Construct a probability distribution for '{node}' for **every possible combination** of parent states.
        3. If a positive correlation exists (e.g., Parent=High -> Child=High), assign higher probability to matching states.
        4. If no clear correlation is reported, assume a uniform or weak prior.
        5. Probabilities in each row must sum to 1.0.

        ### FORMAT EXAMPLE
        {few_shot_text}
        """

        try:
            completion = client.beta.chat.completions.parse(
                model=model_name,
                messages=[
                    {
                        "role": "system",
                        "content": "You are a helpful data scientist.",
                    },
                    {"role": "user", "content": prompt},
                ],
                response_format=NodeCPT,
            )
            parsed_cpt = completion.choices[0].message.parsed

            cpd = _convert_llm_response_to_pgmpy(
                node, parents, node_states, parent_states, parsed_cpt
            )
            bn_model.add_cpds(cpd)
            # print(f"  > Generated CPT for {node}")

        except Exception as e:
            print(f"  ! Failed {node}: {e}. Using Random.")
            fallback = _create_random_cpd(node, parents, state_names)
            bn_model.add_cpds(fallback)

    if bn_model.check_model():
        return bn_model
    return bn_model


def _convert_llm_response_to_pgmpy(
    node, parents, node_states, parent_states, llm_cpt: NodeCPT
):
    """
    Maps the loose LLM dictionary rows to the strict flattened array pgmpy expects.
    pgmpy iteration order: Iterates first parent's states, then second... (Standard lexicographical)
    Actually pgmpy's internal order usually matches itertools.product of parents.
    """

    # Generate all expected parent combinations in pgmpy order
    if not parents:
        # Root node
        expected_combinations = [()]
    else:
        # pgmpy expects evidence order to match the list passed to `evidence` arg
        parent_cards = [len(parent_states[p]) for p in parents]
        # Create list of lists of states
        lists_of_states = [parent_states[p] for p in parents]
        expected_combinations = list(itertools.product(*lists_of_states))

    # Create a lookup for the LLM provided rows
    # Key: Tuple of parent states corresponding to 'parents' list order
    llm_lookup = {}
    for row in llm_cpt.rows:
        # Convert List[KeyValue] back to dictionary for lookup
        cond_dict = {item.key: item.value for item in row.parent_states}

        # Create key tuple sorted by parent order
        key = tuple(cond_dict.get(p) for p in parents)

        # Convert List[ProbEntry] to dict
        dist_dict = {
            item.state: item.probability for item in row.distributions
        }

        llm_lookup[key] = dist_dict


    flat_values = []

    for combo in expected_combinations:
        dist_dict = llm_lookup.get(combo)

        if dist_dict:
            # Extract probs in correct state order
            probs = [dist_dict.get(s, 0.0) for s in node_states]
        else:
            # Fallback: Uniform
            probs = [1.0 / len(node_states)] * len(node_states)

        # Normalize to ensure sum to 1 (LLM math is often slightly off)
        total = sum(probs)
        if total == 0:
            probs = [1.0 / len(node_states)] * len(node_states)
        else:
            probs = [p / total for p in probs]

        flat_values.append(probs)

    values_array = np.array(flat_values).T

    return TabularCPD(
        variable=node,
        variable_card=len(node_states),
        values=values_array,
        evidence=parents,
        evidence_card=[len(parent_states[p]) for p in parents],
        state_names={**parent_states, node: node_states},
    )


def _create_random_cpd(node, parents, state_names):
    # (Same logic as random baseline for fallback)
    card = len(state_names[node])
    parent_cards = [len(state_names[p]) for p in parents]
    n_columns = int(np.prod(parent_cards)) if parent_cards else 1
    rand_values = np.random.dirichlet([1.0] * card, size=n_columns).T
    return TabularCPD(
        variable=node,
        variable_card=card,
        values=rand_values,
        evidence=parents,
        evidence_card=parent_cards,
        state_names=state_names,
    )
