# --------------------------------------------------------------------------
# Copyright 2020 Joel Dunham

# This file is part of DSNetwork.

# DSNetwork is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

# DSNetwork is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.

# You should have received a copy of the GNU General Public License
# along with DSNetwork.  If not, see <https://www.gnu.org/licenses/>.
# --------------------------------------------------------------------------

__version__ = "0.2.4"

from combinationRules import import_and_combine, import_and_calculate_probabilities
from math import sqrt


def calculate_child_marginals(parent_marginals, potentials):
    """
    Calculates the child marginals that should be based on the parent marginals and potentials
    :param parent_marginals: dict of parent node marginals
    :param potentials: dict of potentials for the transition
    :return: dict of calculated child marginals
    """
    calculated_child_marginals = {}
    for parent in potentials:
        if parent in parent_marginals:
            # Use this data; otherwise, was a zero anyway
            for child in potentials[parent]:
                if child not in calculated_child_marginals:
                    calculated_child_marginals[child] = 0
                calculated_child_marginals[child] += potentials[parent][child] * parent_marginals[parent]
    return calculated_child_marginals


def evaluate_network_consistency(network):
    """
    Evaluates the network consistency using an L2 norm based on the parent node marginals * transition potential -
        child node marginals
    :param network: The DSNetwork to evaluate
    :return: the sum of the L2 norm for every node.
    """
    sum_of_difference_norm = 0.0
    for node in network.nodes.values():
        # Check if there are parents to evaluate
        if len(node.parent_transitions) > 0:
            # Evaluate this node
            difference_norm_squared = 0.0
            # Automatically handle if a combination algorithm is needed
            evidence_num = 1
            evidences = {}
            for parent_transition in node.parent_transitions:
                parent_marginals = parent_transition.get_parent().get_marginals()
                potentials = parent_transition.conditionals_parent_to_child
                evidences[evidence_num] = calculate_child_marginals(parent_marginals, potentials)
                evidence_num += 1
            internal_data = import_and_combine(node.combination_method, evidences, None, 0.0)
            calculated_child_marginals = import_and_calculate_probabilities(node.combination_method, internal_data)
            actual_child_marginals = node.get_marginals()
            # Unless the network was set up wrongly, calculated and actual marginal keys match.
            for marginal_key, marginal_value in actual_child_marginals.items():
                difference_norm_squared += pow(marginal_value - calculated_child_marginals[marginal_key], 2)
            # Square root to get back to the L2 norm
            sum_of_difference_norm += sqrt(difference_norm_squared)
    # Return the final consistency check
    return sum_of_difference_norm


def evaluate_network_validity(network):
    """
    Evaluates the network validity using an L2 norm based on the sum of all marginals and all vertical columns in
     the transition potential matrices equal to 1.
    :param network: The DSNetwork to evaluate
    :return: the sum of the L2 norm for every node and vertical column of transition potential matrix - 1.0
    """
    sum_of_error = 0.0
    for node in network.nodes.values():
        marginals = node.get_marginals()
        if marginals is not None:
            sum_of_marginals = 1.0
            for marginal_value in marginals.values():
                sum_of_marginals -= marginal_value
            sum_of_error += sqrt(sum_of_marginals * sum_of_marginals)
    for transition in network.transitions.values():
        conditionals = transition.conditionals_parent_to_child
        if conditionals is not None:
            for child_dict in conditionals.values():
                sum_of_column = 1.0
                for child_potentials in child_dict.values():
                    sum_of_column -= child_potentials
                sum_of_error += sqrt(sum_of_column * sum_of_column)
    return sum_of_error


# Define any thetas that will be used universally
RISK = {
    "CATEGORIES": {
        "LOW": "LOW",
        "MEDIUM": "MEDIUM",
        "HIGH": "HIGH"
    },
    "COMPARE": {
        "LOW": 1,
        "MEDIUM": 2,
        "HIGH": 3
    },
    "REVERSE": {
        1: "LOW",
        2: "MEDIUM",
        3: "HIGH"
    }
}


RESULT_TYPE = {
    "CATEGORIES": {
        "SUCCESS": "SUCCESS",
        "CANCELLED": "CANCELLED",
        "FAILSAFE": "FAILSAFE",
        "FAIL": "FAIL"
    },
    "COMPARE": {
        "SUCCESS": 0,
        "CANCELLED": 1,
        "FAILSAFE": 2,
        "FAIL": 3
    },
    "REVERSE": {
        0: "SUCCESS",
        1: "CANCELLED",
        2: "FAILSAFE",
        3: "FAIL"
    }
}
