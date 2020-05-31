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

import abc
import operator
import numpy as np
from scipy.optimize import minimize, SR1, Bounds, LinearConstraint, NonlinearConstraint
from math import pow, sqrt
from combinationRules import COMBINATION_METHODS, import_and_combine, import_and_calculate_probabilities,\
    import_and_windowed_combine, import_and_combine_datasets
from combinationRulesExtensions import solve_for_parents_with_root_finder
from copy import deepcopy
from functools import partial

"""
Defines the network.  A network is a series of Nodes combined through Transitions.  DECISION nodes
 include NodeDecision definitions that can trigger alerts based on probability masses.  Currently simple, these
 NodeDecision definitions will be layered.
 Inputs are defined through input mechanisms that transform values to nodes, enabling automatic input to the decision
 tree.
"""

ROUNDOFF_DELTA = 1e-3
# Do not zero out values unless they can be truly zeroed out
LEARNING_SOLUTION_SETUP_ROUNDOFF_DELTA = 1e-16
MAX_ADDITIONAL_WEIGHT_DIVISOR = 0.0001

# The types of nodes.
# Analysis: A standard node for combining data and passing that data to other nodes through transitions
# Input: A node that also has a defined input mapping to it.  Only difference from an analysis node in that the input
#        mapping is defined.
# Decision: A node that has a defined output mapping to a decision.
NODE_TYPE = {
    "CATEGORIES": {
        "ANALYSIS": "ANALYSIS",
        "DECISION": "DECISION",
        "INPUT": "INPUT"
    },
    "COMPARE": {
        "ANALYSIS": 1,
        "DECISION": 2,
        "INPUT": 3
    }
}

NOTIFICATION_TYPE = {
    "CATEGORIES": {
        "TRUE": "TRUE",
        "FALSE": "FALSE",
        "ON_CHANGE": "ON_CHANGE",
    },
    "COMPARE": {
        "TRUE": 1,
        "FALSE": 2,
        "ON_CHANGE": 3
    }
}

WEIGHTING_METHOD = {
    "CATEGORIES": {
        "NONE": "NONE",
        "TO_ONE": "TO_ONE",
        "TOTAL": "TOTAL",
        "INVERSE": "INVERSE",
        "INVERSE_TOTAL": "INVERSE_TOTAL"
    },
    "COMPARE": {
        "NONE": 0,
        "TO_ONE": 1,
        "TOTAL": 2,
        "INVERSE": 3,
        "INVERSE_TOTAL": 4
    }
}

NONINFERENCE_WEIGHT_METHOD = {
    "CATEGORIES": {
        "MIN": "MIN",
        "MAX": "MAX",
        "AVERAGE": "AVERAGE"
    },
    "COMPARE": {
        "MIN": 0,
        "MAX": 1,
        "AVERAGE": 2
    }
}

MULTI_PARENT_SOLUTION_TYPE = {
    "CATEGORIES": {
        "OPTIMIZER": "OPTIMIZER",
        "ROOT_FINDER": "ROOT_FINDER",
        "ROOT_FINDER_OPT_FALLBACK": "ROOT_FINDER_OPT_FALLBACK"
    },
    "COMPARE": {
        "OPTIMIZER": 0,
        "ROOT_FINDER": 1,
        "ROOT_FINDER_OPT_FALLBACK": 2
    }
}

UNKNOWN_NAME = "Unknown"


class InputToNode:
    """
    Defines an input to a node - contains any mapping from inputs to node-specific data
    """
    def __init__(self, name="input name", input_nodes=None):
        """
        :param name: string name of the input mapping
        :param input_nodes: list of nodes to which the inputs map
        """
        # The name of the input
        self.name = name
        # The nodes to which this input maps - set internally
        self.nodes = {}
        for input_node in input_nodes:
            self.nodes[input_node.name] = input_node

    def __getstate__(self):
        return self.name, list(self.nodes.keys())

    def __setstate__(self, state):
        self.name, node_names = state
        # Will be overwritten with the actual node upon initialization through the FaultTree class
        self.nodes = {}
        for node_name in node_names:
            self.nodes[node_name] = node_name

    def _propagate_input(self, nodes_input, input_weight=0.0, inference_weight_factor=1.0,
                         input_weighting_method=None):
        """
        Call this function after map_to_inputs has been run to send this data into the node
        :param nodes_input: dict of dict of masses of evidence for the node - first key is node name
        :param input_weight: float weight of the input data 0.0 for no weighting performed
        :param inference_weight_factor: float [0.0, 1.0] if less than 1, further reduces weight for each inference
        :param input_weighting_method: None if using the default network weighting method.  Otherwise, dict of
               node name to weighting method for inputs that have overridden weighting methods.
        """
        input_weight = max(0.0, input_weight)  # Must be >= 0.0
        if input_weight < ROUNDOFF_DELTA:
            # Handling the input weights doesn't matter as the weights are being skipped.  In this case,
            #  just use the normal method of serial combinations
            for node_name, evidence in nodes_input.items():
                if node_name in self.nodes:
                    weighting_method = None
                    if (input_weighting_method is not None) and (node_name in input_weighting_method):
                        weighting_method = input_weighting_method[node_name]
                    # Zero out the input weight to reduce issues with roundoff.
                    self.nodes[node_name].propagate(evidence, input_weight=0.0, weighting_method=weighting_method)
        else:
            # Propagate through all nodes correctly - handling the weighting
            # Where life gets interesting.
            #  There is one constraint and one rule of thumb:
            #  constraint: the total weight into any node must be no more than input_weight since that is the weight
            #              of the evidence into the network.
            #  rule of thumb: the more conditionals through which evidence is propagated, the lower the weight of the
            #                 evidence should be.  This is because direct observation is considered more reliable than
            #                 inferred evidence.  Since this is a rule of thumb, it can be broken to enforce the
            #                 constraint.
            #  How these are implemented: given a network such as below:
            #    a---b---c---d
            #    f---|   |---e
            #    if data is entered with weight at a, f, and d: full weight with observation is entered at a, f, and d.
            #       Inference with 1/3 weight (from a,f,c) is entered at b.  Inference with 1/2 weight (from b, d) is
            #         entered at c. Inference (from b and c) with full weight is entered at e.
            #    if data is entered with weight at a, f, d, and e: full weight with observation is entered at a,f,d,e.
            #       Inference with 1/3 weight is entered at b (from a,f,c) and c (from b,d,e).
            #
            #  Final note: the rule of thumb can be implemented on top of what is stated.  If desired, any inferences
            #   can have further reductions in weight to limit the impact.  However, they can have no more weight than
            #   what is defined above.  inference_weight_factor is used to set this reduction.

            # Bound in the range [0.0, 1.0]
            # If set to 0.0, no inference occurs, but potentials are still rebalanced.
            inference_weight_factor = min(max(inference_weight_factor, 0.0), 1.0)
            for node_name, evidence in nodes_input.items():
                if node_name in self.nodes:
                    # Create the list of other observation nodes at this evidence input
                    other_observation_nodes = list(nodes_input.keys())
                    other_observation_nodes.remove(node_name)
                    # Get the weighting method if overridden
                    weighting_method = None
                    if (input_weighting_method is not None) and (node_name in input_weighting_method):
                        weighting_method = input_weighting_method[node_name]
                    # Input to the node
                    self.nodes[node_name].propagate(nodes_input[node_name], input_weight=input_weight,
                                                    inference_weight_factor=inference_weight_factor,
                                                    other_observation_nodes=other_observation_nodes,
                                                    weighting_method=weighting_method)

    @abc.abstractmethod
    def map_to_input(self, inputs, input_weight=0.0, **kwargs):
        """
        Maps inputs to the node inputs - defined by the user
        :param inputs: user defined
        :param input_weight: weighting information for the inputs - user-defined
        :param kwargs: additional keyword arguments - ignored for the pass through
        """


class NodeDecision:
    """
    Defines a decision output for a node.
    """
    def __init__(self):
        # The name of the decision
        self.name = "decision name"
        # The list of probability mass tuples
        self.__mass = [(UNKNOWN_NAME, )]
        # Sum or any of
        self.sum_mass = False
        # When to notify
        self.notification_type = NOTIFICATION_TYPE["CATEGORIES"]["TRUE"]
        # The condition used to check the probability mass against the value
        self.condition = operator.eq
        # The conditional value for checking the probability mass
        self.value = 0.0
        # Set internally - the full name of the trigger including the node name
        self.full_name = UNKNOWN_NAME + "." + self.name
        # The tree reference to enable the callbacks - not stored
        # Set internally; don't fill out when creating a NodeDecision
        self.tree_ref = None
        # The last state - whether triggered or not
        self.__last_state_triggered = False

    def __getstate__(self):
        return self.name, self.full_name, self.__mass, self.sum_mass, self.condition, self.value,\
               self.__last_state_triggered

    def __setstate__(self, state):
        self.name, self.full_name, self.__mass, self.sum_mass, self.condition, self.value,\
            self.__last_state_triggered = state

    def set_mass(self, masses):
        """
        Set the masses, sorted
        :param masses: iterable of masses
        """
        self.__mass = []
        for mass in masses:
            if isinstance(mass, tuple):
                save_mass = tuple(sorted(mass))
            else:
                save_mass = (mass,)
            self.__mass.append(save_mass)

    def check_trigger(self, masses):
        """
        Given the current node masses, should a notification be sent?
        :param masses: dict of current node masses
        :return: tuple of boolean of whether a notification should be sent and boolean of state triggered
        """
        return_value = False
        in_triggered_state = False
        if self.sum_mass is True:
            mass_sum = 0
            for mass in self.__mass:
                if mass in masses:
                    mass_sum += masses[mass]
            if self.condition(mass_sum, self.value) is True:
                in_triggered_state = True
        else:
            for mass in self.__mass:
                if mass in masses:
                    if self.condition(masses[mass], self.value):
                        in_triggered_state = True
        if (self.notification_type == NOTIFICATION_TYPE["CATEGORIES"]["TRUE"]) and (in_triggered_state is True):
            return_value = True
        elif (self.notification_type == NOTIFICATION_TYPE["CATEGORIES"]["FALSE"]) and (in_triggered_state is False):
            return_value = True
        elif (self.notification_type == NOTIFICATION_TYPE["CATEGORIES"]["ON_CHANGE"]) and\
                (in_triggered_state != self.__last_state_triggered):
            return_value = True
        self.__last_state_triggered = in_triggered_state
        return return_value, self.__last_state_triggered


class BaseStorage:
    """
    Contains any common data and settings between Nodes, Transitions, etc.
    """
    def __init__(self):
        self.name = UNKNOWN_NAME
        self.description = "None"
        # Weight of internal data - could be in integer units or float (such as hours of data) to enable correctly
        #  weighting incoming data
        self.internal_data_weight = 0.0
        # How incoming data will be weighted versus internal data - not saved at this level
        self.weighting_method = WEIGHTING_METHOD["CATEGORIES"]["NONE"]
        # Combination method (not saved, but used for required imports to combine data)
        self.combination_method = None
        # Used for node updates during non-inference updates
        self.noninference_weight_update = NONINFERENCE_WEIGHT_METHOD["CATEGORIES"]["AVERAGE"]

    def copy_data(self, from_instance):
        """
        Copies data weight
        :param from_instance: BaseStorage from which to copy data
        """
        self.internal_data_weight = from_instance.internal_data_weight

    def clear_data(self):
        """
        Clears state data
        """
        self.internal_data_weight = 0.0

    @abc.abstractmethod
    def propagate(self, input_masses=None, source=None, inference=True, input_weight=0.0,
                  inference_weight_factor=1.0, other_observation_nodes=None,
                  weighting_method=None, debug=False):
        """
        Propagates the inputs to all possible output paths from this node or transition.
        Note: should only be called from DSNetwork.input_evidence, as that function manages important information
         for simultaneous updates.  If called directly, the caller should increase the number of evidences in all nodes
         by 1.
        :param input_masses: dict of options to probability masses going into the node
        :param source: None if the data is entered directly.  Otherwise, the source transition
        :param inference: True if inference should be performed, false otherwise
        :param input_weight: float weight of the input data 0.0 for no weighting performed
        :param inference_weight_factor: float [0.0, 1.0] if less than 1, further reduces weight for each inference
        :param other_observation_nodes: if multiple observations are entered simultaneously, this is the list
               of other observation nodes, used to determine how to change input weights
        :param weighting_method: one of WEIGHTING_METHOD["CATEGORIES"] or None if using the default network option
        :param debug: bool whether to print debug output
        """


class Node(BaseStorage):
    """
    Defines a node in the fault tree
    """
    def __init__(self):
        super(Node, self).__init__()
        # Node type (default is input)
        self.type = NODE_TYPE["CATEGORIES"]["INPUT"]
        # The list of options for this node.
        self.options = []
        # This is the dictionary of internal data.  This is used by the Decision class to combine data.
        #  The exact structure of this dictionary is defined by the combination method.
        self.internal_probability_data = {}
        # Only used for non-inference updates, so doesn't need to be saved as part of the state
        self.updates = 0
        # Only used for non-inference updates, so doesn't need to be saved as part of the state
        self.update_evidences = {}
        # Parent transitions (flow down to this node)
        self.parent_transitions = []
        # Child transitions (flow from this node)
        self.child_transitions = []
        # Decisions - only required if this is a DECISION type node
        self.decisions = {}
        # Combination limit
        self.combination_limit = None
        # Used for learning - each situation is different.  If two have the same ID, then they are combined.
        #  This maps an ID to internal_probability_data for use in learning.  These are not retained when the
        #  system is saved (for now), so any learning is captured in the transition matrix to be continued for
        #  subsequent learning.  This requires the same combination method, so changing the combination method
        #  also resets these.
        self.snapshot = {}
        self.snapshot_map = {}  # Used if matching mapped the situation to another
        self.current_snapshot_name = None
        # A zero snapshot is the "from" point for which snapshot deltas are determined
        self.zero_snapshot = None  # Not saved on dump to file either.

    def __getstate__(self):
        # Define the attributes to pickle since some are reconstructed.
        child_transition_names = []
        for transition in self.child_transitions:
            child_transition_names.append(transition.name)
        parent_transition_names = []
        for transition in self.parent_transitions:
            parent_transition_names.append(transition.name)

        return self.name, self.description, self.type, child_transition_names, parent_transition_names, \
            self.internal_probability_data, self.options, self.decisions, self.internal_data_weight, \
            self.combination_method, self.weighting_method, self.combination_limit, self.snapshot, self.snapshot_map

    def __setstate__(self, state):
        # Reconstruct the node from the saved attributes
        self.name, self.description, self.type, self.child_transitions, self.parent_transitions, \
            self.internal_probability_data, self.options, self.decisions, self.internal_data_weight, \
            self.combination_method, self.weighting_method, self.combination_limit, self.snapshot, \
            self.snapshot_map = state
        # Because python is stupid and doesn't run a constructor when this is called
        self.zero_snapshot = 0
        # self.snapshot = {}
        # self.snapshot_map = {}
        self.current_snapshot_name = None

    def set_combination_limit(self, combination_limit):
        """
        Sets the combination limit for this node
        :param combination_limit: None or greater than 1
        :return: boolean True if set False if otherwise
        """
        if (combination_limit is None) or (combination_limit > 1):
            self.combination_limit = combination_limit
            return True
        else:
            return False

    def copy_data(self, from_instance):
        """
        Copies probability data to the given instance (options, internal_probability_data, and internal_data_weight)
        :param from_instance: Node to which to copy data
        """
        super(Node, self).copy_data(from_instance)
        self.options = deepcopy(from_instance.options)
        self.internal_probability_data = deepcopy(from_instance.internal_probability_data)

    def clear_data(self):
        """
        Clears state data
        """
        super(Node, self).clear_data()
        self.internal_probability_data = {}
        self.updates = 0
        self.update_evidences = {}

    def set_zero_snapshot(self, clear=False):
        """
        Sets the zero snapshot - the "from" point for determining changes in situations.
        :param clear: True if clearing the zero snapshot
        """
        if clear is True:
            self.zero_snapshot = None
        else:
            self.zero_snapshot = import_and_calculate_probabilities(self.combination_method,
                                                                    self.internal_probability_data)

    def set_current_snapshot_name(self, name, clear=False):
        """
        Sets the current snapshot name for a learning situation
        :param name: string name of the snapshot.  Can be None if clearing
        :param clear: True to clear the current snapshot name
        """
        if (clear is True) or (name is None):
            self.current_snapshot_name = None
        else:
            self.current_snapshot_name = name

    def set_snapshot(self, name, clear=False, map_name=None):
        """
        Sets a snapshot for learning.  The snapshots are saved as part of state.
        :param clear: boolean - set to True if should clear the snapshot
        :param name: string or integer.  Requires a name so that matching can occur but can be referenced back to the
                parent situation.  Unimportant if clear=True
        :param map_name: string or None: a name to which to map if there is a known mapping
        """
        if clear is True:
            self.snapshot = {}
            self.snapshot_map = {}
        elif name in self.snapshot:  # Preset match
            # Combine
            self.snapshot[name] = import_and_combine_datasets(self.combination_method, self.snapshot[name],
                                                              self.internal_probability_data,
                                                              self.combination_limit)
        elif (map_name is not None) and (map_name in self.snapshot):
            self.snapshot_map[name] = map_name
            # Combine
            self.snapshot[map_name] = import_and_combine_datasets(self.combination_method, self.snapshot[map_name],
                                                                  self.internal_probability_data,
                                                                  self.combination_limit)
        elif map_name is not None:  # Odd, but possible, so still save
            self.snapshot_map[name] = map_name
            self.snapshot[map_name] = deepcopy(self.internal_probability_data)
        else:  # TODO Check for statistical matching
            self.snapshot[name] = deepcopy(self.internal_probability_data)

    def combine_input(self, input_masses, input_weight=0.0, weighting_method=None, debug=False, inference=True):
        """
        Combines the new input with the current state
        :param input_masses: dict of options to probability masses
        :param input_weight: float weight of the input data 0.0 for no weighting performed
        :param weighting_method: one of WEIGHTING_METHOD["CATEGORIES"] or none to use the default network option
        :param debug: bool whether to print debug output
        :param inference: bool whether this is only flow-down (False) or a full network update (True)
        """
        # Use the Dempster-Shafer combination methods to combine the data
        if self.combination_method is None:
            raise ValueError("Node " + self.name + " combine_input: combination_method is none")
        else:
            # Import and combine
            # Determine the relative weight
            if inference is True:
                relative_weight = input_weight
                if weighting_method is None:
                    weighting_method = self.weighting_method
                if weighting_method == WEIGHTING_METHOD["CATEGORIES"]["NONE"]:
                    relative_weight = 0.0
                elif weighting_method == WEIGHTING_METHOD["CATEGORIES"]["TOTAL"]:
                    if self.internal_data_weight > ROUNDOFF_DELTA:
                        # Already using the internal data weight internally in the combination method
                        relative_weight = input_weight
                    else:
                        relative_weight = 0.0  # Standard input - no weighting required
                elif weighting_method == WEIGHTING_METHOD["CATEGORIES"]["INVERSE"]:
                    if input_weight > ROUNDOFF_DELTA:
                        input_weight = 1.0 / input_weight  # Inverse to bring all impacts back to a per weight basis
                        relative_weight = input_weight
                        if "evidence_weight" in self.internal_probability_data:
                            # Reset for proper inverse
                            self.internal_probability_data["evidence_weight"] = 1.0
                    else:
                        relative_weight = 0.0  # Standard input - no weighting required
                elif weighting_method == WEIGHTING_METHOD["CATEGORIES"]["INVERSE_TOTAL"]:
                    if input_weight > ROUNDOFF_DELTA:
                        # Inverse to bring all impacts to a per-weight basis, then divide by the internal data weight
                        #  to adjust the results more slowly
                        input_weight = 1.0 / input_weight
                        # Already using the internal data weight internally in the combination method
                        relative_weight = input_weight
                    else:
                        relative_weight = 0.0  # Standard input - no weighting required
                else:
                    # TO_ONE: relative_weight = input_weight / 1.0 = input_weight
                    if "evidence_weight" in self.internal_probability_data:
                        # Reset for proper inverse
                        self.internal_probability_data["evidence_weight"] = 1.0
                if ("number_of_evidences" in self.internal_probability_data) and\
                        (self.combination_limit is not None):
                    self.internal_probability_data = import_and_windowed_combine(self.combination_method,
                                                                                 input_masses,
                                                                                 self.combination_limit,
                                                                                 self.internal_probability_data,
                                                                                 relative_weight)
                else:
                    self.internal_probability_data = import_and_combine(self.combination_method,
                                                                        input_masses,
                                                                        self.internal_probability_data,
                                                                        relative_weight)
                # New data added to node - add in the experience (with inverse if appropriate)
                # Note that this will track the internal data weight inside the combination method.
                self.internal_data_weight += input_weight
            else:
                # During flow-down updates, all data is seen as equal since this is typically multi-parent cases and
                #  shouldn't skew the data towards the order of update (first one would always have a weight of 1).
                # Rebuilding the network
                if self.updates == 0:
                    self.internal_data_weight = input_weight
                elif self.noninference_weight_update == NONINFERENCE_WEIGHT_METHOD["CATEGORIES"]["AVERAGE"]:
                    # Average the experience
                    self.internal_data_weight = ((self.internal_data_weight * self.updates) + input_weight) / \
                                                (self.updates + 1)
                elif self.noninference_weight_update == NONINFERENCE_WEIGHT_METHOD["CATEGORIES"]["MIN"]:
                    # Min the experience
                    self.internal_data_weight = min(self.internal_data_weight, input_weight)
                else:
                    # Max the experience
                    self.internal_data_weight = max(self.internal_data_weight, input_weight)
                self.updates += 1

                # Save the evidence for an all-at-once combination
                self.update_evidences[self.updates] = deepcopy(input_masses["evidence"])
                if self.updates >= len(self.parent_transitions):
                    # Received all the data - do the combination now
                    self.internal_probability_data = import_and_combine(self.combination_method,
                                                                        self.update_evidences,
                                                                        self.internal_probability_data)
                    if "evidence_weight" in self.internal_probability_data:
                        # The weight must be assigned inside the combination data to enable proper updates
                        self.internal_probability_data["evidence_weight"] = self.internal_data_weight

        # Handle decisions based on the updated masses, assuming there is mass information
        if (self.type == NODE_TYPE["CATEGORIES"]["DECISION"]) and (self.internal_probability_data is not None) and\
                self.internal_probability_data:
            # Run through all decisions and determine if a trigger needs to be made
            for decision_key in self.decisions.keys():
                decision = self.decisions[decision_key]
                probabilities = import_and_calculate_probabilities(self.combination_method,
                                                                   self.internal_probability_data)
                should_notify, decision_state = decision.check_trigger(probabilities)
                if should_notify is True:
                    if decision.tree_ref is not None:
                        decision.tree_ref.trigger_callback(decision.full_name, decision_state)
                    else:
                        raise ValueError("tree_ref in decision " + decision.full_name + " is None - cannot trigger")

    def propagate(self, input_masses=None, source=None, inference=True, input_weight=0.0,
                  inference_weight_factor=1.0, other_observation_nodes=None,
                  weighting_method=None, debug=False):
        """
        Propagates the inputs to all possible output transitions from this node
        :param input_masses: dict of options to probability masses going into the node
        :param source: None if the data is entered directly.  Otherwise, the source transition
        :param inference: True if inference should be performed, false otherwise
        :param input_weight: float weight of the input data 0.0 for no weighting performed
        :param inference_weight_factor: float [0.0, 1.0] if less than 1, further reduces weight for each inference
        :param other_observation_nodes: if multiple observations are entered simultaneously, this is the list
               of other observation nodes, used to determine how to change input weights
        :param weighting_method: one of WEIGHTING_METHOD["CATEGORIES"] or None if using the default network option
        :param debug: bool whether to print debug output
        """
        inference_weight_factor = min(max(inference_weight_factor, 0.0), 1.0)
        if input_masses is not None:
            if (input_weight > ROUNDOFF_DELTA) and (other_observation_nodes is not None) and\
                    (len(other_observation_nodes) > 0) and (source is not None):
                # Check for the specific weighting for this node
                num_input_directions = 1  # One known input direction
                for transition in self.parent_transitions:
                    if transition.name != source.name:
                        if transition.check_for_nodes(self.name, other_observation_nodes) is True:
                            num_input_directions += 1
                for transition in self.child_transitions:
                    if transition.name != source.name:
                        if transition.check_for_nodes(self.name, other_observation_nodes) is True:
                            num_input_directions += 1
                # This input weight is based on the number of inferred observations for this node.
                input_weight /= float(num_input_directions)
            self.combine_input(input_masses, input_weight=input_weight, weighting_method=weighting_method,
                               debug=debug, inference=inference)

        # TODO: Multi-thread this
        probabilities = import_and_calculate_probabilities(self.combination_method,
                                                           self.internal_probability_data)
        if inference is True:
            for transition in self.parent_transitions:
                if (source is None) or (transition.name != source.name):
                    # Use the current masses to propagate.
                    transition.propagate(probabilities, self, inference,
                                         input_weight=input_weight * inference_weight_factor,
                                         inference_weight_factor=inference_weight_factor,
                                         other_observation_nodes=other_observation_nodes,
                                         weighting_method=weighting_method, debug=debug)
        should_propagate_to_children = True
        if inference is False:
            # Only propagate if last parent is the one that has updated
            # Note that it should never be greater than, but don't like using an exact equal just to be safe
            should_propagate_to_children = self.updates >= len(self.parent_transitions)

        if should_propagate_to_children is True:
            for transition in self.child_transitions:
                if inference is True:
                    weight_to_pass = input_weight * inference_weight_factor
                else:
                    # Only pass the internal weight to enable the next level
                    #  to understand the information available to it
                    weight_to_pass = self.internal_data_weight
                if (source is None) or (transition.name != source.name):
                    # Use the current masses to propagate
                    transition.propagate(probabilities, self, inference,
                                         input_weight=weight_to_pass,
                                         inference_weight_factor=inference_weight_factor,
                                         other_observation_nodes=other_observation_nodes,
                                         weighting_method=weighting_method, debug=debug)

    def dump_to_csv(self, file_name):
        """
        Dumps node marginals to CSV lines to enable tracing and debugging.  Appends if possible, otherwise
         creates the new file
        :param file_name: the file name to which to dump
        """
        marginals = self.get_marginals()
        print_line = self.name + "," + str(self.combination_method) + "," + str(self.weighting_method) + ","
        print_line_keys = ",,,"
        print_line_values = ",,,"
        for marginal_key, marginal_value in marginals.items():
            print_line_keys += str(marginal_key).replace(",", "/") + ","
            print_line_values += str(marginal_value) + ","
        if (self.internal_probability_data is not None) and\
                ("last_evidence" in self.internal_probability_data):
            # Include the last evidence
            print_line_keys += ",Last Evidence,"
            print_line_values += ",,"
            for evidence_key, evidence_value in self.internal_probability_data["last_evidence"].items():
                print_line_keys += str(evidence_key).replace(",", "/") + ","
                print_line_values += str(evidence_value) + ","
        if (self.combination_method == COMBINATION_METHODS["MURPHY"]) and\
                (self.internal_probability_data is not None) and\
                ("evidence" in self.internal_probability_data):
            # Also include the internal evidence (the averaged data)
            print_line_keys += ",Evidence Weight,Num Evidences,"
            print_line_values += "," + str(self.internal_probability_data["evidence_weight"]) + "," +\
                str(self.internal_probability_data["number_of_evidences"]) + ","
            for evidence_key, evidence_value in self.internal_probability_data["evidence"].items():
                print_line_keys += str(evidence_key).replace(",", "/") + ","
                print_line_values += str(evidence_value) + ","
        with open(file_name, 'a+')as f:
            f.write(print_line + "\n")
            f.write(print_line_keys + "\n")
            f.write(print_line_values + "\n")

    def check_for_nodes(self, source_name, search_node_names):
        """
        Checks for specific nodes
        :param source_name: the sender name (to know which direction to search)
        :param search_node_names: the node names to find.
        :return: Whether any of the nodes were found.
        """
        if (source_name is not None) and (self.name in search_node_names):
            # Found one
            return True
        else:
            if source_name is None:
                # This is the starting source
                search_node_names = [self.name]
                source_name = self.name  # So it won't interfere with the direction checks
            # Search through all connections except the input
            for transition in self.parent_transitions:
                if transition.name != source_name:  # Don't check source direction
                    found_one = transition.check_for_nodes(self.name, search_node_names)
                    if found_one is True:
                        # Break early
                        return found_one
            for transition in self.child_transitions:
                if transition.name != source_name:  # Don't check source direction
                    found_one = transition.check_for_nodes(self.name, search_node_names)
                    if found_one is True:
                        # Break early
                        return found_one
        # Return final result - did not one of the specified nodes
        return False

    def initialize_unknown(self, initialize_value=0.0):
        """
        Initializes the node to unknown values
        :param initialize_value: value for non-unknown to initialize to if already set,
                otherwise None to not initialize
        """
        self.internal_data_weight = 0.0

        if (self.internal_probability_data is not None) and\
                (initialize_value is not None) and\
                (initialize_value > LEARNING_SOLUTION_SETUP_ROUNDOFF_DELTA):
            current_marginals = import_and_calculate_probabilities(self.combination_method,
                                                                   self.internal_probability_data)
        else:
            current_marginals = {}

        powerset = [tuple(sorted([x for j, x in enumerate(self.options) if (i >> j) & 1]))
                    for i in range(2 ** len(self.options))]
        powerset.remove(())  # No empty set
        all_options = tuple(sorted(self.options))
        node_input = {"evidence": {}}
        set_sum = 0.0
        for option in powerset:
            if option != all_options:
                if (option in current_marginals) and\
                        (current_marginals[option] > LEARNING_SOLUTION_SETUP_ROUNDOFF_DELTA):
                    set_value = initialize_value  # min(current_marginals[option], initialize_value)
                else:
                    set_value = 0.0
                set_sum += set_value
                node_input["evidence"][option] = set_value
        node_input["evidence"][all_options] = 1.0 - set_sum
        # Since different combination methods set up different information, use the combination method to set
        # Also, clear out the internal data (reset from scratch)
        self.internal_probability_data = import_and_combine(self.combination_method, node_input)

    def set_internal_probability_data(self, marginals):
        """
        Cannot set directly since the structure depends on the combination method
        """
        self.internal_data_weight = 0.0
        self.internal_probability_data = import_and_combine(self.combination_method, {"evidence": marginals})

    def get_marginals(self, snapshot_name=None):
        """
        Calculates and returns the margins from the internal data.  If snapshot_name is not None, uses the snapshot
         data if possible
        :param snapshot_name: String, Integer, or None
        :return: dict of options to marginals
        """
        lookup_name = snapshot_name
        if (snapshot_name is not None) and (snapshot_name in self.snapshot_map):
            lookup_name = self.snapshot_map[snapshot_name]
        if (lookup_name is not None) and (lookup_name in self.snapshot):
            internal_data = import_and_combine_datasets(self.combination_method,
                                                        self.snapshot[lookup_name],
                                                        self.internal_probability_data,
                                                        self.combination_limit)
            return deepcopy(import_and_calculate_probabilities(self.combination_method, internal_data))
        else:
            return deepcopy(import_and_calculate_probabilities(self.combination_method, self.internal_probability_data))


class Transition(BaseStorage):
    """
    Defines a transition between nodes
    """
    def __init__(self, parent=None, child=None):
        super(Transition, self).__init__()
        # The transition name is <parent node name>.<child node name> - overwrite the base name
        self.name = UNKNOWN_NAME + "." + UNKNOWN_NAME
        # Parent and child are node references - in the direction of the graph
        self.__parent = parent
        self.__child = child
        self.set_name()
        # Probability mass potentials - the conditionals for going from the parent to the child
        # Indexed by parent option, then by child option
        self.conditionals_parent_to_child = {}
        # Indexed by child option, then by parent option
        self.conditionals_child_to_parent = {}
        # Whether learning is enabled for this transition
        self.enable_learning = False
        # Snapshot of conditionals_parent_to_child for learning purposes
        self.zero_snapshot_conditionals = None
        # Used for weighting changes to marginals when learning transitions
        self.learning_weight = 0.0

    def __getstate__(self):
        # Define the attributes to pickle since some are reconstructed.
        return self.name, self.__parent.name, self.__child.name, self.conditionals_parent_to_child,\
               self.internal_data_weight, self.enable_learning, self.combination_method, self.weighting_method,\
               self.learning_weight

    def __setstate__(self, state):
        # Reconstruct the transition from the saved attributes
        self.name, self.__parent, self.__child, self.conditionals_parent_to_child, self.internal_data_weight,\
            self.enable_learning, self.combination_method, self.weighting_method, self.learning_weight = state
        # Because python is stupid and doesn't run a constructor when this is called
        self.zero_snapshot_conditionals = 0

        # Recreate the child to parent conditionals
        self.create_child_to_parent_conditionals()

    def copy_data(self, from_instance):
        """
        Copies probability data to the given instance (conditionals and conditionals_weight)
        :param from_instance: Node to which to copy data
        """
        super(Transition, self).copy_data(from_instance)
        self.conditionals_parent_to_child = deepcopy(from_instance.conditionals_parent_to_child)
        self.conditionals_child_to_parent = deepcopy(from_instance.conditionals_child_to_parent)

    def clear_data(self):
        """
        Clears state data
        """
        super(Transition, self).clear_data()
        self.conditionals_parent_to_child = {}
        self.conditionals_child_to_parent = {}

    def set_child(self, child):
        self.__child = child
        self.set_name()

    def get_child(self):
        return self.__child

    def set_parent(self, parent):
        self.__parent = parent
        self.set_name()

    def get_parent(self):
        return self.__parent

    def set_name(self):
        parent_name = UNKNOWN_NAME
        child_name = UNKNOWN_NAME
        if (self.__parent is not None) and (isinstance(self.__parent, Node)):
            parent_name = self.__parent.name
        if (self.__child is not None) and (isinstance(self.__child, Node)):
            child_name = self.__child.name
        self.name = parent_name + "." + child_name

    def set_zero_snapshot(self, clear=False):
        """
        Sets a snapshot for learning.  The snapshots are NOT saved as part of state.
        :param clear: boolean - set to True if should clear the snapshot
        """
        if clear is True:
            self.zero_snapshot_conditionals = None
        else:
            self.zero_snapshot_conditionals = deepcopy(self.conditionals_parent_to_child)

    def create_child_to_parent_conditionals(self):
        """
        Mirror the parent to child conditionals for computational speed
        """
        self.conditionals_child_to_parent = {}
        for parent in self.conditionals_parent_to_child.keys():
            for child in self.conditionals_parent_to_child[parent]:
                if child not in self.conditionals_child_to_parent:
                    self.conditionals_child_to_parent[child] = {}
                self.conditionals_child_to_parent[child][parent] =\
                    self.conditionals_parent_to_child[parent][child]

    def combine_input(self, input_masses, input_weight=0.0, weighting_method=None, debug=False, inference=True):
        """
        Combines the new input with the current state
        :param input_masses: dict of options to conditional masses
        :param input_weight: float weight of the input data 0.0 for no weighting performed
        :param weighting_method: one of WEIGHTING_METHOD["CATEGORIES"] or none to use the default option
        :param debug: bool whether to print debug output
        :param inference: bool whether inference is enabled or disabled for this update
        """
        # For now, just replace with the new data, making sure all keys are sorted tuples
        self.conditionals_parent_to_child = {}
        for parent_key in input_masses.keys():
            if isinstance(parent_key, tuple) is False:
                # Convert to a tuple
                new_parent_key = (parent_key,)
            else:
                # Sort the tuple to make sure everything aligns properly
                new_parent_key = tuple(sorted(parent_key))
            self.conditionals_parent_to_child[new_parent_key] = {}
            for child_key in input_masses[parent_key].keys():
                if isinstance(parent_key, tuple) is False:
                    # Convert to a tuple
                    new_child_key = (child_key,)
                else:
                    # Sort the tuple to make sure everything aligns properly
                    new_child_key = tuple(sorted(child_key))
                self.conditionals_parent_to_child[new_parent_key][new_child_key] = input_masses[parent_key][child_key]

        # Recreate the child to parent conditionals
        self.create_child_to_parent_conditionals()
        # Update the internal weight
        self.internal_data_weight += input_weight

    def propagate(self, input_masses=None, source=None, inference=True, input_weight=0.0,
                  inference_weight_factor=1.0, other_observation_nodes=None,
                  weighting_method=None, debug=False):
        """
        Converts source input_masses through the conditional to become an input for the sink.
        :param input_masses: dict of options to probability masses
        :param source: the source node (to tell the transition which direction to go)
        :param inference: Passes along to next node
        :param input_weight: float weight of the input data 0.0 for no weighting performed
        :param inference_weight_factor: float [0.0, 1.0] if less than 1, further reduces weight for each inference
        :param other_observation_nodes: if multiple observations are entered simultaneously, this is the list
               of other observation nodes, used to determine how to change input weights
        :param weighting_method: one of WEIGHTING_METHOD["CATEGORIES"] or None if using the default network option
        :param debug: whether to print debug output
        """
        did_propagate = False
        if source is None:
            # Direct input into this transition
            self.combine_input(input_masses, input_weight=input_weight, weighting_method=weighting_method, debug=debug)
            # Send messages in both directions using the updated conditionals, assuming data has already been set
            # TODO: Multi-thread this
            # Need both parent and child before re-propagating such that the second isn't
            #  changed during the re-propagation of the first
            # Get the last child input, if possible
            child_marginals = import_and_calculate_probabilities(self.__child.combination_method,
                                                                 self.__child.internal_probability_data)
            # Get the last parent input, if possible
            parent_marginals = import_and_calculate_probabilities(self.__parent.combination_method,
                                                                  self.__parent.internal_probability_data)
            if (child_marginals is not None) and (len(child_marginals) > 0):
                self.propagate(child_marginals, source=self.__child, inference=inference, input_weight=input_weight,
                               inference_weight_factor=inference_weight_factor,
                               other_observation_nodes=other_observation_nodes,
                               weighting_method=weighting_method, debug=debug)
            if (parent_marginals is not None) and (len(parent_marginals) > 0):
                self.propagate(parent_marginals, source=self.__parent, inference=inference, input_weight=input_weight,
                               inference_weight_factor=inference_weight_factor,
                               other_observation_nodes=other_observation_nodes,
                               weighting_method=weighting_method, debug=debug)
        elif source.name == self.__child.name:
            # Going to parent
            if self.__parent is None:
                raise ValueError("Transition " + self.name + "propagate: self.parent is None - cannot compute")
            elif input_masses is None:
                raise ValueError("Transition " + self.name + "propagate: input masses is None - cannot compute")
            elif (self.conditionals_child_to_parent is not None) and (len(self.conditionals_child_to_parent) > 0):
                # Convert through the conditionals
                # No need to create local/global ignorance - will be done via the combination later
                parent_marginals = {}
                total_sum = 0
                for child in self.conditionals_child_to_parent:
                    if child in input_masses:
                        # Use this data; otherwise, was a zero anyway
                        for parent in self.conditionals_child_to_parent[child]:
                            if parent not in parent_marginals:
                                parent_marginals[parent] = 0
                            marginal_add = self.conditionals_child_to_parent[child][parent] * input_masses[child]
                            total_sum += marginal_add
                            parent_marginals[parent] += marginal_add
                # Divide through to ensure input to parent adds to 1.0
                if total_sum > ROUNDOFF_DELTA:
                    for parent_key in parent_marginals.keys():
                        parent_marginals[parent_key] /= total_sum
                # Create new evidence from this
                # Do NOT propagate in one of three conditions:
                #  1) The weight factor has dropped to zero (effectively nothing to propagate)
                #  2) All masses in the evidence are zero (usually results from D-S combination)
                #  3) It would propagate to a node observed on this update
                if (inference_weight_factor > ROUNDOFF_DELTA) and (total_sum > ROUNDOFF_DELTA) and\
                        ((other_observation_nodes is None) or (self.__parent.name not in other_observation_nodes)):
                    self.__parent.propagate({"evidence": parent_marginals}, source=self, inference=inference,
                                            input_weight=input_weight,
                                            inference_weight_factor=inference_weight_factor,
                                            other_observation_nodes=other_observation_nodes,
                                            weighting_method=weighting_method, debug=debug)
                # Still rebalance potentials
                did_propagate = True
        elif source.name == self.__parent.name:
            # Going to child
            if self.__child is None:
                raise ValueError("Transition " + self.name + "propagate: self.child is None - cannot compute")
            elif input_masses is None:
                raise ValueError("Transition " + self.name + "propagate: input masses is None - cannot compute")
            elif (self.conditionals_parent_to_child is not None) and (len(self.conditionals_parent_to_child) > 0):
                # Convert through the conditionals
                # No need to create local/global ignorance - will be done via the combination later
                child_marginals = {}
                total_sum = 0
                for parent in self.conditionals_parent_to_child:
                    if parent in input_masses:
                        # Use this data; otherwise, was a zero anyway
                        for child in self.conditionals_parent_to_child[parent]:
                            if child not in child_marginals:
                                child_marginals[child] = 0
                            marginal_add = self.conditionals_parent_to_child[parent][child] * input_masses[parent]
                            total_sum += marginal_add
                            child_marginals[child] += marginal_add
                # Create new evidence from this
                # Do NOT propagate in one of three conditions:
                #  1) The weight factor has dropped to zero (effectively nothing to propagate)
                #  2) All masses in the evidence are zero (usually results from D-S combination)
                #  3) It would propagate to a node observed on this update
                if (inference_weight_factor > 0.0) and (total_sum > ROUNDOFF_DELTA) and \
                        ((other_observation_nodes is None) or (self.__child.name not in other_observation_nodes)):
                    self.__child.propagate({"evidence": child_marginals}, source=self, inference=inference,
                                           input_weight=input_weight,
                                           inference_weight_factor=inference_weight_factor,
                                           other_observation_nodes=other_observation_nodes,
                                           weighting_method=weighting_method, debug=debug)
                did_propagate = True
        else:
            raise ValueError("Transition " + self.name + "propagate: source.name " + source.name +
                             " is unknown - cannot compute")

        if (self.enable_learning is True) and (source is not None) and (did_propagate is True) and\
                (inference is True):
            # After propagating without errors, re-balance the transition
            # This will almost always be necessary other than the first update, so assume it is
            # always necessary.  There will only be a minor hit in processing on the first update.
            # Note: do not rebalance if there are multiple parents of the child.  This requires special processing,
            #  which happens after the update.
            # Inference is True check isn't essential, but ensures that the rebalance routine doesn't give slightly
            #  different results due to roundoff errors.
            if len(self.__child.parent_transitions) == 1:  # There is at least 1 since this transition exists.
                self.rebalance_potentials(debug=debug)

    def initialize_unknown(self, initialize_value=0.0):
        """
        Initializes to unknown values if parent and child set
        :param initialize_value: value for non-unknown to initialize to if already set,
                otherwise None to not initialize
        """
        self.internal_data_weight = 0.0
        if (self.__parent is not None) and (self.__child is not None):
            options = self.get_parent().options
            parent_powerset = [tuple(sorted([x for j, x in enumerate(options) if (i >> j) & 1]))
                               for i in range(2 ** len(options))]
            parent_powerset.remove(())  # No empty set

            options = self.get_child().options
            child_powerset = [tuple(sorted([x for j, x in enumerate(options) if (i >> j) & 1]))
                              for i in range(2 ** len(options))]
            child_powerset.remove(())  # No empty set
            child_all_options = tuple(sorted(options))
            for parent_option in parent_powerset:
                if parent_option in self.conditionals_parent_to_child:
                    previous_conditionals = deepcopy(self.conditionals_parent_to_child[parent_option])
                else:
                    previous_conditionals = {}
                self.conditionals_parent_to_child[parent_option] = {}
                set_sum = 0.0
                for child_option in child_powerset:
                    if child_option != child_all_options:
                        if (initialize_value is not None) and\
                                (initialize_value > LEARNING_SOLUTION_SETUP_ROUNDOFF_DELTA) and\
                                (child_option in previous_conditionals):
                            set_value = initialize_value  # min(previous_conditionals[child_option], initialize_value)
                        else:
                            set_value = 0.0
                        set_sum += set_value
                        self.conditionals_parent_to_child[parent_option][child_option] = set_value
                self.conditionals_parent_to_child[parent_option][child_all_options] = 1.0 - set_sum
            self.create_child_to_parent_conditionals()

    def print_debug_matrix(self, temp_parent_to_child, ls_temp_parent_to_child, adjustments,
                           parent_marginals, child_marginals):
        """
        Outputs debug information for solving the transition matrix
        """
        print_temp_parent_to_child = {}
        for parent_key in temp_parent_to_child.keys():
            print_temp_parent_to_child[parent_key] = {}
            for child_key in temp_parent_to_child[parent_key].keys():
                print_temp_parent_to_child[parent_key][child_key] = temp_parent_to_child[parent_key][child_key]
        for parent_key in ls_temp_parent_to_child.keys():
            for child_key in ls_temp_parent_to_child[parent_key].keys():
                temp_parent_to_child[parent_key][child_key] = ls_temp_parent_to_child[parent_key][child_key]
        if adjustments is not None:
            for parent_key in adjustments.keys():
                for child_key in adjustments[parent_key].keys():
                    temp_parent_to_child[parent_key][child_key] += adjustments[parent_key][child_key]
        error_string = ""
        child_keys = list(child_marginals.keys())
        parent_keys = list(parent_marginals.keys())
        print_sum_of_columns = '%80s' % " " + "  " + '%10s' % " " + '%10s' % " " + "     "
        sum_of_columns = {}
        for child_key_index in range(0, len(child_keys)):
            child_key = child_keys[child_key_index]
            test_child = 0.0
            print_line_answer = '%80s' % str(child_key) + ": " + \
                                '%10f' % child_marginals[child_key] + " = "
            print_line = ""
            for parent_key in parent_keys:
                if parent_key not in sum_of_columns:
                    sum_of_columns[parent_key] = 0.0
                sum_of_columns[parent_key] += temp_parent_to_child[parent_key][child_key]
                test_child += parent_marginals[parent_key] * \
                    temp_parent_to_child[parent_key][child_key]
                if (temp_parent_to_child[parent_key][child_key] < -LEARNING_SOLUTION_SETUP_ROUNDOFF_DELTA) or \
                        (temp_parent_to_child[parent_key][child_key] >
                         (1.0 + LEARNING_SOLUTION_SETUP_ROUNDOFF_DELTA)):
                    print_line += '*%10f' % temp_parent_to_child[parent_key][child_key] + "  "
                else:
                    print_line += ' %10f' % temp_parent_to_child[parent_key][child_key] + "  "
            print_line_answer += '%10f' % test_child + " = ["
            if child_key_index < len(parent_keys):
                print_line += "][" + '%10f' % parent_marginals[parent_keys[child_key_index]] + "] " + \
                              str(parent_keys[child_key_index])
            error_string += print_line_answer + print_line + "\n"
        if len(parent_keys) > len(child_keys):
            for parent_key_index in range(len(child_keys), len(parent_keys)):
                print_line = '%120s' % " "
                for _ in parent_keys:
                    print_line += '%11s' % " "
                print_line += " [" + '%10f' % parent_marginals[parent_keys[parent_key_index]] + "] " + \
                              str(parent_keys[parent_key_index])
                error_string += print_line + "\n"
        # Verify the sum of columns
        for parent_key in parent_keys:
            print_sum_of_columns += '%13f' % sum_of_columns[parent_key]
        error_string += print_sum_of_columns + "\n"
        print(error_string)

    # noinspection PyTypeChecker
    def rebalance_potentials(self, child_marginals=None, debug=False):
        """
        Update the potentials based on the parent and child marginals
        :param child_marginals: dict of child marginals if not using the ones stored in the child Node, otherwise None
                                to use the marginals stored in the child Node
        :param debug: bool whether to print debug output
        """
        parent_snapshot_name = None
        if self.__parent.current_snapshot_name is not None:
            # Use the parent snapshot name and combine
            parent_snapshot_name = self.__parent.current_snapshot_name
            parent_marginals = self.__parent.get_marginals(parent_snapshot_name)
        elif len(self.__parent.snapshot) > 0:
            # TODO: statistically match snapshots and use that name for the parent and the child if the child isn't
            #  given.
            parent_snapshot_name = None  # Will get from the parent matching
            # For now
            parent_marginals = import_and_calculate_probabilities(self.__parent.combination_method,
                                                                  self.__parent.internal_probability_data)
        else:
            parent_marginals = import_and_calculate_probabilities(self.__parent.combination_method,
                                                                  self.__parent.internal_probability_data)

        if child_marginals is None:
            child_marginals = self.__child.get_marginals(parent_snapshot_name)
        # else: use the passed-in values - these will already have to use the current situation or statistical
        #  matching.

        # Capture the transition matrix that will be used as the basis for optimization deltas
        if self.zero_snapshot_conditionals is None:
            transition_for_deltas = deepcopy(self.conditionals_parent_to_child)
        else:
            transition_for_deltas = deepcopy(self.zero_snapshot_conditionals)
        sum_child = 0.0
        has_single_output_child = None
        for child_key, child_marginal in child_marginals.items():
            sum_child += child_marginal
            if abs(1.0 - child_marginal) < LEARNING_SOLUTION_SETUP_ROUNDOFF_DELTA:
                has_single_output_child = child_key
        sum_parent = 0.0
        has_single_input_parent = None
        for parent_key, parent_marginal in parent_marginals.items():
            sum_parent += parent_marginal
            if abs(1.0 - parent_marginal) < LEARNING_SOLUTION_SETUP_ROUNDOFF_DELTA:
                has_single_input_parent = parent_key

        if abs(1.0 - sum_child) > LEARNING_SOLUTION_SETUP_ROUNDOFF_DELTA:
            # Clean it up - doesn't change the results, but enables the solution in almost all cases
            if sum_child > ROUNDOFF_DELTA:
                for child_key in child_marginals.keys():
                    child_marginals[child_key] /= sum_child
            else:
                print("Child marginals do not sum to 1.0 for " + self.name)
        if abs(1.0 - sum_parent) > LEARNING_SOLUTION_SETUP_ROUNDOFF_DELTA:
            # Clean it up - doesn't change the results, but enables the solution in almost all cases
            if sum_parent > ROUNDOFF_DELTA:
                for parent_key in parent_marginals.keys():
                    parent_marginals[parent_key] /= sum_parent
            else:
                print("Parent marginals do not sum to 1.0 for " + self.name)

        # Before attempting to LS solve, the problem can be reduced through 4 steps, 2 of which have the potential to
        #  complete the solution.
        # Steps:
        # 1) If there is a 0 on the input, match the previous potential column since there is no effect.
        # 2) If there is a single 1 in the child marginals, that row has ones.  All other rows are zeros.
        #    The exception is what came from (1).  SOLUTION.
        # 3) If 1 cell remains in a row, it gets the output value (was a 1 for the input).  SOLUTION.
        # 4) If there is a zero in the output, all remaining cells in the row are zeros.
        # 5) Solve via least squares
        # 6) If the LS solution is not sufficient, shift weights to make the solution feasible while adhering to the
        #    constraints.
        solved = False
        temp_parent_to_child = {}
        # Initialize the output
        for parent_key in parent_marginals.keys():
            temp_parent_to_child[parent_key] = {}
            for child_key in child_marginals.keys():
                temp_parent_to_child[parent_key][child_key] = False  # Not solved yet
        # (1) from above
        for parent_key, parent_mass in parent_marginals.items():
            if parent_mass < LEARNING_SOLUTION_SETUP_ROUNDOFF_DELTA:
                # Effectively 0 - match the previous column.  Must match the full column, so ignore (1).
                for child_key in child_marginals.keys():
                    temp_parent_to_child[parent_key][child_key] = transition_for_deltas[parent_key][child_key]
        # This cannot result in a full solution since not all parent masses can be 0
        # (2) from above
        if solved is False:
            if has_single_output_child is not None:
                for parent_key in parent_marginals.keys():
                    for child_key in child_marginals.keys():
                        # Only change values not already set from (1)
                        if temp_parent_to_child[parent_key][child_key] is False:
                            if child_key == has_single_output_child:
                                temp_parent_to_child[parent_key][child_key] = 1.0
                            else:
                                temp_parent_to_child[parent_key][child_key] = 0.0
                solved = True
        # (3) from above
        if (solved is False) and (has_single_input_parent is not None):
            num_unsolved = 0
            for parent_key in parent_marginals.keys():
                for child_key in child_marginals.keys():
                    if parent_key == has_single_input_parent:
                        temp_parent_to_child[parent_key][child_key] = child_marginals[child_key]
                    # Do nothing on the else: already set from (2)
                    if temp_parent_to_child[parent_key][child_key] is False:
                        num_unsolved += 1
            if num_unsolved == 0:
                solved = True
            else:
                print("rebalance_potentials: Error in step (3) - this should have completed the solution.")
        # (4) from above
        if solved is False:
            num_unsolved = 0
            for parent_key in parent_marginals.keys():
                for child_key, child_mass in child_marginals.items():
                    if temp_parent_to_child[parent_key][child_key] is False:
                        # Really has to be ~0 since this one messes things up otherwise
                        if child_mass < LEARNING_SOLUTION_SETUP_ROUNDOFF_DELTA:
                            # All remaining cells in the row should be zero
                            temp_parent_to_child[parent_key][child_key] = 0.0
                        else:
                            # Checking for unsolved
                            num_unsolved += 1
            if num_unsolved == 0:
                solved = True  # Odd...
                print("rebalancing_potentials: Did not expect step (4) to complete the solution.")
        # (5) from above
        ls_temp_parent_to_child = {}
        if solved is False:
            # Least-squares solution for the remaining values
            # The math is as follows:
            # Assume c' is the new child marginals and p' are the new parent marginals
            # Assume T is the previous transition and T' is the desired new transition
            # Assume there are n designed variables (T = m x p matrix where m x p = n)
            # The following constraints are enforced:
            # 1) c' = T'*p'
            # 2) for each col_i(T'), i=1,p sum(col_i(T')) = 1
            # 3) for each T'_i,j, i=1,p, j = 1,m, 0 <= T'_i,j <= 1.
            #    Note: <= 1 is handled through the previous constraint
            # DROP (3) for now, as this is deemed not necessary
            # The optimization goal is min((T'-T) + sum((T'-T)_i,j * learning_weight /
            #                              (max((c'_i-c_i)*(p'_j-p_j), 0.0001)))
            # Re-define the problem as the following: Ax = b
            # x = for all col_i(T'-T), i=1,p concatenate(col_i(T'-T))
            #
            # The problem may be reduced although the math is the same due to the steps completed above.
            # Changes:
            # 1) There are now k design variables where k is the number of unsolved potentials after the process above.
            # 2) c' = T'*p', but any fully solved rows are removed, and fully solved columns are subtracted from c'
            # 3) The remaining values in each column sum to 1.0 - solved values in that column
            # 4) All design variables must still be in the range [0.0, 1.0].
            unsolved_child_marginal_keys = []
            unsolved_parent_marginal_keys = []
            for parent_key in temp_parent_to_child.keys():
                for child_key in temp_parent_to_child[parent_key].keys():
                    if temp_parent_to_child[parent_key][child_key] is False:
                        # Needs to be solved.
                        if child_key not in unsolved_child_marginal_keys:
                            unsolved_child_marginal_keys.append(child_key)
                        if parent_key not in unsolved_parent_marginal_keys:
                            unsolved_parent_marginal_keys.append(parent_key)
            # Hopefully is a subset to make this faster.
            # Create the matrices
            num_child_marginals = len(unsolved_child_marginal_keys)
            num_parent_marginals = len(unsolved_parent_marginal_keys)
            a_matrix = np.zeros((num_child_marginals + num_parent_marginals,
                                 num_child_marginals * num_parent_marginals))
            b_matrix = np.zeros(num_child_marginals + num_parent_marginals)

            # If weighting is to be used, add that now.
            if (self.__parent.zero_snapshot is not None) and (self.__child.zero_snapshot is not None):
                # Valid information - use this for weighting
                learning_weight = self.learning_weight
                # Get the snapshots used for additional weighing
                parent_snapshot = deepcopy(self.__parent.zero_snapshot)
                # TODO: This may pose an issue since it's not using the passed in child marginals -
                #  may be rather wrong if this is creating a combination from multiple parents.
                child_snapshot = deepcopy(self.__child.zero_snapshot)
            else:
                # Set this to zero such that it won't affect anything
                learning_weight = 0.0
                parent_snapshot = deepcopy(parent_marginals)
                child_snapshot = deepcopy(child_marginals)
            # Define the scaling factors
            scaling_factors = {}
            for row_counter in range(0, num_parent_marginals):
                scaling_factors[unsolved_parent_marginal_keys[row_counter]] = {}
                for col_counter in range(0, num_child_marginals):
                    parent_marginal_delta_sq = pow(parent_snapshot[unsolved_parent_marginal_keys[row_counter]] -
                                                   parent_marginals[unsolved_parent_marginal_keys[row_counter]], 2)
                    child_marginal_delta_sq = pow(child_snapshot[unsolved_child_marginal_keys[col_counter]] -
                                                  child_marginals[unsolved_child_marginal_keys[col_counter]], 2)
                    additional_factor = sqrt(learning_weight / max(parent_marginal_delta_sq *
                                                                   child_marginal_delta_sq,
                                                                   MAX_ADDITIONAL_WEIGHT_DIVISOR))
                    scaling_factors[unsolved_parent_marginal_keys[row_counter]][
                        unsolved_child_marginal_keys[col_counter]] = 1.0 + additional_factor

            # Define the sum to 1 constraints.  Remember that it is the remaining amount, not the total now.
            for row_counter in range(0, num_parent_marginals):
                for col_counter in range(0, num_child_marginals):
                    a_matrix[row_counter, col_counter + (row_counter * num_child_marginals)] =\
                        1.0 / scaling_factors[unsolved_parent_marginal_keys[row_counter]][
                        unsolved_child_marginal_keys[col_counter]]
                # Sum the pre-solved values for the column.
                sum_of_solved = 0.0
                sum_of_previous_solved = 0.0
                for child_key in temp_parent_to_child[unsolved_parent_marginal_keys[row_counter]].keys():
                    if temp_parent_to_child[unsolved_parent_marginal_keys[row_counter]][child_key] is not False:
                        sum_of_solved += temp_parent_to_child[unsolved_parent_marginal_keys[row_counter]][child_key]
                        sum_of_previous_solved +=\
                            transition_for_deltas[unsolved_parent_marginal_keys[row_counter]][child_key]
                # Remaining amount is now 1.0 - sum_of_solved - (1.0 - sum_of_previous_solved)
                b_matrix[row_counter] = 1.0 - sum_of_solved - (1.0 - sum_of_previous_solved)

            # Define the c' = T'*p' constraints
            for add_row_counter in range(0, num_child_marginals):
                # row_counter needs to continue from previous
                row_counter = num_parent_marginals + add_row_counter
                b_sum = 0.0
                for col_counter in range(0, num_parent_marginals):
                    a_matrix[row_counter, col_counter * num_child_marginals + row_counter - num_parent_marginals] = \
                        parent_marginals[unsolved_parent_marginal_keys[col_counter]] /\
                        scaling_factors[unsolved_parent_marginal_keys[col_counter]][
                        unsolved_child_marginal_keys[add_row_counter]]
                    b_sum +=\
                        transition_for_deltas[unsolved_parent_marginal_keys[
                            col_counter]][unsolved_child_marginal_keys[add_row_counter]] *\
                        parent_marginals[unsolved_parent_marginal_keys[col_counter]]

                # Remove the already-solved portion.
                child_marginal_key = unsolved_child_marginal_keys[add_row_counter]
                sum_of_solved = 0.0
                sum_of_previous_solved = 0.0
                for parent_key in temp_parent_to_child.keys():
                    if temp_parent_to_child[parent_key][child_marginal_key] is not False:
                        sum_of_solved += temp_parent_to_child[parent_key][child_marginal_key] *\
                                         parent_marginals[parent_key]
                        sum_of_previous_solved += transition_for_deltas[parent_key][child_marginal_key] *\
                            parent_marginals[parent_key]
                # Remove previously solved values
                b_matrix[row_counter] = child_marginals[child_marginal_key] - sum_of_solved -\
                    (b_sum - sum_of_previous_solved)

            if debug is True:
                print(" Transition: " + self.name + " rebalancing potentials.")
            a_matrix_rank = np.linalg.matrix_rank(a_matrix.transpose())
            if (a_matrix_rank < (num_child_marginals + num_parent_marginals)) and (debug is True):
                # Rank deficient
                print("Transition " + self.name + " has a row rank-deficient A matrix ( " + str(a_matrix_rank) + "," +
                      str(num_child_marginals + num_parent_marginals) + "):")
                for row_counter in range(0, num_child_marginals + num_parent_marginals):
                    print_string = "["
                    for col_counter in range(0, num_child_marginals * num_parent_marginals):
                        print_string += '%10f' % a_matrix[row_counter, col_counter]
                        if col_counter != (num_child_marginals * num_parent_marginals) - 1:
                            print_string += ","
                    print_string += "]"
                    print(print_string)

            # A and b matrices are fully defined.  Now to run the least squares.
            new_design_vars = np.linalg.lstsq(a_matrix, b_matrix, rcond=None)[0]
            # Check whether the new values are within the inequality constraints (>= 0)

            valid_values = True
            for parent_counter in range(0, num_parent_marginals):
                parent_key = unsolved_parent_marginal_keys[parent_counter]
                ls_temp_parent_to_child[parent_key] = {}
                for child_counter in range(0, num_child_marginals):
                    child_key = unsolved_child_marginal_keys[child_counter]
                    ls_temp_parent_to_child[parent_key][child_key] = transition_for_deltas[parent_key][child_key] + \
                        float(new_design_vars[(parent_counter * num_child_marginals) + child_counter]) /\
                        scaling_factors[parent_key][child_key]
                    valid_values = valid_values and \
                        (ls_temp_parent_to_child[parent_key][child_key] > -LEARNING_SOLUTION_SETUP_ROUNDOFF_DELTA) and \
                        (ls_temp_parent_to_child[parent_key][child_key] <
                         (1.0 + LEARNING_SOLUTION_SETUP_ROUNDOFF_DELTA))
                    child_counter += 1
                parent_counter += 1
            # Set the new design variables to the conditionals if they are valid
            # noinspection PySimplifyBooleanCheck,PyPep8
            if valid_values == True:
                # Copy back to temp_parent_to_child
                for parent_key in ls_temp_parent_to_child.keys():
                    for child_key in ls_temp_parent_to_child[parent_key].keys():
                        temp_parent_to_child[parent_key][child_key] = ls_temp_parent_to_child[parent_key][child_key]
                # No adjustments required
                solved = True
            elif debug is True:
                # Print out the Least Squares solution before adjusting values to make sure it's working from a valid
                #  start
                # Print the previous transition matrix so we can replicate
                error_string = "Previous transition matrix\n"
                child_keys = list(child_marginals.keys())
                parent_keys = list(parent_marginals.keys())
                for child_key_index in range(0, len(child_keys)):
                    child_key = child_keys[child_key_index]
                    test_child = 0.0
                    print_line = "["
                    for parent_key in parent_keys:
                        test_child += parent_marginals[parent_key] * transition_for_deltas[parent_key][child_key]
                        if (transition_for_deltas[parent_key][child_key] < -LEARNING_SOLUTION_SETUP_ROUNDOFF_DELTA) or \
                                (transition_for_deltas[parent_key][child_key] >
                                 (1.0 + LEARNING_SOLUTION_SETUP_ROUNDOFF_DELTA)):
                            print_line += '*%10f' % transition_for_deltas[parent_key][child_key] + "  "
                        else:
                            print_line += ' %10f' % transition_for_deltas[parent_key][child_key] + "  "
                    print_line += "]"
                    error_string += print_line + "\n"
                print(error_string)
                print("Least squares solution before adjusting values")
                self.print_debug_matrix(temp_parent_to_child, ls_temp_parent_to_child, None, parent_marginals,
                                        child_marginals)
        if solved is False:
            # Adjust the least squares solution to get a valid solution.  While no longer an optimal solution,
            #  it will be feasible and somewhat close to the optimal that was found.
            # The algorithm is similar to linear programming adjustments.
            # Note: there are cases in which the more simple method of never making a good cell bad or a bad cell worse
            #  is not sufficient (see test cases).  Therefore, the simple algorithm is modified a bit to account for
            #  these cases.
            # 1) Order all cells that are invalid below zero.  The order is as follows:
            #    a) By column
            #    b) Any column in which all bad cells (<0 and > 1) sum to zero are highest priority
            #    c) Within a column, from minimum value to maximum value (still all values below zero)
            # 2) Once the invalid below zero cells are ordered, fix each one.
            #    a) Fixing is defined as moving the amount of mass necessary to make that cell within tolerance.  Adjust
            #       the mass to within the column and other columns to balance.
            #    b) Attempt to not make any cells worse (good cells bad or bad cells worse).
            #    c) If (b) cannot be done completely, the final step is to spread the extra mass across all remaining
            #       cells to keep the correct balance.
            # 3) Once a column has been "fixed", that column is no longer allowed to be made bad.  It can still be
            #    adjusted, but not made bad (good cells turned to bad cells).

            # Step (1) - find the order of columns to fix
            columns_to_fix = []
            column_mapping_to_fix = {}
            for parent_key in ls_temp_parent_to_child.keys():
                sum_of_bad = 0.0
                for child_key, potential in ls_temp_parent_to_child[parent_key].items():
                    if (potential < 0.0) or (potential > 1.0):
                        sum_of_bad += potential
                column_mapping_to_fix[parent_key] = sum_of_bad
            for parent_key, sum_of_bad in column_mapping_to_fix.items():
                # See where it fits - also handles the first case since len(columns_to_fix) == 0.
                found = False
                for counter in range(0, len(columns_to_fix)):
                    if sum_of_bad > column_mapping_to_fix[columns_to_fix[counter]]:
                        columns_to_fix.insert(counter, parent_key)
                        found = True
                        break
                if found is False:
                    columns_to_fix.append(parent_key)
            fixed_columns = []  # Not required, but makes it easier to check quickly if a column is fixed
            for bad_parent_key in columns_to_fix:
                # Note that this may include columns that are already good.  In that case, they will be passed over
                #  quickly and listed as "fixed".
                # Order the cells to fix.  Remember that, within a column, cells are never moved from good to bad,
                #  so this order will hold.
                cells_to_fix = []
                for child_key, potential in ls_temp_parent_to_child[bad_parent_key].items():
                    if potential < -LEARNING_SOLUTION_SETUP_ROUNDOFF_DELTA:
                        # Needs to be fixed
                        found = False
                        for counter in range(0, len(cells_to_fix)):
                            if potential < ls_temp_parent_to_child[bad_parent_key][cells_to_fix[counter]]:
                                cells_to_fix.insert(counter, child_key)
                                found = True
                                break
                        if found is False:
                            cells_to_fix.append(child_key)
                # Now have all the child keys in this column to fix, in order from maximum bad to minimum bad
                for bad_child_key in cells_to_fix:
                    change_mass_keys = {bad_parent_key: {}}
                    # The mass is less than zero.
                    # There will never be more than one value in the column that is > 1.0 unless least squares royally
                    #  screwed up, which it doesn't appear to do.
                    # In order to stay as close to the least squares solution as possible, apply as much as necessary to
                    #  values that are greater than 1.  Then, apply the rest uniformly across all
                    #  values in the column with the exception that any application that will push the value outside of
                    #  bounds is distributed over the remaining values in the column.
                    adjustment_mass = -ls_temp_parent_to_child[bad_parent_key][bad_child_key]
                    change_mass_keys[bad_parent_key][bad_child_key] = adjustment_mass
                    for child_key, mass in ls_temp_parent_to_child[bad_parent_key].items():
                        if (child_key != bad_child_key) and (mass > 1.0):
                            mass_to_apply = min(adjustment_mass,
                                                ls_temp_parent_to_child[bad_parent_key][child_key] - 1.0)
                            change_mass_keys[bad_parent_key][child_key] = -mass_to_apply
                            adjustment_mass -= mass_to_apply
                            if debug is True:
                                print("Printing after bad column update with " + str(child_key) + " mass > 1.0 for " +
                                      str(bad_parent_key) +
                                      " and " + str(bad_child_key) + " with remaining mass " + str(adjustment_mass))
                                self.print_debug_matrix(temp_parent_to_child, ls_temp_parent_to_child, change_mass_keys,
                                                        parent_marginals, child_marginals)
                            break  # There will only be one greater than 1.0 in the column.
                    if adjustment_mass > 0.0:
                        # Distribute across all child keys
                        applied_children = [bad_child_key]
                        current_adjustment = adjustment_mass / \
                            float(len(ls_temp_parent_to_child[bad_parent_key].keys()) -
                                  len(applied_children))
                        check_mass_limit = True
                        while len(applied_children) < len(ls_temp_parent_to_child[bad_parent_key].keys()):
                            applied_this_round = False
                            for child_key, mass in ls_temp_parent_to_child[bad_parent_key].items():
                                if check_mass_limit is True:
                                    # Still finding children to which the full adjustment cannot be applied.
                                    if (child_key not in applied_children) and (mass < current_adjustment):
                                        # Apply to this one if possible
                                        if child_key not in change_mass_keys[bad_parent_key]:
                                            change_mass_keys[bad_parent_key][child_key] = 0.0
                                        applied_adjustment = 0.0
                                        if mass > 0.0:
                                            applied_adjustment = mass
                                        change_mass_keys[bad_parent_key][child_key] += -applied_adjustment
                                        # Has been applied
                                        applied_children.append(child_key)
                                        # Spread the extra across the remaining children
                                        current_adjustment += (current_adjustment - applied_adjustment) / \
                                            float(len(ls_temp_parent_to_child[bad_parent_key].keys()) -
                                                  len(applied_children))
                                        # Applied
                                        if debug is True:
                                            print("Printing after bad column update with " + str(child_key) +
                                                  " limitation for " + str(bad_parent_key) + " and " +
                                                  str(bad_child_key))
                                            self.print_debug_matrix(temp_parent_to_child, ls_temp_parent_to_child,
                                                                    change_mass_keys, parent_marginals, child_marginals)
                                        applied_this_round = True
                                elif child_key not in applied_children:
                                    # Apply to this one - this is the last round, so apply to all
                                    if child_key not in change_mass_keys[bad_parent_key]:
                                        change_mass_keys[bad_parent_key][child_key] = 0.0
                                    change_mass_keys[bad_parent_key][child_key] += -current_adjustment
                                    # Has been applied
                                    applied_children.append(child_key)
                                    if debug is True:
                                        print("Printing after bad column " + str(child_key) + " full update for " +
                                              str(bad_parent_key) + " and " + str(bad_child_key))
                                        self.print_debug_matrix(temp_parent_to_child, ls_temp_parent_to_child,
                                                                change_mass_keys, parent_marginals, child_marginals)
                            if applied_this_round is False:
                                # Didn't apply any last round, so can finish the spread
                                check_mass_limit = False
                        # The column has now been appropriately redistributed.  Now that has to be redistributed to the
                        #  other columns.
                    # Distribute to all columns without making them worse.  Afterwards, distribute all remaining to
                    #  the columns that have not been fixed.  That way,
                    #  everything remains consistent, and the columns that still have to be fixed will be fixed.
                    #  Moreover, any "fixed" columns can still take mass, so there is room to fix unfixed columns.
                    adjustment_mass = change_mass_keys[bad_parent_key][bad_child_key]
                    for parent_key in columns_to_fix:
                        if parent_key != bad_parent_key:
                            mass = ls_temp_parent_to_child[parent_key][bad_child_key]
                            if mass > 1.0:
                                change_mass_keys[parent_key] = {}  # Haven't defined this one yet
                                factor_to_new_column = parent_marginals[bad_parent_key] / \
                                    parent_marginals[parent_key]
                                column_compensated_mass = adjustment_mass * factor_to_new_column
                                mass_to_apply = min(column_compensated_mass, mass - 1.0)
                                change_mass_keys[parent_key][bad_child_key] = -mass_to_apply
                                fraction_applied = mass_to_apply / column_compensated_mass

                                # Apply down the rest of the column based on the distribution from the original column
                                #  and the fraction applied this round
                                for child_key in ls_temp_parent_to_child[parent_key].keys():
                                    if (child_key != bad_child_key) and \
                                            (child_key in change_mass_keys[bad_parent_key]):
                                        change_mass_keys[parent_key][child_key] =\
                                            fraction_applied * -change_mass_keys[bad_parent_key][child_key] *\
                                            factor_to_new_column

                                # Convert back to the original column so have remaining to remove
                                adjustment_mass *= (1.0 - fraction_applied)
                                if debug is True:
                                    print("Printing after new column update with " + str(bad_child_key) +
                                          " mass > 1.0 for " + str(bad_parent_key) + " and " + str(bad_child_key) +
                                          " to " + str(parent_key))
                                    self.print_debug_matrix(temp_parent_to_child, ls_temp_parent_to_child,
                                                            change_mass_keys, parent_marginals, child_marginals)
                                # Do not break.  More than one column could have a > 0 value
                    if adjustment_mass > 0.0:
                        # Distribute across all columns as equally as possible.
                        applied_parents = [bad_parent_key]
                        check_mass_limit = True
                        while len(applied_parents) < len(columns_to_fix):
                            applied_this_round = False
                            sum_of_remaining_marginals = 0.0
                            # Apply to all remaining parent columns based on the parent marginal distribution.
                            for marginal_key, marginal_value in parent_marginals.items():
                                if marginal_key not in applied_parents:
                                    sum_of_remaining_marginals += marginal_value
                            for parent_key in columns_to_fix:
                                if parent_key not in applied_parents:
                                    # The factor to move between columns
                                    factor_to_new_column = parent_marginals[parent_key] / sum_of_remaining_marginals
                                    # The mass that applies to the new column
                                    mass_to_new_column = adjustment_mass * factor_to_new_column
                                    column_compensated_mass = mass_to_new_column * parent_marginals[bad_parent_key] / \
                                        parent_marginals[parent_key]
                                    if check_mass_limit is True:
                                        # Still finding children to which the full adjustment cannot be applied.
                                        is_limited = False
                                        limitation_fraction = 1.0
                                        child_key_limitation = None
                                        for child_key in ls_temp_parent_to_child[parent_key].keys():
                                            if child_key in change_mass_keys[bad_parent_key]:
                                                # Will apply to this column, so check if there is a limitation
                                                # Remember that it applies opposite to the original modification
                                                previous_adjustments = 0.0
                                                if (parent_key in change_mass_keys) and \
                                                        (child_key in change_mass_keys[parent_key]):
                                                    previous_adjustments = change_mass_keys[parent_key][child_key]
                                                current_value = ls_temp_parent_to_child[parent_key][child_key] + \
                                                    previous_adjustments
                                                # The adjustment applied to this cell in the matrix
                                                adjustment_to_cell = -column_compensated_mass *\
                                                    change_mass_keys[bad_parent_key][child_key] / \
                                                    change_mass_keys[bad_parent_key][bad_child_key]
                                                if ((adjustment_to_cell < 0.0) and (current_value <= 0.0)) or \
                                                        ((adjustment_to_cell > 0.0) and (current_value >= 1.0)):
                                                    # Already cannot apply to this column
                                                    is_limited = True
                                                    limitation_fraction = 0.0
                                                    child_key_limitation = child_key
                                                    break  # Limited as far as possible
                                                else:
                                                    adjusted_value = current_value + adjustment_to_cell
                                                    if (adjusted_value < 0.0) and (adjustment_to_cell < 0.0):
                                                        is_limited = True
                                                        child_key_limitation = child_key
                                                        # How limited?
                                                        limitation_fraction = min(limitation_fraction, current_value /
                                                                                  abs(adjustment_to_cell))
                                                    elif (adjusted_value > 1.0) and (adjustment_to_cell > 0.0):
                                                        is_limited = True
                                                        child_key_limitation = child_key
                                                        # How limited?
                                                        limitation_fraction = \
                                                            min(limitation_fraction, (1.0 - current_value) /
                                                                abs(adjustment_to_cell))
                                        if is_limited:
                                            # Apply to this one
                                            if parent_key not in change_mass_keys:
                                                change_mass_keys[parent_key] = {}
                                            # Apply down the entire column based on the distribution from the
                                            #  original column
                                            for child_key in ls_temp_parent_to_child[parent_key].keys():
                                                if child_key in change_mass_keys[bad_parent_key]:
                                                    if child_key not in change_mass_keys[parent_key]:
                                                        change_mass_keys[parent_key][child_key] = 0.0
                                                    change_mass_keys[parent_key][child_key] +=\
                                                        -column_compensated_mass * limitation_fraction * \
                                                        change_mass_keys[bad_parent_key][child_key] / \
                                                        change_mass_keys[bad_parent_key][bad_child_key]
                                            # Has been applied
                                            applied_parents.append(parent_key)
                                            # Reduce the remaining mass
                                            adjustment_mass -= mass_to_new_column * limitation_fraction
                                            # Applied
                                            applied_this_round = True
                                            if debug is True:
                                                print("Printing after new column update with " +
                                                      str(child_key_limitation) + " " + str(limitation_fraction) +
                                                      " limitation with remaining adjustment mass " +
                                                      str(adjustment_mass) + " for " + str(bad_parent_key) +
                                                      " and " + str(bad_child_key) + " to " + str(parent_key))
                                                self.print_debug_matrix(temp_parent_to_child, ls_temp_parent_to_child,
                                                                        change_mass_keys, parent_marginals,
                                                                        child_marginals)
                                    else:
                                        if parent_key not in change_mass_keys:
                                            change_mass_keys[parent_key] = {}
                                        # Apply down the entire column based on the distribution from the original
                                        #  column
                                        for child_key in ls_temp_parent_to_child[parent_key].keys():
                                            if child_key in change_mass_keys[bad_parent_key]:
                                                if child_key not in change_mass_keys[parent_key]:
                                                    change_mass_keys[parent_key][child_key] = 0.0
                                                change_mass_keys[parent_key][child_key] += -column_compensated_mass * \
                                                    change_mass_keys[bad_parent_key][child_key] / \
                                                    change_mass_keys[bad_parent_key][bad_child_key]
                                        # Has been applied
                                        applied_parents.append(parent_key)
                                        if debug is True:
                                            print("Printing after new column update for " + str(bad_parent_key) +
                                                  " and " + str(bad_child_key) + " to " + str(parent_key))
                                            self.print_debug_matrix(temp_parent_to_child, ls_temp_parent_to_child,
                                                                    change_mass_keys, parent_marginals, child_marginals)
                            if applied_this_round is False:
                                # Didn't apply any last round, so can finish the spread
                                check_mass_limit = False
                        if (check_mass_limit is True) and (adjustment_mass > 0.0):
                            if debug is True:
                                print("Spreading remaining mass even though it makes some values worse.")
                            # Here is where things get modified.
                            # Went through all columns to which values could be applied, and there is still mass
                            #  remaining.  In order to remain consistent, this mass must be applied.  Apply it to all
                            #  non-fixed columns using the previous equal scheme.
                            sum_of_remaining_marginals = 0.0
                            # Apply to all remaining parent columns based on the parent marginal distribution.
                            for marginal_key, marginal_value in parent_marginals.items():
                                if (marginal_key not in fixed_columns) and (marginal_key != bad_parent_key):
                                    sum_of_remaining_marginals += marginal_value
                            for parent_key in columns_to_fix:
                                if (parent_key not in fixed_columns) and (parent_key != bad_parent_key):
                                    # The factor to move between columns
                                    factor_to_new_column = parent_marginals[parent_key] / sum_of_remaining_marginals
                                    # The mass that applies to the new column
                                    mass_to_new_column = adjustment_mass * factor_to_new_column
                                    column_compensated_mass = mass_to_new_column * parent_marginals[bad_parent_key] / \
                                        parent_marginals[parent_key]
                                    if parent_key not in change_mass_keys:
                                        change_mass_keys[parent_key] = {}
                                    # Apply down the entire column based on the distribution from the original
                                    #  column
                                    for child_key in ls_temp_parent_to_child[parent_key].keys():
                                        if child_key in change_mass_keys[bad_parent_key]:
                                            if child_key not in change_mass_keys[parent_key]:
                                                change_mass_keys[parent_key][child_key] = 0.0
                                            change_mass_keys[parent_key][child_key] += -column_compensated_mass * \
                                                change_mass_keys[bad_parent_key][child_key] / \
                                                change_mass_keys[bad_parent_key][bad_child_key]
                                    # Has been applied
                                    applied_parents.append(parent_key)
                                    if debug is True:
                                        print("Printing after spread column update for " + str(bad_parent_key) +
                                              " and " + str(bad_child_key) + " to " + str(parent_key))
                                        self.print_debug_matrix(temp_parent_to_child, ls_temp_parent_to_child,
                                                                change_mass_keys, parent_marginals, child_marginals)

                    # Apply the changes
                    for parent_key in change_mass_keys.keys():
                        for child_key, adjustment_mass in change_mass_keys[parent_key].items():
                            ls_temp_parent_to_child[parent_key][child_key] += adjustment_mass
                # The column has been fixed
                fixed_columns.append(bad_parent_key)

            # All potentials have been modified
            # Copy back to temp_parent_to_child regardless of whether the adjustments worked.  If they didn't, then
            #  there is a useful error output.
            for parent_key in ls_temp_parent_to_child.keys():
                for child_key in ls_temp_parent_to_child[parent_key].keys():
                    temp_parent_to_child[parent_key][child_key] = ls_temp_parent_to_child[parent_key][child_key]
            # Check whether the solution worked
            calculated_child_marginals = {}
            solved = True
            for parent_key, parent_marginal in parent_marginals.items():
                for child_key, potential in temp_parent_to_child[parent_key].items():
                    if (potential < -ROUNDOFF_DELTA) or (potential > (1.0 + ROUNDOFF_DELTA)):
                        solved = False
                        break
                    if child_key not in calculated_child_marginals:
                        calculated_child_marginals[child_key] = 0.0
                    calculated_child_marginals[child_key] += parent_marginal *\
                        temp_parent_to_child[parent_key][child_key]
                if solved is False:
                    break
            if solved is True:
                # Check the marginals
                for child_key, child_marginal in child_marginals.items():
                    if abs(child_marginal - calculated_child_marginals[child_key]) > ROUNDOFF_DELTA:
                        solved = False
                        break
        if solved is True:
            # Hurrah!
            for parent_key in parent_marginals.keys():
                for child_key in child_marginals.keys():
                    # Also take care of any roundoff errors that occurred to stay in [0.0, 1.0]
                    self.conditionals_parent_to_child[parent_key][child_key] = \
                        max(min(temp_parent_to_child[parent_key][child_key], 1.0), 0.0)
        else:
            # Create an output for analysis - why didn't the solution method work
            error_string = "Transition: " + self.name + "\n"
            child_keys = list(child_marginals.keys())
            parent_keys = list(parent_marginals.keys())
            print_sum_of_columns = '%80s' % " " + "  " + '%10s' % " " + '%10s' % " " + "     "
            sum_of_columns = {}
            for child_key_index in range(0, len(child_keys)):
                child_key = child_keys[child_key_index]
                test_child = 0.0
                print_line_answer = '%80s' % str(child_key) + ": " + \
                                    '%10f' % child_marginals[child_key] + " = "
                print_line = ""
                for parent_key in parent_keys:
                    if parent_key not in sum_of_columns:
                        sum_of_columns[parent_key] = 0.0
                    sum_of_columns[parent_key] += temp_parent_to_child[parent_key][child_key]
                    test_child += parent_marginals[parent_key] * \
                        temp_parent_to_child[parent_key][child_key]
                    if (temp_parent_to_child[parent_key][child_key] < -ROUNDOFF_DELTA) or \
                            (temp_parent_to_child[parent_key][child_key] > (1.0 + ROUNDOFF_DELTA)):
                        print_line += '*%10f' % temp_parent_to_child[parent_key][child_key] + "  "
                    else:
                        print_line += ' %10f' % temp_parent_to_child[parent_key][child_key] + "  "
                print_line_answer += '%10f' % test_child + " = ["
                if child_key_index < len(parent_keys):
                    print_line += "][" + '%10f' % parent_marginals[parent_keys[child_key_index]] + "] " + \
                                  str(parent_keys[child_key_index])
                error_string += print_line_answer + print_line + "\n"
            if len(parent_keys) > len(child_keys):
                for parent_key_index in range(len(child_keys), len(parent_keys)):
                    print_line = '%120s' % " "
                    for _ in parent_keys:
                        print_line += '%11s' % " "
                    print_line += " [" + '%10f' % parent_marginals[parent_keys[parent_key_index]] + "] " + \
                                  str(parent_keys[parent_key_index])
                    error_string += print_line + "\n"
            # Verify the sum of columns
            for parent_key in parent_keys:
                print_sum_of_columns += '%13f' % sum_of_columns[parent_key]
            error_string += print_sum_of_columns + "\n"

            raise ValueError(error_string)

        # Verify the results
        if debug is True:
            print("Least squares solution")
            # Results verification
            child_keys = list(child_marginals.keys())
            parent_keys = list(parent_marginals.keys())
            for child_key_index in range(0, len(child_keys)):
                child_key = child_keys[child_key_index]
                test_child = 0.0
                print_line_answer = '%80s' % str(child_key) + ": " +\
                    '%10f' % child_marginals[child_key] + " = "
                print_line = ""
                for parent_key in parent_keys:
                    test_child += parent_marginals[parent_key] * \
                                  self.conditionals_parent_to_child[parent_key][child_key]
                    print_line += '%10f' % self.conditionals_parent_to_child[parent_key][child_key] + "  "
                print_line_answer += '%10f' % test_child + " = ["
                if child_key_index < len(parent_keys):
                    print_line += "][" + '%10f' % parent_marginals[parent_keys[child_key_index]] + "] " +\
                        str(parent_keys[child_key_index])
                print(print_line_answer + print_line)
            if len(parent_keys) > len(child_keys):
                for parent_key_index in range(len(child_keys), len(parent_keys)):
                    print_line = '%120s' % " "
                    for _ in parent_keys:
                        print_line += '%11s' % " "
                    print_line += " [" + '%10f' % parent_marginals[parent_keys[parent_key_index]] + "] " +\
                        str(parent_keys[parent_key_index])
                    print(print_line)
        # Mirror the transition
        self.create_child_to_parent_conditionals()

    def dump_to_csv(self, file_name):
        """
        Dumps transition conditionals to CSV lines to enable tracing and debugging.  Appends if possible, otherwise
         creates the new file
        :param file_name: the file name to which to dump
        """
        parent_keys = list(self.conditionals_parent_to_child.keys())
        child_keys = list(self.conditionals_child_to_parent.keys())
        with open(file_name, 'a+') as f:
            f.write(self.name + "\n")
            write_line = ",,"
            for parent_key in parent_keys:
                write_line += str(parent_key).replace(",", "/") + ","
            f.write(write_line + "\n")
            for child_key in child_keys:
                write_line = "," + str(child_key).replace(",", "/") + ","
                for parent_key in parent_keys:
                    write_line += str(self.conditionals_parent_to_child[parent_key][child_key]) + ","
                f.write(write_line + "\n")

    def check_for_nodes(self, source_name, search_node_names):
        """
        Checks for specific nodes
        :param source_name: the sender name (to know which direction to search)
        :param search_node_names: the node names to find.
        :return: Whether any of the nodes were found.
        :return: Whether a cycle was found
        """
        if source_name == self.__child.name:
            return self.__parent.check_for_nodes(self.name, search_node_names)
        elif source_name == self.__parent.name:
            return self.__child.check_for_nodes(self.name, search_node_names)
        else:
            raise ValueError("Transition " + self.name + " check_for_nodes: source_name " + source_name +
                             " is unknown - cannot compute")

    def get_propagated_marginals(self, child=True):
        """
        Gets the marginals based on the current other end marginals multiplied by the potentials.  This function is
         primarily use for simultaneously updating multiple parent transitions.  Otherwise, this result should already
         be consistent with the desired end marginals.
        :param child: bool If True: gets the child marginals based on the parent marginals multiplied by the potentials.
        :return: dict of the propagated marginals
        """
        if child is True:
            parent_marginals = import_and_calculate_probabilities(self.__parent.combination_method,
                                                                  self.__parent.internal_probability_data)
            child_marginals = {}
            for parent in self.conditionals_parent_to_child:
                if parent in parent_marginals:
                    # Use this data; otherwise, was a zero anyway
                    for child in self.conditionals_parent_to_child[parent]:
                        if child not in child_marginals:
                            child_marginals[child] = 0
                        child_marginals[child] += self.conditionals_parent_to_child[parent][child] * \
                            parent_marginals[parent]
            return child_marginals
        else:
            child_marginals = import_and_calculate_probabilities(self.__child.combination_method,
                                                                 self.__child.internal_probability_data)
            parent_marginals = {}
            total_sum = 0
            for child in self.conditionals_child_to_parent:
                if child in child_marginals:
                    # Use this data; otherwise, was a zero anyway
                    for parent in self.conditionals_child_to_parent[child]:
                        if parent not in parent_marginals:
                            parent_marginals[parent] = 0
                        marginal_add = self.conditionals_child_to_parent[child][parent] * child_marginals[child]
                        total_sum += marginal_add
                        parent_marginals[parent] += marginal_add
            # Divide through to ensure input to parent adds to 1.0
            if total_sum != 0.0:
                for parent_key in parent_marginals.keys():
                    parent_marginals[parent_key] /= total_sum
            return parent_marginals


class DSNetwork:
    def __init__(self):
        # If a node has multiple parent transitions, determines whether to drive the results of all parent transitions
        #  to the same marginal before combination or allow them to be different.  Different means more flexibility
        #  and potentially easier optimization topology, but has many more design variables, so takes longer.
        #  Driving to the same should be faster to find a feasible solution, but may be harder to find an optimal one.
        self.same_multi_parent_marginals = True  # Default is true
        # Can solve using root finders instead of the multidimensional optimization.  Only works for certain methods.
        self.use_root_finder = MULTI_PARENT_SOLUTION_TYPE["CATEGORIES"]["ROOT_FINDER_OPT_FALLBACK"]  # Default is both
        # The nodes that make up the fault tree - a mapping of node names to nodes (for ease of tree reconstruction)
        self.nodes = {}
        # The transitions that connect between nodes.  These include the conditional values (potentials, might
        #  be probabilities)
        # They are named as parent.child
        self.transitions = {}
        # A mapping of inputs to nodes.  A set of the values of this map are all the input nodes in the tree
        self.__inputs = {}
        # Defines a way to get events out of the network easily - not stored in file
        self.__output_callbacks = {}
        # For non-inference weight updates
        self.noninference_weight_update = NONINFERENCE_WEIGHT_METHOD["CATEGORIES"]["AVERAGE"]
        # Not saved
        self.snapshot_counter = 0

    def __getstate__(self):
        # Define the attributes to pickle since they are reconstructed
        return list(self.nodes.values()), list(self.transitions.values()), list(self.__inputs.values()),\
            self.noninference_weight_update, self.same_multi_parent_marginals, self.use_root_finder

    def __setstate__(self, state):
        # Reconstruct the tree from the saved attributes
        # Don't forget, init isn't run beforehand
        in_nodes, in_transitions, in_inputs, self.noninference_weight_update, self.same_multi_parent_marginals, \
            self.use_root_finder = state
        self.snapshot_counter = 0  # Because python is stupid and doesn't run a constructor when this is called
        self.nodes = {}
        for node in in_nodes:
            self.nodes[node.name] = node
            for decision_key in node.decisions.keys():
                # Set the tree reference for callbacks
                node.decisions[decision_key].tree_ref = self
        self.transitions = {}
        for transition in in_transitions:
            self.transitions[transition.name] = transition
        self.__inputs = {}
        for input_def in in_inputs:
            # Convert from node name to node
            for node_name in input_def.nodes.keys():
                input_def.nodes[node_name] = self.nodes[node_name]
            # Save
            self.__inputs[input_def.name] = input_def

        # Reconstruct the transitions with full child and parent references
        for transition in self.transitions.values():
            transition.set_parent(self.nodes[transition.get_parent()])
            transition.set_child(self.nodes[transition.get_child()])

        # Reconstruct the nodes with full transitions
        for node in self.nodes.values():
            child_transition_names = node.child_transitions
            node.child_transitions = []
            for transition_name in child_transition_names:
                node.child_transitions.append(self.transitions[transition_name])
            parent_transition_names = node.parent_transitions
            node.parent_transitions = []
            for transition_name in parent_transition_names:
                node.parent_transitions.append(self.transitions[transition_name])

    # noinspection PySimplifyBooleanCheck,PyPep8
    def add_link_and_initialize(self, nodes, transitions, combination_method=None, weighting_method=None,
                                initialize_unknown=True):
        """
        Convenience function to add all and link
        :param nodes: An iterable of Node(s)
        :param transitions: An iterable of Transition(s)
        :param combination_method: COMBINATION_METHOD for nodes and transitions that are not already set
        :param weighting_method: WEIGHTING_METHOD
        :param initialize_unknown: bool - whether to initialize the nodes and transitions to unknowns
        :return: Whether it all worked
        """
        all_worked = self.add_nodes(nodes)
        all_worked |= self.add_transitions(transitions)
        # Some of these logical operators are turning into a numpy.bool_, so use == True to guarantee the check works.
        if all_worked == True:
            self.link_nodes()
            self.set_combination_and_weighting_methods(combination_method, weighting_method)  # Override is left false
            if initialize_unknown is True:
                self.initialize_unknown()
            return True
        # Failed somewhere
        return False

    def add_nodes(self, nodes):
        """
        Adds the list of nodes to the network
        :param nodes: An iterable of Node(s)
        :return: Whether all the nodes were added
        """
        all_added = True
        for node in nodes:
            if node.name not in self.nodes:
                self.nodes[node.name] = node
            else:
                all_added = False
        return all_added

    def add_transitions(self, transitions):
        """
        Adds the list of transitions to the network
        :param transitions: An iterable of Transition(s)
        :return: Whether all the transitions were added
        """
        all_added = True
        for transition in transitions:
            if transition.name not in self.transitions:
                self.transitions[transition.name] = transition
            else:
                all_added = False
        return all_added

    def link_nodes(self):
        """
        Once all nodes and transitions are defined, this adds the transitions to the nodes to automatically
         link it all up.  It also checks if links are already done to avoid doubling links.
        """
        for transition_key in self.transitions.keys():
            transition = self.transitions[transition_key]
            if not any(x for x in transition.get_parent().child_transitions if x.name == transition.name):
                # Add to the list
                transition.get_parent().child_transitions.append(transition)
            if not any(x for x in transition.get_child().parent_transitions if x.name == transition.name):
                # Add to the list
                transition.get_child().parent_transitions.append(transition)

    def set_combination_limit(self, combination_limit=None):
        """
        Sets the maximum number of combinations for DS combination types that support this.  This capability is
         available because older information can be combined together and be considered background information,
         perhaps with a different weighting than the new inputs.
        :param combination_limit: None for no limit, positive value greater than 1 for any other limit
        :return boolean whether the limit was set
        """
        if (combination_limit is None) or (combination_limit > 1):
            # Set it
            for node in self.nodes.values():
                node.set_combination_limit(combination_limit)
            return True
        else:
            return False

    def set_combination_and_weighting_methods(self, combination_method=None, weighting_method=None, override=False):
        """
        Sets the new combination method
        :param combination_method: The new combination method from COMBINATION_METHODS
        :param weighting_method: The new weighting method from WEIGHTING_METHOD
        :param override: bool whether to override combination methods that are already set
        """
        if combination_method in COMBINATION_METHODS:
            # Update
            for node in self.nodes.values():
                if (override is True) or (node.combination_method is None):
                    node.combination_method = combination_method
                    node.set_snapshot(name=None, clear=True)  # Have to be reset when changing combination method
            for transition in self.transitions.values():
                if (override is True) or (transition.combination_method is None):
                    transition.combination_method = combination_method
        elif combination_method is not None:
            raise ValueError("FaultTree: Combination Method " + combination_method + " is not valid")

        if weighting_method in WEIGHTING_METHOD["CATEGORIES"]:
            # Update
            for node in self.nodes.values():
                node.weighting_method = weighting_method
            for transition in self.transitions.values():
                transition.weighting_method = weighting_method
        elif weighting_method is not None:
            raise ValueError("FaultTree: Weighting Method " + weighting_method + " is not valid")

    def initialize_unknown(self, initialize_nodes=0.0, initialize_transitions=0.0):
        """
        Initializes all transitions and nodes to full unknown (empty set + map to empty set)
        :param initialize_nodes: value for non-unknown to initialize to if already set,
                otherwise None to not initialize
        :param initialize_transitions: value for non-unknown to initialize to if already set,
                otherwise None to not initialize
        """
        if initialize_nodes is not None:
            for node_key in self.nodes.keys():
                self.nodes[node_key].initialize_unknown(initialize_nodes)
        if initialize_transitions is not None:
            for transition_key in self.transitions.keys():
                self.transitions[transition_key].initialize_unknown(initialize_transitions)

    def enable_learning(self, enable, learning_weight=0.0):
        """
        Enables/Disables the learning in each transition
        :param enable: bool: whether transition learning should be enabled/disabled
        :param learning_weight: float weight used for learning - weighs based on changes to marginals.  Default is 0,
                which is effectively equal weighting.
        """
        for transition in self.transitions.values():
            transition.enable_learning = enable
            transition.learning_weight = learning_weight

    def set_zero_snapshot(self, transitions=True, nodes=True, clear=False):
        """
        Sets zero snapshots - the "from" point for determining deltas
        :param transitions: boolean - if True, this action applies to transitions
        :param nodes: boolean - if True, this action applies to nodes
        :param clear: boolean - if True, clears the snapshot instead of setting it.
        """
        if transitions is True:
            for transition in self.transitions.values():
                transition.set_zero_snapshot(clear)
        if nodes is True:
            for node in self.nodes.values():
                node.set_zero_snapshot(clear)

    def set_current_snapshot_name(self, name, clear=False):
        """
        If using named snapshots, this defines the name of the snapshot.  If not using named snapshots, will need
         to statistically match the current situation with the captured snapshots for each learning update.
        :param name: the name of the current situation
        :param clear: True to clear the current snapshot name
        """
        for node in self.nodes.values():
            node.set_current_snapshot_name(name, clear)

    def set_snapshot_for_learning(self, name=None, clear=False, node_names=None):
        """
        Creates a snapshot of the transition states for learning - uses these values for optimization instead
         of the most recent values.  Enables setting an expected structure and working from there when learning.
        :param name: string or None - string if known named situation, otherwise None for an incrementing integer
        :param clear: boolean - if True, clears the snapshot instead of setting it.
        :param node_names: None or list of string node names to which this snapshot applies
        """
        if name is None:
            name = self.snapshot_counter
            self.snapshot_counter += 1
        for node_name, node in self.nodes.items():
            if node_names is not None:
                if node_name in node_names:
                    node.set_snapshot(name=name, clear=clear)
            else:
                node.set_snapshot(name=name, clear=clear)

    def update_network_without_inference(self):
        """
        Commands the network to run a complete update without inference (so only parent to child, not child to
         parent).  This is primarily used when a network is created with data from various sources and needs to be
         run to be made consistent before using the full capabilities of the network.
         Note: this only runs nodes that do not have parents.  It also assumes all nodes are on equal footing and
          does not take any current levels of information per node into account.
          The weight assigned to each node will be the maximum weight of any of the parent nodes (such that impact to
           the node will be minimal for future effects).
          Nodes without parents will have their data cleared before running this function such that the first
           propagation to that node will assign instead of combine.
        """
        for each_node in self.nodes.values():
            if len(each_node.parent_transitions) > 0:
                # Has parents - must clear the data
                each_node.clear_data()
                # Make sure the method is set correctly
                each_node.noninference_weight_update = self.noninference_weight_update
        for each_node in self.nodes.values():
            if len(each_node.parent_transitions) == 0:
                # Propagate data starting from nodes with no parents
                # Make sure the method is set correctly
                each_node.noninference_weight_update = self.noninference_weight_update
                each_node.propagate(inference=False)

    def dump_to_csv(self, file_name):
        """
        Dumps all transitions and nodes to CSV lines to enable tracing and debugging.  Appends if possible, otherwise
         creates the new file
        :param file_name: the file name to which to dump
        """
        for each_node in self.nodes.values():
            each_node.dump_to_csv(file_name)
        for each_transition in self.transitions.values():
            each_transition.dump_to_csv(file_name)

    def check_for_cycles(self):
        """
        Runs a search through the graph to check for cycles.
        :return: Whether a cycle exists in the graph
        """
        # Run an exhaustive search for now.  Start with each node,
        # and run every path through transitions and nodes to see if the node is
        # returned to.
        # TODO: Make this more efficient
        for node in self.nodes.values():
            found_cycle = node.check_for_nodes(None, None)
            if found_cycle is True:
                # Break early
                return found_cycle
        # Did not find a cycle
        return False

    def add_callback(self, callback_function, trigger_name, node_name):
        """
        Adds a callback for a given trigger name
        :param callback_function: the function to call if the trigger goes off.  Should take the form of a 1 argument
         function
        :param trigger_name: the string trigger upon which the callback should be called.  Matches to a NodeDecision
         name
        :param node_name: the string name of the node that can cause this trigger
        :return bool whether the callback was added
        """
        full_trigger_name = node_name + "." + trigger_name
        if full_trigger_name not in self.__output_callbacks:
            self.__output_callbacks[full_trigger_name] = []  # New list
        # Add to the list
        self.__output_callbacks[full_trigger_name].append(callback_function)
        return True  # Currently always added

    def trigger_callback(self, trigger_name, triggered_state):
        """
        Makes the callback to the appropriate function based on the trigger name
        :param trigger_name: string name of the trigger
        :param triggered_state: boolean of the state of the trigger (True = triggered, False = not triggered)
        """
        if trigger_name in self.__output_callbacks:
            for callback in self.__output_callbacks[trigger_name]:
                callback(triggered_state)  # Call it

    def add_trigger(self, decision, node_name):
        """
        Adds a trigger to a node.
        :param decision:  NodeDecision - fill this out fully - it defines the trigger
        :param node_name: String - the name of the node to which to add the decision
        :return: Whether it was added and, if not, the reason why (tuple)
        """
        if node_name not in self.nodes:
            return False, "Node name not defined"

        if decision.name in self.nodes[node_name].decisions:
            return False, "Trigger already defined by name and node"

        if not isinstance(decision, NodeDecision):
            return False, "decision not an instance of NodeDecision"

        # Otherwise, add to the list of decisions for that node
        # Define the internally set data
        decision.full_name = node_name + "." + decision.name
        decision.tree_ref = self
        # Add
        self.nodes[node_name].decisions[decision.name] = decision
        return True, ""

    def remove_trigger(self, node_name, trigger_name):
        """
        :param node_name: string name of the node from which to remove the trigger
        :param trigger_name: string name of the trigger to remove
        :return: tuple of whether it was removed and reason if not
        """
        if node_name not in self.nodes:
            return False, "Node name not defined"
        if trigger_name not in self.nodes[node_name].decisions:
            return False, "Trigger name not defined in node " + node_name

        self.nodes[node_name].decisions.pop(trigger_name, None)
        return True, ""

    def add_input_mapping(self, input_mapping):
        """
        Adds an input mapping for the network
        :param input_mapping: InputToNode: the input mapping class instantiation
        :return: tuple of whether it was added and, if not, why not
        """
        if input_mapping.nodes is None:
            return False, "Nodes not defined (is None)"

        for node_name in input_mapping.nodes.keys():
            if node_name not in self.nodes:
                return False, "Node name " + node_name + " not defined."

        if input_mapping.name in self.__inputs:
            return False, "Input name " + input_mapping.name + " already defined."

        # Save it
        self.__inputs[input_mapping.name] = input_mapping

    def solve_with_root_finder(self, child_node_name, parent_transition_names, snapshot=None):
        """
        Learns the transitions simultaneously for several parents using a root finder method
        :param child_node_name: The name of the child node that has multiple parents
        :param parent_transition_names: List of parent transition names that need to be simultaneously learned
        :param snapshot: None or string of snapshot name.  If not None, then only parents with that snapshot should
                be updated.
        :return bool whether the solution was successful
        """
        child_node = self.nodes[child_node_name]
        # Must handle snapshot cases.
        # If a name is set, then the name is consistent across all nodes
        if child_node.current_snapshot_name is not None:
            # We can use this
            child_marginals_after = child_node.get_marginals(child_node.current_snapshot_name)
        else:
            # TODO: Check to see if there is statistical matching for parents.  If so, use the most appropriate name
            child_marginals_after = child_node.get_marginals()  # Dictionary
        solved, parent_end_marginals = solve_for_parents_with_root_finder(child_node.combination_method,
                                                                          child_marginals_after,
                                                                          len(parent_transition_names))
        # If a solution was found, update the transitions
        if solved is True:
            for parent_transition_index in range(0, len(parent_transition_names)):
                # Update that parent if applicable
                if (snapshot is None) or\
                        (snapshot in
                         self.transitions[parent_transition_names[parent_transition_index]].get_parent().snapshot) or \
                        (snapshot in
                         self.transitions[parent_transition_names[parent_transition_index]].get_parent().snapshot_map):
                    self.transitions[parent_transition_names[parent_transition_index]].rebalance_potentials(
                        child_marginals=parent_end_marginals[parent_transition_index])
        # Return the results to see if an optimization needs to be tried
        return solved

    def multi_learn(self, child_node_name, parent_transition_names, snapshot=None):
        """
        Learns the transitions simultaneously for several parents
        :param child_node_name: The name of the child node that has multiple parents
        :param parent_transition_names: List of parent transition names that need to be simultaneously learned
        :param snapshot: None or string name of snapshot.  If not None, will be used to restrict which transitions are
                        updated
        """

        if (self.same_multi_parent_marginals is True) and\
                (self.use_root_finder != MULTI_PARENT_SOLUTION_TYPE["CATEGORIES"]["OPTIMIZER"]):
            # First attempt to solve using a root finder.
            #  If the D-S combination method has a defined root finder, then this method should always work.
            if self.solve_with_root_finder(child_node_name, parent_transition_names, snapshot=snapshot) is True:
                return  # Solution is good - don't go further
            elif self.use_root_finder == MULTI_PARENT_SOLUTION_TYPE["CATEGORIES"]["ROOT_FINDER_OPT_FALLBACK"]:
                print("Unable to solve node " + child_node_name + " with root finder and combination method " +
                      self.nodes[child_node_name].combination_method +
                      ".  Continuing with multi-dimensional optimization.")
            else:
                # No fallback
                child_node = self.nodes[child_node_name]
                not_enough_data = False
                if (self.same_multi_parent_marginals is True) and \
                        ((child_node.combination_method == COMBINATION_METHODS["MURPHY"]) or
                         (child_node.combination_method == COMBINATION_METHODS["ZHANG"])) and \
                        (child_node.internal_probability_data["number_of_evidences"] < len(parent_transition_names)):
                    not_enough_data = True
                if not not_enough_data:
                    raise ValueError("Unable to solve for set of calculated child marginals for node " +
                                     child_node_name + " with root finder.")
                else:
                    # No fall back
                    return

        # How this optimization works:
        # Design variables: the desired child marginals produced from each parent marginals multiplied by the
        #   corresponding transition potentials.  If same_multi_parent_marginals is True, then all sets of desired
        #   child marginals are the same, so one set of design variables.  If False, each set is different, so the
        #   number of design variables is the number of child marginals multiplied by the number of applicable parents.
        # Equality constraints: child marginals = D-S combination method (whichever one is being used) of all applicable
        #   desired child marginals => nonlinear constraint (technically, multiple constraints via a single function)
        #   For each set of desired child marginals, the sum must be 1.  This could be one constraint if
        #   same_multi_parent_marginals is True or n constraints if there are n applicable parents. => linear constraint
        # Inequality constraints: bounds on all individual design variables between 0 and 1.  Note that 1 is really
        #   unnecessary since the sum equality constraint above and the lower bound mean that the 1 upper bound is
        #   automatically satisfied.  Due to the algorithm interface, though, it is easier just to state both bounds.
        # Objective function: quadratic objective function based on the difference between the desired child marginals
        #   and the previous child marginals obtained by multiplying the new parent marginals with the old transition
        #   potentials.  In simple terms: the goal is to minimize change to avoid major changes during updates.
        # The starting design vector is the new parent marginals multiplied by the old transition potentials.  Obviously
        #   this will not satisfy the equality constraints since the child marginals have changed.
        child_node = self.nodes[child_node_name]
        # TODO: Use statistical matching with parents to see if a different option should be used if the name is not set
        child_marginals_after = child_node.get_marginals(child_node.current_snapshot_name)  # Dictionary
        child_marginal_names = list(child_marginals_after.keys())
        num_child_marginals = len(child_marginals_after)
        # Get all the before+ child marginals - used for the optimization function to minimize change.
        before_plus_child_marginals = []
        x0 = []
        lower_bounds = []
        upper_bounds = []
        first_parent = True
        for parent_transition_name in parent_transition_names:
            propagated_marginals = self.transitions[parent_transition_name].get_propagated_marginals(True)
            for child_marginal_name in child_marginal_names:
                before_plus_child_marginals.append(propagated_marginals[child_marginal_name])
                # Initialize the search here
                if (self.same_multi_parent_marginals is False) or (first_parent is True):
                    x0.append(propagated_marginals[child_marginal_name])
                    lower_bounds.append(0.0)
                    upper_bounds.append(1.0)  # Not necessary due to sum to 1 and 0 lower bound.
            first_parent = False
        # Create the Bounds object - only necessary if using different values from all parents
        # keep_feasible=True ensures that values outside of the constraints will not be tested, which is essential
        #  for D-S combination algorithms that have unintended effects when values are outside the [0,1] range.
        # Unfortunately, this also seems to make the optimization algorithm fail.  So, relaxing this and controlling
        #  the roundoff error in the results and the D-S combination algorithms.
        bounds = Bounds(lower_bounds, upper_bounds, keep_feasible=False)
        # Create the objective function to minimize
        obj_func = partial(multi_parent_obj_func, before_plus_child_marginals)
        # Handle constraints
        # Define the nonlinear constraints - this is the harder function to define.
        nonlinear_constraint_bound = np.zeros(num_child_marginals)
        child_marginal_counter = 0
        for child_marginal_name in child_marginal_names:
            nonlinear_constraint_bound[child_marginal_counter] = child_marginals_after[child_marginal_name]
            child_marginal_counter += 1
        # noinspection PyTypeChecker
        nonlinear_constraint = NonlinearConstraint(partial(multi_parent_nonlinear_constraint_func,
                                                           child_node.combination_method,
                                                           child_marginal_names,
                                                           len(parent_transition_names)),  # The number of parents
                                                   nonlinear_constraint_bound,
                                                   nonlinear_constraint_bound, jac='2-point', hess=SR1())

        if self.same_multi_parent_marginals is True:
            num_design_variables = num_child_marginals
            constraints = [nonlinear_constraint]
        else:
            num_design_variables = num_child_marginals * len(parent_transition_names)
            num_linear_equality_constraints = len(parent_transition_names)

            # Define the linear constraints
            linear_constraint_matrix = np.zeros((num_linear_equality_constraints,  # Rows
                                                 num_design_variables))  # Columns
            linear_constraint_bound = np.ones(num_linear_equality_constraints)
            for design_var_counter in range(0, num_child_marginals):
                for constraint_counter in range(0, num_linear_equality_constraints):
                    linear_constraint_matrix[constraint_counter,
                                             constraint_counter * num_child_marginals + design_var_counter] = 1
            linear_constraints = LinearConstraint(linear_constraint_matrix, linear_constraint_bound,
                                                  linear_constraint_bound)
            constraints = [linear_constraints, nonlinear_constraint]

        # Can try BFGS() for the Hessian update strategy
        # noinspection PyTypeChecker,PyTypeChecker
        result = minimize(obj_func, np.array(x0), method='trust-constr', jac="2-point", hess=SR1(),
                          options={'verbose': 0}, constraints=constraints, bounds=bounds)
        # TODO: Switch this to checking the success flag.  Need the newest version of scipy, but that requires
        # TODO: an update that Pycharm doesn't support right now, so holding back on that for the moment.
        if (result['status'] == 1) or (result['status'] == 2):
            for parent_transition_index in range(0, len(parent_transition_names)):
                parent_end_marginals = {}
                offset_index = 0
                if num_design_variables != num_child_marginals:
                    # Not a single set
                    offset_index = parent_transition_index * len(child_marginal_names)
                total_sum = 0.0
                for child_marginal_index in range(0, len(child_marginal_names)):
                    # Cap in the [0.0, 1.0] range to handle roundoff error
                    parent_end_marginals[child_marginal_names[child_marginal_index]] =\
                        max(min(result.x[offset_index + child_marginal_index], 1.0), 0.0)
                    total_sum += parent_end_marginals[child_marginal_names[child_marginal_index]]
                # The total_sum is likely not 1.0 due to roundoff error, but needs to be to maintain consistency
                #  for the next set of optimizations.
                for marginal_name in parent_end_marginals.keys():
                    parent_end_marginals[marginal_name] /= total_sum
                # Update that parent if applicable
                if (snapshot is None) or \
                        (snapshot in
                         self.transitions[parent_transition_names[parent_transition_index]].get_parent().snapshot) or \
                        (snapshot in
                         self.transitions[parent_transition_names[parent_transition_index]].get_parent().snapshot_map):
                    self.transitions[parent_transition_names[parent_transition_index]].rebalance_potentials(
                        child_marginals=parent_end_marginals)
        else:
            # May not be a problem - may simply not have enough evidence to allow a solution yet.
            child_node = self.nodes[child_node_name]
            not_enough_data = False
            if (self.same_multi_parent_marginals is True) and\
                    ((child_node.combination_method == COMBINATION_METHODS["MURPHY"]) or
                     (child_node.combination_method == COMBINATION_METHODS["ZHANG"])) and\
                    (child_node.internal_probability_data["number_of_evidences"] < len(parent_transition_names)):
                not_enough_data = True
            if not not_enough_data:
                raise ValueError("Unable to solve for set of calculated child marginals for node " + child_node_name)
            else:
                # Not enough data
                return

    def input_evidence(self, input_name, inputs, input_weight):
        """
        Calls the underlying input and handles network updates
        :param input_name: existing input name
        :param inputs: user defined inputs
        :param input_weight: user-defined weights for the input
        :return tuple of whether the update succeeded and, if not, why
        """
        if input_name not in self.__inputs:
            return False, "Input name " + input_name + " not defined."

        self.__inputs[input_name].map_to_input(inputs, input_weight)

        # Update any transitions for multiple parents of a node
        self.update_multiple_parent_transitions()

    def update_transitions(self, snapshot=None):
        """
        Updates transitions without new evidence.  Useful if not updating transitions on every evidence input.
        :param snapshot: None or string name of snapshot.  If not None, must be in the parent to update the transition
        """
        for each_node in self.nodes.values():
            parent_transition_names_to_learn = []
            for each_parent_transition in each_node.parent_transitions:
                if each_parent_transition.enable_learning is True:
                    parent_transition_names_to_learn.append(each_parent_transition.name)
            if (len(each_node.parent_transitions) == 1) and (len(parent_transition_names_to_learn) > 0):
                # One parent transition - update it if the snapshot is None or appears in the parent
                if (snapshot is None) or\
                        (snapshot in self.transitions[parent_transition_names_to_learn[0]].get_parent().snapshot) or\
                        (snapshot in self.transitions[parent_transition_names_to_learn[0]].get_parent().snapshot_map):
                    # Either the snapshot doesn't matter or it exists in the parent
                    self.transitions[parent_transition_names_to_learn[0]].rebalance_potentials()

        # Finish with any multiple parent transitions
        self.update_multiple_parent_transitions(snapshot=snapshot)

    def update_multiple_parent_transitions(self, snapshot=None):
        """
        Updates the transitions between child and multiple parent nodes.
        :param snapshot: None or string name of snapshot.  If not None, must be in the parent to update the transition
        """
        for each_node_name, each_node in self.nodes.items():
            parent_transition_names_to_learn = []
            for each_parent_transition in each_node.parent_transitions:
                if each_parent_transition.enable_learning is True:
                    parent_transition_names_to_learn.append(each_parent_transition.name)
            if len(each_node.parent_transitions) > 1:
                # There are multiple parent transitions, so learning these were skipped during the individual update
                if len(parent_transition_names_to_learn) > 1:
                    # Learn the parent transitions.  These have to be done together.
                    # This is a special capability.
                    self.multi_learn(each_node_name, parent_transition_names_to_learn, snapshot=snapshot)
                elif len(parent_transition_names_to_learn) > 0:
                    # Only a single one - use the default learning method
                    # This means that learning is not turned on for at least one...an odd situation, but move
                    #  ahead as defined.
                    if (snapshot is None) or \
                            (snapshot in
                             self.transitions[parent_transition_names_to_learn[0]].get_parent().snapshot) or \
                            (snapshot in
                             self.transitions[parent_transition_names_to_learn[0]].get_parent().snapshot_map):
                        # Either the snapshot doesn't matter or it exists in the parent
                        self.transitions[parent_transition_names_to_learn[0]].rebalance_potentials()


def multi_parent_obj_func(before_plus_child_marginals, x):
    """
    An objective function for optimization when multiple parents are involved.  Handles both cases (in which all
     marginals are independently adjusted and in which all sets of marginals are driven to the same set of values).
    :param x: list the design vector
    :param before_plus_child_marginals: list the child marginals based on the updated parents and original potentials
    :return: double objective value
    """
    return_value = 0
    x_counter = 0
    for before_plus_value in before_plus_child_marginals:
        return_value += pow(x[x_counter] - before_plus_value, 2)
        x_counter += 1
        if x_counter >= len(x):
            x_counter = 0
    return return_value


def multi_parent_nonlinear_constraint_func(combination_method, child_marginals_names, num_parents, x):
    """
    A nonlinear constraint function for multi-parent optimization/updates.  Handles both cases (in which all
     marginals are independently adjusted and in which all sets of marginals are driven to the same set of values).
    :param x: list the design vector
    :param combination_method: The D-S combination type
    :param child_marginals_names: list of child marginals names (for ordering the outputs)
    :param num_parents: int number of parents (number of evidence combinations)
    :return: list of values from the combination in the child_marginals_names order
    """
    evidence = {}
    if len(x) == len(child_marginals_names):
        # All values are being driven to the same set, so create n number of evidences
        each_evidence = {}
        for x_counter in range(0, len(x)):
            each_evidence[child_marginals_names[x_counter]] = x[x_counter]
        # Copy the one evidence to all
        for evidence_counter in range(0, num_parents):
            evidence[str(evidence_counter)] = deepcopy(each_evidence)
    else:
        # All evidences are different
        # Create each evidence set
        for evidence_counter in range(0, num_parents):
            evidence_name = str(evidence_counter)
            evidence[evidence_name] = {}
            for child_marginal_counter in range(0, len(child_marginals_names)):
                evidence[evidence_name][child_marginals_names[child_marginal_counter]] =\
                    x[evidence_counter * len(child_marginals_names) + child_marginal_counter]
    # Get the resulting marginals
    probability_data = import_and_combine(combination_method, evidence)
    calculated_child_marginals = import_and_calculate_probabilities(combination_method, probability_data)
    # Return in a form that the optimizer can use
    results = []
    for child_marginals_name in child_marginals_names:
        results.append(calculated_child_marginals[child_marginals_name])
    return results
