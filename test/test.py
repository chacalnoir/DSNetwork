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

import unittest
from network import evaluate_network_consistency
from network.DSNetwork import DSNetwork, Node, Transition, NODE_TYPE, WEIGHTING_METHOD
from network.NetworkGenerator import NetworkGenerator
from combinationRules import import_and_calculate_probabilities, import_and_combine, COMBINATION_METHODS
from combinationRulesExtensions import solve_for_parents_with_root_finder
from functools import reduce
from combinationRules.utilities import the_keys
from combinationRulesExtensions import previously_applicable_set_intersection, MARGINAL_STATE,\
    constrained_solver_for_root_finder
import random
from copy import deepcopy

TEST_DELTA = 0.0001
ZERO_DELTA = 1e-16

PROBABILITY_TO_SET_POTENTIAL = 1.0
NUMBER_OF_UPDATES_PER_TEST = 5
NUMBER_OF_SIMULTANEOUS_UPDATES = 3

MARGINAL_DEFINES = {
    "INDIVIDUAL": "INDIVIDUAL",
    "COMBINED": "COMBINED",
    "LIMITED": "LIMITED"
}


class TestNetwork(unittest.TestCase):
    def setUp(self):
        """
        Create the network
        """
        self.new_generator = NetworkGenerator()
        self.new_generator.create_all_trees()
        # It's been created.  Now to test

    def test_save_load(self):
        """
        Test saving and loading the network
        """
        import pickle

        original_network = self.new_generator.generate_test_tree()

        # Dump to file
        with open('network_save_test.txt', 'wb') as f:
            pickle.dump(original_network, f)

        # Recover to structure
        with open('network_save_test.txt', 'rb') as f:
            new_network = pickle.load(f)

        # Test whether it worked
        self.assertFalse(new_network is None)
        for name in original_network.nodes.keys():
            self.assertTrue(name in new_network.nodes)
            for transition in original_network.nodes[name].parent_transitions:
                # Find the matching transition
                found = False
                for new_transition in new_network.nodes[name].parent_transitions:
                    if new_transition.name == transition.name:
                        found = True
                self.assertTrue(found)
            for transition in original_network.nodes[name].child_transitions:
                # Find the matching transition
                found = False
                for new_transition in new_network.nodes[name].child_transitions:
                    if new_transition.name == transition.name:
                        found = True
                self.assertTrue(found)
        for name in original_network.transitions.keys():
            self.assertTrue(name in new_network.transitions)
            self.assertTrue(original_network.transitions[name].get_parent().name ==
                            new_network.transitions[name].get_parent().name)
            self.assertTrue(original_network.transitions[name].get_child().name ==
                            new_network.transitions[name].get_child().name)

    def test_cycles(self):
        """
        Test whether cycles are detected accurately in the DS network
        """
        no_cycle_network = self.new_generator.test_tree
        self.assertFalse(no_cycle_network.check_for_cycles())

        cycle_network = self.new_generator.cycle_test_tree
        self.assertTrue(cycle_network.check_for_cycles())

    def forward_and_backward_propagation(self):
        """
        Run tests on forward and backward propagation depending on the combination method specified
        """
        test_tree = self.new_generator.test_tree
        # Start by entering data at the transitions
        # Parent at first level, then child
        potential_input_db = {
            ("TRUE",): {
                ("TRUE",): 0.8,
                ("FALSE",): 0.1,
                ("FALSE", "TRUE"): 0.1
            },
            ("FALSE",): {
                ("TRUE",): 0.2,
                ("FALSE",): 0.6,
                ("FALSE", "TRUE"): 0.2
            },
            ("FALSE", "TRUE"): {
                ("TRUE",): 0.05,
                ("FALSE",): 0.05,
                ("FALSE", "TRUE"): 0.9
            }
        }
        test_tree.transitions["D.B"].propagate(potential_input_db)

        # Other transition
        potential_input_dg = {
            ("TRUE",): {
                ("TRUE",): 0.6,
                ("FALSE",): 0.3,
                ("FALSE", "TRUE"): 0.1
            },
            ("FALSE",): {
                ("TRUE",): 0.1,
                ("FALSE",): 0.6,
                ("FALSE", "TRUE"): 0.3
            },
            ("FALSE", "TRUE"): {
                ("TRUE",): 0.1,
                ("FALSE",): 0.2,
                ("FALSE", "TRUE"): 0.7
            }
        }
        test_tree.transitions["D.G"].propagate(potential_input_dg)

        # First node input as evidence
        potential_input_d = {
            1: {
                "TRUE": 0.5,
                "FALSE": 0.3,
                ("FALSE", "TRUE"): 0.2
            }
        }
        test_tree.nodes["D"].propagate(potential_input_d)

        # Check the data - B
        b_probabilities = test_tree.nodes["B"].get_marginals()
        self.assertAlmostEqual(b_probabilities[("TRUE",)], 0.47, 4)
        self.assertAlmostEqual(b_probabilities[("FALSE",)], 0.24, 4)
        self.assertAlmostEqual(b_probabilities[("FALSE", "TRUE")], 0.29, 4)
        # G
        g_probabilities = test_tree.nodes["G"].get_marginals()
        self.assertAlmostEqual(g_probabilities[("TRUE",)], 0.35, 4)
        self.assertAlmostEqual(g_probabilities[("FALSE",)], 0.37, 4)
        self.assertAlmostEqual(g_probabilities[("FALSE", "TRUE")], 0.28, 4)

        # Now test back propagation + forward propagation
        potential_input_b = {
            1: {
                "TRUE": 0.6,
                "FALSE": 0.15,
                ("FALSE", "TRUE"): 0.25
            }
        }
        test_tree.nodes["B"].propagate(potential_input_b)
        # Test the data
        # Start with the assumption that the Dempster-Shafer combinations library has
        # already been tested.

        # B
        new_b_probabilities = test_tree.nodes["B"].get_marginals()
        # Get the correct answers through the D-S combinations library
        test_evidence = {
            0: b_probabilities,
            1: potential_input_b[1]
        }
        correct_answer = import_and_calculate_probabilities(test_tree.nodes["D"].combination_method,
                                                            import_and_combine(test_tree.nodes["D"].combination_method,
                                                            test_evidence))
        for answer_key in correct_answer.keys():
            self.assertAlmostEqual(correct_answer[answer_key], new_b_probabilities[answer_key], 4)

        # D
        new_d_probabilities = test_tree.nodes["D"].get_marginals()
        # Transition the input to the D node
        correct_d_input = {
            "TRUE": new_b_probabilities[("TRUE",)] * potential_input_db[("TRUE",)][("TRUE",)] +
            new_b_probabilities[("FALSE",)] * potential_input_db[("TRUE",)][("FALSE",)] +
            new_b_probabilities[("FALSE", "TRUE")] * potential_input_db[("TRUE",)][("FALSE", "TRUE")],
            "FALSE": new_b_probabilities[("TRUE",)] * potential_input_db[("FALSE",)][("TRUE",)] +
            new_b_probabilities[("FALSE",)] * potential_input_db[("FALSE",)][("FALSE",)] +
            new_b_probabilities[("FALSE", "TRUE")] * potential_input_db[("FALSE",)][("FALSE", "TRUE")],
            ("FALSE", "TRUE"): new_b_probabilities[("TRUE",)] * potential_input_db[("FALSE", "TRUE")][("TRUE",)] +
            new_b_probabilities[("FALSE",)] * potential_input_db[("FALSE", "TRUE")][("FALSE",)] +
            new_b_probabilities[("FALSE", "TRUE")] * potential_input_db[("FALSE", "TRUE")][("FALSE", "TRUE")]
        }
        sum_of_input = 0
        for input_key in correct_d_input.keys():
            sum_of_input += correct_d_input[input_key]
        for input_key in correct_d_input.keys():
            correct_d_input[input_key] /= sum_of_input

        # Get the correct answers through the DS combinations library
        test_evidence = {
            0: potential_input_d[1],
            1: correct_d_input
        }
        correct_answer = import_and_calculate_probabilities(test_tree.nodes["D"].combination_method,
                                                            import_and_combine(test_tree.nodes["D"].combination_method,
                                                            test_evidence))
        for answer_key in correct_answer.keys():
            self.assertAlmostEqual(correct_answer[answer_key], new_d_probabilities[answer_key], 4)

        # G
        new_g_probabilities = test_tree.nodes["G"].get_marginals()
        # Transition the input to the D node
        correct_g_input = {
            "TRUE": new_d_probabilities[("TRUE",)] * potential_input_dg[("TRUE",)][("TRUE",)] +
            new_d_probabilities[("FALSE",)] * potential_input_dg[("FALSE",)][("TRUE",)] +
            new_d_probabilities[("FALSE", "TRUE")] * potential_input_dg[("FALSE", "TRUE")][("TRUE",)],
            "FALSE": new_d_probabilities[("TRUE",)] * potential_input_dg[("TRUE",)][("FALSE",)] +
            new_d_probabilities[("FALSE",)] * potential_input_dg[("FALSE",)][("FALSE",)] +
            new_d_probabilities[("FALSE", "TRUE")] * potential_input_dg[("FALSE", "TRUE")][("FALSE",)],
            ("FALSE", "TRUE"): new_d_probabilities[("TRUE",)] * potential_input_dg[("TRUE",)][("FALSE", "TRUE")] +
            new_d_probabilities[("FALSE",)] * potential_input_dg[("FALSE",)][("FALSE", "TRUE")] +
            new_d_probabilities[("FALSE", "TRUE")] * potential_input_dg[("FALSE", "TRUE")][("FALSE", "TRUE")]
        }
        sum_of_input = 0
        for input_key in correct_g_input.keys():
            sum_of_input += correct_g_input[input_key]
        for input_key in correct_g_input.keys():
            correct_g_input[input_key] /= sum_of_input

        # Get the correct answers through the DS combinations library
        test_evidence = {
            0: g_probabilities,
            1: correct_g_input
        }
        correct_answer = import_and_calculate_probabilities(test_tree.nodes["D"].combination_method,
                                                            import_and_combine(test_tree.nodes["D"].combination_method,
                                                            test_evidence))
        for answer_key in correct_answer.keys():
            self.assertAlmostEqual(correct_answer[answer_key], new_g_probabilities[answer_key], 4)

    def test_dempster_shafer(self):
        """
        Test the network with a Dempster Shafer combination method
        """
        self.new_generator.test_tree.set_combination_and_weighting_methods(
            combination_method=COMBINATION_METHODS["DEMPSTER_SHAFER"], override=True)
        self.forward_and_backward_propagation()

    def test_murphy(self):
        """
        Test the network with a Murphy combination method
        """
        self.new_generator.test_tree.set_combination_and_weighting_methods(
            combination_method=COMBINATION_METHODS["MURPHY"], override=True)
        self.forward_and_backward_propagation()

    def test_yager(self):
        """
        Test the network with a Yager combination method
        """
        self.new_generator.test_tree.set_combination_and_weighting_methods(
            combination_method=COMBINATION_METHODS["YAGER"], override=True)
        self.forward_and_backward_propagation()

    def test_zhang(self):
        """
        Test the network with a Zhang combination method
        """
        self.new_generator.test_tree.set_combination_and_weighting_methods(
            combination_method=COMBINATION_METHODS["ZHANG"], override=True)
        self.forward_and_backward_propagation()

    def test_specific_learning(self):
        debug = False
        a_node = Node()
        a_node.name = "A"
        a_node.type = NODE_TYPE["CATEGORIES"]["INPUT"]
        a_node.options = ["A_O1", "A_O2", "A_O3"]

        c_node = Node()
        c_node.name = "C"
        c_node.type = NODE_TYPE["CATEGORIES"]["INPUT"]
        c_node.options = ["C_O1", "C_O2", "C_O3"]

        ac_transition = Transition(a_node, c_node)
        tree = DSNetwork()
        tree.add_link_and_initialize([a_node, c_node],
                                     [ac_transition],
                                     combination_method=COMBINATION_METHODS["MURPHY"],
                                     weighting_method=WEIGHTING_METHOD["CATEGORIES"]["TO_ONE"],
                                     initialize_unknown=False)
        tree.enable_learning(False)  # Only when specifically run
        a_evidence = {
            "evidence": {
                ('A_O1', 'A_O2'): 8.968321585713485e-09,
                ('A_O1', 'A_O2', 'A_O3'): 1.9406054340177627e-11,
                ('A_O1', 'A_O3'): 2.714656644731106e-07,
                ('A_O1', ): 0.8574163540032125,
                ('A_O2', 'A_O3'): 1.3501152457670091e-08,
                ('A_O2', ): 0.04703944472431756,
                ('A_O3', ): 0.09554390731792536
            }
        }
        tree.nodes["A"].propagate(a_evidence, inference=False, input_weight=1.0, debug=debug)

        c_evidence = {
            "evidence": {
                ('C_O1', 'C_O2'): 3.314506783820986e-09,
                ('C_O1', 'C_O2', 'C_O3'): 5.462173768796587e-11,
                ('C_O1', 'C_O3'): 2.505942250144573e-07,
                ('C_O1', ): 2.1786705103673725e-05,
                ('C_O2', 'C_O3'): 4.0190442849250264e-07,
                ('C_O2', ): 2.2185140821554828e-05,
                ('C_O3', ): 0.9999993469487781
            }
        }
        tree.nodes["C"].propagate(c_evidence, inference=False, input_weight=1.0, debug=debug)

        tree.transitions["A.C"].initialize_unknown()

        # Learn
        tree.transitions["A.C"].rebalance_potentials(debug=debug)

    def test_learning(self):
        """
        Tests whether nodes and transitions can be learned from unknown states
        """
        debug = False
        test_tree = self.new_generator.test_tree
        test_tree.initialize_unknown()  # Initialize
        test_tree.enable_learning(True)

        # First node input as evidence
        potential_input_d = {
            1: {
                "TRUE": 0.5,
                "FALSE": 0.3,
                ("FALSE", "TRUE"): 0.2
            }
        }
        test_tree.nodes["D"].propagate(potential_input_d, debug=debug)

        # That should make it through to the b node. No learning, since the b node didn't change at all.
        conditionals = test_tree.transitions["D.B"].conditionals_parent_to_child
        for parent_key in conditionals.keys():
            for child_key in conditionals[parent_key].keys():
                if child_key == ("FALSE", "TRUE"):
                    self.assertAlmostEqual(1.0, conditionals[parent_key][child_key], 4)
                else:
                    self.assertAlmostEqual(0.0, conditionals[parent_key][child_key], 4)

        # That should make it through to the g node. No learning, since the g node didn't change at all.
        conditionals = test_tree.transitions["D.G"].conditionals_parent_to_child
        for parent_key in conditionals.keys():
            for child_key in conditionals[parent_key].keys():
                if child_key == ("FALSE", "TRUE"):
                    self.assertAlmostEqual(1.0, conditionals[parent_key][child_key], 4)
                else:
                    self.assertAlmostEqual(0.0, conditionals[parent_key][child_key], 4)

        # Now it gets interesting - enter data at the child level - at the B node
        potential_input_b = {
            1: {
                "TRUE": 0.2,
                "FALSE": 0.4,
                ("FALSE", "TRUE"): 0.4
            }
        }
        test_tree.nodes["B"].propagate(potential_input_b, debug=debug)

        # Now verify the results
        # D.B
        conditionals = test_tree.transitions["D.B"].conditionals_parent_to_child
        parent_marginals = import_and_calculate_probabilities(test_tree.nodes["D"].combination_method,
                                                              test_tree.nodes["D"].internal_probability_data)
        child_marginals = import_and_calculate_probabilities(test_tree.nodes["B"].combination_method,
                                                             test_tree.nodes["B"].internal_probability_data)
        calculated_child_marginals = {}
        for parent_key in parent_marginals.keys():
            if parent_key in conditionals:
                vector_sum = 0
                for child_key in conditionals[parent_key].keys():
                    # Check positive value, including round-off error
                    self.assertTrue(conditionals[parent_key][child_key] >= -0.0001)

                    vector_sum += conditionals[parent_key][child_key]
                    if child_key not in calculated_child_marginals:
                        calculated_child_marginals[child_key] = 0
                    calculated_child_marginals[child_key] +=\
                        conditionals[parent_key][child_key] * parent_marginals[parent_key]
                # Check sum to 1
                self.assertAlmostEqual(1.0, vector_sum, 4)
        for child_key in child_marginals.keys():
            self.assertTrue(child_key in calculated_child_marginals)
            self.assertAlmostEqual(child_marginals[child_key], calculated_child_marginals[child_key], 4)

        # D.G
        conditionals = test_tree.transitions["D.G"].conditionals_parent_to_child
        child_marginals = import_and_calculate_probabilities(test_tree.nodes["G"].combination_method,
                                                             test_tree.nodes["G"].internal_probability_data)
        calculated_child_marginals = {}
        for parent_key in parent_marginals.keys():
            if parent_key in conditionals:
                vector_sum = 0
                for child_key in conditionals[parent_key].keys():
                    # Check positive value, including round-off error
                    self.assertTrue(conditionals[parent_key][child_key] >= -0.0001)

                    vector_sum += conditionals[parent_key][child_key]
                    if child_key not in calculated_child_marginals:
                        calculated_child_marginals[child_key] = 0
                    calculated_child_marginals[child_key] += \
                        conditionals[parent_key][child_key] * parent_marginals[parent_key]
                # Check sum to 1
                self.assertAlmostEqual(1.0, vector_sum, 4)
        for child_key in child_marginals.keys():
            self.assertTrue(child_key in calculated_child_marginals)
            self.assertAlmostEqual(child_marginals[child_key], calculated_child_marginals[child_key], 4)

    def test_learning_two_level(self):
        """
        Tests whether nodes and transitions can be learned from unknown states for more than one level
        """
        test_tree = self.new_generator.test_tree
        test_tree.initialize_unknown()  # Initialize
        test_tree.enable_learning(True)

        # Enter data multiple times at the parent and grandchild levels to see if things start to propagate
        for test_iter in range(0, 5):
            # Parent level
            potential_input_parent = {
                1: {
                    "TRUE": 0.5,
                    "FALSE": 0.3,
                    ("FALSE", "TRUE"): 0.2
                }
            }
            test_tree.nodes["D"].propagate(potential_input_parent)

            # Grandchild level
            potential_input_grandchild = {
                1: {
                    "TRUE": 0.2,
                    "FALSE": 0.4,
                    ("FALSE", "TRUE"): 0.4
                }
            }
            test_tree.nodes["H"].propagate(potential_input_grandchild)

    def test_multi_learn(self):
        """
        Test learning multiple transitions simultaneously
        """
        test_tree = self.new_generator.multi_combination_test_tree
        test_tree.enable_learning(True)
        test_tree.same_multi_parent_marginals = True
        # The data has already been initialized to unknown.  Now add data
        evidence = {
            "D": {"evidence": {
                "TRUE": 0.2,
                "FALSE": 0.4,
                ("FALSE", "TRUE"): 0.05,
                "NOT_SURE": 0.1,
                ("FALSE", "NOT_SURE"): 0.1,
                ("FALSE", "TRUE", "NOT_SURE"): 0.15
                }
            },
            "B": {"evidence": {
                "TRUE": 0.3,
                "FALSE": 0.2,
                ("FALSE", "TRUE"): 0.1,
                "NOT_SURE": 0.15,
                ("FALSE", "NOT_SURE"): 0.15,
                ("FALSE", "TRUE", "NOT_SURE"): 0.1
                }
            },
            "G": {"evidence": {
                "TRUE": 0.6,
                "FALSE": 0.3,
                ("FALSE", "TRUE"): 0.0,
                "NOT_SURE": 0.05,
                ("FALSE", "NOT_SURE"): 0.0,
                ("FALSE", "TRUE", "NOT_SURE"): 0.05
                }
            }
        }
        test_tree.input_evidence("TestInputs", evidence, input_weight=0.0)  # No weighting

        evidence_2 = {
            "D": {"evidence": {
                "TRUE": 0.1,
                "FALSE": 0.5,
                ("FALSE", "TRUE"): 0.1,
                "NOT_SURE": 0.05,
                ("FALSE", "NOT_SURE"): 0.05,
                ("FALSE", "TRUE", "NOT_SURE"): 0.2
                }
            },
            "B": {"evidence": {
                "TRUE": 0.25,
                "FALSE": 0.25,
                ("FALSE", "TRUE"): 0.1,
                "NOT_SURE": 0.1,
                ("FALSE", "NOT_SURE"): 0.1,
                ("FALSE", "TRUE", "NOT_SURE"): 0.2
                }
            },
            "G": {"evidence": {
                "TRUE": 0.4,
                "FALSE": 0.4,
                ("FALSE", "TRUE"): 0.05,
                "NOT_SURE": 0.05,
                ("FALSE", "NOT_SURE"): 0.05,
                ("FALSE", "TRUE", "NOT_SURE"): 0.05
                }
            }
        }

        test_tree.input_evidence("TestInputs", evidence_2, input_weight=0.0)  # No weighting

    def test_simple_reverse_solver(self):
        """
        Tests a specific known case of the reverse solver failing
        """
        child_marginals = {
            ('b', ): 0.3,
            ('a', 'b'): 0.3,
            ('a', ): 0.4
        }
        solved, results = solve_for_parents_with_root_finder(COMBINATION_METHODS["DEMPSTER_SHAFER"],
                                                             child_marginals, 2)
        self.assertTrue(solved is True,
                        msg="{} for num parents {} and num options {}".format(COMBINATION_METHODS["DEMPSTER_SHAFER"],
                                                                              2, 2))
        # Combine back forward to see if the results are valid
        # Forward combine
        combined_data = import_and_combine(COMBINATION_METHODS["DEMPSTER_SHAFER"], results)
        # Get the marginals
        combined_marginals = import_and_calculate_probabilities(COMBINATION_METHODS["DEMPSTER_SHAFER"], combined_data)
        # Check
        # Due to roundoff error in the root finding, other cases are not as accurate
        test_delta = TEST_DELTA
        for marginal_key, marginal_value in combined_marginals.items():
            self.assertAlmostEqual(marginal_value, child_marginals[marginal_key], delta=test_delta,
                                   msg="{} for num parents {} and num options {}".format(
                                       COMBINATION_METHODS["DEMPSTER_SHAFER"], 2, 2))

    def test_specific_reverse_solver(self):
        """
        Tests a specific known case of the reverse solver failing
        """
        num_parents = 3
        child_marginals = {
            ('c',): 0.0,
            ('a', 'c'): 0.3,
            ('b', ): 0.0,
            ('a', 'b'): 0.3,
            ('a', ): 0.0,
            ('b', 'c'): 0.4,
            ('a', 'b', 'c'): 0.0
        }
        solved, results = solve_for_parents_with_root_finder(COMBINATION_METHODS["DEMPSTER_SHAFER"],
                                                             child_marginals, num_parents)
        self.assertTrue(solved is True,
                        msg="{} for num parents {} and num options {}".format(COMBINATION_METHODS["DEMPSTER_SHAFER"],
                                                                              num_parents, 3))
        # Combine back forward to see if the results are valid
        # Forward combine
        combined_data = import_and_combine(COMBINATION_METHODS["DEMPSTER_SHAFER"], results)
        # Get the marginals
        combined_marginals = import_and_calculate_probabilities(COMBINATION_METHODS["DEMPSTER_SHAFER"], combined_data)
        # Check
        # Due to roundoff error in the root finding, other cases are not as accurate
        test_delta = TEST_DELTA
        for marginal_key, marginal_value in combined_marginals.items():
            self.assertAlmostEqual(marginal_value, child_marginals[marginal_key], delta=test_delta,
                                   msg="{} for num parents {} and num options {}".format(
                                       COMBINATION_METHODS["DEMPSTER_SHAFER"], num_parents, 3))

    def test_specific_reverse_solver_2(self):
        """
        Tests a specific known case of the reverse solver failing
        """
        num_parents = 3
        child_marginals = {
            ('c',): 0.004182517220232905,
            ('a', 'c'): 0.7181052796955626,
            ('b', ): 0.17264142227467427,
            ('a', 'b'): 0.0075129011907618,
            ('a', ): 0.0032756406763107997,
            ('b', 'c'): 0.05621283952987573,
            ('a', 'b', 'c'): 0.03806939941258188
        }
        solved, results = solve_for_parents_with_root_finder(COMBINATION_METHODS["DEMPSTER_SHAFER"],
                                                             child_marginals, num_parents)
        self.assertFalse(solved is True,
                        msg="{} for num parents {} and num options {}".format(COMBINATION_METHODS["DEMPSTER_SHAFER"],
                                                                              num_parents, 3))
        # Combine forward to see if the results are valid
        combined_data = import_and_combine(COMBINATION_METHODS["DEMPSTER_SHAFER"], results)
        # Get the marginals
        combined_marginals = import_and_calculate_probabilities(COMBINATION_METHODS["DEMPSTER_SHAFER"], combined_data)
        # Check
        # Due to roundoff error in the root finding, other cases are not as accurate
        test_delta = TEST_DELTA
        for marginal_key, marginal_value in combined_marginals.items():
            self.assertAlmostEqual(marginal_value, child_marginals[marginal_key], delta=test_delta,
                                   msg="{} for num parents {} and num options {}".format(
                                       COMBINATION_METHODS["DEMPSTER_SHAFER"], num_parents, 3))

    def test_specific_reverse_solver_3(self):
        """
        Tests a specific known case of the reverse solver failing
        """
        num_parents = 4
        child_marginals = {
            ('a', 'b'): 2.4028357005915588e-05,
            ('a', 'b', 'c'): 2.997050756976387e-08,
            ('a', 'c'): 0.001662614220194217,
            ('a', ): 0.9824931820954771,
            ('b', 'c'): 7.841446128252039e-06,
            ('b', ): 5.21718572068943e-05,
            ('c', ): 0.01576013205348013
        }
        solved, results = solve_for_parents_with_root_finder(COMBINATION_METHODS["DEMPSTER_SHAFER"],
                                                             child_marginals, num_parents)
        self.assertTrue(solved is True,
                        msg="{} for num parents {} and num options {}".format(COMBINATION_METHODS["DEMPSTER_SHAFER"],
                                                                              num_parents, 3))
        # Combine forward to see if the results are valid
        combined_data = import_and_combine(COMBINATION_METHODS["DEMPSTER_SHAFER"], results)
        # Get the marginals
        combined_marginals = import_and_calculate_probabilities(COMBINATION_METHODS["DEMPSTER_SHAFER"], combined_data)
        # Check
        # Due to roundoff error in the root finding, other cases are not as accurate
        test_delta = TEST_DELTA
        for marginal_key, marginal_value in combined_marginals.items():
            self.assertAlmostEqual(marginal_value, child_marginals[marginal_key], delta=test_delta,
                                   msg="{} for num parents {} and num options {}".format(
                                       COMBINATION_METHODS["DEMPSTER_SHAFER"], num_parents, 3))

    def test_incremental_work(self):
        child_marginals = {
            ('c',): 0.004182517220232905,
            ('a', 'c'): 0.7181052796955626,
            ('b',): 0.17264142227467427,
            ('a', 'b'): 0.0075129011907618,
            ('a',): 0.0032756406763107997,
            ('b', 'c'): 0.05621283952987573,
            ('a', 'b', 'c'): 0.03806939941258188
        }

        evidence = {
            0: {
                ('a', 'b'): 0.01838630707425276,
                ('a', 'b', 'c'): 0.19511381143471593,
                ('a', 'c'): 0.6744692375652888,
                ('a',): -0.012204992232466294,
                ('b', 'c'): 0.11194031887858505,
                ('b',): 0.19842649457885103,
                ('c',): -0.07788071956109632
            },
            1: {
                ('a', 'b'): 0.01838630707425276,
                ('a', 'b', 'c'): 0.19511381143471593,
                ('a', 'c'): 0.6744692375652888,
                ('a',): -0.012204992232466294,
                ('b', 'c'): 0.11194031887858505,
                ('b',): 0.19842649457885103,
                ('c',): -0.07788071956109632
            }
        }
        issue_sets = [('a', ), ('c', )]
        universal_set = ('a', 'b', 'c')

        # The parent that isn't being changed to zeros is known.  The rest of the parents will need to change.
        value_states = {}
        for evidence_key, evidence_dict in evidence.items():
            value_states[evidence_key] = {}
            for evidence_marginal in evidence_dict.keys():
                value_states[evidence_key][evidence_marginal] = MARGINAL_STATE["FLEXIBLE"]

        for issue_set in issue_sets:
            sum_of_previously_solved_applicables = \
                previously_applicable_set_intersection(evidence, issue_set, universal_set, len(evidence) - 1, 1.0)
            child_marginal = child_marginals[issue_set]
            failed_frac = sum_of_previously_solved_applicables / child_marginal
            # Calculate the required offset (failed_delta / num_of_parents)
            frac = failed_frac * 10.0

            for marginal_key in evidence[0].keys():
                if (len(marginal_key) > len(issue_set)) and (len(marginal_key) < len(universal_set)) and \
                        (set(marginal_key).intersection(set(issue_set)) == set(issue_set)) and\
                        (value_states[0][marginal_key] != MARGINAL_STATE["SET"]):
                    # Change this one
                    evidence[0][marginal_key] /= frac
                    # Should not be changed after this
                    value_states[0][marginal_key] = MARGINAL_STATE["RECOMMENDED"]

            solved = constrained_solver_for_root_finder(evidence, value_states, child_marginals,
                                                        universal_set, len(issue_set))
            if solved is False:
                print("Cannot solve")
        # Normalize the evidence
        for evidence_key, evidence_dict in evidence.items():
            f = sum(list(evidence_dict.values()))
            if f != 0.0:
                for i in evidence_dict.keys():
                    evidence_dict[i] /= f
        # Forward combine
        combined_data = import_and_combine(COMBINATION_METHODS["DEMPSTER_SHAFER"], evidence)
        # Get the marginals
        combined_marginals = import_and_calculate_probabilities(COMBINATION_METHODS["DEMPSTER_SHAFER"], combined_data)
        test_delta = TEST_DELTA
        for marginal_key, marginal_value in child_marginals.items():
            self.assertAlmostEqual(marginal_value, combined_marginals[marginal_key], delta=test_delta,
                                   msg="{} for num parents {} and num options {}".format(
                                       COMBINATION_METHODS["DEMPSTER_SHAFER"], 2, 3))
        # Check the negatives
        for evidence_key, evidence_dict in evidence.items():
            for evidence_marginal, evidence_value in evidence_dict.items():
                self.assertTrue(evidence_value > -0.00001,
                                msg="Evidence {} and key {} has negative value {}".format(evidence_key,
                                                                                          evidence_marginal,
                                                                                          evidence_value))

    def test_incremental_work_2(self):
        child_marginals = {
            ('a', 'b'): 0.01038799918308785,
            ('a', 'b', 'c'): 1.1736188165662087e-05,
            ('a', 'c'): 0.07432717516894283,
            ('a', ): 0.7522994031953376,
            ('b', 'c'): 0.004691022290639421,
            ('b', ): 0.0016426366662198875,
            ('c', ): 0.15664002730760682
        }

        evidence = {
            0: {
                ('a', 'b'): 0.19555145682083577,
                ('a', 'b', 'c'): 0.022725268404332626,
                ('a', 'c'): 0.3977483306844805,
                ('a', ):  0.32639901562538814,
                ('b', 'c'): 0.14481436405367742,
                ('b', ): -0.1073141721699543,
                ('c', ): 0.05239848993573076
            },
            1: {
                ('a', 'b'): 0.19555145682083577,
                ('a', 'b', 'c'): 0.022725268404332626,
                ('a', 'c'): 0.3977483306844805,
                ('a',): 0.32639901562538814,
                ('b', 'c'): 0.14481436405367742,
                ('b',): -0.1073141721699543,
                ('c',): 0.05239848993573076
            },
            2: {
                ('a', 'b'): 0.19555145682083577,
                ('a', 'b', 'c'): 0.022725268404332626,
                ('a', 'c'): 0.3977483306844805,
                ('a',): 0.32639901562538814,
                ('b', 'c'): 0.14481436405367742,
                ('b',): -0.1073141721699543,
                ('c',): 0.05239848993573076
            }
        }
        issue_sets = [('b', )]
        universal_set = ('a', 'b', 'c')

        # The parent that isn't being changed to zeros is known.  The rest of the parents will need to change.
        value_states = {}
        for evidence_key, evidence_dict in evidence.items():
            value_states[evidence_key] = {}
            for evidence_marginal in evidence_dict.keys():
                value_states[evidence_key][evidence_marginal] = MARGINAL_STATE["FLEXIBLE"]

        for issue_set in issue_sets:
            sum_of_previously_solved_applicables = \
                previously_applicable_set_intersection(evidence, issue_set, universal_set, len(evidence) - 1, 1.0)
            child_marginal = child_marginals[issue_set]
            failed_frac = sum_of_previously_solved_applicables / child_marginal
            # Calculate the required offset (failed_delta / num_of_parents)
            frac = failed_frac * 10.0

            for marginal_key in evidence[0].keys():
                if (len(marginal_key) > len(issue_set)) and \
                        (set(marginal_key).intersection(set(issue_set)) == set(issue_set)) and\
                        (value_states[0][marginal_key] != MARGINAL_STATE["SET"]):
                    # Change this one
                    evidence[0][marginal_key] /= frac
                    # Should not be changed after this
                    value_states[0][marginal_key] = MARGINAL_STATE["RECOMMENDED"]

            solved = constrained_solver_for_root_finder(evidence, value_states, child_marginals,
                                                        universal_set, len(issue_set))
            if solved is False:
                print("Cannot solve")
        # Normalize the evidence
        for evidence_key, evidence_dict in evidence.items():
            f = sum(list(evidence_dict.values()))
            if f != 0.0:
                for i in evidence_dict.keys():
                    evidence_dict[i] /= f
        # Forward combine
        combined_data = import_and_combine(COMBINATION_METHODS["DEMPSTER_SHAFER"], evidence)
        # Get the marginals
        combined_marginals = import_and_calculate_probabilities(COMBINATION_METHODS["DEMPSTER_SHAFER"], combined_data)
        test_delta = TEST_DELTA
        for marginal_key, marginal_value in child_marginals.items():
            self.assertAlmostEqual(marginal_value, combined_marginals[marginal_key], delta=test_delta,
                                   msg="{} for num parents {} and num options {}".format(
                                       COMBINATION_METHODS["DEMPSTER_SHAFER"], 2, 3))
        # Check the negatives - expected to be negative for this test since this solution doesn't work.
        for evidence_key, evidence_dict in evidence.items():
            for evidence_marginal, evidence_value in evidence_dict.items():
                if evidence_value < -0.00001:
                    print("Evidence {} and key {} has negative value {}".format(evidence_key,
                                                                                evidence_marginal,
                                                                                evidence_value))

    def test_incremental_work_3(self):
        child_marginals = {
            ('a', 'b'): 0.00051857921539869,
            ('a', 'b', 'c'): 7.739704790559122e-07,
            ('a', 'c'): 0.16475656790296164,
            ('a', ): 0.659153954057475,
            ('b', 'c'): 4.796653747988763e-05,
            ('b', ): 0.0004127840977211419,
            ('c', ): 0.17510937421848466
        }

        original_solution = {
            0: {
                ('a', 'b'): 0.37522174128116775,
                ('a', 'b', 'c'): 0.004592800397248642,
                ('a', 'c'): 0.4031079120761195,
                ('a',): 0.07619677947860899,
                ('b', 'c'): 0.10598426442084881,
                ('b',): 0.020588896286661103,
                ('c',): 0.014307606059345172
            },
            1: {
                ('a', 'b'): 0.001324168679468981,
                ('a', 'b', 'c'): 0.052168340847835025,
                ('a', 'c'): 0.30732969052672887,
                ('a',): 0.4746913091304802,
                ('b', 'c'): 0.004605054159360209,
                ('b',): 0.005132892677836978,
                ('c',): 0.15474854397828977
            },
            2: {
                ('a', 'b'): 0.017643332120394775,
                ('a', 'b', 'c'): 0.0025520799782750197,
                ('a', 'c'): 0.8855466933208422,
                ('a',): 0.001663135955337057,
                ('b', 'c'): 0.003581790598083251,
                ('b',): 0.0004061826538267447,
                ('c',): 0.08860678537324082
            }
        }

        evidence = {
            0: {
                ('a', 'b'): 0.38,
                ('a', 'b', 'c'): 0.0092,
                ('a', 'c'): 0.40,
                ('a', ): 0.076,
                ('b', 'c'): 0.11,
                ('b', ): 0.021,
                ('c', ): 0.014
            },
            1: {
                ('a', 'b'): 0.0013,
                ('a', 'b', 'c'): 0.07,
                ('a', 'c'): 0.31,
                ('a', ): 0.47,
                ('b', 'c'): 0.0046,
                ('b', ): 0.0051,
                ('c', ): 0.15
            },
            2: {
                ('a', 'b'): 0.017643332120394775,
                ('a', 'b', 'c'): 0.0025520799782750197,
                ('a', 'c'): 0.8855466933208422,
                ('a', ): 0.001663135955337057,
                ('b', 'c'): 0.003581790598083251,
                ('b', ): 0.0004061826538267447,
                ('c', ): 0.08860678537324082
            }
        }
        universal_set = ('a', 'b', 'c')

        solved, results = solve_for_parents_with_root_finder(COMBINATION_METHODS["DEMPSTER_SHAFER"],
                                                             child_marginals, 3)

        # The parent that isn't being changed to zeros is known.  The rest of the parents will need to change.
        value_states = {}
        for counter in range(0, 3):
            value_states[counter] = {}
            for evidence_marginal in evidence[counter].keys():
                if counter < 2:
                    value_states[counter][evidence_marginal] = MARGINAL_STATE["RECOMMENDED"]
                else:
                    value_states[counter][evidence_marginal] = MARGINAL_STATE["FLEXIBLE"]

        solved = constrained_solver_for_root_finder(evidence, value_states, child_marginals, universal_set, 1)
        if solved is False:
            print("Cannot solve")
        # Normalize the evidence
        for evidence_key, evidence_dict in evidence.items():
            f = sum(list(evidence_dict.values()))
            if f != 0.0:
                for i in evidence_dict.keys():
                    evidence_dict[i] /= f
        # Forward combine
        combined_data = import_and_combine(COMBINATION_METHODS["DEMPSTER_SHAFER"], evidence)
        # Get the marginals
        combined_marginals = import_and_calculate_probabilities(COMBINATION_METHODS["DEMPSTER_SHAFER"], combined_data)
        test_delta = TEST_DELTA
        for marginal_key, marginal_value in child_marginals.items():
            self.assertAlmostEqual(marginal_value, combined_marginals[marginal_key], delta=test_delta,
                                   msg="{} for num parents {} and num options {}".format(
                                       COMBINATION_METHODS["DEMPSTER_SHAFER"], 2, 3))
        # Check the negatives
        for evidence_key, evidence_dict in evidence.items():
            for evidence_marginal, evidence_value in evidence_dict.items():
                self.assertTrue(evidence_value > -0.00001,
                                msg="Evidence {} and key {} has negative value {}".format(evidence_key,
                                                                                          evidence_marginal,
                                                                                          evidence_value))

    def run_reverse_solver(self, combination_methods, max_number_of_parents, options, num_tests, defined_combined):
        """
        Runs the reverse solver and tests.  Called from another function
        :param combination_methods: list of COMBINATION_METHODS
        :param max_number_of_parents: int maximum number of parents (> 1)
        :param options: list of marginal keys
        :param num_tests: int number of tests to perform per combination method and max_number_of_parents
        :param defined_combined: MARGINAL_DEFINES whether to started with combined children, individual, or limited
        """
        # Define the powerset of the options
        thetas = list(reduce(lambda a, b: a | set(the_keys(b)), options, set()))
        # The global ignorance set might be in here
        powerset = [tuple(sorted([x for j, x in enumerate(thetas) if (i >> j) & 1])) for i in range(2 ** len(thetas))]
        # Remove the null set from the powerset
        powerset.remove(())

        # Go through all options and randomly pick the values to test
        for test_number in range(0, num_tests):
            # Iterate through the number of parents
            for number_of_parents in range(2, max_number_of_parents + 1):
                # Generate the individual evidences
                test_evidence = {}
                if defined_combined == MARGINAL_DEFINES["INDIVIDUAL"]:
                    for evidence_num in range(0, number_of_parents):
                        total_sum = 0
                        test_evidence[evidence_num] = {}
                        last_marginal_key = None
                        random.shuffle(powerset)
                        for marginal_key in powerset:
                            test_evidence[evidence_num][marginal_key] = random.random() * (1.0 - total_sum)
                            total_sum += test_evidence[evidence_num][marginal_key]
                            last_marginal_key = marginal_key
                        if total_sum < 1.0:
                            # Make sure it adds to 1.0
                            test_evidence[evidence_num][last_marginal_key] += 1.0 - total_sum
                elif defined_combined == MARGINAL_DEFINES["COMBINED"]:
                    # Use same evidence for all
                    total_sum = 0
                    test_evidence[0] = {}
                    last_marginal_key = None
                    random.shuffle(powerset)
                    for marginal_key in powerset:
                        test_evidence[0][marginal_key] = random.random() * (1.0 - total_sum)
                        total_sum += test_evidence[0][marginal_key]
                        last_marginal_key = marginal_key
                    if total_sum < 1.0:
                        # Make sure it adds to 1.0
                        test_evidence[0][last_marginal_key] += 1.0 - total_sum
                    # Copy to all others
                    for evidence_num in range(1, number_of_parents):
                        test_evidence[evidence_num] = deepcopy(test_evidence[0])
                else:
                    # For testing various limitation methods
                    total_max = random.random()  # Ensures that a broader range of tests get run
                    for evidence_num in range(0, number_of_parents):
                        total_sum = 0
                        test_evidence[evidence_num] = {}
                        powerset.sort(key=len)
                        universal_set = powerset[len(powerset) - 1]
                        for marginal_key in powerset:
                            # First find max value
                            max_value = total_max - total_sum
                            if set(marginal_key) != set(universal_set):
                                for previous_key in test_evidence[evidence_num].keys():
                                    if (set(previous_key).intersection(set(marginal_key)) == set(previous_key)) and\
                                            (test_evidence[evidence_num][previous_key] > ZERO_DELTA):
                                        # Is a subset and is not zero
                                        max_value = min(max_value, test_evidence[evidence_num][previous_key])
                            test_evidence[evidence_num][marginal_key] = random.random() * max_value
                            total_sum += test_evidence[evidence_num][marginal_key]
                        if total_sum < 1.0:
                            # Make sure it adds to 1.0
                            for marginal_key in test_evidence[evidence_num].keys():
                                test_evidence[evidence_num][marginal_key] /= total_sum
                # Iterate through the combination methods using the same set of evidences
                for combination_method in combination_methods:
                    # Combine forward to create a valid set of child marginals to test.
                    evidence = {}
                    for evidence_num in range(0, number_of_parents):
                        evidence[evidence_num] = deepcopy(test_evidence[evidence_num])
                    # Forward combine
                    combined_data = import_and_combine(combination_method, evidence)
                    # Get the marginals
                    combined_marginals = import_and_calculate_probabilities(combination_method, combined_data)
                    # Handle roundoff error since negatives mess up the reverse solver.
                    for marginal_key, marginal_value in combined_marginals.items():
                        combined_marginals[marginal_key] = max(min(marginal_value, 1.0), 0.0)

                    solved, results = solve_for_parents_with_root_finder(combination_method,
                                                                         combined_marginals,
                                                                         number_of_parents)
                    self.assertTrue(solved is True,
                                    msg="{} for num parents {} and num options {}".format(combination_method,
                                                                                          number_of_parents,
                                                                                          len(options)))
                    # Forward combine again the results
                    evidence = {}
                    for evidence_num in range(0, number_of_parents):
                        evidence[evidence_num] = deepcopy(results[evidence_num])
                    # Forward combine
                    combined_data = import_and_combine(combination_method, evidence)
                    # Get the marginals
                    results_combined_marginals = import_and_calculate_probabilities(combination_method, combined_data)
                    # Check
                    # Due to roundoff error in the root finding, other cases are not as accurate
                    test_delta = TEST_DELTA
                    for marginal_key, marginal_value in results_combined_marginals.items():
                        self.assertAlmostEqual(marginal_value, combined_marginals[marginal_key], delta=test_delta,
                                               msg="{} for num parents {} and num options {}".format(combination_method,
                                                                                                     number_of_parents,
                                                                                                     len(options)))

    def test_dempster_shafer_reverse_solver(self):
        """
        Tests whether solutions can be obtained for a variety of situations with Dempster-Shafer, Yager, and Murphy
        """
        # Run the two combination methods that work easily first, then run the two that may have to have disparate
        #  parent marginals
        solution_method = [COMBINATION_METHODS["MURPHY"],
                           COMBINATION_METHODS["ZHANG"]]
        # Go up to 5 parents.
        max_parents = 5
        num_tests = 20
        # Two option set
        options = ['a', 'b']
        self.run_reverse_solver(solution_method, max_parents, options, num_tests,
                                defined_combined=MARGINAL_DEFINES["INDIVIDUAL"])

        # Three option set
        options = ['a', 'b', 'c']
        self.run_reverse_solver(solution_method, max_parents, options, num_tests,
                                defined_combined=MARGINAL_DEFINES["INDIVIDUAL"])

        # Four option set
        options = ['a', 'b', 'c', 'd']
        self.run_reverse_solver(solution_method, max_parents, options, num_tests,
                                defined_combined=MARGINAL_DEFINES["INDIVIDUAL"])

        # For methods that can't handle individual parents
        solution_method = [COMBINATION_METHODS["DEMPSTER_SHAFER"]]
        # Two option set
        options = ['a', 'b']
        self.run_reverse_solver(solution_method, max_parents, options, num_tests,
                                defined_combined=MARGINAL_DEFINES["INDIVIDUAL"])  # Two options is not an issue

        # Three option set
        options = ['a', 'b', 'c']
        self.run_reverse_solver(solution_method, max_parents, options, num_tests,
                                defined_combined=MARGINAL_DEFINES["LIMITED"])

        # Four option set
        options = ['a', 'b', 'c', 'd']
        self.run_reverse_solver(solution_method, max_parents, options, num_tests,
                                defined_combined=MARGINAL_DEFINES["LIMITED"])

    def test_specific_case_for_ds_solver(self):
        """
        Creates a 2-node DS network and tests
        """
        test_network = DSNetwork()
        p_node = Node()
        p_node.name = "P"
        p_node.type = NODE_TYPE["CATEGORIES"]["INPUT"]
        p_node.options = ["C_O1", "C_O2", "C_O3"]

        c_node = Node()
        c_node.name = "C"
        c_node.type = NODE_TYPE["CATEGORIES"]["INPUT"]
        c_node.options = ["G_O1", "G_O2", "G_O3"]

        pc_transition = Transition(p_node, c_node)
        test_network.add_link_and_initialize([p_node, c_node], [pc_transition],
                                             combination_method=COMBINATION_METHODS["ZHANG"],
                                             weighting_method=WEIGHTING_METHOD["CATEGORIES"]["TO_ONE"])
        p_marginals = {
            ("C_O2",): 0.084063,
            ("C_O1", "C_O3"): 0.031115,
            ("C_O1", "C_O2"): 0.029386,
            ("C_O3",): 0.069411,
            ("C_O1", "C_O2", "C_O3"): 0.015064,
            ("C_O1",): 0.747977,
            ("C_O2", "C_O3"): 0.022983
        }
        # Overwrite for specific values
        test_network.nodes["P"].set_internal_probability_data(p_marginals)

        c_marginals = {
            ("G_O2",): 0.001518,
            ("G_O1",): 0.990946,
            ("G_O3",): 0.000003,
            ("G_O1", "G_O2", "G_O3"): 0.000894,
            ("G_O1", "G_O3"): 0.000002,
            ("G_O2", "G_O3"): 0.000007,
            ("G_O1", "G_O2"): 0.006629
        }
        # Overwrite for specific values
        test_network.nodes["C"].set_internal_probability_data(c_marginals)

        # Set the current transition potentials; otherwise LS will give different results
        test_network.transitions["P.C"].conditionals_parent_to_child = {
            ("C_O2",): {
                ("G_O2",): 0.220580,
                ("G_O1",): 0.552859,
                ("G_O3",): 0.186522,
                ("G_O1", "G_O2", "G_O3"): 0.039436,
                ("G_O1", "G_O3"): 0.000163,
                ("G_O2", "G_O3"): 0.000349,
                ("G_O1", "G_O2"): 0.000093
            },
            ("C_O1", "C_O3"): {
                ("G_O2",): 0.203441,
                ("G_O1",): 0.567417,
                ("G_O3",): 0.188754,
                ("G_O1", "G_O2", "G_O3"): 0.039759,
                ("G_O1", "G_O3"): 0.000169,
                ("G_O2", "G_O3"): 0.000363,
                ("G_O1", "G_O2"): 0.000096
            },
            ("C_O1", "C_O2"): {
                ("G_O2",): 0.108735,
                ("G_O1",): 0.650181,
                ("G_O3",): 0.198900,
                ("G_O1", "G_O2", "G_O3"): 0.041433,
                ("G_O1", "G_O3"): 0.000202,
                ("G_O2", "G_O3"): 0.000434,
                ("G_O1", "G_O2"): 0.000115
            },
            ("C_O3",): {
                ("G_O2",): 0.177873,
                ("G_O1",): 0.590490,
                ("G_O3",): 0.190804,
                ("G_O1", "G_O2", "G_O3"): 0.040175,
                ("G_O1", "G_O3"): 0.000177,
                ("G_O2", "G_O3"): 0.000379,
                ("G_O1", "G_O2"): 0.000101
            },
            ("C_O1", "C_O2", "C_O3"): {
                ("G_O2",): 0.292638,
                ("G_O1",): 0.492045,
                ("G_O3",): 0.176761,
                ("G_O1", "G_O2", "G_O3"): 0.038059,
                ("G_O1", "G_O3"): 0.000134,
                ("G_O2", "G_O3"): 0.000286,
                ("G_O1", "G_O2"): 0.000076
            },
            ("C_O1",): {
                ("G_O2",): 0.000000,
                ("G_O1",): 0.978257,
                ("G_O3",): 0.021743,
                ("G_O1", "G_O2", "G_O3"): 0.000000,
                ("G_O1", "G_O3"): 0.000000,
                ("G_O2", "G_O3"): 0.000000,
                ("G_O1", "G_O2"): 0.000000
            },
            ("C_O2", "C_O3"): {
                ("G_O2",): 0.000000,
                ("G_O1",): 0.761016,
                ("G_O3",): 0.192534,
                ("G_O1", "G_O2", "G_O3"): 0.042863,
                ("G_O1", "G_O3"): 0.000958,
                ("G_O2", "G_O3"): 0.002077,
                ("G_O1", "G_O2"): 0.000553
            }
        }

        # Learn
        test_network.transitions["P.C"].rebalance_potentials(debug=False)

    def test_specific_case_2_for_ds_solver(self):
        """
        Creates a 2-node DS network and tests
        """
        test_network = DSNetwork()
        p_node = Node()
        p_node.name = "P"
        p_node.type = NODE_TYPE["CATEGORIES"]["INPUT"]
        p_node.options = ["C_O1", "C_O2", "C_O3"]

        c_node = Node()
        c_node.name = "C"
        c_node.type = NODE_TYPE["CATEGORIES"]["INPUT"]
        c_node.options = ["G_O1", "G_O2", "G_O3"]

        pc_transition = Transition(p_node, c_node)
        test_network.add_link_and_initialize([p_node, c_node], [pc_transition],
                                             combination_method=COMBINATION_METHODS["ZHANG"],
                                             weighting_method=WEIGHTING_METHOD["CATEGORIES"]["TO_ONE"])
        p_marginals = {
            ("C_O2",): 0.011018,
            ("C_O1", "C_O3"): 0.078597,
            ("C_O1", "C_O2"): 0.001266,
            ("C_O3",): 0.188667,
            ("C_O1", "C_O2", "C_O3"): 0.000095,
            ("C_O1",): 0.711591,
            ("C_O2", "C_O3"): 0.008767
        }
        # Overwrite for specific values
        test_network.nodes["P"].set_internal_probability_data(p_marginals)

        c_marginals = {
            ("G_O2",): 0.977065,
            ("G_O1",): 0.001096,
            ("G_O3",): 0.005891,
            ("G_O1", "G_O2", "G_O3"): 0.000033,
            ("G_O1", "G_O3"): 0.000025,
            ("G_O2", "G_O3"): 0.015883,
            ("G_O1", "G_O2"): 0.000007
        }
        # Overwrite for specific values
        test_network.nodes["C"].set_internal_probability_data(c_marginals)

        # Set the current transition potentials; otherwise LS will give different results
        test_network.transitions["P.C"].conditionals_parent_to_child = {
            ("C_O2",): {
                ("G_O2",): 0.973339,
                ("G_O1",): 0.002788,
                ("G_O3",): 0.002076,
                ("G_O1", "G_O2", "G_O3"): 0.010522,
                ("G_O1", "G_O3"): 0.000271,
                ("G_O2", "G_O3"): 0.010767,
                ("G_O1", "G_O2"): 0.000238
            },
            ("C_O1", "C_O3"): {
                ("G_O2",): 0.203441,
                ("G_O1",): 0.567417,
                ("G_O3",): 0.188754,
                ("G_O1", "G_O2", "G_O3"): 0.039759,
                ("G_O1", "G_O3"): 0.000169,
                ("G_O2", "G_O3"): 0.000363,
                ("G_O1", "G_O2"): 0.000096
            },
            ("C_O1", "C_O2"): {
                ("G_O2",): 0.988039,
                ("G_O1",): 0.001195,
                ("G_O3",): 0.000863,
                ("G_O1", "G_O2", "G_O3"): 0.005311,
                ("G_O1", "G_O3"): 0.000124,
                ("G_O2", "G_O3"): 0.004347,
                ("G_O1", "G_O2"): 0.000121
            },
            ("C_O3",): {
                ("G_O2",): 0.971972,
                ("G_O1",): 0.002901,
                ("G_O3",): 0.002126,
                ("G_O1", "G_O2", "G_O3"): 0.011569,
                ("G_O1", "G_O3"): 0.000291,
                ("G_O2", "G_O3"): 0.010872,
                ("G_O1", "G_O2"): 0.000270
            },
            ("C_O1", "C_O2", "C_O3"): {
                ("G_O2",): 0.995950,
                ("G_O1",): 0.000401,
                ("G_O3",): 0.000288,
                ("G_O1", "G_O2", "G_O3"): 0.001830,
                ("G_O1", "G_O3"): 0.000042,
                ("G_O2", "G_O3"): 0.001447,
                ("G_O1", "G_O2"): 0.000042
            },
            ("C_O1",): {
                ("G_O2",): 0.785496,
                ("G_O1",): 0.031929,
                ("G_O3",): 0.026248,
                ("G_O1", "G_O2", "G_O3"): 0.004259,
                ("G_O1", "G_O3"): 0.002270,
                ("G_O2", "G_O3"): 0.149090,
                ("G_O1", "G_O2"): 0.000708
            },
            ("C_O2", "C_O3"): {
                ("G_O2",): 0.988039,
                ("G_O1",): 0.001195,
                ("G_O3",): 0.000863,
                ("G_O1", "G_O2", "G_O3"): 0.005312,
                ("G_O1", "G_O3"): 0.000124,
                ("G_O2", "G_O3"): 0.004347,
                ("G_O1", "G_O2"): 0.000121
            }
        }

        # Learn
        test_network.transitions["P.C"].rebalance_potentials(debug=False)

    def test_specific_case_3_for_ds_solver(self):
        """
        Creates a 2-node DS network and tests
        """
        test_network = DSNetwork()
        p_node = Node()
        p_node.name = "P"
        p_node.type = NODE_TYPE["CATEGORIES"]["INPUT"]
        p_node.options = ["B_O1", "B_O2"]

        c_node = Node()
        c_node.name = "C"
        c_node.type = NODE_TYPE["CATEGORIES"]["INPUT"]
        c_node.options = ["E_O1", "E_O2"]

        pc_transition = Transition(p_node, c_node)
        test_network.add_link_and_initialize([p_node, c_node], [pc_transition],
                                             combination_method=COMBINATION_METHODS["ZHANG"],
                                             weighting_method=WEIGHTING_METHOD["CATEGORIES"]["TO_ONE"])
        p_marginals = {
            ("B_O1",): 0.199829,
            ("B_O2",): 0.787300,
            ("B_O1", "B_O2"): 0.012872
        }
        # Overwrite for specific values
        test_network.nodes["P"].set_internal_probability_data(p_marginals)

        c_marginals = {
            ("E_O1",): 0.002742,
            ("E_O2",): 0.995609,
            ("E_O1", "E_O2"): 0.001649
        }
        # Overwrite for specific values
        test_network.nodes["C"].set_internal_probability_data(c_marginals)

        # Set the current transition potentials; otherwise LS will give different results
        test_network.transitions["P.C"].conditionals_parent_to_child = {
            ("B_O1",): {
                ("E_O1",): 0.000040,
                ("E_O2",): 0.888450,
                ("E_O1", "E_O2"): 0.111510
            },
            ("B_O2",): {
                ("E_O1",): 0.035253,
                ("E_O2",): 0.946713,
                ("E_O1", "E_O2"): 0.018034
            },
            ("B_O1", "B_O2"): {
                ("E_O1",): 0.000419,
                ("E_O2",): 0.999581,
                ("E_O1", "E_O2"): 0.000000
            }
        }

        # Learn
        test_network.transitions["P.C"].rebalance_potentials(debug=False)

    def test_specific_case_for_presentation(self):
        """
        Creates a 2-node DS network and tests
        """
        test_network = DSNetwork()
        p_node = Node()
        p_node.name = "P"
        p_node.type = NODE_TYPE["CATEGORIES"]["INPUT"]
        p_node.options = ["B_O1", "B_O2"]

        c_node = Node()
        c_node.name = "C"
        c_node.type = NODE_TYPE["CATEGORIES"]["INPUT"]
        c_node.options = ["E_O1", "E_O2"]

        pc_transition = Transition(p_node, c_node)
        test_network.add_link_and_initialize([p_node, c_node], [pc_transition],
                                             combination_method=COMBINATION_METHODS["ZHANG"],
                                             weighting_method=WEIGHTING_METHOD["CATEGORIES"]["TO_ONE"])
        p_marginals = {
            ("B_O1",): 0.2,
            ("B_O2",): 0.5,
            ("B_O1", "B_O2"): 0.3
        }
        # Overwrite for specific values
        test_network.nodes["P"].set_internal_probability_data(p_marginals)

        c_marginals = {
            ("E_O1",): 0.5,
            ("E_O2",): 0.3,
            ("E_O1", "E_O2"): 0.2
        }
        # Overwrite for specific values
        test_network.nodes["C"].set_internal_probability_data(c_marginals)

        # Set the current transition potentials; otherwise LS will give different results
        test_network.transitions["P.C"].conditionals_parent_to_child = {
            ("B_O1",): {
                ("E_O1",): 0.0,
                ("E_O2",): 0.0,
                ("E_O1", "E_O2"): 1.0
            },
            ("B_O2",): {
                ("E_O1",): 0.0,
                ("E_O2",): 0.0,
                ("E_O1", "E_O2"): 1.0
            },
            ("B_O1", "B_O2"): {
                ("E_O1",): 0.0,
                ("E_O2",): 0.0,
                ("E_O1", "E_O2"): 1.0
            }
        }

        # Learn
        test_network.transitions["P.C"].rebalance_potentials(debug=False)

    def set_transitions_for_no_learning(self, network_dict):
        """
        Duplicated from testForPaper because Python is too stupid to allow me to re-use the functions without trying
         to run all the tests there also
        """
        combination_name_in_use = list(network_dict.keys())[0]
        # All the networks are the same, but with different combination methods
        for transition_name in network_dict[combination_name_in_use].transitions.keys():
            # Only set some potentials, not all.
            if random.random() < PROBABILITY_TO_SET_POTENTIAL:
                # Set the transition potentials
                for parent_key in network_dict[combination_name_in_use].transitions[transition_name]. \
                        conditionals_parent_to_child.keys():
                    potential_sum = 0.0
                    l_child_key = None  # Last child key
                    # Randomly set the potentials, making sure they don't go above a sum of 1.0
                    for child_key in network_dict[combination_name_in_use].transitions[transition_name]. \
                            conditionals_parent_to_child[parent_key].keys():
                        potential = random.random() * (1.0 - potential_sum)
                        # Set them all in here
                        for network in network_dict.values():
                            network.transitions[transition_name].conditionals_parent_to_child[parent_key][child_key] = \
                                potential
                        potential_sum += potential
                        l_child_key = child_key
                    # Handle cases (almost all cases) in which the total potential does not sum to 1.0
                    if (potential_sum < 1.0) and (l_child_key is not None):
                        for network in network_dict.values():
                            network.transitions[transition_name].conditionals_parent_to_child[parent_key][
                                l_child_key] += \
                                1.0 - potential_sum

    def create_updates(self, network):
        """
        Duplicated from testForPaper because Python is too stupid to allow me to re-use the functions without trying
         to run all the tests there also
        """
        updates = []
        for update_counter in range(0, NUMBER_OF_UPDATES_PER_TEST):
            set_of_updates = {}
            for simultaneous_counter in range(0, NUMBER_OF_SIMULTANEOUS_UPDATES):
                each_update = {}
                node_num = random.randint(0, len(network.nodes.keys()) - 1)
                node_name = list(network.nodes.keys())[node_num]  # Get a random node name
                each_update["evidence"] = {}
                marginals = network.nodes[node_name].get_marginals()  # Get the full list of keys to input for masses
                # Define the update masses
                sum_input = 0.0
                last_marginal_key = None
                # Shuffle the marginals each time
                keys_list = list(marginals.keys())
                random.shuffle(keys_list)
                for marginal_key in keys_list:
                    input_mass = random.random() * (1.0 - sum_input)
                    each_update["evidence"][marginal_key] = input_mass
                    sum_input += input_mass
                    last_marginal_key = marginal_key
                # Make sure each update sums to 1.0
                if (sum_input < 1.0) and (last_marginal_key is not None):
                    each_update["evidence"][last_marginal_key] += 1.0 - sum_input
                # Add to the set of updates
                set_of_updates[node_name] = each_update
            # Add to the full list of updates
            updates.append(set_of_updates)
        # Return the list of updates
        return updates

    def test_non_inference(self):
        """
        Testing whether non-inference updates change the data
        """
        tree = self.new_generator.complex_evaluation_test_tree
        tree.set_combination_and_weighting_methods(combination_method=COMBINATION_METHODS["MURPHY"],
                                                   weighting_method=WEIGHTING_METHOD["CATEGORIES"]["TOTAL"],
                                                   override=True)
        tree.initialize_unknown()
        tree.enable_learning(False)
        # Initialize all transitions and nodes for this test
        self.set_transitions_for_no_learning({COMBINATION_METHODS["MURPHY"]: tree})
        updates = self.create_updates(tree)
        for update_number in range(0, NUMBER_OF_UPDATES_PER_TEST):
            tree.input_evidence("TestInputs", updates[update_number], input_weight=1.0)
        # Should have a fully set up network now.
        before_no_inference_update_marginals = {
            "K": tree.nodes["K"].get_marginals(),
            "D": tree.nodes["D"].get_marginals()
        }
        # Non-inference update
        prior_consistency = evaluate_network_consistency(tree)
        tree.update_network_without_inference()
        after_no_inference_update_marginals = {
            "K": tree.nodes["K"].get_marginals(),
            "D": tree.nodes["D"].get_marginals()
        }
        current_consistency = evaluate_network_consistency(tree)
        # Compare to see if they agree
        for each_node_name, each_node_marginals in before_no_inference_update_marginals.items():
            for marginal_key, marginal_value in each_node_marginals.items():
                self.assertAlmostEqual(marginal_value,
                                       after_no_inference_update_marginals[each_node_name][marginal_key],
                                       delta=prior_consistency,
                                       msg="Marginal key {} not equal after no inference update with ".format(marginal_key)
                                           + "prior consistency {} and current consistency {}".format(prior_consistency,
                                                                                                      current_consistency))
