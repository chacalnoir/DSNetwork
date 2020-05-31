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

"""
Run tests to produce output for the paper about this network
"""

from network.NetworkGenerator import NetworkGenerator
from network.DSNetwork import WEIGHTING_METHOD, MULTI_PARENT_SOLUTION_TYPE
from network import evaluate_network_consistency
from combinationRules import COMBINATION_METHODS
import random
import timeit

PROBABILITY_TO_SET_POTENTIAL = 0.8
NUMBER_OF_TESTS_PER_CASE = 30
NUMBER_OF_UPDATES_PER_TEST = 30
NUMBER_OF_SIMULTANEOUS_UPDATES = 3


def set_transitions_for_no_learning(network_dict):
    """
    Sets transitions in the given network for updating without learning
    Randomly selects which transition potentials to set, and randomly sets the potentials.
    Because this shouldn't affect the speed of computation or whether there is consistency afterwards,
     it can remain random throughout the tests.

    :param network_dict a dict of combination method to DSNetwork for which to set transition potentials
    """
    combination_name_in_use = list(network_dict.keys())[0]
    # All the networks are the same, but with different combination methods
    for transition_name in network_dict[combination_name_in_use].transitions.keys():
        # Only set some potentials, not all.
        if random.random() < PROBABILITY_TO_SET_POTENTIAL:
            # Set the transition potentials
            for parent_key in network_dict[combination_name_in_use].transitions[transition_name].\
                    conditionals_parent_to_child.keys():
                potential_sum = 0.0
                l_child_key = None  # Last child key
                # Randomly set the potentials, making sure they don't go above a sum of 1.0
                for child_key in network_dict[combination_name_in_use].transitions[transition_name].\
                        conditionals_parent_to_child[parent_key].keys():
                    potential = random.random() * (1.0 - potential_sum)
                    # Set them all in here
                    for network in network_dict.values():
                        network.transitions[transition_name].conditionals_parent_to_child[parent_key][child_key] =\
                            potential
                    potential_sum += potential
                    l_child_key = child_key
                # Handle cases (almost all cases) in which the total potential does not sum to 1.0
                if (potential_sum < 1.0) and (l_child_key is not None):
                    for network in network_dict.values():
                        network.transitions[transition_name].conditionals_parent_to_child[parent_key][l_child_key] +=\
                            1.0 - potential_sum


def create_updates(network, inject_unknown=False):
    """
    Creates all the updates for a single parent test.  Needs a network to understand what updates are available.
    :param network: An example network for creating updates
    :param inject_unknown: bool whether to inject unknowns as part of evidence or to not to better see what is learnt
    """
    updates = []
    for update_counter in range(0, NUMBER_OF_UPDATES_PER_TEST):
        set_of_updates = {}
        node_name_options = list(network.nodes.keys())
        for simultaneous_counter in range(0, NUMBER_OF_SIMULTANEOUS_UPDATES):
            each_update = {}
            node_num = random.randint(0, len(node_name_options) - 1)
            node_name = node_name_options[node_num]  # Get a random node name
            node_name_options.remove(node_name)  # Make sure it's used only once
            each_update["evidence"] = {}
            marginals = network.nodes[node_name].get_marginals()  # Get the full list of keys to input for masses
            # Define the update masses
            sum_input = 0.0
            last_marginal_key = None
            # Shuffle the marginals for better distributions
            keys_list = list(marginals.keys())
            if inject_unknown is False:
                # Do not add unknown information in order to make it very clear what is actually learnt
                universal_key = None
                for marginal_key in keys_list:
                    if (universal_key is None) or (len(marginal_key) > len(universal_key)):
                        universal_key = marginal_key
                if universal_key is not None:
                    keys_list.remove(universal_key)
                    each_update["evidence"][universal_key] = 0.0
            random.shuffle(keys_list)
            for marginal_key in keys_list:
                input_mass = random.random() * (1.0 - sum_input)
                each_update["evidence"][marginal_key] = min(max(input_mass, 0.0), 1.0)  # Make sure stays within bounds
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


def evaluate_network_unknown(network):
    """
    Determines the fraction of complete unknown in nodes and transitions
    :param network: a DSNetwork to evaluate
    :return: float fraction of complete unknown in nodes and transitions
    """
    total_size = 0.0
    total_unknown = 0.0
    for each_node in network.nodes.values():
        node_marginals = each_node.get_marginals()
        # Find the unknown (universal set) and associated mass
        universal_mass = 0.0
        universal_set = None
        for marginal_key, marginal_mass in node_marginals.items():
            if (universal_set is None) or (len(marginal_key) > len(universal_set)):
                universal_set = marginal_key
                universal_mass = marginal_mass
        total_size += 1.0  # Since the masses always sum to this
        total_unknown += universal_mass
    for each_transition in network.transitions.values():
        potentials = each_transition.conditionals_parent_to_child
        for child_potentials_per_parent in potentials.values():
            # Find the unknown (universal set) and associated mass
            universal_mass = 0.0
            universal_set = None
            for potential_key, potential_mass in child_potentials_per_parent.items():
                if (universal_set is None) or (len(potential_key) > len(universal_set)):
                    universal_set = potential_key
                    universal_mass = potential_mass
            total_size += 1.0  # Since the potential masses per parent always sum to this
            total_unknown += universal_mass
    return total_unknown / total_size


# noinspection PyBroadException
def run_test(trees, updates, output_filename, test_name, simultaneous_updates, input_weight,
             initialize_transitions=False):
    """
    Runs the test and outputs the results to CSV
    :param trees: dict of the trees to run
    :param updates: dict of the updates to perform, or None if the updates should be generated here
    :param output_filename: string filename to output CSV results
    :param test_name: string test name
    :param simultaneous_updates: bool of whether to update simultaneously or serially
    :param input_weight: weight of each input (0.0 for no weighting, 1.0 for weighting)
    :param initialize_transitions: True for not learning, False for learning
    """
    # Initialize all checks
    number_of_failures = {}
    run_time = {}
    consistency = {}  # Maintains a calculation of how different the parent * transition is from the child
    unknown_data = {}
    weight = {}
    passed_tests = {}
    generate_updates = False
    if updates is None:
        generate_updates = True
    for test_counter in range(0, NUMBER_OF_TESTS_PER_CASE):
        # Reset all trees
        for tree in trees.values():
            tree.initialize_unknown()
        if initialize_transitions is True:
            # Because there is no learning, the transitions need to be defined, randomly, and only some of them.
            set_transitions_for_no_learning(trees)  # All set the same way

        if generate_updates is True:
            updates = create_updates(trees[list(trees.keys())[0]])

        for tree_type, tree in trees.items():
            if tree_type not in passed_tests:
                passed_tests[tree_type] = 0  # Initialize

            failed = False  # For handling failed tests
            start = timeit.default_timer()
            for update_number in range(0, NUMBER_OF_UPDATES_PER_TEST):
                if simultaneous_updates is True:
                    evidence = updates[update_number]
                    try:
                        tree.input_evidence("TestInputs", evidence, input_weight=input_weight)
                    except Exception as e:
                        # Failed to optimize - don't stop the tests, just record the number of failures
                        failed = True
                        print("Failed for type " + tree_type + " due to " + str(e))
                        if tree_type in number_of_failures:
                            number_of_failures[tree_type] += 1
                        else:
                            number_of_failures[tree_type] = 1
                else:
                    for node_name in updates[update_number].keys():
                        the_update = updates[update_number][node_name]
                        evidence = {node_name: the_update}  # Make it serial, not part of a set
                        try:
                            tree.input_evidence("TestInputs", evidence, input_weight=0.0)  # No weighting
                        except Exception as e:
                            # Failed to optimize - don't stop the tests, just record the number of failures
                            failed = True
                            print("Failed for type " + tree_type + " due to " + str(e))
                            if tree_type in number_of_failures:
                                number_of_failures[tree_type] += 1
                            else:
                                number_of_failures[tree_type] = 1
                        if failed is True:
                            break  # Stop updates - the network failed to update properly
                if failed is True:
                    break  # Stop updates - the network failed to update properly
            stop = timeit.default_timer()
            # Evaluate results
            # Time to propagate
            if not failed:
                this_run_time = (stop - start) / len(tree.nodes)  # Per node
                if tree_type in run_time:
                    run_time[tree_type] = (run_time[tree_type] * passed_tests[tree_type] +
                                           this_run_time) / (float(passed_tests[tree_type]) + 1.0)
                else:
                    run_time[tree_type] = this_run_time

                # Final consistency
                consistency_check = evaluate_network_consistency(tree) / len(tree.nodes)  # Per node basis
                if tree_type in consistency:
                    consistency[tree_type] = (consistency[tree_type] * passed_tests[tree_type] +
                                              consistency_check) / (float(passed_tests[tree_type]) + 1.0)
                else:
                    consistency[tree_type] = consistency_check

                # Explainability
                # For now, evaluate as a combination of two metrics:
                #  (1) How much unknown information remains (how well the network explains the data it receives)
                #      Depends on combination type and data, so needs learning and non-learning to compare.
                #  (2) Differences between expected experience and actual experience per update per node
                unknown_check = evaluate_network_unknown(tree)
                if tree_type in unknown_data:
                    unknown_data[tree_type] = (unknown_data[tree_type] * passed_tests[tree_type] +
                                               unknown_check) / (float(passed_tests[tree_type]) + 1.0)
                else:
                    unknown_data[tree_type] = unknown_check

                if input_weight < 0.0001:
                    # Because no weighting is involved, experience per node will be number of updates * number of
                    #  simultaneous updates
                    if tree_type in weight:
                        total_weight = NUMBER_OF_UPDATES_PER_TEST * NUMBER_OF_SIMULTANEOUS_UPDATES
                        weight[tree_type] = (weight[tree_type] * passed_tests[tree_type] +
                                             total_weight) / (float(passed_tests[tree_type]) + 1.0)
                    else:
                        weight[tree_type] = NUMBER_OF_UPDATES_PER_TEST * NUMBER_OF_SIMULTANEOUS_UPDATES
                else:
                    total_weight = 0.0
                    for each_node in tree.nodes.values():
                        total_weight += each_node.internal_data_weight
                    total_weight /= len(tree.nodes)
                    if tree_type in weight:
                        weight[tree_type] = (weight[tree_type] * passed_tests[tree_type] +
                                             total_weight) / (float(passed_tests[tree_type]) + 1.0)
                    else:
                        weight[tree_type] = total_weight

                # Averaging based on passed tests only
                passed_tests[tree_type] += 1
        print("      " + str(test_counter) + " Completed")

    # Update the run time to seconds per update
    for tree_type in trees.keys():
        if tree_type in run_time:
            run_time[tree_type] /= float(NUMBER_OF_UPDATES_PER_TEST * NUMBER_OF_SIMULTANEOUS_UPDATES)

    # Print results to CSV file
    with open(output_filename, 'a+')as f:
        f.write(test_name + "\n")  # Test name and tree (single-parent, multi-parent, complex)
        for tree_type in trees.keys():
            print_string = ",,,,," + tree_type + ","
            if tree_type in number_of_failures:
                print_string += str(number_of_failures[tree_type]) + ","
            else:
                print_string += "0,"
            if tree_type in run_time:
                print_string += str(run_time[tree_type]) + ","
            else:
                print_string += "-,"
            if tree_type in consistency:
                print_string += str(consistency[tree_type]) + ","
            else:
                print_string += "-,"
            if tree_type in unknown_data:
                print_string += str(unknown_data[tree_type]) + ","
            else:
                print_string += "-,"
            if tree_type in weight:
                print_string += str(weight[tree_type]) + "\n"
            else:
                print_string += "-\n"
            f.write(print_string)


def test_network_for_paper(output_filename):
    """
    Runs all tests for the paper
    :param output_filename string output filename of the CSV to which to print results
    """
    # TODO: How do we evaluate whether the end results "makes sense" in terms of the inputs?

    # The combination methods to be tested
    test_combination_methods = [COMBINATION_METHODS["DEMPSTER_SHAFER"],
                                COMBINATION_METHODS["MURPHY"],
                                COMBINATION_METHODS["ZHANG"],
                                COMBINATION_METHODS["OVERWRITE"]]  # For baseline
    # Whether each test has a random set of updates or whether they are the same between tests of the same type of tree
    randomize_all_updates = True
    updates = None

    # The network generator that creates the test networks
    net_gen = NetworkGenerator()
    # For each set of tests, generate the trees
    trees = {}
    for combination_method in test_combination_methods:
        tree = net_gen.generate_network_evaluation_test_tree()
        # The weighting method ensures that the primary focus is on being a single update when
        #  simultaneous updates occur, instead of weighing for experience over time.
        tree.set_combination_and_weighting_methods(combination_method=combination_method,
                                                   weighting_method=WEIGHTING_METHOD["CATEGORIES"]["TO_ONE"],
                                                   override=True)
        tree.initialize_unknown()  # Make sure they are initialized for creating the list of updates
        trees[combination_method] = tree

    # Create the list of updates
    if randomize_all_updates is False:
        updates = create_updates(trees[test_combination_methods[0]])

    # Print header to CSV file
    with open(output_filename, 'w+')as f:  # Overwrite for headers since first line of file
        # Insert headers since this is the first output
        f.write("Learning,Update,Weighting,Multi-Parent-Method,Tree,Combination Type,Failures,Run Time,Consistency," +
                "Unknown Fraction,Weight\n")

    # First tests - no learning, and only update one node at a time
    for tree in trees.values():
        tree.enable_learning(False)
    # Run the test
    test_name = "No Learning,Single-Update,No Weight,,Single-Parent"
    print("Test 1: " + test_name)
    run_test(trees, updates, output_filename, test_name, False, 0.0, True)

    # Second tests - with learning, and only update one node at a time
    for tree in trees.values():
        tree.enable_learning(True)
    # Run the test
    test_name = "Learning,Single-Update,No Weight,,Single-Parent"
    print("Test 2: " + test_name)
    run_test(trees, updates, output_filename, test_name, False, 0.0, False)

    # Third tests - with learning, update multiple nodes simultaneously, and with no weighting
    # Run the test
    test_name = "Learning,Multi-Update,No Weight,,Single-Parent"
    print("Test 3: " + test_name)
    run_test(trees, updates, output_filename, test_name, True, 0.0, False)

    # Fourth tests - with learning, update multiple nodes simultaneously, and with weighting
    # Run the test
    test_name = "Learning,Multi-Update,Weight,,Single-Parent"
    print("Test 4: " + test_name)
    run_test(trees, updates, output_filename, test_name, True, 1.0, False)

    # Moving to multi-parents
    test_combination_methods = [COMBINATION_METHODS["DEMPSTER_SHAFER"],
                                COMBINATION_METHODS["OVERWRITE"],
                                COMBINATION_METHODS["MURPHY"],
                                COMBINATION_METHODS["ZHANG"]]

    # For each set of tests, generate the trees
    trees = {}
    for combination_method in test_combination_methods:
        multi_parent_tree = net_gen.generate_multi_parent_network_evaluation_test_tree()
        multi_parent_tree.set_combination_and_weighting_methods(combination_method=combination_method,
                                                                weighting_method=
                                                                WEIGHTING_METHOD["CATEGORIES"]["TO_ONE"],
                                                                override=True)
        multi_parent_tree.initialize_unknown()
        multi_parent_tree.enable_learning(True)
        trees[combination_method] = multi_parent_tree

    # Create the list of updates
    if randomize_all_updates is False:
        updates = create_updates(trees[test_combination_methods[0]])

    # Fifth tests - multiple parents with learning and weighting.
    # Solve with optimization only.
    for tree in trees.values():
        # Only via optimization on the first run
        tree.use_root_finder = MULTI_PARENT_SOLUTION_TYPE["CATEGORIES"]["OPTIMIZER"]
    # Run the test
    test_name = "Learning,Multi-Update,Weight,Optimization,Multi-Parent"
    print("Test 5: " + test_name)
    run_test(trees, updates, output_filename, test_name, True, 1.0, False)

    # Sixth tests - multiple parent tests with learning and weighting.
    # Solve with novel capabilities
    for tree in trees.values():
        # Use root finder.  Don't use the fallback to avoid messing up timing tests.
        tree.use_root_finder = MULTI_PARENT_SOLUTION_TYPE["CATEGORIES"]["ROOT_FINDER"]
    # Run the test
    test_name = "Learning,Multi-Update,Weight,Root Finder,Multi-Parent"
    print("Test 6: " + test_name)
    run_test(trees, updates, output_filename, test_name, True, 1.0, False)

    # Moving to complex networks
    trees = {}
    for combination_method in test_combination_methods:
        complex_tree = net_gen.generate_complex_evaluation_test_tree()
        complex_tree.set_combination_and_weighting_methods(combination_method=combination_method,
                                                           weighting_method=WEIGHTING_METHOD["CATEGORIES"]["TO_ONE"],
                                                           override=True)
        complex_tree.initialize_unknown()
        complex_tree.enable_learning(True)
        complex_tree.use_root_finder = MULTI_PARENT_SOLUTION_TYPE["CATEGORIES"]["ROOT_FINDER"]
        trees[combination_method] = complex_tree

    # Create the list of updates
    if randomize_all_updates is False:
        updates = create_updates(trees[test_combination_methods[0]])

    # Seventh tests - complex network with individual node updates (no weight effectively)
    # Run the test
    test_name = "Learning,Single-Update,No Weight,Root Finder,Complex"
    print("Test 7: " + test_name)
    run_test(trees, updates, output_filename, test_name, False, 0.0, False)

    # Eighth tests - complex network with simultaneous node updates and weighting
    # Run the test
    test_name = "Learning,Multi-Update,Weight,Root Finder,Complex"
    print("Test 8: " + test_name)
    run_test(trees, updates, output_filename, test_name, True, 1.0, False)

    print("Tests completed")


# Run the tests
test_network_for_paper("./test_output.csv")
