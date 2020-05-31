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
from network.DSNetwork import WEIGHTING_METHOD
from network.NetworkGenerator import NetworkGenerator
from combinationRules import COMBINATION_METHODS
import random
from metrics import PlotMetrics

TEST_DELTA = 0.0001
ZERO_DELTA = 1e-16

PROBABILITY_TO_SET_POTENTIAL = 1.0
NUMBER_OF_UPDATES_PER_TEST = 60
NUMBER_OF_SIMULTANEOUS_UPDATES = 3


class TestMetrics(unittest.TestCase):
    def setUp(self):
        """
        Create the network
        """
        self.new_generator = NetworkGenerator()
        self.new_generator.create_all_trees()
        # It's been created.  Now to test

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

    def test_plotting(self):
        """
        Testing plotting
        """
        plots = PlotMetrics.PlotMetrics(by_time=False)
        tree = self.new_generator.network_evaluation_test_tree
        tree.set_combination_and_weighting_methods(combination_method=COMBINATION_METHODS["MURPHY"],
                                                   weighting_method=WEIGHTING_METHOD["CATEGORIES"]["TOTAL"],
                                                   override=True)
        tree.initialize_unknown()
        tree.enable_learning(True)
        updates = self.create_updates(tree)
        for update_number in range(0, NUMBER_OF_UPDATES_PER_TEST):
            tree.input_evidence("TestInputs", updates[update_number], input_weight=1.0)
            for name, data in tree.nodes.items():
                plots.add_data(name, data, update_number, update_number)  # Time same as step
            for name, data in tree.transitions.items():
                plots.add_data(name, data, update_number, update_number)  # Time same as step
        # Should have a fully set up network now.
        # Show the plots
        # plots.plot_data(plot_names=["E", "B.E"], plot_internal=True, interval=500, repeat_delay=1000)
        # plots.plot_ambiguities(plot_names=["E", "B.E"], plot_internal=True, interval=500, repeat_delay=1000)
        limits = {
            "E_O1": {"HIGH": 0.2},
            "E_O2": {"LOW": 0.3},
        }
        plots.plot_limited(limits, plot_names=["E"], plot_internal=True, interval=500, repeat_delay=1000)

    def test_loading_animate(self):
        """
        Loads data and plots
        """
        plots = PlotMetrics.PlotMetrics(by_time=True)
        plots.read_data("op_1_OPERATOR_RISK_increment.csv")
        plots.plot_data(plot_internal=True, repeat=False, interval=100, repeat_delay=1000, filepath=".")
        plots.plot_ambiguities(plot_internal=True, repeat=False, interval=100, repeat_delay=1000, filepath=".")
        limits = {
            "HIGH": {"HIGH": 0.1}
        }
        plots.plot_limited(limits, plot_internal=True, repeat=False, interval=100, repeat_delay=1000, filepath=".")

    def test_loading_stacked(self):
        """
        Loads data and plots
        """
        plots = PlotMetrics.PlotMetrics(by_time=True)
        plots.read_data("op_1_OPERATOR_RISK_increment.csv")
        # plots.read_data("1_DYNAMICS_RISK_INTERNAL_RISK_increment.csv")
        plots.plot_data(plot_internal=True, animate=False, area_labels=True)
        plots.plot_ambiguities(plot_internal=True, animate=False, area_labels=True)
        limits = {
            "HIGH": {"HIGH": 0.1}
        }
        plots.plot_limited(limits, plot_internal=True, animate=False, area_labels=True)

    def test_loading_stacked_save(self):
        """
        Loads data and plots
        """
        plots = PlotMetrics.PlotMetrics(by_time=True)
        plots.read_data("op_1_OPERATOR_RISK_increment.csv")
        plots.plot_data(plot_internal=True, animate=False, filepath=".")
        plots.plot_ambiguities(plot_internal=True, animate=False, filepath=".")
        limits = {
            "HIGH": {"HIGH": 0.1}
        }
        plots.plot_limited(limits, plot_internal=True, animate=False, filepath=".")
