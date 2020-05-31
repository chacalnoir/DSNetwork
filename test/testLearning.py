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
For testing learning and retaining background knowledge
"""
from network.DSNetwork import COMBINATION_METHODS, WEIGHTING_METHOD
from network.NetworkGenerator import NetworkGenerator
from metrics import PlotMetrics

INITIALIZE_ZERO_MARGINAL = 0.05
LEARNING_WEIGHT = 10.0


def teach_network(network):
    """
    Teaches the network using test information designed to evoke a specific result
    Node 1: A, B, (A,B)
    Node 2: C, D, (C,D)
    Transition maps Node 1 to Node 2
    The goal is the following general mapping:

              A       B       (A,B)
    C         0.9    0.1        0.05
    D         0.1    0.4        0.15
    (C,D)     0.0    0.5        0.8
    To test this, we will give the following scenarios:
    1)
    a) High weight A, low weight B, almost no weight (A,B) to high weight C, low weight D, no weight (C,D)
    b) Low weight A, High weight B, low weight (A,B) to mid weight D, (C,D) low weight C
    That should result appropriately.  That's the simple case.
    2)
    a) High weight A, low weight B, almost no weight (A,B) to high weight C, low weight D, no weight (C,D)
    b) High weight D, high weight B, low weight everything else
    c) High weight C,D, High weight B, low weight everything else
    That should result appropriately.  That's the more complex case.
    Does this make sense?  Should (c) actually update (b), or should it balance out (b)?  See if we can balance out
    (b) since we don't want to lose information already learned.
    # In case (2), the result switches because we don't keep previous information.  That is expected.  Moreover, that
     may be encouraged, because it allows the system to learn that a change has occurred.  I would say, however, that
     learning that a change has occurred should use D-S; it shouldn't switch because of the situation changes the
     teacher gives it.
    # How to mechanize this?  For learning purposes, can we keep the snapshots of the child that relate to the snapshot
     of the parent?  That's a complicated setup, but it allows things to play out over time, using the same rules as
     developed for learning before.  I think this plan is better.  Remember, even if scenarios overlap in the parent
     distribution, that is okay.  The point is eventually to use some level of statistically significant difference to
     determine individual situations, and update accordingly.  In the end, that would be used for all nodes, allowing
     the nodes that didn't change between given situations to combine and continue learning for their same "situation".
    :param network: a DSNetwork with 2 nodes for testing
    """
    # Set up for plotting
    plots = PlotMetrics.PlotMetrics(by_time=False)
    # Set up the network for learning
    network.enable_learning(True, LEARNING_WEIGHT)  # Must be learning for this part
    # Set for all
    network.set_combination_and_weighting_methods(COMBINATION_METHODS["MURPHY"],
                                                  WEIGHTING_METHOD["CATEGORIES"]["TOTAL"],
                                                  True)
    network.set_combination_limit(2)  # Just running Murphy with weighting
    # Clear the network.
    network.initialize_unknown()
    network.set_zero_snapshot()  # Set the zero snapshot for all deltas- especially for the nodes
    network.set_current_snapshot_name(name="option_a_case")
    for name, data in network.nodes.items():
        plots.add_data(name, data, 0, 0)  # Time same as step
    for name, data in network.transitions.items():
        plots.add_data(name, data, 0, 0)  # Time same as step

    evidence = {
        "Node_1": {
            "evidence": {
                ("OPTION_A",): 0.9,
                ("OPTION_B",): 0.08,
                ("OPTION_A", "OPTION_B"): 0.02
            }
        },
        "Node_2": {
            "evidence": {
                ("OPTION_C",): 0.9,
                ("OPTION_D",): 0.1,
                ("OPTION_C", "OPTION_D"): 0.0
            }
        }
    }
    network.input_evidence("LearningInputs", evidence, input_weight=1.0)
    for name, data in network.nodes.items():
        plots.add_data(name, data, 1, 1)  # Time same as step
    for name, data in network.transitions.items():
        plots.add_data(name, data, 1, 1)  # Time same as step
    network.set_snapshot_for_learning(name="option_a_case")
    # Next section
    network.initialize_unknown(initialize_nodes=INITIALIZE_ZERO_MARGINAL, initialize_transitions=None)
    network.set_zero_snapshot()
    network.set_current_snapshot_name(name="option_b_case")

    evidence = {
        "Node_1": {
            "evidence": {
                ("OPTION_A",): 0.05,
                ("OPTION_B",): 0.9,
                ("OPTION_A", "OPTION_B"): 0.05
            }
        },
        "Node_2": {
            "evidence": {
                ("OPTION_C",): 0.1,
                ("OPTION_D",): 0.7,
                ("OPTION_C", "OPTION_D"): 0.2
            }
        }
    }
    network.input_evidence("LearningInputs", evidence, input_weight=1.0)
    for name, data in network.nodes.items():
        plots.add_data(name, data, 2, 2)  # Time same as step
    for name, data in network.transitions.items():
        plots.add_data(name, data, 2, 2)  # Time same as step
    network.set_snapshot_for_learning(name="option_b_case")
    plots.plot(["Node_1.Node_2"], plot_evidence=True, visualization_type=PlotMetrics.VISUALIZATION_TYPE["DATA"],
               area_labels=True)

    # Should be approximately as expected now
    test = 0
    # Reset and test alternate method
    network.initialize_unknown()
    network.set_snapshot_for_learning(clear=True)
    network.set_current_snapshot_name(name=None, clear=True)

    # Now run the alternate method
    network.set_zero_snapshot()
    network.set_current_snapshot_name(name="option_a_case")
    evidence = {
        "Node_1": {
            "evidence": {
                ("OPTION_A",): 0.9,
                ("OPTION_B",): 0.08,
                ("OPTION_A", "OPTION_B"): 0.02
            }
        },
        "Node_2": {
            "evidence": {
                ("OPTION_C",): 0.9,
                ("OPTION_D",): 0.1,
                ("OPTION_C", "OPTION_D"): 0.0
            }
        }
    }
    network.input_evidence("LearningInputs", evidence, input_weight=1.0)
    network.set_snapshot_for_learning(name="option_a_case")
    # Next section
    network.initialize_unknown(initialize_nodes=INITIALIZE_ZERO_MARGINAL, initialize_transitions=None)
    network.set_zero_snapshot()
    network.set_current_snapshot_name(name="option_b_case")

    evidence = {
        "Node_1": {
            "evidence": {
                ("OPTION_A",): 0.05,
                ("OPTION_B",): 0.9,
                ("OPTION_A", "OPTION_B"): 0.05
            }
        },
        "Node_2": {
            "evidence": {
                ("OPTION_C",): 0.1,
                ("OPTION_D",): 0.85,
                ("OPTION_C", "OPTION_D"): 0.05
            }
        }
    }
    network.input_evidence("LearningInputs", evidence, input_weight=1.0)
    network.set_snapshot_for_learning(name="option_b_case")

    # Next section
    network.initialize_unknown(initialize_nodes=INITIALIZE_ZERO_MARGINAL, initialize_transitions=None)
    network.set_zero_snapshot()
    network.set_current_snapshot_name(name="option_b_case")

    evidence = {
        "Node_1": {
            "evidence": {
                ("OPTION_A",): 0.05,
                ("OPTION_B",): 0.9,
                ("OPTION_A", "OPTION_B"): 0.05
            }
        },
        "Node_2": {
            "evidence": {
                ("OPTION_C",): 0.1,
                ("OPTION_D",): 0.05,
                ("OPTION_C", "OPTION_D"): 0.85
            }
        }
    }
    network.input_evidence("LearningInputs", evidence, input_weight=1.0)
    network.set_snapshot_for_learning(name="option_b_case")

    # Should be correct now, but likely will have switched instead of added
    test = 1


net_gen = NetworkGenerator()
test_network = net_gen.generate_two_node_tree()
teach_network(test_network)
