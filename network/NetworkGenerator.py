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

# Creates a fault tree, enabling modification to the saved version once that is dumped through pickle
from network.DSNetwork import DSNetwork, Node, NODE_TYPE, Transition, WEIGHTING_METHOD
from combinationRules import COMBINATION_METHODS
from network.inputFunctions.PassThroughInputs import PassThroughInputs


# noinspection PyMethodMayBeStatic
class NetworkGenerator:
    """
    Generate the network to enable dumping to file for modifications later
    """
    def __init__(self):
        """
        Only create all trees if desired
        """
        self.basic_mr_tree = None
        self.test_tree = None
        self.multi_combination_test_tree = None
        self.cycle_test_tree = None
        self.network_evaluation_test_tree = None
        self.multi_parent_network_evaluation_test_tree = None
        self.complex_evaluation_test_tree = None
        self.two_node_tree = None

    def create_all_trees(self):
        """
        Define the default trees here.
        """
        # Create the first fault tree
        self.basic_mr_tree = self.generate_basic_mr_fault_tree()

        # Create a test tree
        self.test_tree = self.generate_test_tree()

        # Create a test tree for multi-combination
        self.multi_combination_test_tree = self.generate_multi_combination_test_tree()

        # Create a test tree with a cycle
        self.cycle_test_tree = self.generate_cycle_test_tree()

        # Create a test tree for evaluating the network algorithms
        self.network_evaluation_test_tree = self.generate_network_evaluation_test_tree()

        # Create a test tree for evaluating multi-parent algorithms
        self.multi_parent_network_evaluation_test_tree = self.generate_multi_parent_network_evaluation_test_tree()

        # Create a more complex test tree for evaluating the network algorithms
        self.complex_evaluation_test_tree = self.generate_complex_evaluation_test_tree()

        # Create a 2 node tree
        self.two_node_tree = self.generate_two_node_tree()

    def generate_basic_mr_fault_tree(self):
        """
        Creates the fault tree for a basic multirotor
        """
        tree = DSNetwork()

        batt_volt_node = Node()
        batt_volt_node.name = "battery voltage"
        batt_volt_node.type = NODE_TYPE["CATEGORIES"]["INPUT"]
        batt_volt_node.options = ["GOOD", "HIGH", "LOW", "DEAD"]
        tree.nodes[batt_volt_node.name] = batt_volt_node

        batt_amp_node = Node()
        batt_amp_node.name = "battery amp"
        batt_amp_node.type = NODE_TYPE["CATEGORIES"]["INPUT"]
        batt_volt_node.options = ["GOOD", "LOW", "DEAD"]
        tree.nodes[batt_amp_node.name] = batt_amp_node

        # Finalize for use
        tree.set_combination_and_weighting_methods(COMBINATION_METHODS["EVIDENTIAL_COMBINATION"],
                                                   WEIGHTING_METHOD["CATEGORIES"]["NONE"])
        return tree

    def generate_test_tree(self):
        """
        Generates the test tree from the paper + additions
           D
        B     G
              H
              I
              J
        """
        d_node = Node()
        d_node.name = "D"
        d_node.type = NODE_TYPE["CATEGORIES"]["INPUT"]
        d_node.options = ["TRUE", "FALSE"]

        b_node = Node()
        b_node.name = "B"
        b_node.type = NODE_TYPE["CATEGORIES"]["INPUT"]
        b_node.options = ["TRUE", "FALSE"]

        g_node = Node()
        g_node.name = "G"
        g_node.type = NODE_TYPE["CATEGORIES"]["INPUT"]
        g_node.options = ["TRUE", "FALSE"]

        h_node = Node()
        h_node.name = "H"
        h_node.type = NODE_TYPE["CATEGORIES"]["INPUT"]
        h_node.options = ["TRUE", "FALSE"]

        i_node = Node()
        i_node.name = "I"
        i_node.type = NODE_TYPE["CATEGORIES"]["INPUT"]
        i_node.options = ["TRUE", "FALSE", "NOT_SURE"]

        j_node = Node()
        j_node.name = "J"
        j_node.type = NODE_TYPE["CATEGORIES"]["INPUT"]
        j_node.options = ["TRUE", "FALSE", "NOT_SURE"]

        db_transition = Transition(d_node, b_node)
        dg_transition = Transition(d_node, g_node)
        gh_transition = Transition(g_node, h_node)
        hi_transition = Transition(h_node, i_node)
        ij_transition = Transition(i_node, j_node)

        # Add to tree, link, and initialize
        tree = DSNetwork()
        tree.add_link_and_initialize([d_node, b_node, g_node, h_node, i_node, j_node],
                                     [db_transition, dg_transition,
                                      gh_transition, hi_transition, ij_transition],
                                     combination_method=COMBINATION_METHODS["DEMPSTER_SHAFER"],
                                     weighting_method=WEIGHTING_METHOD["CATEGORIES"]["NONE"],
                                     initialize_unknown=False)  # For some of the tests, don't initialize
        return tree

    def generate_cycle_test_tree(self):
        """
        Generates a test tree with a cycle in it
        """
        d_node = Node()
        d_node.name = "D"
        d_node.type = NODE_TYPE["CATEGORIES"]["INPUT"]
        d_node.options = ["TRUE", "FALSE"]

        b_node = Node()
        b_node.name = "B"
        b_node.type = NODE_TYPE["CATEGORIES"]["INPUT"]
        b_node.options = ["TRUE", "FALSE"]

        g_node = Node()
        g_node.name = "G"
        g_node.type = NODE_TYPE["CATEGORIES"]["INPUT"]
        g_node.options = ["TRUE", "FALSE"]

        f_node = Node()
        f_node.name = "F"
        f_node.type = NODE_TYPE["CATEGORIES"]["INPUT"]
        f_node.options = ["TRUE", "FALSE"]

        db_transition = Transition(d_node, b_node)
        dg_transition = Transition(d_node, g_node)
        gf_transition = Transition(g_node, f_node)
        bf_transition = Transition(b_node, f_node)
        tree = DSNetwork()
        tree.add_link_and_initialize([d_node, b_node, g_node, f_node],
                                     [db_transition, dg_transition, gf_transition, bf_transition],
                                     combination_method=COMBINATION_METHODS["DEMPSTER_SHAFER"],
                                     weighting_method=WEIGHTING_METHOD["CATEGORIES"]["NONE"])
        return tree

    def generate_multi_combination_test_tree(self):
        """
        Generates a test tree designed to exercise the multi-combination capability

        D    B
          G
        """
        d_node = Node()
        d_node.name = "D"
        d_node.type = NODE_TYPE["CATEGORIES"]["INPUT"]
        d_node.options = ["TRUE", "FALSE", "NOT_SURE"]

        b_node = Node()
        b_node.name = "B"
        b_node.type = NODE_TYPE["CATEGORIES"]["INPUT"]
        b_node.options = ["TRUE", "FALSE", "NOT_SURE"]

        g_node = Node()
        g_node.name = "G"
        g_node.type = NODE_TYPE["CATEGORIES"]["INPUT"]
        g_node.options = ["TRUE", "FALSE", "NOT_SURE"]

        # Transitions
        dg_transition = Transition(d_node, g_node)
        bg_transition = Transition(b_node, g_node)
        tree = DSNetwork()
        tree.add_link_and_initialize([d_node, b_node, g_node], [dg_transition, bg_transition],
                                     combination_method=COMBINATION_METHODS["MURPHY"],
                                     weighting_method=WEIGHTING_METHOD["CATEGORIES"]["TOTAL"])
        # The inputs function
        tree.add_input_mapping(PassThroughInputs("TestInputs", [d_node, b_node, g_node]))
        return tree

    def generate_network_evaluation_test_tree(self):
        """
        Generates a test tree for evaluating the network.  Does not contain any multi-parent nodes.
                A3
           B2         C3
        D2    E2   F3    G3
        """
        a_node = Node()
        a_node.name = "A"
        a_node.type = NODE_TYPE["CATEGORIES"]["INPUT"]
        a_node.options = ["A_O1", "A_O2", "A_O3"]

        b_node = Node()
        b_node.name = "B"
        b_node.type = NODE_TYPE["CATEGORIES"]["INPUT"]
        b_node.options = ["B_O1", "B_O2"]

        c_node = Node()
        c_node.name = "C"
        c_node.type = NODE_TYPE["CATEGORIES"]["INPUT"]
        c_node.options = ["C_O1", "C_O2", "C_O3"]

        d_node = Node()
        d_node.name = "D"
        d_node.type = NODE_TYPE["CATEGORIES"]["INPUT"]
        d_node.options = ["D_O1", "D_O2"]

        e_node = Node()
        e_node.name = "E"
        e_node.type = NODE_TYPE["CATEGORIES"]["INPUT"]
        e_node.options = ["E_O1", "E_O2"]

        f_node = Node()
        f_node.name = "F"
        f_node.type = NODE_TYPE["CATEGORIES"]["INPUT"]
        f_node.options = ["F_O1", "F_O2", "F_O3"]

        g_node = Node()
        g_node.name = "G"
        g_node.type = NODE_TYPE["CATEGORIES"]["INPUT"]
        g_node.options = ["G_O1", "G_O2", "G_O3"]

        ab_transition = Transition(a_node, b_node)
        ac_transition = Transition(a_node, c_node)
        bd_transition = Transition(b_node, d_node)
        be_transition = Transition(b_node, e_node)
        cf_transition = Transition(c_node, f_node)
        cg_transition = Transition(c_node, g_node)

        # Add to tree, link, and initialize
        tree = DSNetwork()
        tree.add_link_and_initialize([a_node, b_node, c_node, d_node, e_node, f_node, g_node],
                                     [ab_transition, ac_transition, bd_transition,
                                      be_transition, cf_transition, cg_transition],
                                     combination_method=COMBINATION_METHODS["DEMPSTER_SHAFER"],
                                     weighting_method=WEIGHTING_METHOD["CATEGORIES"]["NONE"])
        # The inputs function
        tree.add_input_mapping(PassThroughInputs("TestInputs", [a_node, b_node, c_node, d_node, e_node, f_node,
                                                                g_node]))
        return tree

    def generate_multi_parent_network_evaluation_test_tree(self):
        """
        Generates a test tree for evaluating the network.  Contains multi-parent nodes
        A3  B2 E2 F3 G3
         \ /    \ | /
          C3      D2     H3  J2
           \      |      /  /
            \     |    /  /
              \   |  / /
                \ |/ /
                  K3
        """
        a_node = Node()
        a_node.name = "A"
        a_node.type = NODE_TYPE["CATEGORIES"]["INPUT"]
        a_node.options = ["A_O1", "A_O2", "A_O3"]

        b_node = Node()
        b_node.name = "B"
        b_node.type = NODE_TYPE["CATEGORIES"]["INPUT"]
        b_node.options = ["B_O1", "B_O2"]

        c_node = Node()
        c_node.name = "C"
        c_node.type = NODE_TYPE["CATEGORIES"]["INPUT"]
        c_node.options = ["C_O1", "C_O2", "C_O3"]

        d_node = Node()
        d_node.name = "D"
        d_node.type = NODE_TYPE["CATEGORIES"]["INPUT"]
        d_node.options = ["D_O1", "D_O2"]

        e_node = Node()
        e_node.name = "E"
        e_node.type = NODE_TYPE["CATEGORIES"]["INPUT"]
        e_node.options = ["E_O1", "E_O2"]

        f_node = Node()
        f_node.name = "F"
        f_node.type = NODE_TYPE["CATEGORIES"]["INPUT"]
        f_node.options = ["F_O1", "F_O2", "F_O3"]

        g_node = Node()
        g_node.name = "G"
        g_node.type = NODE_TYPE["CATEGORIES"]["INPUT"]
        g_node.options = ["G_O1", "G_O2", "G_O3"]

        h_node = Node()
        h_node.name = "H"
        h_node.type = NODE_TYPE["CATEGORIES"]["INPUT"]
        h_node.options = ["H_O1", "H_O2", "H_O3"]

        j_node = Node()
        j_node.name = "J"
        j_node.type = NODE_TYPE["CATEGORIES"]["INPUT"]
        j_node.options = ["J_O1", "J_O2"]

        k_node = Node()
        k_node.name = "K"
        k_node.type = NODE_TYPE["CATEGORIES"]["INPUT"]
        k_node.options = ["K_O1", "K_O2", "K_O3"]

        ac_transition = Transition(a_node, c_node)
        bc_transition = Transition(b_node, c_node)
        ck_transition = Transition(c_node, k_node)
        ed_transition = Transition(e_node, d_node)
        fd_transition = Transition(f_node, d_node)
        gd_transition = Transition(g_node, d_node)
        dk_transition = Transition(d_node, k_node)
        hk_transition = Transition(h_node, k_node)
        jk_transition = Transition(j_node, k_node)

        tree = DSNetwork()
        tree.add_link_and_initialize([a_node, b_node, c_node, d_node, e_node, f_node, g_node, h_node, j_node, k_node],
                                     [ac_transition, bc_transition, ck_transition, ed_transition,
                                      fd_transition, gd_transition, dk_transition, hk_transition, jk_transition],
                                     combination_method=COMBINATION_METHODS["DEMPSTER_SHAFER"],
                                     weighting_method=WEIGHTING_METHOD["CATEGORIES"]["NONE"])
        # The inputs function
        tree.add_input_mapping(PassThroughInputs("TestInputs", [a_node, b_node, c_node, d_node, e_node, f_node,
                                                                g_node, h_node, j_node, k_node]))
        return tree

    def generate_complex_evaluation_test_tree(self):
        """
        Generates a more complex test tree for evaluating the network algorithms.
        A3  B2 E2 F3 G3
         \ /    \ | /
          C3      D2      H3
           \      |\     / |\
            \     | M3 /  J2 L3
              \   |  /
                \ |/
                  K3
        """
        a_node = Node()
        a_node.name = "A"
        a_node.type = NODE_TYPE["CATEGORIES"]["INPUT"]
        a_node.options = ["A_O1", "A_O2", "A_O3"]

        b_node = Node()
        b_node.name = "B"
        b_node.type = NODE_TYPE["CATEGORIES"]["INPUT"]
        b_node.options = ["B_O1", "B_O2"]

        c_node = Node()
        c_node.name = "C"
        c_node.type = NODE_TYPE["CATEGORIES"]["INPUT"]
        c_node.options = ["C_O1", "C_O2", "C_O3"]

        d_node = Node()
        d_node.name = "D"
        d_node.type = NODE_TYPE["CATEGORIES"]["INPUT"]
        d_node.options = ["D_O1", "D_O2"]

        e_node = Node()
        e_node.name = "E"
        e_node.type = NODE_TYPE["CATEGORIES"]["INPUT"]
        e_node.options = ["E_O1", "E_O2"]

        f_node = Node()
        f_node.name = "F"
        f_node.type = NODE_TYPE["CATEGORIES"]["INPUT"]
        f_node.options = ["F_O1", "F_O2", "F_O3"]

        g_node = Node()
        g_node.name = "G"
        g_node.type = NODE_TYPE["CATEGORIES"]["INPUT"]
        g_node.options = ["G_O1", "G_O2", "G_O3"]

        h_node = Node()
        h_node.name = "H"
        h_node.type = NODE_TYPE["CATEGORIES"]["INPUT"]
        h_node.options = ["H_O1", "H_O2", "H_O3"]

        j_node = Node()
        j_node.name = "J"
        j_node.type = NODE_TYPE["CATEGORIES"]["INPUT"]
        j_node.options = ["J_O1", "J_O2"]

        k_node = Node()
        k_node.name = "K"
        k_node.type = NODE_TYPE["CATEGORIES"]["INPUT"]
        k_node.options = ["K_O1", "K_O2", "K_O3"]

        l_node = Node()
        l_node.name = "L"
        l_node.type = NODE_TYPE["CATEGORIES"]["INPUT"]
        l_node.options = ["L_O1", "L_O2", "L_O3"]

        m_node = Node()
        m_node.name = "M"
        m_node.type = NODE_TYPE["CATEGORIES"]["INPUT"]
        m_node.options = ["M_O1", "M_O2", "M_O3"]

        ac_transition = Transition(a_node, c_node)
        bc_transition = Transition(b_node, c_node)
        ck_transition = Transition(c_node, k_node)
        ed_transition = Transition(e_node, d_node)
        fd_transition = Transition(f_node, d_node)
        gd_transition = Transition(g_node, d_node)
        dk_transition = Transition(d_node, k_node)
        dm_transition = Transition(d_node, m_node)
        hk_transition = Transition(h_node, k_node)
        hj_transition = Transition(h_node, j_node)
        hl_transition = Transition(h_node, l_node)
        tree = DSNetwork()
        tree.add_link_and_initialize([a_node, b_node, c_node, d_node, e_node, f_node, g_node, h_node, j_node, k_node,
                                      l_node, m_node],
                                     [ac_transition, bc_transition, ck_transition, ed_transition, fd_transition,
                                      gd_transition, dk_transition, dm_transition, hk_transition, hj_transition,
                                      hl_transition],
                                     combination_method=COMBINATION_METHODS["DEMPSTER_SHAFER"],
                                     weighting_method=WEIGHTING_METHOD["CATEGORIES"]["NONE"])
        # The inputs function
        tree.add_input_mapping(PassThroughInputs("TestInputs", [a_node, b_node, c_node, d_node, e_node, f_node,
                                                                g_node, h_node, j_node, k_node, l_node, m_node]))
        return tree

    def generate_two_node_tree(self):
        """
        Generates a tree to simply test background retention while learning
                           Node 1
                               |
                               |
                               |
                               |
                               |
                               |
                               |
                               |
                            Node 2
        """

        node_1 = Node()
        node_1.name = "Node_1"
        node_1.type = NODE_TYPE["CATEGORIES"]["INPUT"]
        node_1.options = ["OPTION_A", "OPTION_B"]

        node_2 = Node()
        node_2.name = "Node_2"
        node_2.type = NODE_TYPE["CATEGORIES"]["INPUT"]
        node_2.options = ["OPTION_C", "OPTION_D"]

        node_transition = Transition(node_1, node_2)
        tree = DSNetwork()
        tree.add_link_and_initialize([node_1, node_2],
                                     [node_transition],
                                     combination_method=COMBINATION_METHODS["MURPHY"],
                                     weighting_method=WEIGHTING_METHOD["CATEGORIES"]["TO_ONE"]
                                     )
        # Check for cycles since these cannot exist for the tree
        if tree.check_for_cycles() is True:
            raise ValueError("There is a cycle in the traffic light tree")

        # The inputs function
        # Two input functions - one for learning the transitions between nodes, the other for evaluating the
        #  nodes after the transitions have been learned
        tree.add_input_mapping(PassThroughInputs("LearningInputs", [node_1, node_2]))
        tree.add_input_mapping(PassThroughInputs("EvaluationInputs", [node_1]))
        return tree
