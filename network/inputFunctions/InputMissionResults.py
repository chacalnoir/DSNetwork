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
Defines an input mapping from mission results to the appropriate risk analysis node
"""

from network.DSNetwork import InputToNode, WEIGHTING_METHOD
from network import RISK, RESULT_TYPE
from copy import deepcopy


# Default values - can be overridden through kwargs
# Used to reduce weighting of inference information if the network still thrashes too much.
INFERENCE_REDUCTION = 1.0
UNKNOWN_IF_SUCCESS = 0.5
AMBIGUOUS_IF_SUCCESS = 0.2
UNKNOWN_IF_FAIL = 0.4
AMBIGUOUS_IF_FAIL = 0.2
UNKNOWN_IF_FAILSAFE = 0.4


class InputMissionResults(InputToNode):
    """
    Define the input
    """
    def map_to_input(self, inputs, operation_hours=0.0, **kwargs):
        """
        Map the inputs
        :param inputs: dict(string, dict of "result" to RESULT_TYPE and "risk" to risk input) dictionary of node names
                       to bool of whether each node succeeded or failed.
        :param operation_hours: float - the hours for which this new evidence applies
        :param kwargs: additional keyword arguments - ignored for the pass through
        """
        # Set defaults
        inference_reduction = INFERENCE_REDUCTION
        unknown_if_success = UNKNOWN_IF_SUCCESS
        ambiguous_if_success = AMBIGUOUS_IF_SUCCESS
        unknown_if_fail = UNKNOWN_IF_FAIL
        ambiguous_if_fail = AMBIGUOUS_IF_FAIL
        unknown_if_failsafe = UNKNOWN_IF_FAILSAFE

        if kwargs is not None:
            if "inference_reduction" in kwargs:
                inference_reduction = kwargs["inference_reduction"]
            if "unknown_if_success" in kwargs:
                unknown_if_success = kwargs["unknown_if_success"]
            if "ambiguous_if_success" in kwargs:
                ambiguous_if_success = kwargs["ambiguous_if_success"]
            if "unknown_if_fail" in kwargs:
                unknown_if_fail = kwargs["unknown_if_fail"]
            if "ambiguous_if_fail" in kwargs:
                ambiguous_if_fail = kwargs["ambiguous_if_fail"]
            if "unknown_if_failsafe" in kwargs:
                unknown_if_failsafe = kwargs["unknown_if_failsafe"]

        # TODO: Refine this input mapping
        # TODO: The true goal - map from GUST high/med/low per flight
        evidence = {}
        weighting_method = {}  # For failures, need a reverse weighting method
        for node_name, result in inputs.items():
            if result["result"] != RESULT_TYPE["CATEGORIES"]["SUCCESS"]:
                # Failures and failsafes have an inverse effect on the network
                weighting_method[node_name] = WEIGHTING_METHOD["CATEGORIES"]["INVERSE_TOTAL"]

            if ("risk" in result) and (result["risk"]):
                # The risk is not empty - it is already mapped.
                evidence[node_name] = {
                    "evidence": deepcopy(result["risk"])
                }
            elif result["result"] == RESULT_TYPE["CATEGORIES"]["FAIL"]:
                evidence[node_name] = {
                    "evidence": {
                        RISK["CATEGORIES"]["HIGH"]: 1.0 - unknown_if_fail - ambiguous_if_fail,
                        (RISK["CATEGORIES"]["HIGH"], RISK["CATEGORIES"]["LOW"], RISK["CATEGORIES"]["MEDIUM"]):
                            unknown_if_fail,
                        (RISK["CATEGORIES"]["HIGH"], RISK["CATEGORIES"]["MEDIUM"]): ambiguous_if_fail
                    }
                }
            elif result["result"] == RESULT_TYPE["CATEGORIES"]["FAILSAFE"]:
                evidence[node_name] = {
                    "evidence": {
                        RISK["CATEGORIES"]["MEDIUM"]: 1.0 - unknown_if_failsafe,
                        (RISK["CATEGORIES"]["HIGH"], RISK["CATEGORIES"]["LOW"], RISK["CATEGORIES"]["MEDIUM"]):
                            unknown_if_failsafe,
                    }
                }
            else:
                evidence[node_name] = {
                    "evidence": {
                        RISK["CATEGORIES"]["LOW"]: 1.0 - unknown_if_success - ambiguous_if_success,
                        (RISK["CATEGORIES"]["HIGH"], RISK["CATEGORIES"]["LOW"], RISK["CATEGORIES"]["MEDIUM"]):
                            unknown_if_success,
                        (RISK["CATEGORIES"]["LOW"], RISK["CATEGORIES"]["MEDIUM"]): ambiguous_if_success
                    }
                }

        # Propagate the data
        self._propagate_input(evidence, input_weight=operation_hours, inference_weight_factor=inference_reduction,
                              input_weighting_method=weighting_method)
