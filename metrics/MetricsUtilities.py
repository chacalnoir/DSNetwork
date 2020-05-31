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
Collection of utilities for metrics that support, but don't contain, plotting
"""
from ast import literal_eval


AMBIGUOUS_LABELS = ["KNOWN", "AMBIGUOUS", "UNKNOWN"]
LIMIT_LABELS = ["INSIDE", "OUTSIDE", "UNKNOWN"]
LIMIT_LABELS_HIGH_LOW = ["INSIDE", "OUTSIDE_LOW", "OUTSIDE_HIGH", "UNKNOWN"]


def compare_ds_labels(elem1, elem2):
    """
    First compare by length (by ambiguity), then alphabetically
    :param elem1: first element
    :param elem2: second element
    """
    if isinstance(elem1, tuple) and isinstance(elem2, tuple):
        if len(elem1) < len(elem2):
            return -1
        elif len(elem2) < len(elem1):
            return 1
        else:
            return 0
    elif elem1 < elem2:
        return -1
    elif elem2 < elem1:
        return 1
    else:
        return 0


def get_tuple_name(string_name):
    """
    Get a tuple name from a string name
    :param string_name: string name to convert to tuple
    :return: tuple name
    """
    string_name = string_name.replace("/", ",")
    tuple_name = literal_eval(string_name)
    if isinstance(tuple_name, tuple) is True:
        store_key = tuple(sorted(tuple_name))
    else:
        store_key = (tuple_name,)
    return store_key


def calculate_each_ambiguity_data(data):
    """
    Calculates the ambiguity data for a single frame
    :param data: mass name to mass dict
    :return: dict of KNOWN, AMBIGUOUS, UNKNOWN to masses
    """
    # Ambiguity is all data that is longer than 1 in the tuple but less than the maximum length.
    max_length = 2
    mass_of_max_length = 0.0
    ambiguity_data = {}
    for label in AMBIGUOUS_LABELS:
        ambiguity_data[label] = 0.0
    for mass_name, mass in data.items():
        if len(mass_name) == 1:
            # Unambiguous
            ambiguity_data["KNOWN"] += mass
        elif (len(mass_name) > 1) and (len(mass_name) < max_length):
            # Ambiguous
            ambiguity_data["AMBIGUOUS"] += mass
        elif len(mass_name) == max_length:
            # Not sure yet if ambiguous or unknown
            mass_of_max_length += mass
        elif len(mass_name) > max_length:
            # New max length
            ambiguity_data["AMBIGUOUS"] += mass_of_max_length  # Already multiplied by TO_PERCENT
            max_length = len(mass_name)
            mass_of_max_length = mass  # Assign to the new longest known length
    # Once done, assign the max length to unknown
    ambiguity_data["UNKNOWN"] += mass_of_max_length  # Already multiplied by TO_PERCENT
    return ambiguity_data


def check_limit(test_value, limit_value):
    """
    Checks the limit and returns the INSIDE and OUTSIDE values in a tuple
    :return tuple of INSIDE and OUTSIDE values
    """
    inside = 0.0
    outside = 0.0
    if limit_value >= 0.0:
        # Ceiling
        if test_value <= limit_value:
            # Okay
            inside += test_value
        else:
            # Partially okay
            inside += limit_value
            outside += test_value - limit_value
    else:
        # Lower bound
        if test_value >= abs(limit_value):
            # Okay
            inside += test_value
        else:
            # Partially okay
            inside += test_value
            outside += abs(limit_value) - test_value
    return inside, outside


def calculate_each_limit_data(limits, data, separate_low_high=False):
    """
    Calculates data that is inside, outside, and unknown of the limits
    :param limits: dict of mass name (theta) to "PLAUSIBILITY" and/or "BELIEF" to mass value.
                  Negative means floor, positive means ceiling.
    :param data: dict of mass names to masses
    :param separate_low_high: bool whether to separate the low/high results
    :return: dict of INSIDE, OUTSIDE (or OUTSIDE_LOW, OUTSIDE_HIGH), UNKNOWN (or UNKNOWN_LOW, UNKNOWN_HIGH) to masses
    """
    limits_data = {}
    if separate_low_high is False:
        for label in LIMIT_LABELS:
            limits_data[label] = 0.0
    else:
        for label in LIMIT_LABELS_HIGH_LOW:
            limits_data[label] = 0.0

    bounds = {}
    tuple_limits = {}
    for name in limits.keys():
        if isinstance(name, tuple) is True:
            store_key = tuple(sorted(name))
        else:
            store_key = (name,)
        if (len(store_key) == 1) and (len(limits[name].keys()) <= 2):
            bounds[store_key] = {
                "BELIEF": 0.0,
                "PLAUSIBILITY": 0.0
            }
            tuple_limits[store_key] = {}
            for limit, value in limits[name].items():
                tuple_limits[store_key][limit] = value
        else:
            raise Exception("Limits must only be on thetas, not combinations thereof.")

    # For handling unknown data
    unknown_length = 2
    for name in data.keys():
        if len(name) > unknown_length:
            unknown_length = len(name)
    unknown_mass = 0.0

    # Determine upper and lower bounds for each mass name in the data.
    for mass_name, mass in data.items():
        if len(mass_name) == unknown_length:
            # Used for visualization later
            unknown_mass += mass

        # Still need to apply the unknown data
        for name in bounds.keys():
            if set(name).intersection(set(mass_name)) == set(name):
                # Applies
                bounds[name]["PLAUSIBILITY"] += mass
                # Maybe applies to lower also
                if len(mass_name) == 1:
                    bounds[name]["BELIEF"] += mass

    # Upper and lower bounds have been created for each name in the ordering.  Apply the limits now
    for name, limit in tuple_limits.items():
        for limit_side, limit_value in limit.items():
            # Should be at most 2, and -ve means lower bound (floor), +ve means higher bound (ceiling)
            save_key = "OUTSIDE"
            if limit_side == "PLAUSIBILITY":
                if separate_low_high is True:
                    save_key = "OUTSIDE_HIGH"
                inside, without = check_limit(bounds[name][limit_side], limit_value)
                unknown_applicable = 0.0
                if limit_value > 0.0:
                    # Unknown can only drive plausibility down
                    unknown_applicable = min(unknown_mass, without)
                limits_data["INSIDE"] += inside
                limits_data["UNKNOWN"] += unknown_applicable
                limits_data[save_key] += without - unknown_applicable
                # For the end
                unknown_mass -= unknown_applicable
            elif limit_side == "BELIEF":
                # Checking lower limit
                if separate_low_high is True:
                    save_key = "OUTSIDE_LOW"
                inside, without = check_limit(bounds[name][limit_side], limit_value)
                unknown_applicable = 0.0
                if limit_value < 0.0:
                    # Unknown can only drive belief up
                    unknown_applicable = min(unknown_mass, without)
                limits_data["INSIDE"] += inside
                limits_data["UNKNOWN"] += unknown_applicable
                limits_data[save_key] += without - unknown_applicable
                # For the end
                unknown_mass -= unknown_applicable
            else:
                # Unknown limit - don't do anything
                raise Exception("Limit types can only be PLAUSIBILITY or BELIEF.")
    # Record the rest of the unknown that wasn't already assigned
    limits_data["UNKNOWN"] += unknown_mass

    # Finally, normalize
    total_mass = 0.0
    for data_value in limits_data.values():
        total_mass += data_value
    if total_mass > 0.0:  # Should be
        for data_name in limits_data.keys():
            # Shift to percent
            limits_data[data_name] = (limits_data[data_name] / total_mass)
    # Return the results
    return limits_data
