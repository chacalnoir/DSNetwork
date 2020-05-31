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

__version__ = '0.2.4'

from combinationRules.utilities import the_keys
from functools import reduce
from combinationRules import COMBINATION_METHODS
from math import pow, sqrt
from numpy import roots, array
import numpy as np
from itertools import permutations
from copy import deepcopy

ZERO_DELTA = 1e-4
ZERO_COEFFICIENT_DELTA = 1e-16
REPAIR_TOLERANCE = 0.2
MARGINAL_STATE = {
    "FLEXIBLE": "FLEXIBLE",
    "RECOMMENDED": "RECOMMENDED",
    "SET": "SET"
}


def previously_applicable_set_intersection(before_child_marginals, goal_intersection, current_intersection,
                                           current_parent, current_result):
    """
    Recursive algorithm which enables the full set of intersections up to the number of parents
    :param before_child_marginals: The dict of already-solved marginals
    :param goal_intersection: The goal set
    :param current_intersection: The current set intersection
    :param current_parent: The current parent number (counts backwards to the stop point)
    :param current_result: The current multiplication factor
    :return: The full set of previously applicable intersection values
    """
    return_result = 0.0
    for solved_marginal_key, solved_marginal_value in before_child_marginals[current_parent].items():
        # Only applies to sets of length longer than the goal
        if len(solved_marginal_key) > len(goal_intersection):
            if current_parent == 0:
                # Final step
                if set(solved_marginal_key).intersection(set(current_intersection)) == set(goal_intersection):
                    # This one applies
                    return_result += current_result * solved_marginal_value
            else:
                # Have more levels to go
                next_level_intersection = set(solved_marginal_key).intersection(set(current_intersection))
                if set(goal_intersection).intersection(next_level_intersection) == set(goal_intersection):
                    # This one could contain the goal intersection - have to keep going down
                    return_result += previously_applicable_set_intersection(before_child_marginals, goal_intersection,
                                                                            next_level_intersection, current_parent - 1,
                                                                            current_result * solved_marginal_value)
    # Go back up the chain
    return return_result


def universal_set_solution(partial_solution, value_states, child_marginals, universal_set):
    """
    Solves for the universal set value that will make the remaining universal set values the same
    :param partial_solution: dict of dicts of marginal keys to marginal values
    :param value_states: dict of dicts of which values cannot be adjusted (are already solved)
    :param child_marginals: dict of the original child marginal values
    :param universal_set: the universal set
    :return: tuple of solved and the value that solves this problem.
    """
    # First, find the remaining number of universal set values to calculate
    already_set = 1.0
    number_to_solve = 0
    for parent_num, parent_dict in partial_solution.items():
        if value_states[parent_num][universal_set] == MARGINAL_STATE["FLEXIBLE"]:
            number_to_solve += 1
        else:
            already_set *= partial_solution[parent_num][universal_set]
    try:
        before_child_marginal = pow(child_marginals[universal_set] / already_set,
                                    1.0 / float(number_to_solve))
        return True, before_child_marginal
    except Exception as e:
        print(str(e))
        print("universal_set_solution: Child marginal for universal set {} was less than zero.".
              format(child_marginals[universal_set]) + " Must be in [0.0, 1.0]")
        return False, 0.0


def root_finder_solution(partial_solution, value_states, child_marginals, marginal_key, universal_set):
    """
    Solves for the solution that will make the remaining values the same
    :param partial_solution: dict of dicts of marginal keys to marginal values
    :param value_states: dict of dicts of which values cannot be adjusted (are already solved)
    :param child_marginals: dict of the original child marginal values
    :param marginal_key: the key for which this solution is being run
    :param universal_set: the key for the universal set
    :return: tuple of solved and the value that solves this problem.
    """
    # Root finding method required since solving out all possible combinations seems like a bad idea
    number_of_parents = len(partial_solution)

    # Go through all the power cases.  Determine where each applies based on the number of unsolved values in
    #  the case.
    coefficients_by_power = {}
    for set_size in range(1, number_of_parents + 1):  # Include the total number of parents
        # Append each coefficient as it's calculated
        # The way this works is that the power is the number of non-"marginal_key" values that
        #  must be multiplied by the marginal key, and it must run through all permutations.  For
        #  example, given three options (a, b, c), if we are looking at marginal_key 'a' and
        #  power 2, then the possible combinations are (a,b,c)(a,b,c) for each pair of parents,
        #  (a,b)(a,b) for each pair of parents, and (a,c)(a,c) for each pair of parents.  All are
        #  summed up and result int the sum_of_applicables, which is the coefficient.
        for each_group in permutations(partial_solution.keys(), r=set_size):
            should_run = True
            if set_size > 1:  # combinations_with_replacement doesn't seem to work
                for each_pos in range(0, set_size - 1):
                    if each_group[each_pos + 1] <= each_group[each_pos]:
                        should_run = False
            if should_run is True:
                multiplier = 1.0  # The start value given what is already solved.
                power = 0
                # Only include parent dictionaries that are not part of the set.
                reduced_set_counter = 0
                partial_solution_without_removed_parents = deepcopy(partial_solution)
                renumbered_partial_solution_without_removed = {}
                for each_id in each_group:
                    if value_states[each_id][marginal_key] == MARGINAL_STATE["FLEXIBLE"]:
                        power += 1
                    else:
                        multiplier *= partial_solution[each_id][marginal_key]
                    removed_dict = partial_solution_without_removed_parents.pop(each_id, None)
                    if removed_dict is None:
                        print("root_finder_solution: parent " + str(each_id) +
                              " removal failed for - this shouldn't happen")
                # Renumber the remaining dictionaries
                for partial_dict in partial_solution_without_removed_parents.values():
                    renumbered_partial_solution_without_removed[reduced_set_counter] = deepcopy(partial_dict)
                    reduced_set_counter += 1
                # Run the method
                if len(renumbered_partial_solution_without_removed) > 0:
                    coefficient = \
                        previously_applicable_set_intersection(renumbered_partial_solution_without_removed,
                                                               marginal_key, marginal_key,
                                                               len(renumbered_partial_solution_without_removed) - 1,
                                                               multiplier)
                else:
                    coefficient = multiplier
                if power not in coefficients_by_power:
                    coefficients_by_power[power] = coefficient
                else:
                    coefficients_by_power[power] += coefficient

    # Handle special information for the last coefficient (marginal_key^0 term)
    # Find any previously solved values which apply to this calculation.  Note that this is actually
    #  rather complicated because it involves doing the set intersection number-of-parents times and
    #  seeing if that result is equal to the marginal_key
    sum_of_previously_solved_applicables = \
        previously_applicable_set_intersection(partial_solution, marginal_key, universal_set,
                                               number_of_parents - 1, 1.0)

    # Add the value
    if 0 not in coefficients_by_power:
        coefficients_by_power[0] = -child_marginals[marginal_key] + sum_of_previously_solved_applicables
    else:
        coefficients_by_power[0] += -child_marginals[marginal_key] + sum_of_previously_solved_applicables

    highest_power = max(list(coefficients_by_power.keys()))
    # Make sure all coefficients have been assigned
    for power in range(0, highest_power):
        if power not in coefficients_by_power:
            coefficients_by_power[power] = 0.0

    # Initialize the return values
    before_child_marginal = 0.0
    solved = True
    if highest_power == 1:
        # Linear - simple - solve for the intercept
        if abs(coefficients_by_power[1]) > ZERO_COEFFICIENT_DELTA:
            before_child_marginal = -coefficients_by_power[0] / coefficients_by_power[1]
        else:
            print("Unable to solve for linear root due to zero coefficient")
            return False, before_child_marginal
    elif highest_power == 2:
        # Quadratic formula to speed things up (-b +/-sqrt(b^2 - 4*a*c)) / (2*a)
        # Simplified since some information is known: (-b + sqrt(b^2 - 4*a*c)) / (2 * a)
        before_child_marginal =\
            (-coefficients_by_power[1] + sqrt(pow(coefficients_by_power[1], 2) - 4.0 * coefficients_by_power[2] *
                                              coefficients_by_power[0])) / (2.0 * coefficients_by_power[2])
    else:
        # Higher power than 2 - don't solve out any more, just run the root solver and evaluate for the best answer
        # Create the coefficients list
        coefficients = []
        for power in range(highest_power, -1, -1):
            coefficients.append(coefficients_by_power[power])
        try:
            found_roots = roots(array(coefficients))
            # Handle rounding
            found_roots = found_roots.real[abs(found_roots.imag) < ZERO_DELTA]
            # Limit in the range [0.0, 1.0]
            viable_roots = found_roots[found_roots <= (1.0 + ZERO_DELTA)]
            positive_viable_roots = viable_roots[viable_roots >= (0.0 - ZERO_DELTA)]

            if len(positive_viable_roots) > 1:
                viable_true_positives = positive_viable_roots[positive_viable_roots >= 0.0]
                if len(viable_true_positives) > 0:
                    # Have a solution - always choose the minimum if more than one to maintain
                    #  flexibility.
                    # Always clip to the right range since dealing with roundoff
                    before_child_marginal = min(1.0, max(0.0, np.min(viable_true_positives)))
                    if len(viable_true_positives) > 1:
                        greater_than_zero_deltas = 0
                        for viable_root in viable_true_positives:
                            if viable_root > ZERO_DELTA:
                                greater_than_zero_deltas += 1
                        if greater_than_zero_deltas > 1:
                            # Only print if there is actually an issue with several non-zero results.
                            print_str = "Too many positive solutions (" + str(len(viable_true_positives)) + \
                                        ") which are ("
                            for viable_root in viable_true_positives:
                                print_str += str(viable_root) + ","
                            print_str += ").  Need to work on this"
                            print(print_str)
                else:
                    # Take the best option which is effectively zero
                    before_child_marginal = min(1.0, max(0.0, np.min(positive_viable_roots)))
            elif len(positive_viable_roots) > 0:
                # Have a single solution - good.
                before_child_marginal = min(1.0, max(0.0, np.min(positive_viable_roots)))
            elif len(viable_roots) > 0:
                # Have at least one negative solution.  This means the method won't work for this
                #  particular combination, but that is okay.  We need to get a bad answer, store it,
                #  and modify to fix it.
                # We know it is less than 0.0, so we take the maximum to get as close to 0.0 as possible
                before_child_marginal = np.max(viable_roots)
            else:
                # Find the number of options
                num_options = 0
                for child_key in child_marginals.keys():
                    if len(child_key) == 1:
                        num_options += 1
                print("No viable solution with number of parents " + str(number_of_parents) + " and options: " +
                      str(num_options) + ".  Need to work on this.")
                # Shouldn't see this happen.  Usually means roundoff error.
                solved = False
        except ValueError as e:
            print("Unable to solve for roots due to " + str(e))
            solved = False

    # Return the solution
    return solved, before_child_marginal


def constrained_solver_for_root_finder(partial_repairs, value_states, child_marginals, universal_set, min_length):
    """
    Takes a partial set of before_child_marginals and re-solves to balance the equations
    :param partial_repairs: dict of dicts of marginal keys to values (or None)
    :param value_states: dict of dicts of values that cannot be changed at this point
    :param child_marginals: dict of the original combined child marginals
    :param universal_set: the universal set
    :param min_length: the minimum length to fix
    :return whether it solved
    """
    # Start from the universal set and work backwards
    current_length = len(universal_set)
    powerset = list(child_marginals.keys())

    while current_length >= min_length:
        remove_list = []
        for marginal_key in powerset:
            # Check if any of the marginals for that key are still mobile
            mobile_values = 0
            for parent_dict in value_states.values():
                if parent_dict[marginal_key] == MARGINAL_STATE["FLEXIBLE"]:
                    mobile_values += 1
            if (len(marginal_key) == current_length) and (mobile_values > 0):
                if current_length == len(universal_set):
                    # The universal set is special
                    solved, before_child_marginal = universal_set_solution(partial_repairs, value_states,
                                                                           child_marginals, universal_set)
                    if solved is False:
                        # Break early
                        return False
                else:
                    # Use the root finder for all remaining options because, due to the immobile values,
                    #  we don't know if this is quadratic or something else.
                    solved, before_child_marginal = root_finder_solution(partial_repairs, value_states,
                                                                         child_marginals, marginal_key, universal_set)
                    if solved is False:
                        # Break early
                        return False

                # Assign to unsolved parents
                for repaired_num, repaired_dict in partial_repairs.items():
                    if value_states[repaired_num][marginal_key] == MARGINAL_STATE["FLEXIBLE"]:
                        repaired_dict[marginal_key] = before_child_marginal

                # Finally, remove it from the set
                remove_list.append(marginal_key)

        # Reduce the current length by 1
        current_length -= 1
        # Reduce the list
        for remove_item in remove_list:
            powerset.remove(remove_item)
    return True  # It worked


def fix_negatives_for_root_finder(before_child_marginals, child_marginals, value_states, attempt_iteration=False):
    """
    Takes a partial set of before_child_marginals that has been solved but has negative values due to an inability
     to solve for some values in the current level assuming all values are equal.  This routine modifies the results
     to be valid and allow for the solution method to continue.
    :param before_child_marginals: dict of dicts of marginal keys to values
    :param child_marginals: dict of the original combined child marginals.
    :param value_states: dict of dicts of the ability to change values given any repairs already done.
    :param attempt_iteration: bool whether to attempt to fix issues via iteration
    :return whether solved
    """
    repaired_solution = {}
    num_parents = 0
    current_set_length = None
    universal_set = None
    corrected_issue_sets = []  # Each issue set can be corrected at most once
    current_issue_set = None
    for parent_key, parent_dict in before_child_marginals.items():
        repaired_solution[parent_key] = {}
        num_parents += 1
        for marginal_key, marginal_value in parent_dict.items():
            # Since we have to look through all the marginal values and keys anyway
            repaired_solution[parent_key][marginal_key] = before_child_marginals[parent_key][marginal_key]
            # Get the universal set
            if (universal_set is None) or (len(marginal_key) > len(universal_set)):
                universal_set = marginal_key
            # Get the current set length - will be the same as the length in any of the issue sets
            if (current_set_length is None) or (len(marginal_key) < current_set_length):
                current_set_length = len(marginal_key)
            # Check if this set has to be repaired.  Should only apply to current set length items
            if (marginal_value < 0.0) and (current_issue_set is None):
                current_issue_set = marginal_key
    # Fix each issue
    while current_issue_set is not None:
        # First, ensure that a previous fix has not fixed this issue.
        if repaired_solution[0][current_issue_set] < ZERO_DELTA:
            # First check: if the combined value of the issue set is zero and the current solution is negative, there is
            #  ONLY one way to solve this: one parent must have the same solution as the combined solution and all other
            #  parents must be the universal set.  Because there can be no negatives, a zero is REQUIRED to get this to
            #  work.
            if abs(child_marginals[current_issue_set]) < ZERO_DELTA:
                # Effectively zero - this is a special case.
                # In order for this to work, there must be a lot of zeros.  For any marginals that contribute to the
                #  sum_of_previously_solved_applicables, those marginals in all parents except one must be zero, except.
                #  for the universal set.  The universal set for those parents will have to be modified to compensate
                #  for all other solved values, and the universal set in the remaining parent will compensate to
                #  correctly solve for the universal set.
                # Get the affected sets
                affected_sets = {}
                delta_to_zero = {}  # Calculate which marginal set of marginals should not be zeroed.
                for marginal_key in child_marginals.keys():
                    if (len(marginal_key) > len(current_issue_set)) and (len(marginal_key) < len(universal_set)) and\
                            (set(marginal_key).intersection(set(current_issue_set)) == set(current_issue_set)):
                        # This one has an impact but is not the universal set
                        for parent_num in before_child_marginals.keys():
                            if parent_num not in affected_sets:
                                affected_sets[parent_num] = {}
                            if parent_num not in delta_to_zero:
                                delta_to_zero[parent_num] = 0.0
                            # Minimize the cost of this change
                            delta_to_zero[parent_num] += before_child_marginals[parent_num][marginal_key]
                            # Store the current value to figure out how to modify it.
                            affected_sets[parent_num][marginal_key] = before_child_marginals[parent_num][marginal_key]
                max_cost = 0.0
                no_change_parent = 0
                for parent_num, cost in delta_to_zero.items():
                    if cost > max_cost:
                        max_cost = cost
                        no_change_parent = parent_num
                # The parent that isn't being changed to zeros is known.  The rest of the parents will need to change.
                for parent_num, cost in delta_to_zero.items():
                    if parent_num != no_change_parent:
                        # Change this parent
                        for marginal_key in before_child_marginals[parent_num].keys():
                            if marginal_key in affected_sets[parent_num]:
                                # Change this one
                                if value_states[parent_num][marginal_key] != MARGINAL_STATE["SET"]:
                                    repaired_solution[parent_num][marginal_key] = 0.0
                                    # Cannot be changed after this
                                    value_states[parent_num][marginal_key] = MARGINAL_STATE["SET"]
                                elif (value_states[parent_num][marginal_key] == MARGINAL_STATE["SET"]) and\
                                        (repaired_solution[parent_num][marginal_key] > ZERO_DELTA):
                                    print("fix_negatives_for_root_finder: previously set SET state " +
                                          str(marginal_key) + " cannot be changed to zero - cannot solve this issue.")
                                    return False
                            elif set(marginal_key) == set(universal_set):
                                # This is required because otherwise the universal set can be solved to 0.0 for
                                #  all values when the child marginal for the universal set is 0.0.  That in turn
                                #  creates a divide-by-zero scenario which causes the solver to fail.
                                # Goes up to balance, but constrain to <= 1.0
                                repaired_solution[parent_num][marginal_key] =\
                                    min(1.0, repaired_solution[parent_num][marginal_key] + cost)
                                # The value to keep during this solution, but can be changed later
                                value_states[parent_num][marginal_key] = MARGINAL_STATE["RECOMMENDED"]
                # Now all values that need to be zeroed have been.  Next we need to boost the values in the remaining
                #  set to balance.
                # This is an interesting version of the solver.  It is now more tightly constrained but must still
                #  correctly backwards solve the required values.
                # Rebalance remaining values
                solved = constrained_solver_for_root_finder(repaired_solution, value_states, child_marginals,
                                                            universal_set,  len(current_issue_set))
                if solved is False:
                    # Break early
                    return False
            elif attempt_iteration is True:
                # This is the hard case - it's a bad situation, and we don't know how to precisely fix it.  First,
                #  get some metrics that may help.
                # Get the previously solved applicables value.
                sum_of_previously_solved_applicables = \
                    previously_applicable_set_intersection(repaired_solution, current_issue_set, universal_set,
                                                           len(repaired_solution) - 1, 1.0)
                original_sum = sum_of_previously_solved_applicables
                # Get the child marginal value.
                combined_marginal = child_marginals[current_issue_set]
                # The issue is caused by sum_of_previously_solved_applicables being greater than combined_marginal.
                # The question then becomes what to change and by how much to bring that into tolerance.
                # If brought just barely into tolerance, the resulting parent values for issue_set will be zero.
                # TODO: Perform calculations to see what changes have the desired effect.
                failed_fraction = sum_of_previously_solved_applicables / combined_marginal
                # From here, trying an iterative approach.  There is not enough information to do a single step
                #  calculation.  However, it does appear that by spreading the values that affect the issue set,
                #  a valid (non-optimal) solution can be obtained.  Since this is a corner case, we don't require
                #  optimality.  Start by getting the best result (either a valid or as close to valid as possible)
                #  by spreading the impact values.  If that doesn't solve the problem, then move to offset, then
                #  distribution.
                # Goal: failed_fraction <= 1.0
                original_repaired_solution = deepcopy(repaired_solution)
                original_value_states = deepcopy(value_states)
                current_spread_multiplier = 1.0  # Starting guess
                current_offset = 0.0  # Starting guess
                current_distribution = 1.0  # Starting guess (1.0 is fully weighted out, 0.0 is evenly distributed).
                current_dimension = 1  # 1 is spread, 2 is offset, 3 is distribution
                spread_attempts = {
                    0.0: repaired_solution[0][current_issue_set]
                }
                max_iterations = 10
                iteration = 0
                valid_solution = False
                while (not valid_solution) and (iteration < max_iterations):
                    if current_dimension == 1:  # Working on the spread
                        # Solve based on the current spread
                        # First, reset the repaired solution and value states
                        repaired_solution = deepcopy(original_repaired_solution)
                        value_states = deepcopy(original_value_states)

                        # Update (n is odd) ? (n-1)/2 : n/2 parents where n is the total number of parents.
                        # TODO: Which ones to update?  The first ones, I guess...
                        odd_number = 0
                        if len(repaired_solution) % 2 == 1:
                            num_to_fix = (len(repaired_solution) - 1) / 2
                            odd_number = 1
                        else:
                            num_to_fix = len(repaired_solution) / 2
                        fix_counter = 0
                        for parent_num, parent_dict in repaired_solution.items():
                            for marginal_key in parent_dict.keys():
                                if (len(marginal_key) > len(current_issue_set)) and \
                                        (set(marginal_key).intersection(set(current_issue_set)) ==
                                         set(current_issue_set)) and \
                                        (value_states[parent_num][marginal_key] != MARGINAL_STATE["SET"]):
                                    if (num_parents > 2) or (len(marginal_key) < len(universal_set)):
                                        # If there are more than 2 parents, then the universal set affects this.
                                        #  Otherwise, it doesn't.
                                        if fix_counter < num_to_fix:
                                            # For the middle one of an odd number,
                                            #  leave the same but recommend the setting
                                            parent_dict[marginal_key] /= failed_fraction * current_spread_multiplier
                                        # Should not be changed after this
                                        value_states[parent_num][marginal_key] = MARGINAL_STATE["RECOMMENDED"]
                            # Exit when appropriate
                            fix_counter += 1
                            if ((fix_counter == num_to_fix) and (odd_number == 0)) or (fix_counter > num_to_fix):
                                break

                        # Rebalance remaining values
                        solved = constrained_solver_for_root_finder(repaired_solution, value_states, child_marginals,
                                                                    universal_set, len(current_issue_set))
                        if solved is False:
                            # Could not solve - went too far
                            spread_attempts[current_spread_multiplier] = False
                        else:
                            # Solved - retain the information
                            # Remember, all values for the issue_set will be the same, so grab from [0]
                            spread_attempts[current_spread_multiplier] = repaired_solution[0][current_issue_set]
                            if repaired_solution[0][current_issue_set] >= 0.0:
                                valid_solution = True  # Found a working solution
                        iteration += 1  # Finished this iteration
                        if valid_solution is False:
                            # Determine the next spread attempt
                            sorted_tested_spreads = sorted(spread_attempts.keys())
                            current_index = sorted_tested_spreads.index(current_spread_multiplier)
                            if spread_attempts[current_spread_multiplier] is False:  # Could not solve - has to back off
                                # Half with the next lowest spread attempt since we know that one was good
                                current_spread_multiplier = (current_spread_multiplier -
                                                             sorted_tested_spreads[current_index - 1]) / 2.0 \
                                                             + sorted_tested_spreads[current_index - 1]
                            else:  # Determine whether to go forward or backwards based on current trend
                                # Note that it is always greater than index 0 since index 0 is for zero spread
                                if spread_attempts[current_spread_multiplier] >\
                                        spread_attempts[sorted_tested_spreads[current_index - 1]]:
                                    # Going in the right direction - move forward based on simple extrapolation
                                    slope = (spread_attempts[current_spread_multiplier] -
                                             spread_attempts[sorted_tested_spreads[current_index - 1]]) /\
                                             (current_spread_multiplier - sorted_tested_spreads[current_index - 1])
                                    additional_movement = (0.0 - spread_attempts[current_spread_multiplier]) / slope
                                    current_spread_multiplier += additional_movement
                                else:
                                    # Going in the wrong direction - back off by 1/2 (just to get as close as possible
                                    #  before next dimension)
                                    current_spread_multiplier = (current_spread_multiplier -
                                                                 sorted_tested_spreads[current_index - 1]) / 2.0\
                                                                 + sorted_tested_spreads[current_index - 1]
                if valid_solution is False:
                    # It was never solved
                    return False
            else:
                return False
        # Clean up the solution and check for the next issue set
        for parent_dict in repaired_solution.values():
            sum_of_values = 0.0
            for marginal_value in parent_dict.values():
                sum_of_values += marginal_value
            if abs(sum_of_values) > ZERO_DELTA:
                for marginal_key, marginal_value in parent_dict.items():
                    parent_dict[marginal_key] /= sum_of_values
        corrected_issue_sets.append(current_issue_set)
        current_issue_set = None
        # Find the next issue set
        for parent_key, parent_dict in repaired_solution.items():
            for marginal_key, marginal_value in parent_dict.items():
                # Check if this set has to be repaired.  Should only apply to current set length items
                if (marginal_value < 0.0) and (current_issue_set is None):
                    current_issue_set = marginal_key

    # Assign back the repaired solution
    for parent_key, parent_dict in before_child_marginals.items():
        for marginal_key in parent_dict.keys():
            before_child_marginals[parent_key][marginal_key] = repaired_solution[parent_key][marginal_key]
    return True


def solve_for_parents_with_root_finder(combination_method, child_marginals, number_of_parents):
    """
    Assumes the parent marginals (before the combination method is applied) are all the same.
    :param combination_method: the combinationRules COMBINATION_METHOD
    :param child_marginals: dict of child marginals (tuples to doubles)
    :param number_of_parents: positive int number of parents
    :return: tuple of bool (whether the solution was found) and dict (solution of parent marginals tuples to doubles)
    """

    # Make sure this is working with a full set of marginals.
    full_set_marginals = {}
    # List the full set of inputs
    thetas = list(reduce(lambda a, b: a | set(the_keys(b)), child_marginals.keys(), set()))
    # The global ignorance set might be in here
    powerset = [tuple(sorted([x for j, x in enumerate(thetas) if (i >> j) & 1])) for i in range(2 ** len(thetas))]
    # Remove the null set from the powerset
    powerset.remove(())
    # Make sure this is working with tuples
    # Find the universal set while iterating through
    max_len = 0
    universal_set = None
    for marginal_key, marginal_value in child_marginals.items():
        if isinstance(marginal_key, tuple) is False:
            new_key = (marginal_key,)
        else:
            new_key = tuple(sorted(marginal_key))
        full_set_marginals[new_key] = marginal_value
        if len(new_key) > max_len:
            max_len = len(new_key)
            universal_set = new_key
    for marginal_key in powerset:
        if marginal_key not in full_set_marginals:
            full_set_marginals[marginal_key] = 0.0

    if number_of_parents < 1:
        # No solution - invalid problem
        return False, full_set_marginals
    elif number_of_parents == 1:
        # Simple solution since all D-S combination methods should follow this axiom
        return True, full_set_marginals

    # An actual problem - check whether there are available options
    # Evidential reasoning also works because that reduces to Dempster's rule under equal weight and reliability.
    if (combination_method == COMBINATION_METHODS["DEMPSTER_SHAFER"]) or\
            (combination_method == COMBINATION_METHODS["MURPHY"]) or\
            (combination_method == COMBINATION_METHODS["ZHANG"]):
        # Define the return
        before_child_marginals = {}
        value_states = {}
        for parent_num in range(0, number_of_parents):
            before_child_marginals[parent_num] = {}  # Each parent is defined separately, even if they all end the same
            value_states[parent_num] = {}

        # Start from the universal set and work backwards
        current_length = max_len
        while current_length > 0:
            remove_list = []
            for marginal_key in powerset:
                if len(marginal_key) == current_length:
                    # This is one to do this round
                    # Initially, all values are flexible
                    for parent_num in range(0, number_of_parents):
                        value_states[parent_num][marginal_key] = MARGINAL_STATE["FLEXIBLE"]

                    if current_length == max_len:
                        # The universal set is special - use the solution routine designed for this set.
                        solved, before_child_marginal = universal_set_solution(before_child_marginals, value_states,
                                                                               full_set_marginals, universal_set)
                    else:
                        solved, before_child_marginal = root_finder_solution(before_child_marginals, value_states,
                                                                             full_set_marginals, marginal_key,
                                                                             universal_set)
                    if solved is False:
                        # Break early
                        print("Solution failed for combination method " + combination_method +
                              " at key " + str(marginal_key))
                        return solved, before_child_marginals

                    # Assign to all parents initially.  Modify later if required
                    for before_child_marginal_dict in before_child_marginals.values():
                        before_child_marginal_dict[marginal_key] = before_child_marginal

                    # Finally, remove it from the set
                    remove_list.append(marginal_key)

            # Check to see if modifications are required (if a bad value was found).
            has_negatives = False
            for before_child_marginal_dict in before_child_marginals.values():
                for marginal_key in remove_list:
                    if before_child_marginal_dict[marginal_key] < 0.0:
                        has_negatives = True
                        break
            if has_negatives is True:
                # Fix the solution so far
                solved = fix_negatives_for_root_finder(before_child_marginals, full_set_marginals, value_states)
                if solved is False:
                    # Break early
                    print("Solution failed for combination method " + combination_method + ".")
                    return solved, before_child_marginals

            # Reduce the current length by 1
            current_length -= 1
            # Reduce the list
            for remove_item in remove_list:
                powerset.remove(remove_item)

        # Verify no negatives remain
        solved = True
        for parent_dict in before_child_marginals.values():
            for marginal_value in parent_dict.values():
                if marginal_value < 0.0:
                    solved = False
                    break
        # Normalize - won't affect negatives
        for before_child_marginal_dict in before_child_marginals.values():
            total_sum = sum(list(before_child_marginal_dict.values()))
            if abs(total_sum - 1.0) > ZERO_COEFFICIENT_DELTA:  # Only if not already normalized - need it really close
                if total_sum > ZERO_DELTA:
                    # Normalize
                    iterate_set = before_child_marginal_dict.keys()
                    for marginal_key in iterate_set:
                        before_child_marginal_dict[marginal_key] /= total_sum
                else:
                    print("Total sum is zero for combination method " + combination_method + ".  Unable to normalize.")
        # Return the results
        return solved, before_child_marginals
    else:
        # Cannot solve this way
        return False, {1: full_set_marginals}
