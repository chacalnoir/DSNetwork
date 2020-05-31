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
Adds plotting capabilities for the metrics to enable animated charts for easier data visualization
"""

from metrics.MetricsUtilities import calculate_each_limit_data, calculate_each_ambiguity_data,\
    compare_ds_labels, get_tuple_name
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from copy import deepcopy
import csv
from os import sep
from functools import cmp_to_key
import numpy as np
# from statistics import median_low
# For GIF generation
# from os import system

PLOT_TYPE = {
    "RISK": "RISK",
    "INTERNAL_DATA": "INTERNAL_DATA",
    "EVIDENCE": "EVIDENCE",
    "POTENTIALS": "POTENTIALS",
    "ANIMATION": "ANIMATION",
    "ANIMATE_INTERNAL": "ANIMATE_INTERNAL",
    "ANIMATE_EVIDENCE": "ANIMATE_EVIDENCE"
}

VISUALIZATION_TYPE = {
    "DATA": "DATA",
    "AMBIGUITY": "AMBIGUITY",
    "LIMITS": "LIMITS"
}

DATA_COLOUR_MAP = {
    ('HIGH', ): "red",
    ('LOW', ): "green",
    ('MEDIUM', ): "yellow",
    ('HIGH', 'LOW'): "palegoldenrod",
    ('HIGH', 'MEDIUM'): "orange",
    ('LOW', 'MEDIUM'): "blue",
    ('HIGH', 'LOW', 'MEDIUM'): "gray",
    "INSIDE": "green",
    "UNKNOWN": "gray",
    "OUTSIDE": "red",
    "OUTSIDE_LOW": "darkred",
    "OUTSIDE_HIGH": "lightcoral",
    "KNOWN": "blue",
    "AMBIGUOUS": "deepskyblue",
}

TRANSITION_COLOUR_MAP = {
    ('HIGH', ): {
        ('HIGH', ): "darkred",
        ('LOW', ): "mistyrose",
        ('MEDIUM', ): "salmon",
        ('HIGH', 'LOW'): "red",
        ('HIGH', 'MEDIUM'): "firebrick",
        ('LOW', 'MEDIUM'): "tomato",
        ('HIGH', 'LOW', 'MEDIUM'): "rosybrown"
    },
    ('LOW', ): {
        ('HIGH', ): "darkgreen",
        ('LOW', ): "green",
        ('MEDIUM', ): "lime",
        ('HIGH', 'LOW'): "seagreen",
        ('HIGH', 'MEDIUM'): "forestgreen",
        ('LOW', 'MEDIUM'): "springgreen",
        ('HIGH', 'LOW', 'MEDIUM'): "palegreen"
    },
    ('MEDIUM', ): {
        ('HIGH', ): "olive",
        ('LOW', ): "lawngreen",
        ('MEDIUM', ): "yellow",
        ('HIGH', 'LOW'): "khaki",
        ('HIGH', 'MEDIUM'): "olivedrab",
        ('LOW', 'MEDIUM'): "lightgreen",
        ('HIGH', 'LOW', 'MEDIUM'): "lightgoldenrodyellow"
    },
    ('HIGH', 'LOW'): {
        ('HIGH', ): "darkgoldenrod",
        ('LOW', ): "lemonchiffon",
        ('MEDIUM', ): "gold",
        ('HIGH', 'LOW'): "palegoldenrod",
        ('HIGH', 'MEDIUM'): "goldenrod",
        ('LOW', 'MEDIUM'): "yellowgreen",
        ('HIGH', 'LOW', 'MEDIUM'): "beige"
    },
    ('HIGH', 'MEDIUM'): {
        ('HIGH', ): "sienna",
        ('LOW', ): "navajowhite",
        ('MEDIUM', ): "bisque",
        ('HIGH', 'LOW'): "peachpuff",
        ('HIGH', 'MEDIUM'): "orange",
        ('LOW', 'MEDIUM'): "antiquewhite",
        ('HIGH', 'LOW', 'MEDIUM'): "linen"
    },
    ('LOW', 'MEDIUM'): {
        ('HIGH', ): "purple",
        ('LOW', ): "aquamarine",
        ('MEDIUM', ): "cyan",
        ('HIGH', 'LOW'): "darkturquoise",
        ('HIGH', 'MEDIUM'): "deeppink",
        ('LOW', 'MEDIUM'): "blue",
        ('HIGH', 'LOW', 'MEDIUM'): "lightskyblue"
    },
    ('HIGH', 'LOW', 'MEDIUM'): {
        ('HIGH', ): "black",
        ('LOW', ): "gainsboro",
        ('MEDIUM', ): "darkgray",
        ('HIGH', 'LOW'): "silver",
        ('HIGH', 'MEDIUM'): "dimgray",
        ('LOW', 'MEDIUM'): "lightgray",
        ('HIGH', 'LOW', 'MEDIUM'): "gray"
    }
}
TO_PERCENT = 100.0  # Percent multiplier


def create_error_plot(data, title, plot_max_min=False):
    fig, axs = plt.subplots(len(data))
    counter = 0
    for each_data in data:
        if len(data) > 1:
            ax = axs[counter]
        else:
            ax = axs
        if plot_max_min is False:
            ax.errorbar(np.array(each_data["labels"]), np.array(each_data["means"]),
                        np.array(each_data["standard_deviations"]), linestyle='None', marker='^', capsize=5)
        else:
            test_bar = np.array([each_data["mins"], each_data["maxs"]])
            test_means = np.array(each_data["means"])
            ax.errorbar(np.array(each_data["labels"]), test_means, test_bar,
                        linestyle='None', marker='^', capsize=5)
        ax.grid(b=True, which='major', axis='y')
        ax.set_title(each_data["title"], size=15)
        for tick in ax.xaxis.get_major_ticks():
            tick.label.set_fontsize(15)
        for tick in ax.yaxis.get_major_ticks():
            tick.label.set_fontsize(15)
        counter += 1
    fig.suptitle(title, fontsize=25)
    plt.show()


def create_violin_plot(data, title, plot_max_min=False, y_label=None):
    fig, axs = plt.subplots(len(data))
    counter = 0
    for each_data in data:
        if len(data) > 1:
            ax = axs[counter]
        else:
            ax = axs
        test = np.array([each_data["values"]])
        parts = ax.violinplot(test[0], showextrema=True, showmedians=True)

        ax.grid(b=True, which='major', axis='y')
        ax.set_title(each_data["title"], size=15)
        for tick in ax.xaxis.get_major_ticks():
            tick.label.set_fontsize(15)
        for tick in ax.yaxis.get_major_ticks():
            tick.label.set_fontsize(15)
        ax.set_xticks(np.arange(1, len(each_data["labels"]) + 1))
        ax.set_xticklabels(np.array(each_data["labels"]))
        if y_label is not None:
            ax.set_ylabel(y_label)
            ax.yaxis.label.set_fontsize(15)

        for pc in parts['bodies']:
            pc.set_facecolor('lightcoral')
            pc.set_edgecolor('black')
            pc.set_alpha(1)
        counter += 1
    fig.suptitle(title, fontsize=25)
    plt.show()


def create_histogram_plot(data, plot_names, title, bins=10):
    """
    Creates histograms of the data
    :param data: Dictionary of data for plots
    :param plot_names: The names of the datasets in data to plot
    :param title: Overall title
    :param bins: Number of bins for the histogram
    """
    num_figures = 0
    for each_data in data:
        if each_data["title"] in plot_names:
            num_figures += len(each_data["labels"])
    fig, axs = plt.subplots(num_figures)
    counter = 0
    for each_data in data:
        if each_data["title"] in plot_names:
            for label_counter in range(0, len(each_data["labels"])):
                if num_figures > 1:
                    ax = axs[counter]
                else:
                    ax = axs
                ax.hist(each_data["values"][label_counter], bins)
                ax.grid(b=True, which='major', axis='y')
                ax.set_title(each_data["title"] + "_" + each_data["labels"][label_counter], size=15)
                for tick in ax.xaxis.get_major_ticks():
                    tick.label.set_fontsize(15)
                for tick in ax.yaxis.get_major_ticks():
                    tick.label.set_fontsize(15)
                counter += 1
    fig.suptitle(title, fontsize=25)
    plt.show()


def update_plot(frame, *fargs):
    """
    The update function for the figures
    :param frame: the update frame
    :param fargs: includes the data for each one (data_to_plot(dict), axis, name(string))
    """
    frame_data = fargs[0][frame]
    ax = fargs[1]
    name = fargs[2]
    ax.clear()
    ax.axis('equal')

    # Convert from frame data to display data
    masses = []
    labels = []
    for data_name, data in frame_data.items():
        if isinstance(data, dict) is True:
            # Dual level data
            for child_name, mass in data.items():
                masses.append(mass)
                labels.append(str(data_name) + "." + str(child_name))
        else:
            # Single level data
            masses.append(data)
            labels.append(data_name)
    ax.pie(masses, labels=labels, autopct='%1.1f%%', shadow=True, startangle=140)
    ax.set_title(name + " " + str(frame))


def calculate_node_ambiguity_data(data):
    """
    Calculates the ambiguity data
    :param data: the dictionary of index to mass dictionary
    :return: dict of ambiguity data by index
    """
    ambiguity_data = {}
    for index, data in data.items():
        ambiguity_data[index] = calculate_each_ambiguity_data(data)
    return ambiguity_data


def calculate_transition_ambiguity_data(data):
    """
    Calculates the ambiguity data for transitions.  This is defined as anything that is going from ambiguous to
     ambiguous, ambiguous to known, or known to ambiguous.  Unknown is defined as unknown on either the mapping from
     or mapping to side.  Known requires both the from and to sides to be known
    :param data: the dictionary of index to potentials dictionary
    :return: dict of ambiguity data by index
    """
    # Ambiguity is all data that is longer than 1 in the tuple but less than the maximum length.
    ambiguity_data = {}
    for index, data in data.items():
        max_length = 2
        mass_of_max_length = {
            "KNOWN": 0.0,
            "AMBIGUOUS": 0.0,
            "UNKNOWN": 0.0
        }
        ambiguity_data[index] = {
            "KNOWN": 0.0,
            "AMBIGUOUS": 0.0,
            "UNKNOWN": 0.0
        }
        for parent_mass_name, parent_to_child_dict in data.items():
            parent_ambiguity_data = calculate_each_ambiguity_data(parent_to_child_dict)
            if len(parent_mass_name) == 1:
                # Unambiguous parent, so assignments are one-to-one from the parent_ambiguity_data
                for key_name, mass in parent_ambiguity_data.items():
                    ambiguity_data[index][key_name] += parent_ambiguity_data[key_name]
            elif (len(parent_mass_name) > 1) and (len(parent_mass_name) < max_length):
                # Ambiguous - KNOWN and AMBIGUOUS go to AMBIGUOUS.  UNKNOWN goes to UNKNOWN
                ambiguity_data[index]["AMBIGUOUS"] += parent_ambiguity_data["AMBIGUOUS"] +\
                    parent_ambiguity_data["KNOWN"]
                ambiguity_data[index]["UNKNOWN"] += parent_ambiguity_data["UNKNOWN"]
            elif len(parent_mass_name) == max_length:
                # Not sure yet if ambiguous or unknown
                for key_name, mass in parent_ambiguity_data.items():
                    mass_of_max_length[key_name] += parent_ambiguity_data[key_name]
            elif len(parent_mass_name) > max_length:
                # New max length
                ambiguity_data[index]["AMBIGUOUS"] += mass_of_max_length["AMBIGUOUS"] + \
                                                      mass_of_max_length["KNOWN"]
                ambiguity_data[index]["UNKNOWN"] += mass_of_max_length["UNKNOWN"]
                max_length = len(parent_mass_name)
                # Assign to the longest mass name length
                for key_name, mass in parent_ambiguity_data.items():
                    mass_of_max_length[key_name] = parent_ambiguity_data[key_name]
        # Once done, assign the max length to unknown
        ambiguity_data[index]["UNKNOWN"] += mass_of_max_length["KNOWN"] + mass_of_max_length["AMBIGUOUS"] + \
            mass_of_max_length["UNKNOWN"]
    return ambiguity_data


def calculate_limit_data(limits, data, separate_low_high=False):
    """
    Calculates data that is inside, outside, and unknown of the limits
    :param limits: dict of name to belief and plausibility limits
    :param data: dict of mass names to masses
    :param separate_low_high: bool whether to separate the low/high results
    :return: dict of index to INSIDE, OUTSIDE, UNKNOWN to masses
    """
    limit_data = {}
    for index in data.keys():
        limit_data[index] = calculate_each_limit_data(limits, data[index], separate_low_high)
    return limit_data


def add_area_labels(labels, x_data, y_data, ax):
    """
    Adds the area labels
    :param labels: list of labels
    :param x_data: list of x data
    :param y_data: list of lists of y data
    :param ax: the axis to which to add the labels
    """
    # Create the labels
    counter = 0
    for label in labels:
        y_index = y_data[counter].index(max(y_data[counter]))
        x_position = x_data[y_index]
        y_position = y_data[counter][y_index] * 0.5
        for prev_counter in range(0, counter):
            y_position += y_data[prev_counter][y_index]
        ax.text(x_position, y_position, label)
        counter += 1


def get_plot_layout(has_some_internal_data, has_some_evidence, plot_names, plot_evidence, plot_internal,
                    node_by_name, transition_by_name, saving_animation):
    """
    Gets the plot layout information including columns, rows, and whether there are evidence and internal data
    :param has_some_internal_data: bool of whether there is some internal data (reduces loops)
    :param has_some_evidence: bool of whether there is some evidence (reduces loops)
    :param plot_names: optional list of names to plot or None
    :param plot_evidence: bool of whether to plot evidence
    :param plot_internal: bool of whether to plot internal data
    :param node_by_name: dict of node information by node name
    :param transition_by_name: None or dict of transition information by transition name
    :param saving_animation: bool whether the goal is to save an animation
    :return: tuple of (rows, columns, nodes)
    """
    has_internal_data = has_some_internal_data
    has_evidence = has_some_evidence
    num_nodes = len(node_by_name)
    if plot_names is not None:
        num_nodes = 0
        has_internal_data = False
        has_evidence = False
        for name in node_by_name.keys():
            if name in plot_names:
                num_nodes += 1
                if (PLOT_TYPE["INTERNAL_DATA"] in node_by_name[name]) and\
                        (len(node_by_name[name][PLOT_TYPE["INTERNAL_DATA"]]) > 0):
                    has_internal_data = True
                if (PLOT_TYPE["EVIDENCE"] in node_by_name[name]) and\
                        (len(node_by_name[name][PLOT_TYPE["EVIDENCE"]]) > 0):
                    has_evidence = True
    if transition_by_name is None:
        num_transitions = 0
    else:
        num_transitions = len(transition_by_name)
        if plot_names is not None:
            num_transitions = 0
            for name in transition_by_name.keys():
                if name in plot_names:
                    num_transitions += 1

    num_cols = max(num_nodes, num_transitions)
    num_rows = 0
    if num_nodes > 0:
        num_rows += 1
        if (plot_evidence is True) and (has_evidence is True):
            num_rows += 1
        if (plot_internal is True) and (has_internal_data is True):
            num_rows += 1
    if num_transitions > 0:
        num_rows += 1
    if saving_animation is True:
        # So that only one appears per file
        num_rows = 1
        num_cols = 1
    return num_rows, num_cols, num_nodes


def get_ax(ax, num_rows, num_cols, row_counter, col_counter):
    """
    Returns the correct axis since matplotlib changes whether they are a single value or a 1...n dimensional array
    :param ax: the set of axes for this plot
    :param num_rows: int number of rows
    :param num_cols: int number of columns
    :param row_counter: int current row
    :param col_counter: int current column
    :return: axis
    """
    if (num_rows > 1) and (num_cols > 1):
        return_ax = ax[row_counter][col_counter]
    elif num_rows > 1:
        return_ax = ax[row_counter]
    elif num_cols > 1:
        return_ax = ax[col_counter]
    else:
        return_ax = ax
    return return_ax


def create_animation(data, fig, ax, title, filepath, repeat, interval, repeat_delay):
    """
    Creates an animated pie chart
    :param data: dict of data to animate
    :param fig: figure to attach the animation to
    :param ax: the axis to which to set the values
    :param title: string title for animation
    :param filepath: string path for file to save or None to display
    :param repeat: bool whether to repeat the animation
    :param interval: time between updates
    :param repeat_delay: time after each animation sequence before it repeats
    :return save_anim: the saved animation
    """
    save_anim = FuncAnimation(fig, update_plot, frames=sorted(data.keys()),
                              fargs=(data, ax, title),
                              repeat=repeat, interval=interval, repeat_delay=repeat_delay)
    if filepath is not None:
        filename_no_ext = filepath + sep + title
        filename = filename_no_ext + ".mp4"
        # filename_gif = filename_no_ext + ".gif"
        fps = round(1000 / interval)
        save_anim.save(filename, writer='ffmpeg', fps=fps)
        # To remember how to go to GIF if desired later.  Much larger than mp4
        # convert_string = "ffmpeg -i " + filename + " " + filename_gif
        # system(convert_string)
    return save_anim


def create_stack_plot(data, data_colour_map, transition_data_colour_map, ax, area_labels, title, by_time=True):
    """
    Creates a stack plot
    :param data: dict of data to plot from x values to dict of y values
    :param data_colour_map: dict of colour mapping from y value names to colours
    :param transition_data_colour_map: None or dict of dict of colour mapping from y value names to colours
    :param ax: the axis to use for plotting
    :param area_labels: bool whether to add area labels
    :param title: string title for the plot
    :param by_time: bool whether indexed by time
    """
    # Stackplot
    x_data = sorted(list(data.keys()))
    # Doesn't apply to transition ambiguities
    is_transition_data = isinstance(data[x_data[0]][list(data[x_data[0]].keys())[0]], dict)
    y_data = []
    colours = []
    found_all_colours = True
    if is_transition_data is False:
        # Node or transition ambiguities (one level deep)
        # These also include evidence, which may not have all labels for each input, so we need to check through
        #  all
        labels = []
        for index in x_data:
            for label in data[index].keys():
                if label not in labels:
                    labels.append(label)
        labels = sorted(labels, key=cmp_to_key(compare_ds_labels))
        for label in labels:
            # Initialize the array
            y_data.append([])
            if label in data_colour_map:
                colours.append(data_colour_map[label])
            else:
                found_all_colours = False
        for index in x_data:
            counter = 0
            for label in labels:
                if label in data[index]:
                    y_data[counter].append(data[index][label] * TO_PERCENT)
                else:
                    # Remember, the evidence may not contain all values
                    y_data[counter].append(0.0)
                counter += 1
    else:
        # Transition data (two levels deep)
        parent_labels = list(data[x_data[0]].keys())
        child_labels = sorted(list(data[x_data[0]][parent_labels[0]]))
        labels = []
        for parent_label in parent_labels:
            for child_label in child_labels:
                # Initialize the array
                y_data.append([])
                # Create the labels
                labels.append(str(parent_label) + "." + str(child_label))
                if (transition_data_colour_map is not None) and (parent_label in transition_data_colour_map) and\
                        (child_label in transition_data_colour_map[parent_label]):
                    colours.append(transition_data_colour_map[parent_label][child_label])
                else:
                    found_all_colours = False
        for index in x_data:
            counter = 0
            for parent_label in parent_labels:
                for child_label in child_labels:
                    if (parent_label in data[index]) and (child_label in data[index][parent_label]):
                        y_data[counter].append(data[index][parent_label][child_label] * TO_PERCENT)
                    else:
                        # Remember, not all data may exist for each update
                        y_data[counter].append(0.0)
                    counter += 1
    if found_all_colours is False:
        colours = None  # Can't do colours because the data wasn't specified.
    ax.stackplot(x_data, y_data, labels=labels, colors=colours)
    # Create the labels
    if area_labels is True:
        add_area_labels(labels, x_data, y_data, ax)
    ax.set_title(title)
    ax.set_ylabel("Percent")
    ax.set_xlim(x_data[0], x_data[len(x_data) - 1])
    if by_time is True:
        ax.set_xlabel("Time (s)")
    else:
        ax.set_xlabel("Step")
    ax.legend(loc='upper right')


class PlotMetrics:
    def __init__(self, by_time=False):
        """
        Creates the member variables
        :param by_time: bool whether to create plots based on time or steps
        """
        # Contains any plots for risk nodes
        # The structure is name: dict of "RISK", "EVIDENCE", "INTERNAL_DATA" to marginals
        self.node_by_name = {}
        # Contains any plots for risk potentials (transitions)
        # The structure is name: dict of "POTENTIALS" to potentials
        self.transition_by_name = {}
        # For the figures
        self.by_time = by_time
        self.fig = None
        self.ax = None
        self.has_some_internal_data = False
        self.has_some_evidence = False

    def read_data(self, file_path):
        """
        Reads the data in from a file
        :param file_path: string file path containing the CSV output of a node or transition from the DSNetwork
                          dump_to_csv functions
        """
        with open(file_path) as csvfile:
            # CSV file
            read_csv = csv.reader(csvfile, delimiter=',')
            begin_frame = True
            mass_name_line = False
            frame = -1
            time = 0.0
            is_transition = False
            has_evidence = False
            has_internal_data = False
            # Initialize data (even though not required, because the editor is too stupid to follow the logic path)
            method = None
            name = None
            mass_names = []
            masses = {}
            evidence_names = []
            evidences = {}
            internal_mass_names = []
            internal_masses = {}

            for row in read_csv:
                if begin_frame is True:
                    # Initialize frame data
                    method = None
                    mass_names = []
                    masses = {}
                    evidence_names = []
                    evidences = {}
                    internal_mass_names = []
                    internal_masses = {}
                    has_evidence = False
                    has_internal_data = False
                    # First line of the frame - gives the name, etc.
                    frame += 1
                    name = row[0]
                    is_transition = "." in name
                    if is_transition is False:
                        method = row[1]  # Combination method for the node
                    begin_frame = False  # No longer the beginning
                    mass_name_line = True
                elif mass_name_line is True:
                    # First line after the begin line, which means it's mass names
                    if is_transition is False:
                        column = 3
                        while row[column] != "":
                            mass_names.append(get_tuple_name(row[column]))
                            column += 1
                        column += 1
                        if row[column] == "Last Evidence":
                            # Has evidence
                            has_evidence = True
                            while row[column] != "":
                                evidence_names.append(get_tuple_name(row[column]))
                                column += 1
                            column += 1  # Bump one more
                        # Ends when the column is empty
                        if method == "MURPHY":
                            has_internal_data = True
                            column += 2  # Go to internal data if it is present
                            # Internal data as well
                            while row[column] != "":
                                internal_mass_names.append(get_tuple_name(row[column]))
                                column += 1
                    else:
                        column = 2
                        time = frame  # No time data
                        while row[column] != "":
                            mass_names.append(get_tuple_name(row[column]))
                            column += 1
                        # Ends when the column is empty
                    mass_name_line = False
                else:
                    # Just a data line
                    if is_transition is False:
                        # Single line of data
                        column = 3
                        for mass_name in mass_names:
                            masses[mass_name] = float(row[column])
                            column += 1
                        # Empty - end of mass data
                        column += 1

                        if has_evidence is True:
                            for evidence_name in evidence_names:
                                evidences[evidence_name] = float(row[column])
                                column += 1
                            column += 1
                        if has_internal_data is True:
                            # Get the time
                            time = float(row[column + 1])
                            column += 3
                            # Internal data as well
                            for internal_mass_name in internal_mass_names:
                                internal_masses[internal_mass_name] = float(row[column])
                                column += 1
                        else:
                            # No time data
                            time = frame
                        # Frame done - store data
                        self.add_mass_data(name, masses, frame, time, evidences, internal_masses)
                        # Next frame
                        begin_frame = True
                    elif row[0] != "":  # Multiple lines of data - won't know the end until the next frame
                        # Found the end - the potential map has been created
                        self.add_mass_data(name, masses, frame, time)
                        # This is also the begin_frame for the next frame
                        # Initialize frame data, but only what is necessary since it's already a transition
                        mass_names = []
                        masses = {}
                        # First line of the frame - gives the name, etc.
                        frame += 1
                        name = row[0]
                        mass_name_line = True
                    else:
                        column = 1
                        child_mass_name = get_tuple_name(row[column])
                        column += 1
                        # Get the parent to child to potential mapping for this child
                        for parent_mass_name in mass_names:
                            if parent_mass_name not in masses:
                                masses[parent_mass_name] = {}
                            masses[parent_mass_name][child_mass_name] = float(row[column])
                            column += 1

    def add_mass_data(self, name, masses, step, time, evidences=None, internal_masses=None):
        """
        Adds a frame of node or transition data
        :param name: The node name
        :param masses: dict of mass names to masses
        :param step: the current step (integer)
        :param time: the current time (hours)
        :param evidences: dict of last evidence names to masses (may be None)
        :param internal_masses: dict of internal mass data to masses (may be None)
        """
        if self.by_time is True:
            index = time
        else:
            index = step

        if "." in name:
            # Transition
            if name not in self.transition_by_name:
                # Create
                self.transition_by_name[name] = {}
                self.transition_by_name[name][PLOT_TYPE["POTENTIALS"]] = {}

            # Add the data
            self.transition_by_name[name][PLOT_TYPE["POTENTIALS"]][index] =\
                deepcopy(masses)
        else:
            # Node
            if name not in self.node_by_name:
                # Create
                self.node_by_name[name] = {}
                self.node_by_name[name][PLOT_TYPE["RISK"]] = {}
                if (internal_masses is not None) and (len(internal_masses) > 0):
                    self.node_by_name[name][PLOT_TYPE["INTERNAL_DATA"]] = {}
                if (evidences is not None) and (len(evidences) > 0):
                    self.node_by_name[name][PLOT_TYPE["EVIDENCE"]] = {}

            # Add data
            self.node_by_name[name][PLOT_TYPE["RISK"]][index] = masses
            if (internal_masses is not None) and (len(internal_masses) > 0):
                self.node_by_name[name][PLOT_TYPE["INTERNAL_DATA"]][index] = deepcopy(internal_masses)
                self.has_some_internal_data = True
            if (evidences is not None) and (len(evidences) > 0):
                self.node_by_name[name][PLOT_TYPE["EVIDENCE"]][index] = deepcopy(evidences)
                self.has_some_evidence = True

    def add_data(self, name, data, step, time):
        """
        Adds a single set of data.  If the plot isn't already created, creates and stores the plot
        :param name: The node or transition name (transition names include a "." in the name)
        :param data: The data (DSNetwork Node or Transition that includes the data)
        :param step: The current step (integer)
        :param time: The current time (hours)
        """
        if "." in name:
            # Transition
            self.add_mass_data(name, data.conditionals_parent_to_child, step, time)
        else:
            # Node
            last_evidence = None
            internal_evidence = None
            if "last_evidence" in data.internal_probability_data:
                last_evidence = data.internal_probability_data["last_evidence"]
            if ("evidence" in data.internal_probability_data) and\
                    (isinstance(data.internal_probability_data["evidence"][
                                    list(data.internal_probability_data["evidence"].keys())[0]], dict) is False):
                internal_evidence = data.internal_probability_data["evidence"]
            self.add_mass_data(name, data.get_marginals(), step, time, last_evidence, internal_evidence)

    def plot(self, plot_names=None, plot_internal=False, plot_evidence=False,
             visualization_type=VISUALIZATION_TYPE["DATA"], limits=None,
             animate=False, repeat=True, interval=200, repeat_delay=None, filepath=None, area_labels=False,
             data_colour_map=None, transition_colour_map=None, separate_low_high=False):
        """
        Plotting utility that does all options and keeps everything consistent in one function.  Typically not
         used directly by external callers since the other convenience functions are easier to use.
        :param plot_names: list of names to plot.  Leave as None to plot all
        :param plot_internal: bool whether to plot the internal data as well as the current decision data
        :param plot_evidence: bool whether to plot the latest evidence as well as the current decision data
        :param visualization_type: VISUALIZATION_TYPE the type of visualization to create
        :param limits: None or dict of node name to upper and lower limits for plotting
                        (mass name to PLAUSIBILITY, BELIEF to limits)
        :param animate: bool whether to animate or use stackplot
        :param repeat: whether to repeat the plot or pause at the end of the animation
        :param interval: milliseconds time between each update
        :param repeat_delay: whether to delay before repeating
        :param filepath: None to display, string path to save
        :param area_labels: Whether to show area labels
        :param data_colour_map: dict of labels to colours.  Can be NONE to use the default
        :param transition_colour_map: dict of dict of labels to colours for transitions data.  Can be NONE.
        :param separate_low_high: bool whether to separate the low/high results for limit plots
        """
        num_rows, num_cols, num_nodes = get_plot_layout(self.has_some_internal_data, self.has_some_evidence,
                                                        plot_names, plot_evidence, plot_internal,
                                                        self.node_by_name, self.transition_by_name,
                                                        (filepath is not None) and (animate is True))

        if num_cols > 0:
            self.fig, self.ax = plt.subplots(nrows=num_rows, ncols=num_cols)
            row_counter = 0
            col_counter = 0
            for name, all_data in self.node_by_name.items():
                if ((plot_names is None) or (name in plot_names)) and\
                        ((visualization_type != VISUALIZATION_TYPE["LIMITS"]) or (name in limits)):
                    # Get the correct ax
                    ax = get_ax(self.ax, num_rows, num_cols, row_counter, col_counter)
                    # Get the plot data
                    if visualization_type == VISUALIZATION_TYPE["DATA"]:
                        plot_data = all_data[PLOT_TYPE["RISK"]]
                        name_addition = "_Data"
                    elif visualization_type == VISUALIZATION_TYPE["AMBIGUITY"]:
                        plot_data = calculate_node_ambiguity_data(all_data[PLOT_TYPE["RISK"]])
                        name_addition = "_Ambiguity"
                    else:  # LIMIT
                        plot_data = calculate_limit_data(limits[name], all_data[PLOT_TYPE["RISK"]], separate_low_high)
                        name_addition = "_Limits"
                    # Plot each
                    if animate is True:
                        all_data[PLOT_TYPE["ANIMATION"]] = \
                            create_animation(plot_data, self.fig, ax, name + "_Decision" + name_addition, filepath,
                                             repeat, interval, repeat_delay)
                    else:
                        # Stackplot
                        colour_map = data_colour_map
                        if colour_map is None:
                            colour_map = DATA_COLOUR_MAP
                        else:
                            # Append any additional colours, but don't overwrite
                            colour_map = dict(DATA_COLOUR_MAP)
                            colour_map.update(data_colour_map)
                        create_stack_plot(plot_data, colour_map, None, ax, area_labels,
                                          name + "_Decision" + name_addition, self.by_time)
                    if (plot_evidence is True) and (PLOT_TYPE["EVIDENCE"] in all_data) and \
                            (len(all_data[PLOT_TYPE["EVIDENCE"]]) > 0):
                        # Get the correct ax.
                        ax = get_ax(self.ax, num_rows, num_cols, row_counter + 1, col_counter)
                        # Get the plot data
                        if visualization_type == VISUALIZATION_TYPE["DATA"]:
                            plot_data = all_data[PLOT_TYPE["EVIDENCE"]]
                            name_addition = "_Data"
                        elif visualization_type == VISUALIZATION_TYPE["AMBIGUITY"]:
                            plot_data = calculate_node_ambiguity_data(all_data[PLOT_TYPE["EVIDENCE"]])
                            name_addition = "_Ambiguity"
                        else:  # LIMIT
                            plot_data = calculate_limit_data(limits[name], all_data[PLOT_TYPE["EVIDENCE"]],
                                                             separate_low_high)
                            name_addition = "_Limits"
                        if animate is True:
                            all_data[PLOT_TYPE["ANIMATE_EVIDENCE"]] = \
                                create_animation(plot_data, self.fig, ax, name + "_Evidence" + name_addition, filepath,
                                                 repeat, interval, repeat_delay)
                        else:
                            # Stackplot
                            colour_map = data_colour_map
                            if colour_map is None:
                                colour_map = DATA_COLOUR_MAP
                            else:
                                # Append any additional colours, but don't overwrite
                                colour_map = dict(DATA_COLOUR_MAP)
                                colour_map.update(data_colour_map)
                            create_stack_plot(plot_data, colour_map, None, ax, area_labels,
                                              name + "_Evidence" + name_addition, self.by_time)
                    if (plot_internal is True) and (PLOT_TYPE["INTERNAL_DATA"] in all_data) and \
                            (len(all_data[PLOT_TYPE["INTERNAL_DATA"]]) > 0):
                        # Get the correct ax.
                        ax = get_ax(self.ax, num_rows, num_cols, row_counter + 2, col_counter)
                        # Get the plot data
                        if visualization_type == VISUALIZATION_TYPE["DATA"]:
                            plot_data = all_data[PLOT_TYPE["INTERNAL_DATA"]]
                            name_addition = "_Data"
                        elif visualization_type == VISUALIZATION_TYPE["AMBIGUITY"]:
                            plot_data = calculate_node_ambiguity_data(all_data[PLOT_TYPE["INTERNAL_DATA"]])
                            name_addition = "_Ambiguity"
                        else:  # LIMIT
                            plot_data = calculate_limit_data(limits[name], all_data[PLOT_TYPE["INTERNAL_DATA"]],
                                                             separate_low_high)
                            name_addition = "_Limits"
                        if animate is True:
                            all_data[PLOT_TYPE["ANIMATE_INTERNAL"]] = \
                                create_animation(plot_data, self.fig, ax, name + "_Internal" + name_addition, filepath,
                                                 repeat, interval, repeat_delay)
                        else:
                            # Stackplot
                            colour_map = data_colour_map
                            if colour_map is None:
                                colour_map = DATA_COLOUR_MAP
                            else:
                                # Append any additional colours, but don't overwrite
                                colour_map = dict(DATA_COLOUR_MAP)
                                colour_map.update(data_colour_map)
                            create_stack_plot(plot_data, colour_map, None, ax, area_labels,
                                              name + "_Internal" + name_addition, self.by_time)
                    col_counter += 1  # Index for each node

            if (num_nodes > 0) and (num_rows > 1):
                # Index to the transition row
                row_counter = num_rows - 1

            col_counter = 0
            for name, all_data in self.transition_by_name.items():
                if ((plot_names is None) or (name in plot_names)) and\
                        (visualization_type != VISUALIZATION_TYPE["LIMITS"]):
                    # Get the correct ax
                    ax = get_ax(self.ax, num_rows, num_cols, row_counter, col_counter)
                    # Get the plot data
                    if visualization_type == VISUALIZATION_TYPE["DATA"]:
                        plot_data = all_data[PLOT_TYPE["POTENTIALS"]]
                        name_addition = "_Data"
                    else:  # AMBIGUITY
                        plot_data = calculate_transition_ambiguity_data(all_data[PLOT_TYPE["POTENTIALS"]])
                        name_addition = "_Ambiguity"
                    # Plot each
                    if animate is True:
                        all_data[PLOT_TYPE["ANIMATION"]] = \
                            create_animation(plot_data, self.fig, ax, name + "_Potentials" + name_addition,
                                             filepath, repeat, interval, repeat_delay)
                    else:
                        # Stackplot
                        colour_map = data_colour_map
                        if colour_map is None:
                            colour_map = DATA_COLOUR_MAP
                        else:
                            # Append any additional colours, but don't overwrite
                            colour_map = dict(DATA_COLOUR_MAP)
                            colour_map.update(data_colour_map)
                        two_level_colour_map = transition_colour_map
                        if two_level_colour_map is None:
                            two_level_colour_map = TRANSITION_COLOUR_MAP
                            # This one doesn't append since more complex.  Either the user supplies all colours,
                            #  or the defaults are used.
                        create_stack_plot(plot_data, colour_map, two_level_colour_map, ax, area_labels,
                                          name + "_Potentials" + name_addition, self.by_time)
                    col_counter += 1  # Index for each transition

            # Show the plots if configured to do so
            if filepath is None:
                plt.show()
            elif animate is False:
                # Save the stackplot
                if visualization_type == VISUALIZATION_TYPE["DATA"]:
                    plt.savefig(filepath + sep + "data_plot.png")
                elif visualization_type == VISUALIZATION_TYPE["AMBIGUITY"]:
                    plt.savefig(filepath + sep + "ambiguity_plot.png")
                else:  # Limits
                    plt.savefig(filepath + sep + "limit_plot.png")
        # else: nothing to plot
