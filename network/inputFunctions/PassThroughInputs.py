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

from network.DSNetwork import InputToNode


class PassThroughInputs(InputToNode):
    """
    Defines a simple input that passes through all inputs to the correct nodes.
    """
    def map_to_input(self, inputs, input_weight=0.0, **kwargs):
        """
        Maps the inputs (which is evidence, so no mapping required
        :param inputs: evidence - no mapping required
        :param input_weight: float - no mapping required
        :param kwargs: additional keyword arguments - ignored for the pass through
        """
        self._propagate_input(inputs, input_weight=input_weight)
