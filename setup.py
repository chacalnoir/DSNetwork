#!/usr/bin/env python

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

from distutils.core import setup

setup(name='DSBeliefNetwork',
      version='0.2.4',
      description='Python implementation of Dempster-Shafer Belief Network',
      author='Joel Dunham',
      author_email='joel.ph.dunham@gmail.com',
      url='https://github.com/chacalnoir/DSNetwork',
      packages=['network', 'combinationRulesExtensions', 'network.inputFunctions', 'metrics'],
      )
