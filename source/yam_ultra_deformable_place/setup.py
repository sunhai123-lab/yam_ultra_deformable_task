# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Install the small project-local Python package."""

from setuptools import find_packages, setup

# Installation operation
setup(
    name="yam_ultra_deformable_place",
    version="0.1.0",
    description="YAM Ultra kinematics for the Isaac Lab Newton deformable-placement task",
    packages=find_packages(),
    install_requires=[],
    python_requires=">=3.12",
    zip_safe=False,
)
