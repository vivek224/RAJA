###############################################################################
#
# Copyright (c) 2016-18, Lawrence Livermore National Security, LLC.
#
# Produced at the Lawrence Livermore National Laboratory
#
# LLNL-CODE-689114
#
# All rights reserved.
#
# This file is part of RAJA.
#
# For details about use and distribution, please read RAJA/LICENSE.
#
###############################################################################

raja_add_executable(
  NAME ex0-vecAdd
  SOURCES ex0-vecAdd.cpp)

raja_add_executable(
  NAME ex1-mult
  SOURCES ex1-mult.cpp)

raja_add_executable(
  NAME ex2-triad
  SOURCES ex2-triad.cpp)

raja_add_executable(
  NAME ex3-dot
  SOURCES ex3-dot.cpp)

raja_add_executable(
  NAME ex4-omp-dot
  SOURCES ex4-omp-dot.cpp)


