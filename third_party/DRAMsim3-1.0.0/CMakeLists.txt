# SPDX-License-Identifier: MIT
cmake_minimum_required(VERSION 3.10)
project(dramsim3_public_snapshot LANGUAGES CXX)

add_library(dramsim3 STATIC
  src/bankstate.cc
  src/channel_state.cc
  src/command_queue.cc
  src/common.cc
  src/configuration.cc
  src/controller.cc
  src/dram_system.cc
  src/hmc.cc
  src/memory_system.cc
  src/refresh.cc
  src/simple_stats.cc
  src/timing.cc
)
target_compile_features(dramsim3 PUBLIC cxx_std_11)
target_compile_definitions(dramsim3 PUBLIC FMT_HEADER_ONLY=1)
target_include_directories(dramsim3 PUBLIC
  ${CMAKE_CURRENT_SOURCE_DIR}/src
  ${CMAKE_CURRENT_SOURCE_DIR}/ext/fmt/include
  ${CMAKE_CURRENT_SOURCE_DIR}/ext/headers
)
set_target_properties(dramsim3 PROPERTIES POSITION_INDEPENDENT_CODE ON)

