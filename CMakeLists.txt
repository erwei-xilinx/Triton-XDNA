# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

project(triton-xdna)

cmake_minimum_required(VERSION 3.24)

find_package(Python3 REQUIRED COMPONENTS Interpreter)

find_package(Git REQUIRED)

set(PIP_INSTALL_ARGS)  # Pip install command args
list(APPEND PIP_INSTALL_ARGS "--no-build-isolation")
list(APPEND PIP_INSTALL_ARGS "-vvv")

set(ENV TRITON_PLUGIN_DIRS="${pwd}/amd_triton_npu;${pwd}/third_party/triton_shared")

set(PIP_INSTALL_ENVS)  # Pip install environmental vars
if(DEFINED PIP_INSTALL_TARGET)
	list(APPEND PIP_INSTALL_ENVS "PIP_INSTALL_TARGET=${PIP_INSTALL_TARGET}")
endif()
list(APPEND PIP_INSTALL_ENVS "TRITON_BUILD_WITH_CLANG_LLD=true")

add_custom_target(triton-build ALL
	COMMAND ${PIP_INSTALL_ENVS}
		"TRITON_PLUGIN_DIRS=${CMAKE_CURRENT_SOURCE_DIR}/third_party/triton_shared\;${CMAKE_CURRENT_SOURCE_DIR}/amd_triton_npu"
		${Python3_EXECUTABLE} -m pip install ${PIP_INSTALL_ARGS} .
	WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
	USES_TERMINAL
)
