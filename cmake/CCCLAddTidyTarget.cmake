include_guard(GLOBAL)

find_program(CCCL_CLANG_TIDY clang-tidy)

#[=======================================================================[.rst:
cccl_tidy_init
--------------

Initialize ``clang-tidy`` support and define the global ``tidy`` target. It must be called
before adding any CCCL ``clang-tidy`` targets.

Subsequent calls to this functions are no-ops.

Result Variables
^^^^^^^^^^^^^^^^

  ``CCCL_TIDY_INITIALIZED`` set to true in the parent scope.

#]=======================================================================]
function(cccl_tidy_init)
  list(APPEND CMAKE_MESSAGE_CONTEXT "tidy_init")

  if (CCCL_TIDY_INITIALIZED)
    return()
  endif()

  set(CCCL_TIDY_INITIALIZED TRUE)
  set(CCCL_TIDY_INITIALIZED TRUE PARENT_SCOPE)

  if (CCCL_CLANG_TIDY)
    add_custom_target(tidy COMMENT "running clang-tidy")

    return()
  endif()

  add_custom_target(
    tidy
    COMMENT "clang-tidy is broken"
    COMMAND ${CMAKE_COMMAND} -E echo "-- ERROR: Could not locate clang-tidy"
    COMMAND
      ${CMAKE_COMMAND} -E false # to signal the error
  )
endfunction()

#[=======================================================================[.rst:
cccl_add_tidy_target
--------------------

Create per-source ``clang-tidy`` targets and attach them to the global ``tidy`` target.

.. note::

  :command:`cccl_tidy_init` must be called before using this function to establish the
  global ``tidy`` target.

If ``clang-tidy`` has not been found, this function is a no-op.

Passing the same source file multiple times is allowed. A target is created for it only
once.

If ``SOURCES`` is empty, this function does nothing.

Arguments
^^^^^^^^^

``SOURCES``
  List of source files to analyze. Paths may be absolute or relative.  Relative paths are
  resolved against ``CMAKE_CURRENT_SOURCE_DIR``.

#]=======================================================================]
function(cccl_add_tidy_target)
  list(APPEND CMAKE_MESSAGE_CONTEXT "add_tidy_target")

  set(options)
  set(one_value_args)
  set(multi_value_args SOURCES)

  cmake_parse_arguments(
    _cccl
    "${options}"
    "${one_value_args}"
    "${multi_value_args}"
    ${ARGN}
  )

  if (NOT CCCL_TIDY_INITIALIZED)
    message(FATAL_ERROR "Must call cccl_tidy_init() first")
  endif()

  if (_cccl_UNPARSED_ARGUMENTS)
    message(FATAL_ERROR "Unrecognized arguments: ${_cccl_UNPARSED_ARGUMENTS}")
  endif()

  if (NOT CCCL_CLANG_TIDY)
    return()
  endif()

  foreach (src IN LISTS _cccl_SOURCES)
    cmake_path(SET src NORMALIZE "${src}")
    if (NOT IS_ABSOLUTE "${src}")
      cmake_path(SET src NORMALIZE "${CMAKE_CURRENT_SOURCE_DIR}/${src}")
    endif()

    cmake_path(
      RELATIVE_PATH src
      BASE_DIRECTORY "${CCCL_TOPLEVEL_DIRECTORY}"
      OUTPUT_VARIABLE rel_src
    )
    string(MAKE_C_IDENTIFIER "${rel_src}_tidy" tidy_target)

    if (TARGET "${tidy_target}")
      # We have seen this file before
      return()
    endif()

    add_custom_target(
      "${tidy_target}"
      DEPENDS "${src}"
      COMMAND
        ${CCCL_CLANG_TIDY} #
        --use-color #
        --quiet #
        --extra-arg=-Wno-error=unused-command-line-argument #
        -p "${CMAKE_BINARY_DIR}" #
        "${src}"
      COMMENT "clang-tidy ${rel_src}"
    )

    add_dependencies(tidy "${tidy_target}")
  endforeach()
endfunction()
