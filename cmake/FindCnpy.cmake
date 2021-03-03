# Defines the Cnpy::Cnpy IMPORTED target if the cnpy library is found

find_library(Cnpy_LIBRARY cnpy)
find_path(
  Cnpy_INCLUDE_DIR
  NAMES cnpy.h
  PATH_SUFFIXES include
)
mark_as_advanced(
  Cnpy_LIBRARY
  Cnpy_INCLUDE_DIR
)

# zlib is a public dependency of cnpy - its header is required by cnpy.h
if(Cnpy_FIND_QUIETLY)
  set(_FIND_ZLIB_ARG QUIET)
endif()
find_package(ZLIB ${_FIND_ZLIB_ARG})

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(
  Cnpy
  REQUIRED_VARS Cnpy_LIBRARY
                Cnpy_INCLUDE_DIR
                ZLIB_FOUND
)

if(Cnpy_FOUND)
  add_library(Cnpy::Cnpy UNKNOWN IMPORTED)
  set_target_properties(Cnpy::Cnpy
    PROPERTIES
      IMPORTED_LOCATION "${Cnpy_LIBRARY}"
      IMPORTED_LINK_INTERFACE_LANGUAGES "CXX"
      INTERFACE_INCLUDE_DIRECTORIES "${Cnpy_INCLUDE_DIR}"
      IMPORTED_LINK_INTERFACE_LIBRARIES "ZLIB::ZLIB"
  )
endif()
