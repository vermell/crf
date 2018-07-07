cmake_minimum_required(VERSION 2.8 FATAL_ERROR)

project("ConditionalRandomFields")
SET(CMAKE_CXX_COMPILER g++)

set (CMAKE_CXX_STANDARD 14)
set(OpenMP_CXX_FLAGS -fopenmp)


set(GCO_CXX_FLAGS -lcgco)
set(GCO_LINKER_FLAGS -lgco)

set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS} -O2")
set (CMAKE_CXX_FLAGS "-L/home/bilboi/programming/pyGCO ${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS} -fPIC -lpython2.7 -lpthread -ldl -lutil -lm -lboost_python -lboost_numpy ${GCO_CXX_FLAGS}")
set (CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS} ${GCO_LINKER_FLAGS}")


include_directories(${CMAKE_CURRENT_SOURCE_DIR})
include_directories("/home/bilboi/programming/eigen")

include_directories("/home/bilboi/programming/LBFGS/include")
include_directories("/usr/include/python2.7")
include_directories("/home/bilboi/programming/pyGCO/gco_source")

add_subdirectory(src)



set(CONFIGURED_ONCE TRUE CACHE INTERNAL
    "A flag showing that CMake has configured at least once.")
