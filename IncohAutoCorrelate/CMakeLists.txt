cmake_minimum_required (VERSION 2.8.12)

project (IncohAutoCorrelate)


# Set the output folder where your program will be created
set(CMAKE_BINARY_DIR ${CMAKE_SOURCE_DIR}/bin)
set(EXECUTABLE_OUTPUT_PATH ${CMAKE_BINARY_DIR})
set(LIBRARY_OUTPUT_PATH ${CMAKE_BINARY_DIR}/lib)


# The following folder will be included
include_directories("${PROJECT_SOURCE_DIR}")



add_library(AC1D STATIC AC1D.cpp)
add_library(ACMesh STATIC ACMesh.cpp)
add_library(Crystal STATIC Crystal.cpp)
add_library(Detector STATIC Detector.cpp)
add_library(PPP STATIC PPP.cpp)
add_library(ProfileTime STATIC ProfileTime.cpp)
add_library(RunIAC STATIC RunIAC.cpp)
add_library(Settings STATIC Settings.cpp)
add_library(ArrayOperators STATIC ArrayOperators.cpp)
add_library(Simulator STATIC Simulator.cpp)
add_library(Statistics STATIC Statistics.cpp)



include_directories(/usr/include/eigen3)
include_directories(/usr/lib64 /opt/opencl/intel/lib64)
include_directories(/opt/opencl/intel/include/)
include_directories(/usr/lib64)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -lOpenCL -fopenmp -lgomp -lz -lm -lsz -ldl -lhdf5 -lhdf5_cpp -lpthread ")
#add_compile_options(-std=13 -lOpenCL -fopenmp -lgomp -lz -lm -lsz -ldl -lhdf5 -lhdf5_cpp -lpthread)

add_executable(IncohAutoCorr ${PROJECT_SOURCE_DIR}/main.cpp)

target_link_libraries (IncohAutoCorr AC1D ACMesh Crystal Detector PPP ProfileTime RunIAC Settings ArrayOperators Simulator Statistics )


configure_file(${CMAKE_CURRENT_SOURCE_DIR}/kernel.cl ${EXECUTABLE_OUTPUT_PATH} COPYONLY)