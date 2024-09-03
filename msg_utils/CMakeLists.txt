cmake_minimum_required(VERSION 3.5)

project(msg_utils) 

MESSAGE("building library msg_utils")

find_package(CUDA REQUIRED)
find_package(MPI REQUIRED)

ENABLE_TESTING()

IF(NOT CMAKE_BUILD_TYPE)
	SET(CMAKE_BUILD_TYPE Release)
ENDIF(NOT CMAKE_BUILD_TYPE)

SET(CUDA_PROPAGATE_HOST_FLAGS OFF)
SET(CUDA_SEPARABLE_COMPILATION ON)

# ADD_COMPILE_DEFINITIONS(USE_GPU)

STRING(TOUPPER ${CMAKE_BUILD_TYPE} CMAKE_BUILD_TYPE_UPPER)

IF (CMAKE_BUILD_TYPE_UPPER STREQUAL "RELEASE")
	IF ( "${CMAKE_CXX_COMPILER_ID}" MATCHES "Clang" )
		SET(CMAKE_CXX_FLAGS_RELEASE "$ENV{CXXFLAGS} -std=c++11 -stdlib=libc++ -O3 -fPIC -pthread")
	ELSE( "${CMAKE_CXX_COMPILER_ID}" MATCHES "Clang" )
		SET(CMAKE_CXX_FLAGS_RELEASE "$ENV{CXXFLAGS} -std=c++11 -O3 -fPIC -pthread")
	ENDIF ( "${CMAKE_CXX_COMPILER_ID}" MATCHES "Clang" )
	SET(CUDA_NVCC_FLAGS_RELEASE "-std=c++11;-O3;-D USE_GPU;-gencode=arch=compute_60,code=sm_60;-gencode=arch=compute_61,code=sm_61;-gencode=arch=compute_70,code=sm_70;-gencode=arch=compute_75,code=sm_75;-gencode=arch=compute_80,code=sm_80")
elseif (CMAKE_BUILD_TYPE_UPPER STREQUAL "DEBUG")
	IF ( "${CMAKE_CXX_COMPILER_ID}" MATCHES "Clang" )
		SET(CMAKE_CXX_FLAGS_DEBUG "$ENV{CXXFLAGS} -std=c++11 -stdlib=libc++ -O0 -Wall -g -ggdb -fPIC -pthread -D DEBUG")
	ELSE( "${CMAKE_CXX_COMPILER_ID}" MATCHES "Clang" )
		SET(CMAKE_CXX_FLAGS_DEBUG "$ENV{CXXFLAGS} -std=c++11 -O0 -Wall -g -ggdb -fPIC -pthread -D DEBUG")
	ENDIF ( "${CMAKE_CXX_COMPILER_ID}" MATCHES "Clang" )
	SET(CUDA_NVCC_FLAGS_DEBUG "-std=c++11;-g;-G;-D DEBUG;-D USE_GPU;-gencode=arch=compute_60,code=sm_60;-gencode=arch=compute_61,code=sm_61;-gencode=arch=compute_70,code=sm_70;-gencode=arch=compute_75,code=sm_75;-gencode=arch=compute_80,code=sm_80")
endif (CMAKE_BUILD_TYPE_UPPER STREQUAL "RELEASE")

FILE(GLOB_RECURSE CPU_SRC "./msg_utils/*.cpp" "./msg_utils/*.c")
FILE(GLOB_RECURSE GPU_SRC "./msg_utils/*.cu")
FILE(GLOB GPU_TEST "test/*.cu" "test/*.cpp" "test/*.c")

include_directories(${MPI_INCLUDE_PATH} ./include)

add_library(msg_utils STATIC ${CPU_SRC})
set_target_properties(msg_utils PROPERTIES LINKER_LANGUAGE CXX)
cuda_add_library(msg_utils_cu STATIC ${GPU_SRC} ${CPU_SRC})
set_target_properties(msg_utils_cu PROPERTIES LINKER_LANGUAGE CXX)

FOREACH(exec_path ${GPU_TEST})
	#MESSAGE("EXEC_PATH: ${exec_path}")
	GET_FILENAME_COMPONENT(exec_file ${exec_path} NAME_WE)
	#MESSAGE("EXEC_FILE: ${exec_file}")
	CUDA_ADD_EXECUTABLE(${exec_file} ${exec_path})
	TARGET_LINK_LIBRARIES(${exec_file} msg_utils_cu msg_utils nccl ${MPI_LIBRARIES})
	SET_TARGET_PROPERTIES(${exec_file} PROPERTIES LINKER_LANGUAGE CXX)
	ADD_TEST(NAME ${exec_file}_test COMMAND mpirun -n 4 $<TARGET_FILE:${exec_file}>) 
ENDFOREACH()
