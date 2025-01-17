CMAKE_MINIMUM_REQUIRED(VERSION 3.0)

PROJECT (BSim)

SET(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${PROJECT_SOURCE_DIR}/test/catch2")
MESSAGE(STATUS "CMAKE_MODULE_PATH:" ${CMAKE_MODULE_PATH})
INCLUDE(CTest)
INCLUDE(Catch)

SET(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
SET(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
SET(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
MESSAGE(STATUS "CMAKE_BINARY_DIR:" ${CMAKE_BINARY_DIR})
MESSAGE(STATUS "CMAKE_CURRENT_BINARY_DIR:" ${CMAKE_CURRENT_BINARY_DIR})

# SET(Python_ADDITIONAL_VERSIONS "3.4" "3.5" "3.8")


ENABLE_TESTING()

FIND_PACKAGE(PythonLibs 3 REQUIRED)
INCLUDE_DIRECTORIES(${PYTHON_INCLUDE_DIRS})


FIND_PACKAGE(CUDA REQUIRED)
FIND_PACKAGE(MPI REQUIRED)
FIND_PACKAGE(BLAS REQUIRED)


MESSAGE(STATUS "C Compiler: " ${CMAKE_C_COMPILER_ID} " " ${CMAKE_C_COMPILER_VERSION})
MESSAGE(STATUS "C++ Compiler: " ${CMAKE_CXX_COMPILER_ID} " " ${CMAKE_CXX_COMPILER_VERSION})
MESSAGE(STATUS "CUDA Version: " ${CUDA_VERSION})
MESSAGE(STATUS "MPI Library: ${MPI_LIBRARIES}")
MESSAGE(STATUS "BLAS Libraries: ${BLAS_LIBRARIES}")

IF(NOT CMAKE_BUILD_TYPE)
	SET(CMAKE_BUILD_TYPE Release)
ENDIF(NOT CMAKE_BUILD_TYPE)

STRING(TOUPPER ${CMAKE_BUILD_TYPE} CMAKE_BUILD_TYPE_UPPER)

if (CMAKE_BUILD_TYPE_UPPER STREQUAL "RELEASE")
	IF ( "${CMAKE_CXX_COMPILER_ID}" MATCHES "Clang" )
		SET(CMAKE_CXX_FLAGS_RELEASE "$ENV{CXXFLAGS} -std=c++11 -stdlib=libc++ -O3 -fPIC -pthread")
	ELSE( "${CMAKE_CXX_COMPILER_ID}" MATCHES "Clang" )
		SET(CMAKE_CXX_FLAGS_RELEASE "$ENV{CXXFLAGS} -std=c++11 -O3 -fPIC -pthread")
	ENDIF ( "${CMAKE_CXX_COMPILER_ID}" MATCHES "Clang" )
	SET(CUDA_NVCC_FLAGS_RELEASE "-std=c++11;-O3;-gencode=arch=compute_60,code=sm_60;-gencode=arch=compute_61,code=sm_61;-gencode=arch=compute_70,code=sm_70;-gencode=arch=compute_75,code=sm_75;-gencode=arch=compute_80,code=sm_80")
	# SET(CUDA_NVCC_FLAGS_RELEASE "-std=c++11;-O3;-arch=compute_60;-code=sm_60;-lcufft")
elseif (CMAKE_BUILD_TYPE_UPPER STREQUAL "DEBUG")
	IF ( "${CMAKE_CXX_COMPILER_ID}" MATCHES "Clang" )
		SET(CMAKE_CXX_FLAGS_DEBUG "$ENV{CXXFLAGS} -std=c++11 -stdlib=libc++ -O0 -Wall -g -ggdb -fPIC -D DEBUG -pthread")
	ELSE( "${CMAKE_CXX_COMPILER_ID}" MATCHES "Clang" )
		SET(CMAKE_CXX_FLAGS_DEBUG "$ENV{CXXFLAGS} -std=c++11 -O0 -Wall -g -ggdb -fPIC -D DEBUG -pthread")
	ENDIF ( "${CMAKE_CXX_COMPILER_ID}" MATCHES "Clang" )
	#SET(CUDA_NVCC_FLAGS_DEBUG "-std=c++11;-g;-G;-lineinfo;-Xcompiler;-arch=sm_60")
	SET(CUDA_NVCC_FLAGS_DEBUG "-std=c++11;-O0;-g;-G;-D DEBUG;-lcufft;-gencode=arch=compute_60,code=sm_60;-gencode=arch=compute_61,code=sm_61;-gencode=arch=compute_70,code=sm_70;-gencode=arch=compute_75,code=sm_75;-gencode=arch=compute_80,code=sm_80")
endif (CMAKE_BUILD_TYPE_UPPER STREQUAL "RELEASE")

OPTION(USE_DOUBLE "Use double" OFF) # Disabled by default
IF(USE_DOUBLE)
	ADD_DEFINITIONS(-DUSE_DOUBLE)
ENDIF(USE_DOUBLE)

OPTION(USE_LOG "Use log" OFF) # Disabled by default
IF(USE_LOG)
	ADD_DEFINITIONS(-DLOG_DATA)
ENDIF(USE_LOG)

OPTION(USE_PROF "Use profile" OFF) # Disabled by default
IF(USE_PROF)
	ADD_DEFINITIONS(-DPROF)
ENDIF(USE_PROF)

OPTION(USE_GPU "Use gpu" ON) # Disabled by default

SET(CUDA_PROPAGATE_HOST_FLAGS OFF)
SET(CUDA_SEPARABLE_COMPILATION ON)

SET(CPU_LIB bsim)
SET(GPU_LIB bsim_gpu)

INCLUDE_DIRECTORIES(${MPI_INCLUDE_PATH})
INCLUDE_DIRECTORIES(${BLAS_INCLUDE_PATH})

ADD_SUBDIRECTORY(msg_utils)

# SET(AUTO_TIMESTAMP "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/.timestamp")
SET(AUTO_CPU_SRC "${PROJECT_SOURCE_DIR}/src/base/TypeFunc.cpp")
SET(AUTO_GPU_SRC "${PROJECT_SOURCE_DIR}/src/base/TypeFunc.cu")

SET(GEN_TYPE_FUNC_SCRIPT "${PROJECT_SOURCE_DIR}/script/generate_type_functions.py")
# SET(GEN_DATA_FUNC_SCRIPT "${PROJECT_SOURCE_DIR}/bsim/data_generator.py")

FILE(GLOB_RECURSE CPU_SRC "src/*.cpp" "src/*.c")
FILE(GLOB_RECURSE GPU_SRC "src/*.cu")

LIST(REMOVE_ITEM CPU_SRC "${AUTO_CPU_SRC}")
LIST(REMOVE_ITEM GPU_SRC "${AUTO_GPU_SRC}")

FILE(GLOB CPU_TEST "test/cpu/*.cpp" "test/cpu/*.c")
FILE(GLOB GPU_TEST "test/gpu/*.cpp" "test/gpu/*.c" "test/gpu/*.cu")
FILE(GLOB MPI_TEST "test/mpi/*.cpp" "test/mpi/*.c" "test/mpi/*.cu" "multi-area-model/*.cpp")
FILE(GLOB UNIT_TEST_CPU "test/unit/*.cpp" "test/unit/*.c")
FILE(GLOB UNIT_TEST_GPU "test/unit/*.cu")
FILE(GLOB C_SCRIPTS "script/*.cpp" "script/*.c")
FILE(GLOB TOOLS "test/tool/*.cpp" "test/tool/*.c")

FILE(COPY ${PROJECT_SOURCE_DIR}/CLEAN_ALL.cmake DESTINATION ${CMAKE_BINARY_DIR})

# ADD_CUSTOM_COMMAND(OUTPUT ${AUTO_TIMESTAMP}
# 	COMMAND echo "Generating model data func and ${AUTO_TIMESTAMP}"
# 	COMMAND python3 ${GEN_DATA_FUNC_SCRIPT}
# 	COMMAND touch ${AUTO_TIMESTAMP}
# 	MAIN_DEPENDENCY ${GEN_DATA_FUNC_SCRIPT}
# 	COMMENT "Generate data func"
# 	)

ADD_CUSTOM_COMMAND(OUTPUT ${AUTO_CPU_SRC} ${AUTO_GPU_SRC}
	COMMAND echo "Generating ${AUTO_CPU_SRC} and ${AUTO_GPU_SRC}"
	COMMAND python3 ${GEN_TYPE_FUNC_SCRIPT}
	MAIN_DEPENDENCY src/base/TypeFunc.h
	DEPENDS src/base/type.h # ${GEN_TYPE_FUNC_SCRIPT}
	COMMENT "Generate type func"
	)

ADD_CUSTOM_TARGET(GEN_TYPE_FUNC_CPU DEPENDS ${AUTO_CPU_SRC})
ADD_CUSTOM_TARGET(GEN_TYPE_FUNC_GPU DEPENDS ${AUTO_GPU_SRC})



IF(USE_GPU)
	ADD_DEFINITIONS(-DUSE_GPU)
	CUDA_ADD_LIBRARY(${GPU_LIB} STATIC ${CPU_SRC} ${AUTO_CPU_SRC} ${GPU_SRC} ${AUTO_GPU_SRC} ${AUTO_TIMESTAMP})
	ADD_DEPENDENCIES(${GPU_LIB} GEN_TYPE_FUNC_GPU)

	SET(TARGET_LIB ${GPU_LIB} ${CUDA_CUFFT_LIBRARIES} msg_utils_cu nccl)

	FOREACH(exec_path ${GPU_TEST})
		#MESSAGE("EXEC_PATH: ${exec_path}")
		GET_FILENAME_COMPONENT(exec_file ${exec_path} NAME_WE)
		#MESSAGE("EXEC_FILE: ${exec_file}")
		CUDA_ADD_EXECUTABLE(${exec_file} ${exec_path})
		CUDA_ADD_CUFFT_TO_TARGET(${exec_file})
		TARGET_LINK_LIBRARIES(${exec_file} ${TARGET_LIB} ${CUDA_CUFFT_LIBRARIES} ${MPI_LIBRARIES}  blas ${BLAS_LIBRARIES} ${PYTHON_LIBRARIES})
	ENDFOREACH()

	FOREACH(exec_path ${UNIT_TEST_GPU})
		GET_FILENAME_COMPONENT(exec_file ${exec_path} NAME_WE)
		CUDA_ADD_EXECUTABLE(${exec_file} ${exec_path})
		CUDA_ADD_CUFFT_TO_TARGET(${exec_file})
		TARGET_LINK_LIBRARIES(${exec_file} ${TARGET_LIB} ${CUDA_CUFFT_LIBRARIES} ${MPI_LIBRARIES} blas ${BLAS_LIBRARIES} ${PYTHON_LIBRARIES})
		# CATCH_DISCOVER_TESTS(${exec_file})
		IF (${exec_file} MATCHES "multi_node_level")
			ADD_TEST(NAME ${exec_file}_test COMMAND mpirun -n 4 $<TARGET_FILE:${exec_file}>) 
		ELSE()
			ADD_TEST(NAME ${exec_file}_test COMMAND ${exec_file}) 
		ENDIF()
	ENDFOREACH()
ELSE(USE_GPU)
	ADD_LIBRARY(${CPU_LIB} STATIC ${CPU_SRC} ${AUTO_CPU_SRC} ${AUTO_TIMESTAMP})
	SET_TARGET_PROPERTIES(${CPU_LIB} PROPERTIES LINKER_LANGUAGE CXX)
	ADD_DEPENDENCIES(${CPU_LIB} GEN_TYPE_FUNC_CPU) 

	SET(TARGET_LIB ${CPU_LIB} msg_utils)
ENDIF(USE_GPU)

FOREACH(exec_path ${CPU_TEST})
	#MESSAGE("EXEC_PATH: ${exec_path}")
	GET_FILENAME_COMPONENT(exec_file ${exec_path} NAME_WE)
	#MESSAGE("EXEC_FILE: ${exec_file}")
	ADD_EXECUTABLE(${exec_file} ${exec_path})
	TARGET_LINK_LIBRARIES(${exec_file} ${TARGET_LIB} ${MPI_LIBRARIES} ${BLAS_LIBRARIES})
ENDFOREACH()

FOREACH(exec_path ${UNIT_TEST_CPU})
	GET_FILENAME_COMPONENT(exec_file ${exec_path} NAME_WE)
	ADD_EXECUTABLE(${exec_file} ${exec_path})
	TARGET_LINK_LIBRARIES(${exec_file} ${TARGET_LIB} ${MPI_LIBRARIES} blas ${BLAS_LIBRARIES} ${PYTHON_LIBRARIES})
	# CATCH_DISCOVER_TESTS(${exec_file})
	IF (${exec_file} MATCHES "multi_node")
		ADD_TEST(NAME ${exec_file}_test COMMAND mpirun -n 2 $<TARGET_FILE:${exec_file}>) 
	ELSE()
		ADD_TEST(NAME ${exec_file}_test COMMAND ${exec_file}) 
	ENDIF()
ENDFOREACH()


FOREACH(exec_path ${MPI_TEST})
	#MESSAGE("EXEC_PATH: ${exec_path}")
	GET_FILENAME_COMPONENT(exec_file ${exec_path} NAME_WE)
	#MESSAGE("EXEC_FILE: ${exec_file}")
	CUDA_ADD_EXECUTABLE(${exec_file} ${exec_path})
	CUDA_ADD_CUFFT_TO_TARGET(${exec_file})
	TARGET_LINK_LIBRARIES(${exec_file} ${TARGET_LIB} ${MPI_LIBRARIES} blas ${BLAS_LIBRARIES} ${PYTHON_LIBRARIES})
ENDFOREACH()


FOREACH(c_exe ${C_SCRIPTS})
	GET_FILENAME_COMPONENT(exec_file ${c_exe} NAME_WE)
	ADD_EXECUTABLE(${exec_file} ${c_exe})
	# TARGET_LINK_LIBRARIES(${exec_file} ${TARGET_LIB} ${CUDA_CUFFT_LIBRARIES} blas ${BLAS_LIBRARIES} ${PYTHON_LIBRARIES})
ENDFOREACH()


FOREACH(c_tool ${TOOLS})
	GET_FILENAME_COMPONENT(exec_file ${c_tool} NAME_WE)
	ADD_EXECUTABLE(${exec_file} ${c_tool})
	TARGET_LINK_LIBRARIES(${exec_file} ${TARGET_LIB} ${MPI_LIBRARIES} ${BLAS_LIBRARIES} ${PYTHON_LIBRARIES})
ENDFOREACH()

#ADD_CUSTOM_COMMAND(TARGET multi-gpusim_test
#	POST_BUILD
#	COMMAND cp ${CMAKE_BINARY_DIR}/bin/multi-gpusim_test ${CMAKE_BINARY_DIR}/../shared
#	COMMENT "Copy multi-gpusim_test to shared dir"  
#	)

ADD_CUSTOM_TARGET(clean-all
	COMMAND ${CMAKE_BUILD_TOOL} clean
	COMMAND ${CMAKE_COMMAND} -P CLEAN_ALL.cmake
	# COMMAND rm -f ${AUTO_CPU_SRC} ${AUTO_GPU_SRC}
	# COMMAND rm -f ${RUNTIME_OUTPUT_DIRECTORY}/timestamp
	)
