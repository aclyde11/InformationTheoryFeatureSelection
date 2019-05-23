project(feature_selector)


if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

set(CMAKE_CXX_FLAGS_RELEASE "-O3")


find_package(Eigen3 3.3 REQUIRED NO_MODULE)

if (UNIX AND NOT APPLE)
    set(LINUX TRUE)
endif ()
if (APPLE)
    set(CMAKE_C_COMPILER "/usr/local/Cellar/llvm/8.0.0_1/bin/clang")
    set(CMAKE_CXX_COMPILER "/usr/local/Cellar/llvm/8.0.0_1/bin/clang++")
    set(OPENMP_LIBRARIES "/usr/local/Cellar/llvm/8.0.0_1/lib")
    set(OPENMP_INCLUDES "/usr/local/Cellar/llvm/8.0.0_1/include")

    OPTION(USE_OpenMP "Use OpenMP to enamble <omp.h>" ON)

    # Find OpenMP
    if (APPLE AND USE_OpenMP)
        if (CMAKE_C_COMPILER_ID MATCHES "Clang")
            set(OpenMP_C "${CMAKE_C_COMPILER}")
            set(OpenMP_C_FLAGS "-fopenmp=libomp -Wno-unused-command-line-argument")
            set(OpenMP_C_LIB_NAMES "libomp" "libgomp" "libiomp5")
            set(OpenMP_libomp_LIBRARY ${OpenMP_C_LIB_NAMES})
            set(OpenMP_libgomp_LIBRARY ${OpenMP_C_LIB_NAMES})
            set(OpenMP_libiomp5_LIBRARY ${OpenMP_C_LIB_NAMES})
        endif ()
        if (CMAKE_CXX_COMPILER_ID MATCHES "Clang")
            set(OpenMP_CXX "${CMAKE_CXX_COMPILER}")
            set(OpenMP_CXX_FLAGS "-fopenmp=libomp -Wno-unused-command-line-argument")
            set(OpenMP_CXX_LIB_NAMES "libomp" "libgomp" "libiomp5")
            set(OpenMP_libomp_LIBRARY ${OpenMP_CXX_LIB_NAMES})
            set(OpenMP_libgomp_LIBRARY ${OpenMP_CXX_LIB_NAMES})
            set(OpenMP_libiomp5_LIBRARY ${OpenMP_CXX_LIB_NAMES})
        endif ()
    endif ()

    if (USE_OpenMP)
        find_package(OpenMP REQUIRED)
    endif (USE_OpenMP)

    if (OPENMP_FOUND)
        include_directories("${OPENMP_INCLUDES}")
        link_directories("${OPENMP_LIBRARIES}")
        set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
        set (CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
    endif (OPENMP_FOUND)
endif ()

if(LINUX)
    find_package(OpenMP REQUIRED) # Find the package
    if (OPENMP_FOUND)
        set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
        set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
        set (CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
    endif()
endif()


add_executable(${PROJECT_NAME} main.cpp)
if(LINUX)
    find_package(OpenMP REQUIRED) # Find the package
    target_link_libraries(${PROJECT_NAME} ${OpenMP_CXX_LIBRARIES})
endif()


target_link_libraries(${PROJECT_NAME}
        sub::milib
        Eigen3::Eigen
        cxxopts
        )

