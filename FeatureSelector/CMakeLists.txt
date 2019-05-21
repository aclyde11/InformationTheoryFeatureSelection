project(feature_selector)

find_package (Eigen3 3.3 REQUIRED NO_MODULE)


add_executable(${PROJECT_NAME} main.cpp)
target_link_libraries(${PROJECT_NAME}
        sub::milib
        Eigen3::Eigen
        )