# ETIQ:

project(myAQP)
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
cmake_minimum_required(VERSION 3.14)
set(Torch_DIR /usr/local/libtorch/share/cmake/Torch) # specify the path of libTorch

find_package(Torch REQUIRED)    
find_package(NumCpp 2.1.0 REQUIRED)
 
add_library( myAQP
             SHARED
             myAQP.cpp)
 
target_link_libraries(${PROJECT_NAME} NumCpp::NumCpp) # link NumCpp library
target_link_libraries(myAQP ${TORCH_LIBRARIES}) # link libTorch library

set_property(TARGET myAQP PROPERTY CXX_STANDARD 14)


# ETEQ:

# project(myAQP)
# set(CMAKE_CXX_STANDARD 14)
# set(CMAKE_CXX_STANDARD_REQUIRED ON)
# cmake_minimum_required(VERSION 3.14)

# set(Torch_DIR /usr/local/libtorch/share/cmake/Torch) 

# find_package(Torch REQUIRED)   
# find_package(NumCpp 2.1.0 REQUIRED) 
 
# add_executable(ETEQ ETEQ.cpp)
# target_link_libraries(ETEQ "${TORCH_LIBRARIES}")
# target_link_libraries(ETEQ NumCpp::NumCpp)