cmake_minimum_required(VERSION 2.8)
project(main)

# Define executable path
set(EXECUTABLE_OUTPUT_PATH "../")

# Set packages 
find_package(OpenCV REQUIRED)
find_package(onnxruntime REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS} ${onnxruntime_INCLUDE_DIRS})

# Set executable
add_executable(main main.cpp)
target_link_libraries(main ${OpenCV_LIBS} ${onnxruntime_LIBRARIES})
