
file(GLOB CXX_SRC *.cpp)
file(GLOB CU_SRC *.cu)
set_source_files_properties(App.cpp CudaModel.cpp Camera.cpp Light.cpp Utils.cpp ModelLoader.cpp PROPERTIES LANGUAGE CUDA)

file(GLOB SHADERS "shaders/*/*.glsl")
source_group("shaders" FILES SHADERS)

add_executable(cuWave ${CXX_SRC} ${CXX_CU_SRC} ${CU_SRC})
add_dependencies(cuWave glm glew imgui tinyobjloader glfw cxxopts nativefiledialog)

target_link_libraries(cuWave ${LINK_LIBS})
add_dependencies(cuWave copy_shader_files copy_benchmark_script)

set_target_properties(cuWave PROPERTIES CUDA_STANDARD 14 CUDA_SEPARABLE_COMPILATION ON CUDA_RESOLVE_DEVICE_SYMBOLS ON)

add_custom_target(
        copy_shader_files
        ${CMAKE_COMMAND} -E copy_directory
        ${CMAKE_SOURCE_DIR}/src/shaders ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/shaders/
        COMMENT "Copying shader files for target: cuWave" VERBATIM 
        )

add_custom_target(
        copy_benchmark_script
        ${CMAKE_COMMAND} -E copy
        ${CMAKE_SOURCE_DIR}/benchmark.sh ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}
        COMMENT "Copying benchmark script for target: cuWave" VERBATIM 
        )

add_custom_command(
        TARGET cuWave
        COMMAND ${CMAKE_SOURCE_DIR}/patches/error_parsing_fix.sh ${CMAKE_BINARY_DIR}/.project
        DEPENDS ${CMAKE_BINARY_DIR}/.project
        COMMENT "Fixing error parsing" VERBATIM
        )
