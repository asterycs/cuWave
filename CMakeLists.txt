cmake_minimum_required(VERSION 3.10)

# Parameters
set(ENABLE_CUDA ON CACHE BOOL "Enable Cuda features") # Enable by default

# Set project language
project(cuWave LANGUAGES C CXX CUDA)
add_definitions(-DENABLE_CUDA)

# This is for figuring out whether to ues lib64 or lib
include(GNUInstallDirs)

# Define some variables
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -Werror -Wno-error=int-in-bool-context -Wno-error=unused-parameter -Wconversion -Wno-error=conversion -fasynchronous-unwind-tables -fstack-protector-all -Werror=format-security -Werror=implicit-function-declaration -Wdisabled-optimization -Wformat -Wcast-align -Wformat-security -Wmissing-braces -Wparentheses  -Wpointer-arith -Wredundant-decls -Wreturn-type -Wsign-compare -Wuninitialized -Wunreachable-code -Wunused -Wunused-label -Wunused-value  -Wunused-variable")
add_definitions(-D_GLIBCXX_DEBUG)
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${CMAKE_SOURCE_DIR}/cmake/modules/)
set(CMAKE_CXX_STANDARD 17)
set_property(GLOBAL PROPERTY USE_FOLDERS ON)
include(ExternalProject)

find_package(OpenMP)
if (OPENMP_FOUND)
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}") # CXX_FLAGS not detected properly
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
endif()

# Pre-installed dependencies
find_package(CUDA REQUIRED)
CUDA_SELECT_NVCC_ARCH_FLAGS(CUDA_ARCH_FLAGS Auto)
list(APPEND CMAKE_CUDA_FLAGS ${CUDA_ARCH_FLAGS})
string(REPLACE ";" " " CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS}") # Phew...
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -lineinfo --Werror cross-execution-space-call -Xptxas -dlcm=ca --ptxas-options=-v --compiler-options=\"${CMAKE_CXX_FLAGS}\"")

find_package(X11 REQUIRED)
find_package(OpenGL REQUIRED)
find_package(Threads REQUIRED)
find_package(PkgConfig REQUIRED)
pkg_check_modules(GTK2 REQUIRED gtk+-3.0)

find_program(CCACHE_FOUND ccache)
if(CCACHE_FOUND)
    message(STATUS "CCache found")
    set_property(GLOBAL PROPERTY RULE_LAUNCH_COMPILE ccache)
    set_property(GLOBAL PROPERTY RULE_LAUNCH_LINK ccache)
endif(CCACHE_FOUND)

# tinyobjloader
ExternalProject_Add(tinyobjloader
    PREFIX "${CMAKE_BINARY_DIR}/ext/tinyobjloader"
    GIT_REPOSITORY https://github.com/syoyo/tinyobjloader.git
    GIT_TAG "v1.0.6"
    GIT_PROGRESS 1
    UPDATE_DISCONNECTED 1
    LOG_DOWNLOAD 1
    LOG_BUILD 1
    LOG_INSTALL 1
    CMAKE_ARGS
       "-DCMAKE_INSTALL_PREFIX=<INSTALL_DIR>"
       "-DCMAKE_BUILD_TYPE=Release"
       )

ExternalProject_Get_Property(tinyobjloader INSTALL_DIR)
set(TINYOBJLOADER_INCLUDE_DIRS ${INSTALL_DIR}/include)
set(TINYOBJLOADER_LIBRARIES "${INSTALL_DIR}/${CMAKE_INSTALL_LIBDIR}/${CMAKE_STATIC_LIBRARY_PREFIX}tinyobjloader${CMAKE_STATIC_LIBRARY_SUFFIX}")

# Native file dialog
ExternalProject_Add(nativefiledialog
    PREFIX "${CMAKE_BINARY_DIR}/ext/nativefiledialog"
    GIT_REPOSITORY https://github.com/mlabbe/nativefiledialog.git
    GIT_PROGRESS 1
    CONFIGURE_COMMAND ""
    BUILD_COMMAND cd ${CMAKE_BINARY_DIR}/ext/nativefiledialog/src/nativefiledialog/build/gmake_linux && ${CMAKE_MAKE_PROGRAM}
    INSTALL_COMMAND ""
    UPDATE_DISCONNECTED 1
    )

set(NATIVEFILEDIALOG_INCLUDE_DIRS "${CMAKE_BINARY_DIR}/ext/nativefiledialog/src/nativefiledialog/src/include")
set(NATIVEFILEDIALOG_LIBRARIES "${CMAKE_BINARY_DIR}/ext/nativefiledialog/src/nativefiledialog/build/lib/Release/x64/${CMAKE_STATIC_LIBRARY_PREFIX}nfd${CMAKE_STATIC_LIBRARY_SUFFIX}")


# glm
ExternalProject_Add(
    glm
    PREFIX ${CMAKE_BINARY_DIR}/ext/glm
    GIT_REPOSITORY https://github.com/g-truc/glm.git
    GIT_TAG 0.9.9.2
    GIT_PROGRESS 1
    UPDATE_DISCONNECTED 1
    LOG_DOWNLOAD 1
    UPDATE_COMMAND ""
    CONFIGURE_COMMAND ""
    BUILD_COMMAND ""
    INSTALL_COMMAND ""
    )

ExternalProject_Get_Property(glm SOURCE_DIR)
set(GLM_INCLUDE_DIRS ${SOURCE_DIR})


# cxxopts
ExternalProject_Add(cxxopts
    PREFIX "${CMAKE_BINARY_DIR}/ext/cxxopts"
    GIT_REPOSITORY https://github.com/jarro2783/cxxopts.git
    GIT_TAG 2_0
    GIT_PROGRESS 1
    UPDATE_DISCONNECTED 1
    LOG_DOWNLOAD 1
    LOG_BUILD 1
    LOG_INSTALL 1
    INSTALL_DIR "${CMAKE_BINARY_DIR}/ext/cxxopts"
    CMAKE_ARGS
       "-DCMAKE_INSTALL_PREFIX=${CMAKE_BINARY_DIR}/ext/cxxopts"
       "-DCMAKE_BUILD_TYPE=Release"
       "-DCXXOPTS_BUILD_EXAMPLES=OFF"
       "-DCXXOPTS_BUILD_TESTS=OFF"
    )
ExternalProject_Get_Property(cxxopts SOURCE_DIR)
set(CXXOPTS_INCLUDE_DIRS ${SOURCE_DIR}/include)

# glfw
ExternalProject_Add(
    glfw
    PREFIX ${CMAKE_BINARY_DIR}/ext/glfw
    GIT_REPOSITORY https://github.com/glfw/glfw.git
    GIT_TAG 3.2.1
    GIT_PROGRESS 1
    UPDATE_DISCONNECTED 1
    LOG_DOWNLOAD 1
    LOG_INSTALL 1
    CMAKE_ARGS
        "-DCMAKE_INSTALL_PREFIX=<INSTALL_DIR>"
        "-DCMAKE_BUILD_TYPE=Release"
        "-DGLFW_BUILD_EXAMPLES=OFF"
        "-DGLFW_BUILD_TESTS=OFF"
        "-DGLFW_BUILD_DOCS=OFF"
    )

ExternalProject_Get_Property(glfw INSTALL_DIR)
set(GLFW_INCLUDE_DIRS ${INSTALL_DIR}/include)
set(GLFW_LIBRARIES ${INSTALL_DIR}/lib/${CMAKE_STATIC_LIBRARY_PREFIX}glfw3${CMAKE_STATIC_LIBRARY_SUFFIX})

# glew
ExternalProject_Add(glew
    PREFIX ${CMAKE_BINARY_DIR}/ext/glew
    GIT_REPOSITORY https://github.com/nigels-com/glew.git
    GIT_TAG glew-2.1.0 
    UPDATE_DISCONNECTED 1
    LOG_DOWNLOAD 1
    LOG_CONFIGURE 1
    LOG_BUILD 1
    LOG_INSTALL 1
    PATCH_COMMAND cd ${CMAKE_BINARY_DIR}/ext/glew/src/glew/auto/
 && ${CMAKE_MAKE_PROGRAM}
    CONFIGURE_COMMAND ${CMAKE_COMMAND} -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=<INSTALL_DIR> <SOURCE_DIR>/build/cmake
    )

ExternalProject_Get_Property(glew INSTALL_DIR)
set(GLEW_INCLUDE_DIRS ${INSTALL_DIR}/include)
set(GLEW_LIBRARIES ${INSTALL_DIR}/${CMAKE_INSTALL_LIBDIR}/${CMAKE_STATIC_LIBRARY_PREFIX}GLEW${CMAKE_STATIC_LIBRARY_SUFFIX})

# imgui
ExternalProject_Add(
    imgui
    DEPENDS glew
    PREFIX ${CMAKE_BINARY_DIR}/ext/imgui
    GIT_REPOSITORY https://github.com/ocornut/imgui.git
    GIT_TAG v1.53
    GIT_PROGRESS 1
    UPDATE_DISCONNECTED 1
    LOG_DOWNLOAD 1
    PATCH_COMMAND ${CMAKE_COMMAND} -E copy ${CMAKE_SOURCE_DIR}/patches/CMakeLists_imgui.txt <SOURCE_DIR>/CMakeLists.txt COMMAND git apply ${CMAKE_SOURCE_DIR}/patches/imgui_change_gl_loader.patch || true
    CMAKE_ARGS
        "-DCMAKE_INSTALL_PREFIX=<INSTALL_DIR>"
        "-DGLFW_INCLUDE_DIRS=${GLFW_INCLUDE_DIRS}"
        "-DGLEW_INCLUDE_DIRS=${GLEW_INCLUDE_DIRS}"
        "-DCMAKE_BUILD_TYPE=Release"
    )

ExternalProject_Get_Property(imgui INSTALL_DIR)
set(IMGUI_INCLUDE_DIRS ${INSTALL_DIR}/include)
set(IMGUI_LIBRARIES ${INSTALL_DIR}/lib/${CMAKE_STATIC_LIBRARY_PREFIX}imgui${CMAKE_STATIC_LIBRARY_SUFFIX})

ExternalProject_Add(devil
    PREFIX ${CMAKE_BINARY_DIR}/ext/devil
    GIT_REPOSITORY "https://github.com/DentonW/DevIL"
    UPDATE_DISCONNECTED 1
    CONFIGURE_COMMAND ${CMAKE_COMMAND} -DCMAKE_INSTALL_PREFIX=<INSTALL_DIR>
        <SOURCE_DIR>/DevIL
    CMAKE_ARGS
        "-DCMAKE_BUILD_TYPE=Release"
)

ExternalProject_Get_Property(devil INSTALL_DIR)
set(IL_INCLUDE_DIRS ${INSTALL_DIR}/include)
set(IL_LIBRARIES ${INSTALL_DIR}/lib/${CMAKE_SHARED_LIBRARY_PREFIX}IL${CMAKE_SHARED_LIBRARY_SUFFIX};${INSTALL_DIR}/lib/${CMAKE_SHARED_LIBRARY_PREFIX}ILU${CMAKE_SHARED_LIBRARY_SUFFIX};${INSTALL_DIR}/lib/${CMAKE_SHARED_LIBRARY_PREFIX}ILUT${CMAKE_SHARED_LIBRARY_SUFFIX})

include_directories(
    SYSTEM
    ${IL_INCLUDE_DIRS}
    ${GLEW_INCLUDE_DIRS}
    ${GLFW_INCLUDE_DIRS}
    ${IMGUI_INCLUDE_DIRS}
    ${TINYOBJLOADER_INCLUDE_DIRS}
    ${NATIVEFILEDIALOG_INCLUDE_DIRS}
    ${GTK2_INCLUDE_DIRS}
    ${IL_INCLUDE_DIR}
    ${CUDA_INCLUDE_DIRS}
    ${CXXOPTS_INCLUDE_DIRS}
    ${GLM_INCLUDE_DIRS}
)


set(LINK_LIBS
    ${GLEW_LIBRARIES}
    ${GLFW_LIBRARIES}
    ${IL_LIBRARIES}
    GL
    ${X11_LIBRARIES}
    ${X11_Xcursor_LIB}
    ${X11_Xinerama_LIB}
    ${X11_Xrandr_LIB}
    ${X11_Xxf86vm_LIB}
    ${CMAKE_DL_LIBS}
    ${CMAKE_THREAD_LIBS_INIT}
    ${TINYOBJLOADER_LIBRARIES}
    ${NATIVEFILEDIALOG_LIBRARIES}
    ${GTK2_LIBRARIES}
    ${IL_LIBRARIES}
    ${ILUT_LIBRARIES}
    ${CUDA_LIBRARIES}
    ${CUDA_curand_LIBRARY}
    ${IMGUI_LIBRARIES}
    )

add_subdirectory(src)

