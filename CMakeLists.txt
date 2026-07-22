cmake_minimum_required(VERSION 3.20)
project(MiJuego CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Ruta al SDK precompilado de Filament (motor de render PBR)
set(FILAMENT_DIR "${CMAKE_SOURCE_DIR}/../tools/filament" CACHE PATH "Filament SDK path")

# --- GLFW (ventana + input), se descarga y compila solo ---
include(FetchContent)
set(GLFW_BUILD_DOCS OFF CACHE BOOL "" FORCE)
set(GLFW_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(GLFW_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
FetchContent_Declare(
    glfw
    GIT_REPOSITORY https://github.com/glfw/glfw.git
    GIT_TAG 3.4
)
FetchContent_MakeAvailable(glfw)

add_executable(MiJuego src/main.cpp)

target_include_directories(MiJuego PRIVATE
    "${FILAMENT_DIR}/include"
)

target_link_directories(MiJuego PRIVATE
    "${FILAMENT_DIR}/lib/x86_64/md"
)

# Libs base de Filament necesarias para una escena PBR simple con backend OpenGL
target_link_libraries(MiJuego PRIVATE
    glfw
    filament
    backend
    filabridge
    filaflat
    utils
    geometry
    bluegl
    bluevk
    smol-v
    ibl
    zstd
    opengl32
)

target_compile_definitions(MiJuego PRIVATE FILAMENT_SUPPORTS_OPENGL=1)

# --- Compilar el material .mat a .filamat con matc (herramienta incluida en el SDK) ---
set(MATC_EXE "${FILAMENT_DIR}/bin/matc.exe")
set(MAT_SRC "${CMAKE_SOURCE_DIR}/assets/lit.mat")
set(MAT_OUT "${CMAKE_BINARY_DIR}/assets/lit.filamat")

add_custom_command(
    OUTPUT "${MAT_OUT}"
    COMMAND ${CMAKE_COMMAND} -E make_directory "${CMAKE_BINARY_DIR}/assets"
    COMMAND "${MATC_EXE}" -a opengl -o "${MAT_OUT}" "${MAT_SRC}"
    DEPENDS "${MAT_SRC}"
    COMMENT "Compilando material PBR con matc"
)
add_custom_target(CompileMaterials ALL DEPENDS "${MAT_OUT}")
add_dependencies(MiJuego CompileMaterials)
