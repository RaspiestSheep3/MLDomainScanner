cmake_minimum_required(VERSION 3.20)

project("Neuroevolution" CXX)

# Locate dependencies installed by vcpkg.json
find_package(nlohmann_json CONFIG REQUIRED)
find_package(OpenMP REQUIRED)
find_package(SQLite3 REQUIRED)

# Declare target executable
add_executable(Neuroevolution "Neuroevolution.cpp" "Neuroevolution.h")
set_property(TARGET Neuroevolution PROPERTY CXX_STANDARD 20)

# Link libraries
target_link_libraries(Neuroevolution PRIVATE 
    nlohmann_json::nlohmann_json
    OpenMP::OpenMP_CXX
    SQLite::SQLite3
)

# MSVC Release flags
target_compile_options(Neuroevolution PRIVATE 
    $<$<CONFIG:Release>:/O2 /arch:AVX2 /fp:fast>
)