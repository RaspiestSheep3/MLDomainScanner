cmake_minimum_required (VERSION 3.8)

if (POLICY CMP0141)
  cmake_policy(SET CMP0141 NEW)
  set(CMAKE_MSVC_DEBUG_INFORMATION_FORMAT "$<IF:$<AND:$<C_COMPILER_ID:MSVC>,$<CXX_COMPILER_ID:MSVC>>,$<$<CONFIG:Debug,RelWithDebInfo>:EditAndContinue>,$<$<CONFIG:Debug,RelWithDebInfo>:ProgramDatabase>>")
endif()

project ("Neuroevolution")

add_executable (Neuroevolution "Neuroevolution.cpp" "Neuroevolution.h")

if (CMAKE_VERSION VERSION_GREATER 3.12)
  set_property(TARGET Neuroevolution PROPERTY CXX_STANDARD 20)
endif()

find_package(nlohmann_json CONFIG REQUIRED)
find_package(OpenMP REQUIRED)
target_link_libraries(Neuroevolution PRIVATE nlohmann_json::nlohmann_json)
target_link_libraries(Neuroevolution PRIVATE OpenMP::OpenMP_CXX)

target_compile_options(Neuroevolution PRIVATE 
    $<$<CONFIG:Release>:/O2 /arch:AVX2 /fp:fast>
)