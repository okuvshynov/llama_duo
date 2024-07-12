cmake_minimum_required(VERSION 3.16)

project(serv)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED true)

include(FetchContent)

FetchContent_Declare(
    llama.cpp
    GIT_REPOSITORY https://github.com/ggerganov/llama.cpp.git
    GIT_TAG        b3269
)
FetchContent_MakeAvailable(llama.cpp)

FetchContent_Declare(
    json
    URL https://github.com/nlohmann/json/releases/download/v3.11.3/json.tar.xz
)
FetchContent_MakeAvailable(json)

FetchContent_Declare(
    httplib
    GIT_REPOSITORY https://github.com/yhirose/cpp-httplib.git
    GIT_TAG        v0.15.3
)
FetchContent_MakeAvailable(httplib)


add_executable(duo  duo.cpp)
target_link_libraries(duo  PRIVATE common) # from llama.cpp
target_compile_definitions(duo PRIVATE LLAMA_RPC=ON)

add_executable(serv serv.cpp)
target_link_libraries(serv PRIVATE nlohmann_json::nlohmann_json)
target_link_libraries(serv PRIVATE httplib)
target_link_libraries(serv  PRIVATE common) # from llama.cpp
target_compile_definitions(serv PRIVATE LLAMA_RPC=ON)

if(MSVC)
  target_compile_options(duo  PRIVATE /W4 /WX)
  target_compile_options(serv PRIVATE /W4 /WX)
else()
  target_compile_options(duo  PRIVATE -Wall -Wextra -Wpedantic)
  target_compile_options(serv PRIVATE -Wall -Wextra -Wpedantic)
endif()

