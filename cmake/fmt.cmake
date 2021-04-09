# Download fmt to be made part of the build
FetchContent_Declare(
    fmt
    GIT_REPOSITORY https://github.com/fmtlib/fmt.git
    GIT_TAG master
)

FetchContent_MakeAvailable(fmt)