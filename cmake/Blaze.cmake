# Download Blaze to be made part of the build
FetchContent_Declare(
    blaze
    GIT_REPOSITORY https://bitbucket.org/blaze-lib/blaze.git
    GIT_TAG master
)

FetchContent_MakeAvailable(blaze)