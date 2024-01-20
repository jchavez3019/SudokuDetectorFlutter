project(CmakeGoogleTestTemplateProject CXX)

add_library(LibraryCode SHARED LibraryCode.cpp)
add_library(SudokuDetector SHARED SudokuDetector.cpp)
# add_library(helperFunctions SHARED helperFunctions.cpp)

target_include_directories(LibraryCode
    PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/../include
)

target_include_directories(SudokuDetector
    PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/../include
)

# target_include_directories(helperFunctions
#     PUBLIC
#     ${CMAKE_CURRENT_SOURCE_DIR}/../include
# )

target_link_libraries(SudokuDetector
    ${OpenCV_LIBS}
)