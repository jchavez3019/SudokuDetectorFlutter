project(CmakeGoogleTestTemplateProject CXX)

add_executable(main main.cpp)
add_executable(displayImageExample displayImageExample.cpp)
add_executable(detectSudokuExample detectSudokuExample.cpp)

target_link_libraries(main
    PUBLIC
    LibraryCode
)

target_link_libraries(displayImageExample
    PUBLIC 
    SudokuDetector
)

target_link_libraries(detectSudokuExample
    PUBLIC
    SudokuDetector
)