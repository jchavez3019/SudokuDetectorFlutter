project(CmakeGoogleTestTemplateProject CXX)

add_executable(test_one test_one.cpp)
add_executable(merge_sort_test merge_sort_test.cpp)

target_link_libraries(test_one
    PUBLIC
    gtest_main
    gmock_main
    LibraryCode
)

target_link_libraries(merge_sort_test
    PUBLIC
    gtest_main
    gmock_main
    SudokuDetector
)

add_test(
    NAME test_one
    COMMAND test_one
)

add_test(
    NAME merge_sort_test
    COMMAND merge_sort_test
)