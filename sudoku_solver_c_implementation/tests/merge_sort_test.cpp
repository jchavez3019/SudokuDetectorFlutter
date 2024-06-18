#include <gtest/gtest.h>
#include "helperFunctions.h"
#include <vector>

/* MACROS
    ASSERT_TRUE() -- terminates the test if the argument is not true
    ASSERT_EQ() -- terminates the test if the second argument is not equal to its first argument
    EXPECT_TRUE() -- fails if the argument is not true but continues with testing
    EXPECT_EQ() -- fails if the second argument is not equal to its first argument but continues with testing

    You can explore more functionality in the gtest documentation. 
*/

/* correct order of the list */
const int ordered_values [] = {1,2,3,4,5};
const int list_size = 5;

const int unordered [][list_size] = {
    {1,2,5,4,3},
    {3,4,1,2,5},
    {1,4,3,2,5},
    {3,1,5,2,4},
    {1,2,3,4,5},
    {5,4,3,2,1}
};

const int correct_indices [][list_size] = {
    {0,1,4,3,2},
    {2, 3, 0, 1, 4},
    {0, 3, 2, 1, 4},
    {1, 3, 0, 4, 2},
    {0,1,2,3,4},
    {4,3,2,1,0}
};



using namespace std;
using namespace helper;

TEST(TestHelperSuite, TestMergeOne)
/**
 * @brief Checks the output given that the first element in the given list is correct
 */
{
    const int test_num = 0;

    /* check that the list is sorted correctly */
    int check_arr [list_size];
    for (int i = 0; i < list_size; i++)
        check_arr[i] = unordered[test_num][i];

    mergeSort(check_arr, 0, list_size - 1);

    ASSERT_TRUE(arraysEqual(ordered_values, check_arr, list_size)) << "Merge list is incorrect";

    int check_arr_indices [list_size];
    argMergeSort(unordered[test_num], check_arr_indices, 0, list_size - 1);

    EXPECT_TRUE(arraysEqual(check_arr_indices, correct_indices[test_num], list_size)) << "Sorted indices are incorrect";
}

TEST(TestHelperSuite, TestMergeTwo)
/**
 * @brief Checks the output given that the last element in the given list is correct
 */
{
    const int test_num = 1;

    /* check that the list is sorted correctly */
    int check_arr [list_size];
    for (int i = 0; i < list_size; i++)
        check_arr[i] = unordered[test_num][i];

    mergeSort(check_arr, 0, list_size - 1);

    ASSERT_TRUE(arraysEqual(ordered_values, check_arr, list_size)) << "Merge list is incorrect";

    int check_arr_indices [list_size];
    argMergeSort(unordered[test_num], check_arr_indices, 0, list_size - 1);

    EXPECT_TRUE(arraysEqual(check_arr_indices, correct_indices[test_num], list_size)) << "Sorted indices are incorrect";
}

TEST(TestHelperSuite, TestMergeThree)
/**
 * @brief Checks the output on a list where the first and last element are correct but the middle elements are reversed.
 */
{
    const int test_num = 2;

    /* check that the list is sorted correctly */
    int check_arr [list_size];
    for (int i = 0; i < list_size; i++)
        check_arr[i] = unordered[test_num][i];

    mergeSort(check_arr, 0, list_size - 1);

    ASSERT_TRUE(arraysEqual(ordered_values, check_arr, list_size)) << "Merge list is incorrect";

    int check_arr_indices [list_size];
    argMergeSort(unordered[test_num], check_arr_indices, 0, list_size - 1);

    EXPECT_TRUE(arraysEqual(check_arr_indices, correct_indices[test_num], list_size)) << "Sorted indices are incorrect";
}

TEST(TestHelperSuite, TestMergeFour)
/**
 * @brief Ensures all elements are in the correct order given a list that is out of order
 */
{
    const int test_num = 3;

    /* check that the list is sorted correctly */
    int check_arr [list_size];
    for (int i = 0; i < list_size; i++)
        check_arr[i] = unordered[test_num][i];

    mergeSort(check_arr, 0, list_size - 1);

    ASSERT_TRUE(arraysEqual(ordered_values, check_arr, list_size)) << "Merge list is incorrect";

    int check_arr_indices [list_size];
    argMergeSort(unordered[test_num], check_arr_indices, 0, list_size - 1);

    EXPECT_TRUE(arraysEqual(check_arr_indices, correct_indices[test_num], list_size)) << "Sorted indices are incorrect";
}

TEST(TestHelperSuite, TestMergeFive)
/**
 * @brief Given a list already in order, ensures the output is the same order
 */
{
    const int test_num = 4;

    /* check that the list is sorted correctly */
    int check_arr [list_size];
    for (int i = 0; i < list_size; i++)
        check_arr[i] = unordered[test_num][i];

    mergeSort(check_arr, 0, list_size - 1);

    ASSERT_TRUE(arraysEqual(ordered_values, check_arr, list_size)) << "Merge list is incorrect";

    int check_arr_indices [list_size];
    argMergeSort(unordered[test_num], check_arr_indices, 0, list_size - 1);

    EXPECT_TRUE(arraysEqual(check_arr_indices, correct_indices[test_num], list_size)) << "Sorted indices are incorrect";
}

/* list is reversed */
TEST(TestHelperSuite, TestMergeSix)
/**
 * @brief Checks the output on a list whose order is reversed
 */
{
    const int test_num = 5;

    /* check that the list is sorted correctly */
    int check_arr [list_size];
    for (int i = 0; i < list_size; i++)
        check_arr[i] = unordered[test_num][i];

    mergeSort(check_arr, 0, list_size - 1);

    ASSERT_TRUE(arraysEqual(ordered_values, check_arr, list_size)) << "Merge list is incorrect";

    int check_arr_indices [list_size];
    argMergeSort(unordered[test_num], check_arr_indices, 0, list_size - 1);

    EXPECT_TRUE(arraysEqual(check_arr_indices, correct_indices[test_num], list_size)) << "Sorted indices are incorrect";
}

int main (int argc, char **argv) {
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}