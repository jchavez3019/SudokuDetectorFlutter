#include <gtest/gtest.h>
#include "LibraryCode.hpp"

/* MACROS
    ASSERT_TRUE() -- terminates the test if the argument is not true
    ASSERT_EQ() -- terminates the test if the second argument is not equal to its first argument
    EXPECT_TRUE() -- fails if the argument is not true but continues with testing
    EXPECT_EQ() -- fails if the second argument is not equal to its first argument but continues with testing

    You can explore more functionality in the gtest documentation. 
*/

TEST(TestSuiteSample, TestSample) {

    int loop_sum;
    for (int i = 0; i < 10; i++) {
        loop_sum = sum(2, i);
        EXPECT_EQ(2 + i, loop_sum) << "sum failed on iteration i=" << i;
    }

    EXPECT_TRUE(false) << "Failed intentionally by sending false to expect true macro";

    int ret_sum = sum(2,4);
    ASSERT_EQ(6, ret_sum);
}

int main (int argc, char **argv) {
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}