# CMake Google Test Template

## Summary

This repo serves as an outline for how to create a project with gtest.

* [gtest github](https://github.com/google/googletest)
* [gtest documentation](https://google.github.io/googletest/)
* [gtest video 1](https://www.youtube.com/watch?v=5wI47v4kuxU)
* [gtest video 2](https://www.youtube.com/watch?v=Lp1ifh9TuFI&t=694s)

## Getting Started

From the root directory, you will first create a '\build' directory and navigate to it like so...

```sh
    mkdir build && cd build
```

Next you must run the cmake tool and make all the executable objects. 

```sh
    cmake .. && make clean && make
```

Afterwards, you may run the tests using the command

```sh
    ctest
```

This will display a summary of which tests have passed/failed. For a more detailed log of the errors, the output will tell you to navigate to '/build/Testing/Temporary/LastTest.log'. 

For running executables that are not tests but examples that serve as demos for your project, you can simply navigate to '/build/examples' and run any executable files there. For example, from the root directory you may run

```sh
    ./build/example/main
```

