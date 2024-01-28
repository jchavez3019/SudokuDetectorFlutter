#pragma once
#include <vector>
#include <cstdio>

// Merges two subarrays of array[].
// First subarray is arr[begin..mid]
// Second subarray is arr[mid+1..end]
template <typename T>
void _merge(T* array, int const left, int const mid, int const right, bool getIndices = false, int* argIndices = nullptr)
{   
    if (getIndices && argIndices == nullptr) {
        perror("getIndices set to true but argIndices is nullptr\n");
        return;
    }


    int const subArrayOne = mid - left + 1;
    int const subArrayTwo = right - mid;
 
    // Create temp arrays
    auto *leftArray = new T[subArrayOne], *rightArray = new T[subArrayTwo];
 
    // Copy data to temp arrays leftArray[] and rightArray[]
    for (auto i = 0; i < subArrayOne; i++)
        leftArray[i] = array[left + i];
    for (auto j = 0; j < subArrayTwo; j++)
        rightArray[j] = array[mid + 1 + j];
 
    auto indexOfSubArrayOne = 0, indexOfSubArrayTwo = 0;
    int indexOfMergedArray = left;

    /* copy old indices */
    int* oldIndices;
    if (getIndices) {
        oldIndices = new int[subArrayOne + subArrayTwo];
        for (int i = 0; i < subArrayOne + subArrayTwo; i++)
            oldIndices[i] = argIndices[left + i];
    }
 
    // Merge the temp arrays back into array[left..right]
    while (indexOfSubArrayOne < subArrayOne && indexOfSubArrayTwo < subArrayTwo) {
        if (leftArray[indexOfSubArrayOne] <= rightArray[indexOfSubArrayTwo]) {
            array[indexOfMergedArray] = leftArray[indexOfSubArrayOne];
            if (getIndices) {
                // argIndices[indexOfMergedArray] = left + indexOfSubArrayOne;
                argIndices[indexOfMergedArray] = oldIndices[indexOfSubArrayOne];
            }
            indexOfSubArrayOne++;
        }
        else {
            array[indexOfMergedArray] = rightArray[indexOfSubArrayTwo];
            if (getIndices) {
                // argIndices[indexOfMergedArray] = mid + 1 + indexOfSubArrayTwo;
                argIndices[indexOfMergedArray] = oldIndices[subArrayOne + indexOfSubArrayTwo];
            }
            indexOfSubArrayTwo++;
        }
        indexOfMergedArray++;
    }
 
    // Copy the remaining elements of
    // left[], if there are any
    while (indexOfSubArrayOne < subArrayOne) {
        array[indexOfMergedArray] = leftArray[indexOfSubArrayOne];
        if (getIndices) {
            // argIndices[indexOfMergedArray] = left + indexOfSubArrayOne;
                argIndices[indexOfMergedArray] = oldIndices[indexOfSubArrayOne];
        }
        indexOfSubArrayOne++;
        indexOfMergedArray++;
    }
 
    // Copy the remaining elements of
    // right[], if there are any
    while (indexOfSubArrayTwo < subArrayTwo) {
        array[indexOfMergedArray] = rightArray[indexOfSubArrayTwo];
        if (getIndices) {
            // argIndices[indexOfMergedArray] = mid + 1 + indexOfSubArrayTwo;
            argIndices[indexOfMergedArray] = oldIndices[subArrayOne + indexOfSubArrayTwo];
        }
        indexOfSubArrayTwo++;
        indexOfMergedArray++;
    }
    delete[] leftArray;
    delete[] rightArray;

    if (getIndices)
        delete[] oldIndices;
}

template<typename T>
void _argMergeSortRecurse(T* array, int* argIndices, int const begin, int const end) {

    if (begin >= end)
        return;

    int mid = begin + (end - begin) / 2;
    _argMergeSortRecurse(array, argIndices, begin, mid);
    _argMergeSortRecurse(array, argIndices, mid + 1, end);
    _merge(array, begin, mid, end, true, argIndices);
}
 
namespace helper {
    // begin is for left index and end is right index
    // of the sub-array of arr to be sorted
    template <typename T>
    void mergeSort(T* array, int const begin, int const end)
    {
        if (begin >= end)
            return;
    
        int mid = begin + (end - begin) / 2;
        mergeSort(array, begin, mid);
        mergeSort(array, mid + 1, end);
        _merge(array, begin, mid, end);
    }

    

    template <typename T>
    void argMergeSort(const T* array, int* argIndices, int const begin, int const end) {
        auto *copy_array = new T[end + 1];

        /* create a mutable copy of the array and ordered indices */
        for (int i = 0; i < end + 1; i++)  {
            copy_array[i] = array[i];
            argIndices[i] = i;
        }
            

        _argMergeSortRecurse(copy_array, argIndices, begin, end);

        delete[] copy_array;

        return;
    }

    template <typename T>
    void flipArray(T* array, int array_size) {
        int start = 0;
        int end = array_size - 1;
        T temp;
        while (start < end) {
            temp = array[start];
            array[start] = array[end];
            array[end] = temp;

            start++;
            end--;
        }

        return;
    }

    template <typename T>
    int arraysEqual(const T* array_one, const T* array_two, int array_size) {

        for (int i = 0; i < array_size; i++) {
            if (array_one[i] != array_two[i])
                return 0;
        }
        
        return 1;
    }
}
