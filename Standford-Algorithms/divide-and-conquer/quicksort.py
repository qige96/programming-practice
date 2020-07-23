import random

def _partition(arr, start, end):
    p = random.randint(start, end-1)
    arr[start], arr[p] = arr[p], arr[start]
    pivot = arr[start]
    i, j = start+1, start+1
    while j < end:
        if arr[j] < pivot:
            arr[i], arr[j] = arr[j], arr[i]
            i += 1
        j += 1
    arr[start], arr[i-1] = arr[i-1], arr[start]
    return i-1

def _quick(arr, start, end):
    if start < end:
        p = _partition(arr, start, end)
        _quick(arr, start, p)
        _quick(arr, p+1, end)

def quicksort(arr):
    _quick(arr, 0, len(arr))



if __name__ == "__main__":
    arr = [9, 4, 2, 5, 1, 20, -2, 0, 5, 12]
    quicksort(arr)
    sarr = [-2, 0, 1, 2, 4, 5, 5, 9, 12, 20]
    print(arr)
    assert arr == sarr
