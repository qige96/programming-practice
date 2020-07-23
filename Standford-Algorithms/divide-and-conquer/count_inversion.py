
def _merge_count_split_inv(arr1, arr2):
    new_arr = []
    inv = 0
    while arr1 and arr2:
        if arr1[0] < arr2[0]:
            new_arr.append(arr1.pop(0))
        else:
            new_arr.append(arr2.pop(0))
            inv += len(arr1)
    new_arr.extend(arr1)
    new_arr.extend(arr2)
    return new_arr, inv

def _sort_count(arr):
    n = len(arr)
    if n == 1 or n == 0:
        return arr, 0
    brr, x = _sort_count(arr[:n//2])
    crr, y = _sort_count(arr[n//2:])
    drr, z = _merge_count_split_inv(brr, crr)
    return drr, x + y + z

def count_inversion(arr):
    _, inv = _sort_count(arr)
    return inv

if __name__ == "__main__":
    arr = [1, 20, 6, 4, 5] 
    inv = count_inversion(arr)
    print(inv)
    assert inv == 5
