
def _merge(arr, start, end):
    if end - start == 1:
        return [arr[start]]
    if end - start == 0:
        return []
    mid = (start + end) // 2
    arr1 = _merge(arr, start, mid)
    arr2 = _merge(arr, mid, end)
    i, j, k = 0, 0, 0
    new_arr = [0] * (len(arr1) + len(arr2))
    while i < len(arr1) and j < len(arr2):
        if arr1[i] < arr2[j]:
            new_arr[k] = arr1[i]
            i += 1
            k += 1
        else:
            new_arr[k] = arr2[j]
            j += 1
            k += 1
    while i < len(arr1):
        new_arr[k] = arr1[i]
        i += 1
        k += 1
    while j < len(arr2):
        new_arr[k] = arr2[j]
        j += 1
        k += 1
    return new_arr

def merge(arr):
    if len(arr) <= 1:
        return arr
    mid = len(arr) // 2
    arr1 = merge(arr[:mid])
    arr2 = merge(arr[mid:])
    i, j = 0, 0
    new_arr = []
    while i < len(arr1) and j < len(arr2):
        if arr1[i] < arr2[j]:
            new_arr.append(arr1[i])
            i += 1
        else:
            new_arr.append(arr2[j])
            j += 1

    new_arr.extend(arr1[i:])
    new_arr.extend(arr2[j:])
    return new_arr



def mergesort(arr):
    # return merge(arr)
    return _merge(arr, 0, len(arr))


if __name__ == "__main__":
    arr = [9, 4, 2, 5, 1,20, -2, 0, 5, 12]
    sarr = [-2, 0, 1, 2, 4, 5, 5, 9, 12, 20]
    narr = mergesort(arr)
    print(narr)
    assert sarr == narr
