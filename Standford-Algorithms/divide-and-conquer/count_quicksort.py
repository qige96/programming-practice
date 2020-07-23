import random

n_comparison = 0

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

def _first_partition(arr, start, end):
    p = 0
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


def _last_partition(arr, start, end):
    p = -1
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

def _choose_median(arr, start, end):
    head = start
    tail = end-1
    if len(arr) % 2 == 0:
        middle = arr[(end-1+start)//2 - 1]
    else:
        middle = arr[(end-1+start)//2]
    print(head, middle, tail)
    max_elem = max([arr[head], arr[middle], arr[tail]])
    if max_elem == arr[head]:
        return head
    elif max_elem == arr[middle]:
        return middle
    else:
        return tail


def _median_partition(arr, start, end):
    p = _choose_median(arr, start, end)
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
    global n_comparison
    if start < end:
        p = _median_partition(arr, start, end)
        _quick(arr, start, p)
        _quick(arr, p+1, end)
        n_comparison += p - start - 1
        n_comparison += end - (p + 1) -1

def quicksort(arr):
    _quick(arr, 0, len(arr))



if __name__ == "__main__":
    with open('quicksort.txt', 'r') as f:
        array = f.readlines()
    arr = [int(i[:-1]) for i in array]
    quicksort(arr)
    print(n_comparison)
