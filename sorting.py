def _default_key(x):
    return x

def merge_sort(arr, key=None):
    if key is None:
        key = _default_key
        
    if len(arr) <= 1:
        return arr
    
    mid = len(arr) // 2
    
    left_half = []
    for i in range(mid):
        left_half.append(arr[i])
        
    right_half = []
    for i in range(mid, len(arr)):
        right_half.append(arr[i])
        
    left = merge_sort(left_half, key)
    right = merge_sort(right_half, key)
    
    result = []
    i = 0
    j = 0
    while i < len(left) and j < len(right):
        if key(left[i]) <= key(right[j]):
            result.append(left[i])
            i += 1
        else:
            result.append(right[j])
            j += 1
            
    while i < len(left):
        result.append(left[i])
        i += 1
        
    while j < len(right):
        result.append(right[j])
        j += 1
        
    return result
