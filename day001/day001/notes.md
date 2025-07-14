---
## Date: 2025-07-14  
# Author: Suman Khatri

---

# Logic / Steps to Write Code for Similarity Search Between Query Vector and Key of Knowledge Base (KB)

- Input = query vector, key vector from JSON  
- Loop through each key vector and compute similarity search between each query and key vector  

**Note:**  
1. While applying loop for similarity search, make sure it loops through each key vector.  
2. (You can add more notes here later)

---

# Why only `.shape` attribute works in NumPy array and not in list?

## Python List  
- List is a general-purpose container (like a box) that can hold anything: numbers, strings, other lists, objects.  
- It does **not** have built-in knowledge about its own shape or nested structure.

Example:
```python
a = [[1, 2], [4, 5]]
print(a.shape)  # error: 'list' object has no attribute 'shape'
````

## NumPy Array / Tensor

* It's a specialized library for numerical computing.
* Internally it stores data in fixed-size, typed, and multidimensional arrays.
* It knows exactly how the data is laid out in memory.
* Hence, `.shape` attribute works on NumPy arrays.


