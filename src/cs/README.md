## Discretization

### One-hot encoding

1. Solution representation is a matrix of binary values. In our implementation, the size of the matrix is determined by:

> Buildings/Barangays x Evacuation Spaces

```py
# Given: 5 buildings, 5 evacuation spaces
solution = [
    [0,0,0,0,0], # building 1
    [1,1,1,1,1], # building 2
    [0,0,0,0,0], # building 3
    [1,1,1,1,1], # building 4
    [0,0,0,0,0], # building 5
]
```

1 represents that a building is allocated to an evacuation space.

0 represents that a building is not allocated to an evacuation space.

### Levy Movement using Mantegna's algorithm

To generate levy values, Mantegna's algorithm is used. The process of the movement follows these steps:

1. Generate levy value (step) via Mantegna's algorithm
2. Get the step size (alpha \* step)
3. Generate a new solution using a random nest and step size
4. Convert the new solution to probability via Sigmoid function
5. Convert to binary using probability
