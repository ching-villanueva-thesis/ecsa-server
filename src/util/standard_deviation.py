import math

def sdv(data):
    if len(data) == 0:
        return None  # Handle empty list case

    # Step 1: Calculate the mean
    mean = sum(data) / len(data)
    
    # Step 2: Calculate the variance
    variance = sum((x - mean) ** 2 for x in data) / len(data)
    
    # Step 3: Calculate the standard deviation
    standard_deviation = math.sqrt(variance)
    
    return standard_deviation