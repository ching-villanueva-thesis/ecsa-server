import numpy as np

class Distance:
    def __init__(self, db_coordinates, es_coordinates):
        self.db_coordinates = np.array(db_coordinates)
        self.es_coordinates = np.array(es_coordinates)
        self._distance_matrix = self._to_matrix()

    def _to_matrix(self):
        return np.linalg.norm(self.db_coordinates[:, np.newaxis, :] - self.es_coordinates[np.newaxis, :, :], axis=2)
    
    def get_distance_matrix(self):
        return self._distance_matrix
    
    def fitness(self, solution):
        # Reshape the solution into a 2D array matching the shape of the distance matrix
        solution_matrix = np.array(solution).reshape(len(self.db_coordinates), len(self.es_coordinates))

        # Element-wise multiply the solution matrix with the distance matrix and sum the results
        f = np.sum(self._distance_matrix * solution_matrix)

        # Constraint: In every subarray of length 11, at least one element must be 1
        penalty = 0
        penalty_value = 1000  # Set an appropriate penalty value

        for i in range(0, len(solution), len(self.es_coordinates) - 1): 
            subarray = solution[i:i+len(self.es_coordinates) - 1]

            if 1 not in subarray:  # If no 1s in the subarray, apply a penalty
                penalty += penalty_value

        return f + penalty  # Adding the penalty to the fitness function

    def decode_results(self, result):
        # Create a dictionary to store the shortest allocation per db_coordinate
        shortest_allocations = {}
        allocations = []

        # Define coordinate pointers
        es_pointer = 0
        db_pointer = 0

        # Loop through the result array
        for _, j in enumerate(result):
            if j == 1:
                allocations.append([list(self.db_coordinates[db_pointer]), list(self.es_coordinates[es_pointer])])
                db_coord = tuple(self.db_coordinates[db_pointer])  # Convert to tuple for dict key
                es_coord = tuple(self.es_coordinates[es_pointer])

                # Calculate distance (Haversine or Euclidean)
                distance = np.linalg.norm(np.array(db_coord) - np.array(es_coord))

                # Keep only the shortest distance allocation
                if db_coord not in shortest_allocations or distance < shortest_allocations[db_coord][1]:
                    shortest_allocations[db_coord] = (es_coord, distance)

            es_pointer += 1
            if es_pointer >= len(self.es_coordinates):
                es_pointer = 0
                db_pointer += 1

        # Convert dictionary back to list format
        points = [[list(db), list(es)] for db, (es, _) in shortest_allocations.items()]

        return points, len(allocations)