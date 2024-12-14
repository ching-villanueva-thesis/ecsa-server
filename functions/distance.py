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
    
        return f
