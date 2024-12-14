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
        f = 0
        offset = 0

        for i in range(len(self.db_coordinates)):
            for j in range(len(self.es_coordinates)):
                if(solution[offset + j]):
                    f += self._distance_matrix[i][j]

            offset += len(self.es_coordinates)

        return f
