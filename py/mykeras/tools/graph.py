import numpy as np
import tensorflow as tf


def adjacency_list_to_matrix(adj_list):
    n = len(adj_list)
    matrix = np.zeros((n, n))
    for i, neighbors in enumerate(adj_list):
        for neighbor in neighbors:
            matrix[i][neighbor] = 1
    return matrix

def adjacency_matrix_to_list(matrix):
    n = len(matrix)
    adj_list = [[] for _ in range(n)]
    for i in range(n):
        for j in range(n):
            if matrix[i][j] != 0:
                adj_list[i] = (i, j)
    return adj_list

def test_adjacency_list_to_matrix():
    adj_list = [[1, 2], [0, 2], [0, 1]]
    print(adjacency_list_to_matrix(adj_list))
    adj_list = [[1, 2], [0, 2], [0, 1], [4], [3]]
    print(adjacency_list_to_matrix(adj_list))

def test_adjacency_matrix_to_list():
    adj_list = [[1, 2], [0, 2], [0, 1]]
    adj_mat = adjacency_list_to_matrix(adj_list)
    adj_lst = adjacency_matrix_to_list(adj_mat)
    adj_lst.sort()
    adj_list.sort()
    print(adj_lst)
    print(adj_list)
    
class GraphTests(tf.test.TestCase):
    def test_adj_list_to_matrix(self):
        adj_list = [[1, 2], [0, 2], [0, 1]]
        # TODO: verify this is correct
        self.assertAllEqual(adjacency_list_to_matrix(adj_list), [
            [0, 1, 1],
            [1, 0, 1],
            [1, 1, 0]])
        
        adj_list = [[1, 2], [0, 2], [0, 1], [4], [3]]
        self.assertAllEqual(adjacency_list_to_matrix(adj_list), [
            [0, 1, 1, 0, 0],
            [1, 0, 1, 0, 0],
            [1, 1, 0, 0, 0],
            [0, 0, 0, 0, 1],
            [0, 0, 0, 1, 0]])
        
        adj_list = [[1, 2], [0, 2], [0, 1], [4], [3], []]
        self.assertAllEqual(adjacency_list_to_matrix(adj_list), [
            [0, 1, 1, 0, 0, 0],
            [1, 0, 1, 0, 0, 0],
            [1, 1, 0, 0, 0, 0],
            [0, 0, 0, 0, 1, 0],
            [0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0, 0]])

if __name__ == "__main__":
    if True:
        tf.test.main()
    else:
        test_adjacency_matrix_to_list()
