
import tensorflow as tf


def create_overlapping_chunks(data, chunk_size=5):
    chunks = []
    for i in range(len(data) - chunk_size):
        chunks.append(data[i:i + chunk_size + 1])  # +1 to include the next day's price
    return chunks

def create_nonoverlapping_chunks(data, chunk_size=5):
    chunks = []
    for i in range(0, len(data), chunk_size):
        chunks.append(data[i:i + chunk_size])
    return chunks

class ChunkingTests(tf.test.TestCase):
    
    def test_create_overlapping_chunks(self):
        data = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        chunk_size = 3
        chunks = create_overlapping_chunks(data, chunk_size)
        self.assertEqual(len(chunks), len(data) - chunk_size)
        self.assertAllEqual(chunks[0], [1, 2, 3, 4])
        self.assertAllEqual(chunks[-1], [7, 8, 9, 10])
        self.assertAllEqual(chunks[3], [4, 5, 6, 7])
        self.assertAllEqual(chunks[5], [6, 7, 8, 9])
        
    def test_create_nonoverlapping_chunks(self):
        data = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
        chunk_size = 3
        chunks = create_nonoverlapping_chunks(data, chunk_size)
        self.assertEqual(len(chunks), len(data) // chunk_size)
        self.assertAllEqual(chunks[0], [0, 1, 2])
        self.assertAllEqual(chunks[1], [3, 4, 5])
        self.assertAllEqual(chunks[2], [6, 7, 8])
        self.assertAllEqual(chunks[-1], [9, 10, 11])
        self.assertAllEqual(chunks[3], [9, 10, 11])
        

if __name__ == '__main__':
    tf.test.main()
