"""Test module to test inference.py.
"""
import unittest
from inference import embeddings_loader, cosine_calc

class TestLoader(unittest.TestCase):
    """Class to test infererencing module.
    """

    def test_img_cos_finder(self):
        """Test function to test if cosine distance calculation results 
        in correct calculation of scores.
        """
        img_embedding = [0.5, 0.6, 0.5, 0.7]
        all_embeddings = ['[0.7, 0.1, 0.9, 0.1]',
                        '[0.4, 0.6, 0.2, 0.7]',
                        '[0.5, 3.6, 0.1, 1.7]',
                        '[0.5, 2.6, 0.5, 0.7]',
                        '[0.5, 0.6, 0.1, 0.9]',
                        '[0.3, 0.6, 0.5, 1.7]',
                        '[0.3, 0.4, 0.5, 0.6]',
                        '[0.3, 0.4, 1.5, 0.5]',
                        '[0.5, 0.1, 0.6, 0.4]',
                        '[0.5, 0.4, 0.6, 0.9]',
                        '[0.1, 0.6, 2.7, 1.2]',
                        '[1.1, 0.6, 0.8, 0.4]',
                        '[0.5, 1.2, 0.8, 0.7]']
        file_paths = ['1.jpg', '2.jpg', '3.jpg', '4.jpg', '5.jpg', '6.jpg', '7.jpg', '8.jpg',
                        '9.jpg', '10.jpg', '11.jpg', '12.jpg', '13.jpg']

        # Testing with correct ordering of file names w.r.t. cosine scores.
        self.assertEqual(cosine_calc(img_embedding, all_embeddings, file_paths), \
            ['7.jpg', '10.jpg', '2.jpg', '13.jpg', '5.jpg', '12.jpg', '6.jpg', '9.jpg', \
                '4.jpg', '3.jpg'])

    def test_all_emb_loader(self):
        """Test if csv file containing all embeddings is loaded
        with correct values and correct data types.
        """
        all_emb_path = 'tests/test_data/small_all_embeddings.csv'
        embeddings, file_paths = embeddings_loader(all_emb_path)

        self.assertEqual(embeddings, ['[0.5, 0.6, 0.5, 0.7]', '[0.4, 0.6, 0.2, 0.7]', \
            '[0.5, 3.6, 0.1, 1.7]', '[0.5, 2.6, 0.5, 0.7]', '[0.5, 0.6, 0.1, 0.9]', \
                '[0.3, 0.6, 0.5, 1.7]'])
        self.assertEqual(file_paths, ['shoes1.jpg', 'shoes2.jpg', \
            'shirt1.jpg', 'shirt2.jpg', 'trouser1.jpg', 'trouser2.jpg'])
        self.assertEqual(type(embeddings), list)
        self.assertEqual(type(file_paths), list)
        self.assertEqual(type(embeddings[0]), str)
        self.assertEqual(type(file_paths[0]), str)
