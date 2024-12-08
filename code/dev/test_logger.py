import unittest
import numpy as np
import torch
from train import Logger

class TestLogger(unittest.TestCase):

    def test_logger_scalar(self):

        logger = Logger(['loss', 'accuracy'])
        logger.log_batch(0, 0, 'loss', 0.1)
        logger.log_batch(0, 0, 'accuracy', 1)
        logger.log_batch(0, 1, 'loss', 0.2)
        logger.log_batch(0, 1, 'accuracy', 2)
        logger.end_epoch()
        logger.log_batch(1, 0, 'loss', 0.3)
        logger.log_batch(1, 0, 'accuracy', 3)
        logger.log_batch(1, 1, 'loss', 0.4)
        logger.log_batch(1, 1, 'accuracy', 4)
        logger.end_epoch()

        self.assertAlmostEqual(logger.epoch_data[0]['loss'], 0.15)
        self.assertAlmostEqual(logger.epoch_data[1]['loss'], 0.35)
        self.assertAlmostEqual(logger.epoch_data[0]['accuracy'], 1.5)
        self.assertAlmostEqual(logger.epoch_data[1]['accuracy'], 3.5)

    def test_logger_list(self):

        logger = Logger(['loss', 'accuracy'])
        logger.log_batch(0, 0, 'loss', [0.1])
        logger.log_batch(0, 0, 'accuracy', [1])
        logger.log_batch(0, 1, 'loss', [0.2])
        logger.log_batch(0, 1, 'accuracy', [2])
        logger.end_epoch()
        logger.log_batch(1, 0, 'loss', [0.3])
        logger.log_batch(1, 0, 'accuracy', [3])
        logger.log_batch(1, 1, 'loss', [0.4])
        logger.log_batch(1, 1, 'accuracy', [4])
        logger.end_epoch()

        self.assertAlmostEqual(logger.epoch_data[0]['loss'], 0.15)
        self.assertAlmostEqual(logger.epoch_data[1]['loss'], 0.35)
        self.assertAlmostEqual(logger.epoch_data[0]['accuracy'], 1.5)
        self.assertAlmostEqual(logger.epoch_data[1]['accuracy'], 3.5)

    def test_logger_array(self):

        logger = Logger(['loss', 'accuracy'])
        logger.log_batch(0, 0, 'loss', np.array([0.1]))
        logger.log_batch(0, 0, 'accuracy', np.array([1]))
        logger.log_batch(0, 1, 'loss', np.array([0.2]))
        logger.log_batch(0, 1, 'accuracy', np.array([2]))
        logger.end_epoch()
        logger.log_batch(1, 0, 'loss', np.array([0.3]))
        logger.log_batch(1, 0, 'accuracy', np.array([3]))
        logger.log_batch(1, 1, 'loss', np.array([0.4]))
        logger.log_batch(1, 1, 'accuracy', np.array([4]))
        logger.end_epoch()

        self.assertAlmostEqual(logger.epoch_data[0]['loss'], 0.15)
        self.assertAlmostEqual(logger.epoch_data[1]['loss'], 0.35)
        self.assertAlmostEqual(logger.epoch_data[0]['accuracy'], 1.5)
        self.assertAlmostEqual(logger.epoch_data[1]['accuracy'], 3.5)

    def test_logger_tensor(self):

        logger = Logger(['loss', 'accuracy'])
        logger.log_batch(0, 0, 'loss', torch.tensor([0.1], device='cuda'))
        logger.log_batch(0, 0, 'accuracy', torch.tensor([1], device='cuda'))
        logger.log_batch(0, 1, 'loss', torch.tensor([0.2], device='cuda'))
        logger.log_batch(0, 1, 'accuracy', torch.tensor([2], device='cuda'))
        logger.end_epoch()
        logger.log_batch(1, 0, 'loss', torch.tensor([0.3], device='cuda'))
        logger.log_batch(1, 0, 'accuracy', torch.tensor([3], device='cuda'))
        logger.log_batch(1, 1, 'loss', torch.tensor([0.4], device='cuda'))
        logger.log_batch(1, 1, 'accuracy', torch.tensor([4], device='cuda'))
        logger.end_epoch()

        self.assertAlmostEqual(logger.epoch_data[0]['loss'], 0.15)
        self.assertAlmostEqual(logger.epoch_data[1]['loss'], 0.35)
        self.assertAlmostEqual(logger.epoch_data[0]['accuracy'], 1.5)
        self.assertAlmostEqual(logger.epoch_data[1]['accuracy'], 3.5)

    def test_missing_batch(self):

        logger = Logger(['loss', 'accuracy'])
        logger.log_batch(0, 0, 'loss', [0.1])
        logger.log_batch(0, 0, 'accuracy', [1])

        with self.assertRaises(ValueError):
            logger.log_batch(0, 2, 'loss', [0.2])

    def test_missing_metric(self):

        logger = Logger(['loss', 'accuracy'])
        logger.log_batch(0, 0, 'loss', [0.1])
        logger.log_batch(0, 0, 'accuracy', [1])
        logger.log_batch(0, 1, 'loss', [0.2])
        #logger.log_batch(0, 1, 'accuracy', [2])

        with self.assertRaises(ValueError):
            logger.end_epoch()

    def test_wrong_epoch(self):

        logger = Logger(['loss', 'accuracy'])
        logger.log_batch(0, 0, 'loss', [0.1])

        with self.assertRaises(ValueError):
            logger.log_batch(1, 0, 'accuracy', [1])

                         
if __name__ == '__main__':
    unittest.main()