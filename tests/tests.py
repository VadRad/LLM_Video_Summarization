import unittest
from unittest.mock import MagicMock, patch
import numpy as np

class TestOCRHelper(unittest.TestCase):

    def setUp(self):
        self.filepath = 'test.mp4'
        self.sample_rate = 2
        self.ocr_helper = OCRHelper(self.filepath, self.sample_rate)

    @patch('cv2.VideoCapture')
    def test_perform_video_ocr(self, mock_videocapture):
        mock_videocapture.read.return_value = (True, np.array([[0, 0], [0, 0]]))
        frames = self.ocr_helper.perform_video_ocr()
        self.assertIsNotNone(frames)

    def test__are_similar_frame(self):
        frame1 = Frame(1, np.array([[0, 0], [0, 0]]), 0)
        frame2 = Frame(2, np.array([[0, 0], [0, 0]]), 1)
        result = self.ocr_helper._are_similar_frame(frame1, frame2)
        self.assertTrue(result)

    def test__phash(self):
        image = np.array([[0, 0], [0, 0]])
        result = self.ocr_helper._phash(image)
        self.assertIsNotNone(result)

    def test__ocr(self):
        frame = Frame(1, np.array([[0, 0], [0, 0]]), 0)
        result_frame = self.ocr_helper._ocr(frame)
        self.assertIsNotNone(result_frame.text)


if __name__ == '__main__':
    unittest.main()