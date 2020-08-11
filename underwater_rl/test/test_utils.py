import os
import unittest

from underwater_rl.utils import convert_images_to_video


class TestUtils(unittest.TestCase):
    def setUp(self) -> None:
        self.video_path = os.path.join('assets', 'pong.mp4')
        convert_images_to_video('assets/video', 'assets')

    def tearDown(self) -> None:
        os.remove(self.video_path)

    def test_convert_images_to_video_produces_video(self):
        self.assertTrue(os.path.exists(self.video_path))


if __name__ == '__main__':
    unittest.main()
