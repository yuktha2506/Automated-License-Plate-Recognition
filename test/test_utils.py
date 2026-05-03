from __future__ import annotations

import unittest

from alpr.utils import bbox_iou, looks_like_plate, normalize_plate_text, plate_text_score


class UtilsTests(unittest.TestCase):
    def test_normalize_plate_text_removes_noise(self) -> None:
        self.assertEqual(normalize_plate_text(" mh-12 ab 1234 "), "MH12AB1234")

    def test_looks_like_plate_requires_letters_and_digits(self) -> None:
        self.assertTrue(looks_like_plate("MH12AB1234"))
        self.assertFalse(looks_like_plate("ABCDEFGH"))
        self.assertFalse(looks_like_plate("12345678"))
        self.assertFalse(looks_like_plate("AAAAA11111"))

    def test_plate_text_score_prefers_reasonable_plate_shapes(self) -> None:
        strong = plate_text_score("MH12AB1234", 0.90)
        weak = plate_text_score("ZZ", 0.90)
        self.assertGreater(strong, weak)

    def test_bbox_iou(self) -> None:
        self.assertAlmostEqual(bbox_iou((10, 10, 50, 20), (20, 10, 50, 20)), 0.6666666667, places=3)


if __name__ == "__main__":
    unittest.main()
