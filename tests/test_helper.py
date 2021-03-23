import unittest

from mcrf import helper


class TestHelper(unittest.TestCase):
    def test_ent_reveal(self):
        chars = list("北京北京，我是宇航员翟志刚。我已出舱，感觉良好。")
        tags = [
            "B-LOC", "I-LOC", "B-LOC", "I-LOC", "O",
            "O", "O", "B", "I", "I", "B-PER", "M_PER", "E_PER", "O",
            "O", "O", "O", "O", "O", "O", "O", "O", "O", "O"
        ]
        entities = helper.get_entities_from_tag_seq(chars, tags)
        self.assertEqual(entities["default"], [('宇航员', 'default', 7, 10)])
        self.assertEqual(entities["LOC"], [('北京', 'LOC', 0, 2), ('北京', 'LOC', 2, 4)])
        self.assertEqual(entities["PER"], [('翟志刚', 'PER', 10, 13)])
