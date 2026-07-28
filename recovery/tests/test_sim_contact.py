import unittest
from types import SimpleNamespace

from recovery.teachers.sim_contact import extract_target_gripper_contact


class FakeModel:
    def __init__(self, names):
        self.names = names

    def geom_id2name(self, geom_id):
        return self.names[geom_id]


class SimContactTest(unittest.TestCase):
    def test_bilateral_target_contact_is_attached(self):
        sim = SimpleNamespace(
            model=FakeModel(
                {
                    0: "akita_black_bowl_1_g0",
                    1: "gripper0_leftfinger_collision",
                    2: "gripper0_rightfinger_collision",
                }
            ),
            data=SimpleNamespace(
                ncon=2,
                contact=[
                    SimpleNamespace(geom1=0, geom2=1),
                    SimpleNamespace(geom1=2, geom2=0),
                ],
            ),
        )
        state, attached = extract_target_gripper_contact(
            sim, target_name="akita_black_bowl_1"
        )
        self.assertTrue(attached)
        self.assertEqual(state["num_pairs"], 2)
        self.assertEqual(state["finger_sides"], ["left", "right"])

    def test_unrelated_or_single_contact_is_not_attached(self):
        sim = SimpleNamespace(
            model=FakeModel(
                {
                    0: "akita_black_bowl_1_g0",
                    1: "gripper0_leftfinger_collision",
                    2: "table_collision",
                }
            ),
            data=SimpleNamespace(
                ncon=2,
                contact=[
                    SimpleNamespace(geom1=0, geom2=1),
                    SimpleNamespace(geom1=2, geom2=1),
                ],
            ),
        )
        state, attached = extract_target_gripper_contact(
            sim, target_name="akita_black_bowl_1"
        )
        self.assertFalse(attached)
        self.assertTrue(state["target_gripper_contact"])
        self.assertEqual(state["num_pairs"], 1)


if __name__ == "__main__":
    unittest.main()
