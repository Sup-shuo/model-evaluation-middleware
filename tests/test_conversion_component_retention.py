import unittest

from tools.model_conversion.component_retention import component_retention


class ComponentRetentionTests(unittest.TestCase):
    def test_dense_vision_and_dropped_mtp(self):
        source = ["model.visual.merge.qweight", "model.visual.merge.qzeros", "mtp.fc.weight"]
        output = ["model.visual.merge.weight", "lm_head.weight"]
        self.assertEqual(component_retention(source, output), {"vision": "retained", "mtp": "removed"})

    def test_text_extraction_and_absent_mtp(self):
        self.assertEqual(component_retention(["visual.merge.weight"], ["lm_head.weight"]),
                         {"vision": "removed", "mtp": "absent_in_source"})

    def test_partial_vision_is_not_retained(self):
        self.assertEqual(component_retention(["visual.encoder.weight", "visual.merger.weight"],
                                             ["visual.encoder.weight"])["vision"], "changed")

    def test_ignore_similar_names_and_handle_extra_output(self):
        self.assertEqual(component_retention(["nonvisual.weight", "not_mtp.weight"], ["mtp.fc.weight"]),
                         {"vision": "absent_in_source", "mtp": "output_only"})

    def test_packed_weights_and_projector(self):
        src = ["mtp.fc.weight_packed", "mtp.fc.weight_scale", "mm_projector.proj.weight"]
        dst = ["mtp.fc.weight", "mm_projector.proj.weight"]
        self.assertEqual(component_retention(src, dst), {"vision": "retained", "mtp": "retained"})
