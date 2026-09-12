import unittest

import torch

from models.domain_blocks import get_domain_block_registry


class RegistrySmokeTest(unittest.TestCase):
    canonical_shape = (2, 16, 8)

    def test_default_registry_executes_every_registered_block(self):
        registry = get_domain_block_registry()
        blocks = registry.get_all_blocks()
        self.assertEqual(len(blocks), 15)

        for block in blocks:
            with self.subTest(block=block.name):
                self.assertTrue(block.validate_input_shape(self.canonical_shape))
                module = block.create_module(self.canonical_shape)
                module.eval()
                output = module(torch.randn(*self.canonical_shape))
                expected_shape = tuple(block.get_output_shape(self.canonical_shape))
                self.assertEqual(tuple(output.shape), expected_shape)

    def test_time_mixing_conv_preserves_channel_last_shape(self):
        registry = get_domain_block_registry()
        block = registry.get_block("time_mixing")
        module = block.create_module(self.canonical_shape, mixing_type="conv")
        output = module(torch.randn(*self.canonical_shape))
        self.assertEqual(
            tuple(output.shape),
            tuple(block.get_output_shape(self.canonical_shape, mixing_type="conv")),
        )


if __name__ == "__main__":
    unittest.main()
