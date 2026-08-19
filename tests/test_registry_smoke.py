import unittest

import torch

from models.domain_blocks import get_domain_block_registry


class RegistrySmokeTest(unittest.TestCase):
    def test_default_registry_executes_layer_norm(self):
        registry = get_domain_block_registry()
        self.assertEqual(len(registry.get_all_blocks()), 15)

        module = registry.get_block("layer_norm").create_module((2, 4, 3))
        output = module(torch.randn(2, 4, 3))
        self.assertEqual(tuple(output.shape), (2, 4, 3))


if __name__ == "__main__":
    unittest.main()
