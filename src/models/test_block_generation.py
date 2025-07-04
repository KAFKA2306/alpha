"""
Test script for block generation system
"""

import torch
import numpy as np
from block_factory import BlockFactory, create_block_factory
from block_generation_rules import BlockGenerationRules, create_block_generation_rules
import sys
import traceback


def test_block_generation_rules():
    """Test the block generation rules"""
    print("Testing Block Generation Rules...")
    
    try:
        rules = create_block_generation_rules()
        print(f"✓ Created generation rules with {len(rules.templates)} template categories")
        
        # Test generating a few blocks
        block_specs = rules.generate_diverse_blocks(num_blocks=10, min_diversity_threshold=0.5)
        print(f"✓ Generated {len(block_specs)} block specifications")
        
        # Print sample blocks
        for i, spec in enumerate(block_specs[:3]):
            print(f"  Block {i+1}: {spec['name']} ({spec['category']}) - {spec['description'][:50]}...")
        
        return True
        
    except Exception as e:
        print(f"✗ Error in block generation rules: {e}")
        traceback.print_exc()
        return False


def test_block_factory():
    """Test the block factory"""
    print("\nTesting Block Factory...")
    
    try:
        factory = create_block_factory()
        print("✓ Created block factory")
        
        # Test similarity tester
        from domain_blocks import BatchNormBlock, LayerNormBlock
        block1 = BatchNormBlock()
        block2 = LayerNormBlock()
        
        similarity = factory.similarity_tester.calculate_similarity(block1, block2)
        print(f"✓ Similarity test: {similarity['overall']:.3f}")
        
        return True
        
    except Exception as e:
        print(f"✗ Error in block factory: {e}")
        traceback.print_exc()
        return False


def test_block_creation():
    """Test creating actual blocks"""
    print("\nTesting Block Creation...")
    
    try:
        factory = create_block_factory()
        
        # Generate a few blocks
        blocks = factory.mass_produce_blocks(num_blocks=5, diversity_threshold=0.5)
        print(f"✓ Created {len(blocks)} blocks")
        
        # Test each block
        test_input_shape = (8, 252, 64)  # batch, seq_len, features
        
        for i, block in enumerate(blocks):
            try:
                # Test module creation
                module = block.create_module(test_input_shape)
                print(f"  Block {i+1}: {block.name} - Module created ✓")
                
                # Test forward pass
                test_input = torch.randn(test_input_shape)
                with torch.no_grad():
                    output = module(test_input)
                    print(f"    Input shape: {test_input.shape}, Output shape: {output.shape}")
                
                # Test output shape calculation
                expected_shape = block.get_output_shape(test_input_shape)
                print(f"    Expected output shape: {expected_shape}")
                
            except Exception as e:
                print(f"  Block {i+1}: {block.name} - Error: {e}")
        
        return True
        
    except Exception as e:
        print(f"✗ Error in block creation: {e}")
        traceback.print_exc()
        return False


def test_diversity_metrics():
    """Test diversity metrics"""
    print("\nTesting Diversity Metrics...")
    
    try:
        factory = create_block_factory()
        
        # Generate blocks
        blocks = factory.mass_produce_blocks(num_blocks=10, diversity_threshold=0.4)
        
        # Test diversity
        meets_threshold, diversity_scores = factory.similarity_tester.test_diversity_threshold(
            blocks, threshold=0.4
        )
        
        print(f"✓ Diversity test completed")
        print(f"  Meets threshold: {meets_threshold}")
        print(f"  Diversity scores: {diversity_scores}")
        
        return True
        
    except Exception as e:
        print(f"✗ Error in diversity testing: {e}")
        traceback.print_exc()
        return False


def run_all_tests():
    """Run all tests"""
    print("=" * 60)
    print("BLOCK GENERATION SYSTEM TESTS")
    print("=" * 60)
    
    tests = [
        test_block_generation_rules,
        test_block_factory,
        test_block_creation,
        test_diversity_metrics
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
    
    print(f"\n{'='*60}")
    print(f"TEST RESULTS: {passed}/{total} tests passed")
    print(f"{'='*60}")
    
    return passed == total


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)