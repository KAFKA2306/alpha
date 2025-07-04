#!/usr/bin/env python3
"""
Demo script showing the Alpha Architecture Agent in action.

This script demonstrates:
1. Loading configuration
2. Generating neural network architectures using AI agents
3. Compiling architectures into PyTorch models
4. Evaluating architectural diversity and complexity
"""

import sys
import os
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import torch
import numpy as np
from typing import List

from agents.architecture_agent import ArchitectureAgent, LLMArchitectureGenerator
from models.domain_blocks import get_domain_block_registry
from core.config import get_config


def demonstrate_domain_blocks():
    """Demonstrate the domain block system."""
    print("=" * 60)
    print("DOMAIN BLOCKS DEMONSTRATION")
    print("=" * 60)
    
    # Try to use extended registry first, fallback to basic
    try:
        from models.domain_blocks_extended import get_extended_domain_block_registry
        registry = get_extended_domain_block_registry()
        print("✓ Using EXTENDED domain block registry with all implemented blocks")
    except ImportError:
        registry = get_domain_block_registry()
        print("⚠ Using basic domain block registry (extended blocks not available)")
    
    print(f"Total registered blocks: {len(registry.get_all_blocks())}")
    print(f"Available categories: {', '.join(registry.get_categories())}")
    
    # Show blocks by category
    for category in registry.get_categories():
        blocks = registry.get_blocks_by_category(category)
        print(f"\n{category.upper()} ({len(blocks)} blocks):")
        for block in blocks:
            hyperparams = block.get_hyperparameters()
            if hyperparams:
                print(f"  - {block.name}: {block.description}")
                print(f"    Hyperparameters: {list(hyperparams.keys())}")
            else:
                print(f"  - {block.name}: {block.description}")


def demonstrate_architecture_generation():
    """Demonstrate AI-powered architecture generation."""
    print("\n" + "=" * 60)
    print("ARCHITECTURE GENERATION DEMONSTRATION")
    print("=" * 60)
    
    # Define input shape for Japanese stock data
    # (batch_size, sequence_length, features)
    input_shape = (32, 252, 20)  # 32 stocks, 252 trading days, 20 features
    
    print(f"Input shape: {input_shape}")
    print("(batch_size, sequence_length, features)")
    print("- 32 stocks in a batch")
    print("- 252 trading days (1 year)")
    print("- 20 features (returns, technical indicators, etc.)")
    
    # Initialize architecture agent
    print("\nInitializing Architecture Agent...")
    
    try:
        # Try to use LLM-based generation (may fail if no API keys)
        agent = ArchitectureAgent()
        print("✓ LLM-based agent initialized")
    except Exception as e:
        print(f"⚠ LLM initialization failed: {e}")
        print("  This is expected if API keys are not configured")
        print("  Demonstration will use random generation as fallback")
        
        # Create agent with random generator
        from agents.architecture_agent import ArchitectureCompiler
        agent = ArchitectureAgent(
            generator=None,  # Will use random fallback
            compiler=ArchitectureCompiler()
        )
    
    # Generate a small suite of architectures
    print("\nGenerating 5 sample architectures...")
    
    try:
        architectures = agent.generate_architecture_suite(
            input_shape=input_shape,
            num_architectures=5
        )
        
        print(f"\n✓ Successfully generated {len(architectures)} architectures")
        
        # Display architecture summaries
        for i, arch in enumerate(architectures, 1):
            print(f"\n--- Architecture {i}: {arch.name} ---")
            print(f"ID: {arch.id}")
            print(f"Blocks: {len(arch.blocks)}")
            print(f"Complexity Score: {arch.complexity_score:.2f}")
            print(f"Diversity Score: {arch.diversity_score:.2f}")
            
            print("Block sequence:")
            for j, block_spec in enumerate(arch.blocks):
                block_name = block_spec['name']
                hyperparams = block_spec.get('hyperparameters', {})
                if hyperparams:
                    print(f"  {j+1}. {block_name} {hyperparams}")
                else:
                    print(f"  {j+1}. {block_name}")
            
            if 'description' in arch.metadata:
                print(f"Description: {arch.metadata['description']}")
    
    except Exception as e:
        print(f"✗ Architecture generation failed: {e}")
        return []
    
    return architectures


def demonstrate_model_compilation(architectures: List):
    """Demonstrate compiling architectures into PyTorch models."""
    if not architectures:
        print("\nSkipping model compilation - no architectures available")
        return
    
    print("\n" + "=" * 60)
    print("MODEL COMPILATION DEMONSTRATION")
    print("=" * 60)
    
    agent = ArchitectureAgent()
    
    print(f"Compiling {len(architectures)} architectures into PyTorch models...")
    
    compiled_models = agent.compile_architecture_suite(architectures)
    
    print(f"\n✓ Successfully compiled {len(compiled_models)} models")
    
    # Test models with dummy data
    input_shape = (32, 252, 20)
    dummy_input = torch.randn(input_shape)
    
    print(f"\nTesting models with dummy input {input_shape}...")
    
    for arch_id, model in compiled_models.items():
        try:
            model.eval()
            with torch.no_grad():
                output = model(dummy_input)
            
            arch_name = next((a.name for a in architectures if a.id == arch_id), arch_id)
            print(f"✓ {arch_name}: Input {input_shape} → Output {output.shape}")
            
        except Exception as e:
            print(f"✗ {arch_id}: Failed - {e}")


def demonstrate_architecture_analysis(architectures: List):
    """Demonstrate analysis of generated architectures."""
    if not architectures:
        print("\nSkipping architecture analysis - no architectures available")
        return
    
    print("\n" + "=" * 60)
    print("ARCHITECTURE ANALYSIS")
    print("=" * 60)
    
    registry = get_domain_block_registry()
    
    # Analyze block usage
    block_usage = {}
    category_usage = {}
    
    for arch in architectures:
        for block_spec in arch.blocks:
            block_name = block_spec['name']
            block = registry.get_block(block_name)
            
            # Count block usage
            block_usage[block_name] = block_usage.get(block_name, 0) + 1
            
            # Count category usage
            category = block.category
            category_usage[category] = category_usage.get(category, 0) + 1
    
    print("Block Usage Frequency:")
    sorted_blocks = sorted(block_usage.items(), key=lambda x: x[1], reverse=True)
    for block_name, count in sorted_blocks:
        print(f"  {block_name}: {count} times")
    
    print("\nCategory Usage Frequency:")
    sorted_categories = sorted(category_usage.items(), key=lambda x: x[1], reverse=True)
    for category, count in sorted_categories:
        print(f"  {category}: {count} times")
    
    # Analyze complexity and diversity
    complexities = [arch.complexity_score for arch in architectures]
    diversities = [arch.diversity_score for arch in architectures]
    
    print(f"\nComplexity Statistics:")
    print(f"  Mean: {np.mean(complexities):.2f}")
    print(f"  Std:  {np.std(complexities):.2f}")
    print(f"  Range: {np.min(complexities):.2f} - {np.max(complexities):.2f}")
    
    print(f"\nDiversity Statistics:")
    print(f"  Mean: {np.mean(diversities):.2f}")
    print(f"  Std:  {np.std(diversities):.2f}")
    print(f"  Range: {np.min(diversities):.2f} - {np.max(diversities):.2f}")


def main():
    """Main demonstration function."""
    print("Alpha Architecture Agent - Demonstration")
    print("AI-Powered Neural Network Architecture Generation for Stock Prediction")
    
    # Load configuration
    config = get_config()
    print(f"\nLoaded configuration: {config.project_name} v{config.project_version}")
    
    # Run demonstrations
    demonstrate_domain_blocks()
    
    architectures = demonstrate_architecture_generation()
    
    demonstrate_model_compilation(architectures)
    
    demonstrate_architecture_analysis(architectures)
    
    print("\n" + "=" * 60)
    print("DEMONSTRATION COMPLETE")
    print("=" * 60)
    print("")
    print("Next steps:")
    print("1. Configure API keys for LLM-based generation")
    print("2. Implement data pipeline for Japanese stock data")
    print("3. Create backtesting framework for strategy evaluation")
    print("4. Build ensemble system for strategy combination")
    print("")
    print("For more information, see:")
    print("- docs/architecture/system_architecture.md")
    print("- config/config.yaml")
    print("- src/ directory for implementation details")


if __name__ == "__main__":
    main()