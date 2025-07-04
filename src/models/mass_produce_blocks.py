"""
Mass Production Script for Domain Blocks

This script generates hundreds of diverse domain blocks based on the generation
rules and saves them for integration into the extended registry.
"""

import json
import sys
import os
from typing import Dict, Any, List
from datetime import datetime
import traceback


def mock_torch_imports():
    """Mock torch imports for generation without installation"""
    import sys
    from unittest.mock import MagicMock
    
    # Mock torch
    torch_mock = MagicMock()
    torch_mock.nn = MagicMock()
    torch_mock.nn.Module = object
    torch_mock.nn.Linear = MagicMock
    torch_mock.nn.LayerNorm = MagicMock
    torch_mock.randn = lambda *args: [0] * (args[0] if args else 1)
    torch_mock.tensor = lambda x: x
    
    sys.modules['torch'] = torch_mock
    sys.modules['torch.nn'] = torch_mock.nn
    sys.modules['torch.nn.functional'] = MagicMock()
    
    # Mock numpy
    numpy_mock = MagicMock()
    numpy_mock.mean = lambda x: 0.5
    numpy_mock.prod = lambda x: 1
    sys.modules['numpy'] = numpy_mock


def generate_block_batch(batch_size: int = 50, diversity_threshold: float = 0.4) -> List[Dict[str, Any]]:
    """Generate a batch of diverse blocks"""
    print(f"Generating batch of {batch_size} blocks with diversity threshold {diversity_threshold}...")
    
    mock_torch_imports()
    from block_generation_rules import create_block_generation_rules
    
    rules = create_block_generation_rules()
    
    # Generate multiple attempts with lower threshold to get more blocks
    all_blocks = []
    max_attempts = batch_size * 5  # Generate 5x more attempts than needed
    
    generated_specs = rules.generate_diverse_blocks(
        num_blocks=max_attempts,
        min_diversity_threshold=diversity_threshold
    )
    
    print(f"Generated {len(generated_specs)} block specifications")
    return generated_specs[:batch_size]  # Take only what we need


def create_production_run(target_blocks: int = 200, output_dir: str = "generated_blocks") -> Dict[str, Any]:
    """Create a full production run of blocks"""
    
    print("=" * 80)
    print("DOMAIN BLOCK MASS PRODUCTION")
    print("=" * 80)
    print(f"Target: {target_blocks} diverse blocks")
    print(f"Output directory: {output_dir}")
    print(f"Start time: {datetime.now()}")
    print()
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate blocks in batches to ensure diversity
    all_blocks = []
    batch_size = 50
    diversity_threshold = 0.3  # Lower threshold for more blocks
    
    num_batches = (target_blocks + batch_size - 1) // batch_size
    
    for batch_num in range(num_batches):
        print(f"Batch {batch_num + 1}/{num_batches}:")
        
        try:
            batch = generate_block_batch(batch_size, diversity_threshold)
            all_blocks.extend(batch)
            
            print(f"  Generated {len(batch)} blocks")
            print(f"  Total so far: {len(all_blocks)}")
            
            if len(all_blocks) >= target_blocks:
                all_blocks = all_blocks[:target_blocks]
                break
                
        except Exception as e:
            print(f"  Error in batch {batch_num + 1}: {e}")
            continue
    
    print(f"\nFinal count: {len(all_blocks)} blocks generated")
    
    # Analyze the generated blocks
    analysis = analyze_generated_blocks(all_blocks)
    
    # Save blocks to files
    save_blocks(all_blocks, analysis, output_dir)
    
    return {
        'total_blocks': len(all_blocks),
        'analysis': analysis,
        'output_dir': output_dir,
        'generation_time': datetime.now().isoformat()
    }


def analyze_generated_blocks(blocks: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Analyze the generated blocks for diversity and distribution"""
    
    print("\nAnalyzing generated blocks...")
    
    # Category distribution
    category_counts = {}
    complexity_counts = {}
    component_counts = {}
    
    for block in blocks:
        # Categories
        cat = block['category']
        category_counts[cat] = category_counts.get(cat, 0) + 1
        
        # Complexity
        comp = block['complexity']
        complexity_counts[comp] = complexity_counts.get(comp, 0) + 1
        
        # Components
        for component in block['components']:
            component_counts[component] = component_counts.get(component, 0) + 1
    
    # Parameter diversity
    all_param_keys = set()
    param_value_diversity = {}
    
    for block in blocks:
        for key, value in block['parameters'].items():
            all_param_keys.add(key)
            if key not in param_value_diversity:
                param_value_diversity[key] = set()
            param_value_diversity[key].add(str(value))
    
    # Calculate diversity scores
    total_categories = len(category_counts)
    total_complexities = len(complexity_counts)
    total_components = len(component_counts)
    avg_param_diversity = sum(len(values) for values in param_value_diversity.values()) / len(param_value_diversity) if param_value_diversity else 0
    
    analysis = {
        'category_distribution': category_counts,
        'complexity_distribution': complexity_counts,
        'component_distribution': component_counts,
        'parameter_diversity': {k: len(v) for k, v in param_value_diversity.items()},
        'diversity_metrics': {
            'category_diversity': total_categories,
            'complexity_diversity': total_complexities,
            'component_diversity': total_components,
            'avg_parameter_diversity': avg_param_diversity
        }
    }
    
    print(f"Category diversity: {total_categories} categories")
    print(f"Complexity diversity: {total_complexities} levels") 
    print(f"Component diversity: {total_components} components")
    print(f"Average parameter diversity: {avg_param_diversity:.2f}")
    
    return analysis


def save_blocks(blocks: List[Dict[str, Any]], analysis: Dict[str, Any], output_dir: str):
    """Save blocks and analysis to files"""
    
    print(f"\nSaving blocks to {output_dir}/...")
    
    # Prepare blocks for JSON serialization (remove non-serializable objects)
    serializable_blocks = []
    for block in blocks:
        serializable_block = {
            'name': block['name'],
            'category': block['category'],
            'description': block['description'],
            'parameters': block['parameters'],
            'complexity': block['complexity'],
            'components': block['components'],
            'template_name': block['template'].name_pattern if hasattr(block['template'], 'name_pattern') else 'unknown'
        }
        serializable_blocks.append(serializable_block)
    
    # Save all blocks as JSON
    blocks_file = os.path.join(output_dir, "generated_blocks.json")
    with open(blocks_file, 'w') as f:
        json.dump(serializable_blocks, f, indent=2)
    print(f"Saved {len(serializable_blocks)} blocks to {blocks_file}")
    
    # Save analysis
    analysis_file = os.path.join(output_dir, "generation_analysis.json")
    with open(analysis_file, 'w') as f:
        json.dump(analysis, f, indent=2)
    print(f"Saved analysis to {analysis_file}")
    
    # Save blocks by category
    blocks_by_category = {}
    for block in serializable_blocks:
        cat = block['category']
        if cat not in blocks_by_category:
            blocks_by_category[cat] = []
        blocks_by_category[cat].append(block)
    
    for category, cat_blocks in blocks_by_category.items():
        cat_file = os.path.join(output_dir, f"blocks_{category}.json")
        with open(cat_file, 'w') as f:
            json.dump(cat_blocks, f, indent=2)
        print(f"Saved {len(cat_blocks)} {category} blocks to {cat_file}")
    
    # Create a summary report
    create_summary_report(serializable_blocks, analysis, output_dir)


def create_summary_report(blocks: List[Dict[str, Any]], analysis: Dict[str, Any], output_dir: str):
    """Create a human-readable summary report"""
    
    report_file = os.path.join(output_dir, "generation_report.md")
    
    with open(report_file, 'w') as f:
        f.write("# Domain Block Mass Production Report\n\n")
        f.write(f"Generated at: {datetime.now()}\n\n")
        f.write(f"## Summary\n\n")
        f.write(f"- Total blocks generated: {len(blocks)}\n")
        f.write(f"- Categories covered: {analysis['diversity_metrics']['category_diversity']}\n")
        f.write(f"- Complexity levels: {analysis['diversity_metrics']['complexity_diversity']}\n")
        f.write(f"- Unique components: {analysis['diversity_metrics']['component_diversity']}\n\n")
        
        f.write("## Category Distribution\n\n")
        for category, count in sorted(analysis['category_distribution'].items()):
            percentage = (count / len(blocks)) * 100
            f.write(f"- {category}: {count} blocks ({percentage:.1f}%)\n")
        
        f.write("\n## Complexity Distribution\n\n")
        for complexity, count in sorted(analysis['complexity_distribution'].items()):
            percentage = (count / len(blocks)) * 100
            f.write(f"- {complexity}: {count} blocks ({percentage:.1f}%)\n")
        
        f.write("\n## Sample Blocks\n\n")
        for i, block in enumerate(blocks[:10]):
            f.write(f"### {i+1}. {block['name']}\n")
            f.write(f"- **Category**: {block['category']}\n")
            f.write(f"- **Complexity**: {block['complexity']}\n")
            f.write(f"- **Description**: {block['description']}\n")
            f.write(f"- **Components**: {', '.join(block['components'])}\n")
            f.write(f"- **Parameters**: {', '.join(block['parameters'].keys())}\n\n")
        
        if len(blocks) > 10:
            f.write(f"... and {len(blocks) - 10} more blocks\n\n")
        
        f.write("## Next Steps\n\n")
        f.write("1. Review generated blocks for quality\n")
        f.write("2. Integrate blocks into extended registry\n") 
        f.write("3. Test block functionality\n")
        f.write("4. Run diversity validation\n")
    
    print(f"Created summary report at {report_file}")


def main():
    """Main execution function"""
    
    # Configuration
    TARGET_BLOCKS = 150
    OUTPUT_DIR = "generated_blocks"
    
    try:
        result = create_production_run(TARGET_BLOCKS, OUTPUT_DIR)
        
        print("\n" + "=" * 80)
        print("PRODUCTION COMPLETE")
        print("=" * 80)
        print(f"Generated: {result['total_blocks']} blocks")
        print(f"Output: {result['output_dir']}/")
        print(f"Time: {result['generation_time']}")
        print("\nFiles created:")
        print(f"- generated_blocks.json (all blocks)")
        print(f"- generation_analysis.json (analysis)")
        print(f"- generation_report.md (summary)")
        print(f"- blocks_<category>.json (by category)")
        
        return True
        
    except Exception as e:
        print(f"\n✗ Production failed: {e}")
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)