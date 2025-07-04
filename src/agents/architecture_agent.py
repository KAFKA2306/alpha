from typing import Dict, Any, List, Optional, Tuple, Union
import random
import json
import hashlib
from dataclasses import dataclass
from abc import ABC, abstractmethod

import torch
import torch.nn as nn
from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI, ChatAnthropic
from langchain.schema import HumanMessage, SystemMessage
from langchain.prompts import PromptTemplate

from ..models.domain_blocks import DomainBlockRegistry, DomainBlock, get_domain_block_registry
from ..core.config import get_config


@dataclass
class ArchitectureSpec:
    """Specification for a neural network architecture."""
    
    id: str
    name: str
    blocks: List[Dict[str, Any]]
    connections: List[Tuple[int, int]]
    input_shape: Tuple[int, ...]
    output_shape: Tuple[int, ...]
    hyperparameters: Dict[str, Any]
    complexity_score: float
    diversity_score: float
    metadata: Dict[str, Any]


class ArchitectureGenerator(ABC):
    """Abstract base class for architecture generators."""
    
    @abstractmethod
    def generate_architecture(self, 
                            input_shape: Tuple[int, ...],
                            constraints: Optional[Dict[str, Any]] = None) -> ArchitectureSpec:
        """Generate a single architecture specification."""
        pass
    
    @abstractmethod
    def generate_architectures(self, 
                             input_shape: Tuple[int, ...],
                             num_architectures: int,
                             constraints: Optional[Dict[str, Any]] = None) -> List[ArchitectureSpec]:
        """Generate multiple architecture specifications."""
        pass


class LLMArchitectureGenerator(ArchitectureGenerator):
    """Architecture generator using Large Language Models."""
    
    def __init__(self, 
                 llm_provider: str = "openai",
                 model_name: str = "gpt-4",
                 temperature: float = 0.7,
                 max_tokens: int = 4000):
        
        self.config = get_config()
        self.registry = get_domain_block_registry()
        
        # Initialize LLM
        if llm_provider == "openai":
            self.llm = ChatOpenAI(
                model=model_name,
                temperature=temperature,
                max_tokens=max_tokens,
                openai_api_key=self.config.agent.llm.openai_api_key
            )
        elif llm_provider == "anthropic":
            self.llm = ChatAnthropic(
                model=model_name,
                temperature=temperature,
                max_tokens=max_tokens,
                anthropic_api_key=self.config.agent.llm.anthropic_api_key
            )
        else:
            raise ValueError(f"Unsupported LLM provider: {llm_provider}")
        
        self.generated_architectures = []
    
    def _create_block_description_prompt(self) -> str:
        """Create a prompt describing available domain blocks."""
        
        block_descriptions = []
        for category in self.registry.get_categories():
            blocks = self.registry.get_blocks_by_category(category)
            category_desc = f"\n## {category.upper()} BLOCKS\n"
            
            for block in blocks:
                hyperparams = block.get_hyperparameters()
                hyperparam_str = ""
                if hyperparams:
                    hyperparam_str = f" (hyperparameters: {hyperparams})"
                
                category_desc += f"- {block.name}: {block.description}{hyperparam_str}\n"
            
            block_descriptions.append(category_desc)
        
        return "\n".join(block_descriptions)
    
    def _create_architecture_prompt(self, 
                                   input_shape: Tuple[int, ...],
                                   constraints: Optional[Dict[str, Any]] = None) -> str:
        """Create a prompt for architecture generation."""
        
        constraints = constraints or {}
        max_blocks = constraints.get('max_blocks', 8)
        min_blocks = constraints.get('min_blocks', 3)
        required_categories = constraints.get('required_categories', [])
        
        block_descriptions = self._create_block_description_prompt()
        
        prompt = f"""You are an expert AI architect specializing in neural network design for financial time series prediction. 
        Your task is to generate a creative and effective neural network architecture by combining the provided domain blocks.
        
        ## INPUT SPECIFICATION
        - Input shape: {input_shape}
        - This represents (batch_size, sequence_length, features) for Japanese stock data
        - Sequence length is 252 (trading days in a year)
        - Features include price returns and technical indicators
        
        ## AVAILABLE DOMAIN BLOCKS
        {block_descriptions}
        
        ## CONSTRAINTS
        - Use between {min_blocks} and {max_blocks} blocks
        - Ensure blocks are compatible (output shape of one block matches input shape of next)
        - Must include at least one prediction head block
        - Consider the financial domain - stock prediction requires temporal and cross-sectional features
        - Be creative but practical - avoid overly complex combinations
        
        ## REQUIRED CATEGORIES
        {required_categories if required_categories else 'No specific requirements'}
        
        ## ARCHITECTURE DESIGN PRINCIPLES
        1. **Temporal Patterns**: Include blocks that capture time-series patterns
        2. **Cross-sectional Features**: Include blocks that capture relationships between stocks
        3. **Financial Domain**: Leverage financial-specific blocks for better performance
        4. **Regularization**: Include normalization and dropout for stability
        5. **Diversity**: Create architectures that are different from common patterns
        
        ## OUTPUT FORMAT
        Return a JSON object with the following structure:
        {{
            "name": "descriptive_architecture_name",
            "description": "brief description of the architecture's approach",
            "blocks": [
                {{
                    "name": "block_name",
                    "hyperparameters": {{
                        "param1": value1,
                        "param2": value2
                    }}
                }},
                ...
            ],
            "rationale": "explanation of why this architecture should work well for stock prediction"
        }}
        
        Generate a single, well-thought-out architecture that balances innovation with practicality.
        """
        
        return prompt
    
    def _validate_architecture_spec(self, 
                                   spec_dict: Dict[str, Any], 
                                   input_shape: Tuple[int, ...]) -> bool:
        """Validate that the architecture specification is valid."""
        
        try:
            # Check required fields
            required_fields = ['name', 'blocks', 'description']
            for field in required_fields:
                if field not in spec_dict:
                    return False
            
            # Check that blocks exist in registry
            for block_spec in spec_dict['blocks']:
                if 'name' not in block_spec:
                    return False
                
                block_name = block_spec['name']
                if block_name not in self.registry.get_block_names():
                    return False
            
            # Check that at least one prediction head exists
            prediction_heads = [b for b in spec_dict['blocks'] 
                              if self.registry.get_block(b['name']).category == 'prediction_heads']
            if not prediction_heads:
                return False
            
            return True
            
        except Exception:
            return False
    
    def _calculate_complexity_score(self, blocks: List[Dict[str, Any]]) -> float:
        """Calculate complexity score for an architecture."""
        
        complexity = 0
        for block_spec in blocks:
            block = self.registry.get_block(block_spec['name'])
            
            # Base complexity by category
            category_complexity = {
                'normalization': 1,
                'feature_extraction': 2,
                'mixing': 2,
                'encoding': 3,
                'financial_domain': 3,
                'feature_integration': 2,
                'time_integration': 2,
                'stock_features': 3,
                'attention': 4,
                'feedforward': 2,
                'time_embedding': 1,
                'sequence_models': 4,
                'prediction_heads': 2
            }
            
            complexity += category_complexity.get(block.category, 2)
            
            # Add complexity from hyperparameters
            hyperparams = block_spec.get('hyperparameters', {})
            if 'hidden_size' in hyperparams:
                complexity += hyperparams['hidden_size'] / 100
            if 'num_layers' in hyperparams:
                complexity += hyperparams['num_layers'] * 0.5
        
        return complexity
    
    def _calculate_diversity_score(self, blocks: List[Dict[str, Any]]) -> float:
        """Calculate diversity score compared to existing architectures."""
        
        if not self.generated_architectures:
            return 1.0
        
        # Create signature for this architecture
        block_names = [b['name'] for b in blocks]
        current_signature = set(block_names)
        
        # Calculate Jaccard similarity with existing architectures
        similarities = []
        for existing_arch in self.generated_architectures:
            existing_blocks = [b['name'] for b in existing_arch.blocks]
            existing_signature = set(existing_blocks)
            
            intersection = len(current_signature.intersection(existing_signature))
            union = len(current_signature.union(existing_signature))
            
            similarity = intersection / union if union > 0 else 0
            similarities.append(similarity)
        
        # Return 1 - maximum similarity (higher is more diverse)
        return 1.0 - max(similarities) if similarities else 1.0
    
    def _create_architecture_id(self, blocks: List[Dict[str, Any]]) -> str:
        """Create a unique ID for an architecture."""
        
        # Create deterministic hash based on blocks and their hyperparameters
        content = json.dumps(blocks, sort_keys=True)
        return hashlib.md5(content.encode()).hexdigest()[:12]
    
    def generate_architecture(self, 
                            input_shape: Tuple[int, ...],
                            constraints: Optional[Dict[str, Any]] = None) -> ArchitectureSpec:
        """Generate a single architecture specification using LLM."""
        
        prompt = self._create_architecture_prompt(input_shape, constraints)
        
        # Generate architecture using LLM
        messages = [
            SystemMessage(content="You are an expert neural network architect specializing in financial time series prediction."),
            HumanMessage(content=prompt)
        ]
        
        response = self.llm(messages)
        
        try:
            # Parse LLM response
            response_text = response.content.strip()
            if response_text.startswith('```json'):
                response_text = response_text[7:-3]
            elif response_text.startswith('```'):
                response_text = response_text[3:-3]
            
            spec_dict = json.loads(response_text)
            
            # Validate the specification
            if not self._validate_architecture_spec(spec_dict, input_shape):
                raise ValueError("Invalid architecture specification")
            
            # Calculate scores
            complexity_score = self._calculate_complexity_score(spec_dict['blocks'])
            diversity_score = self._calculate_diversity_score(spec_dict['blocks'])
            
            # Create architecture specification
            arch_spec = ArchitectureSpec(
                id=self._create_architecture_id(spec_dict['blocks']),
                name=spec_dict['name'],
                blocks=spec_dict['blocks'],
                connections=[(i, i+1) for i in range(len(spec_dict['blocks']) - 1)],
                input_shape=input_shape,
                output_shape=(input_shape[0], 1),  # Default to single output
                hyperparameters={},
                complexity_score=complexity_score,
                diversity_score=diversity_score,
                metadata={
                    'description': spec_dict.get('description', ''),
                    'rationale': spec_dict.get('rationale', ''),
                    'generated_by': 'LLM',
                    'llm_provider': self.llm.__class__.__name__
                }
            )
            
            self.generated_architectures.append(arch_spec)
            return arch_spec
            
        except Exception as e:
            # Fallback to random generation if LLM fails
            print(f"LLM generation failed: {e}. Falling back to random generation.")
            return self._generate_random_architecture(input_shape, constraints)
    
    def _generate_random_architecture(self, 
                                    input_shape: Tuple[int, ...],
                                    constraints: Optional[Dict[str, Any]] = None) -> ArchitectureSpec:
        """Generate a random architecture as fallback."""
        
        constraints = constraints or {}
        max_blocks = constraints.get('max_blocks', 8)
        min_blocks = constraints.get('min_blocks', 3)
        
        # Select random blocks
        available_blocks = self.registry.get_all_blocks()
        prediction_heads = [b for b in available_blocks if b.category == 'prediction_heads']
        other_blocks = [b for b in available_blocks if b.category != 'prediction_heads']
        
        num_other_blocks = random.randint(min_blocks - 1, max_blocks - 1)
        selected_blocks = random.sample(other_blocks, num_other_blocks)
        selected_blocks.append(random.choice(prediction_heads))
        
        # Create block specifications
        block_specs = []
        for block in selected_blocks:
            hyperparams = block.get_hyperparameters()
            selected_hyperparams = {}
            
            for param, values in hyperparams.items():
                if isinstance(values, list):
                    selected_hyperparams[param] = random.choice(values)
                else:
                    selected_hyperparams[param] = values
            
            block_specs.append({
                'name': block.name,
                'hyperparameters': selected_hyperparams
            })
        
        # Calculate scores
        complexity_score = self._calculate_complexity_score(block_specs)
        diversity_score = self._calculate_diversity_score(block_specs)
        
        # Create architecture specification
        arch_spec = ArchitectureSpec(
            id=self._create_architecture_id(block_specs),
            name=f"random_arch_{len(self.generated_architectures)}",
            blocks=block_specs,
            connections=[(i, i+1) for i in range(len(block_specs) - 1)],
            input_shape=input_shape,
            output_shape=(input_shape[0], 1),
            hyperparameters={},
            complexity_score=complexity_score,
            diversity_score=diversity_score,
            metadata={
                'description': 'Randomly generated architecture',
                'generated_by': 'Random',
                'num_blocks': len(block_specs)
            }
        )
        
        self.generated_architectures.append(arch_spec)
        return arch_spec
    
    def generate_architectures(self, 
                             input_shape: Tuple[int, ...],
                             num_architectures: int,
                             constraints: Optional[Dict[str, Any]] = None) -> List[ArchitectureSpec]:
        """Generate multiple architecture specifications."""
        
        architectures = []
        constraints = constraints or {}
        
        # Generate architectures with diversity constraints
        for i in range(num_architectures):
            # Add diversity pressure by updating constraints
            current_constraints = constraints.copy()
            current_constraints['diversity_iteration'] = i
            
            try:
                arch = self.generate_architecture(input_shape, current_constraints)
                architectures.append(arch)
                
                # Add some randomness to avoid getting stuck
                if i % 10 == 0:
                    current_constraints['random_seed'] = random.randint(0, 1000)
                    
            except Exception as e:
                print(f"Failed to generate architecture {i}: {e}")
                continue
        
        return architectures


class ArchitectureCompiler:
    """Compile architecture specifications into PyTorch models."""
    
    def __init__(self):
        self.registry = get_domain_block_registry()
    
    def compile_architecture(self, spec: ArchitectureSpec) -> nn.Module:
        """Compile an architecture specification into a PyTorch model."""
        
        class GeneratedModel(nn.Module):
            def __init__(self, spec: ArchitectureSpec, registry: DomainBlockRegistry):
                super().__init__()
                self.spec = spec
                self.registry = registry
                
                # Create modules for each block
                self.blocks = nn.ModuleList()
                current_shape = spec.input_shape
                
                for i, block_spec in enumerate(spec.blocks):
                    block = registry.get_block(block_spec['name'])
                    hyperparams = block_spec.get('hyperparameters', {})
                    
                    # Create module
                    module = block.create_module(current_shape, **hyperparams)
                    self.blocks.append(module)
                    
                    # Update shape for next block
                    current_shape = block.get_output_shape(current_shape, **hyperparams)
            
            def forward(self, x):
                # Forward pass through all blocks
                for block in self.blocks:
                    x = block(x)
                return x
        
        return GeneratedModel(spec, self.registry)
    
    def validate_architecture(self, spec: ArchitectureSpec) -> bool:
        """Validate that an architecture can be compiled successfully."""
        
        try:
            model = self.compile_architecture(spec)
            
            # Test with dummy input
            dummy_input = torch.randn(spec.input_shape)
            output = model(dummy_input)
            
            return True
            
        except Exception as e:
            print(f"Architecture validation failed: {e}")
            return False


class ArchitectureAgent:
    """Main agent for generating and managing neural network architectures."""
    
    def __init__(self, 
                 generator: Optional[ArchitectureGenerator] = None,
                 compiler: Optional[ArchitectureCompiler] = None):
        
        self.config = get_config()
        self.generator = generator or LLMArchitectureGenerator(
            llm_provider=self.config.agent.llm.provider,
            model_name=self.config.agent.llm.model,
            temperature=self.config.agent.llm.temperature,
            max_tokens=self.config.agent.llm.max_tokens
        )
        self.compiler = compiler or ArchitectureCompiler()
        
        self.generated_architectures = []
        self.compiled_models = {}
    
    def generate_architecture_suite(self, 
                                   input_shape: Tuple[int, ...],
                                   num_architectures: Optional[int] = None) -> List[ArchitectureSpec]:
        """Generate a suite of diverse architectures for evaluation."""
        
        num_architectures = num_architectures or self.config.agent.max_architectures
        
        # Generate architectures
        architectures = self.generator.generate_architectures(
            input_shape=input_shape,
            num_architectures=num_architectures,
            constraints={
                'max_blocks': 10,
                'min_blocks': 3,
                'diversity_threshold': self.config.agent.diversity_threshold
            }
        )
        
        # Filter and validate architectures
        valid_architectures = []
        for arch in architectures:
            if self.compiler.validate_architecture(arch):
                valid_architectures.append(arch)
            else:
                print(f"Invalid architecture: {arch.name}")
        
        self.generated_architectures.extend(valid_architectures)
        
        # Sort by diversity score (higher is better)
        valid_architectures.sort(key=lambda x: x.diversity_score, reverse=True)
        
        return valid_architectures
    
    def compile_architecture_suite(self, 
                                 architectures: List[ArchitectureSpec]) -> Dict[str, nn.Module]:
        """Compile a suite of architectures into PyTorch models."""
        
        compiled_models = {}
        
        for arch in architectures:
            try:
                model = self.compiler.compile_architecture(arch)
                compiled_models[arch.id] = model
                self.compiled_models[arch.id] = model
                
            except Exception as e:
                print(f"Failed to compile architecture {arch.name}: {e}")
                continue
        
        return compiled_models
    
    def get_architecture_summary(self, arch_id: str) -> Dict[str, Any]:
        """Get a summary of an architecture."""
        
        arch = next((a for a in self.generated_architectures if a.id == arch_id), None)
        if not arch:
            raise ValueError(f"Architecture {arch_id} not found")
        
        return {
            'id': arch.id,
            'name': arch.name,
            'num_blocks': len(arch.blocks),
            'complexity_score': arch.complexity_score,
            'diversity_score': arch.diversity_score,
            'blocks': [b['name'] for b in arch.blocks],
            'categories': list(set(self.registry.get_block(b['name']).category for b in arch.blocks)),
            'metadata': arch.metadata
        }
    
    def export_architecture_specs(self, filename: str):
        """Export all generated architectures to a JSON file."""
        
        export_data = []
        for arch in self.generated_architectures:
            export_data.append({
                'id': arch.id,
                'name': arch.name,
                'blocks': arch.blocks,
                'connections': arch.connections,
                'input_shape': arch.input_shape,
                'output_shape': arch.output_shape,
                'hyperparameters': arch.hyperparameters,
                'complexity_score': arch.complexity_score,
                'diversity_score': arch.diversity_score,
                'metadata': arch.metadata
            })
        
        with open(filename, 'w') as f:
            json.dump(export_data, f, indent=2)
    
    def load_architecture_specs(self, filename: str):
        """Load architecture specifications from a JSON file."""
        
        with open(filename, 'r') as f:
            data = json.load(f)
        
        loaded_architectures = []
        for item in data:
            arch = ArchitectureSpec(
                id=item['id'],
                name=item['name'],
                blocks=item['blocks'],
                connections=item['connections'],
                input_shape=tuple(item['input_shape']),
                output_shape=tuple(item['output_shape']),
                hyperparameters=item['hyperparameters'],
                complexity_score=item['complexity_score'],
                diversity_score=item['diversity_score'],
                metadata=item['metadata']
            )
            loaded_architectures.append(arch)
        
        self.generated_architectures.extend(loaded_architectures)
        return loaded_architectures