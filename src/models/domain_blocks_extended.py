# Extended Domain Blocks Registry with all implemented blocks
from typing import Dict, Any, List, Optional, Union, Tuple
import torch
import torch.nn as nn
from .domain_blocks import DomainBlock, DomainBlockRegistry

# Import all block modules
try:
    from .blocks.normalization_blocks import (
        BatchNormBlock as ExtBatchNormBlock, LayerNormBlock as ExtLayerNormBlock, 
        AdaptiveInstanceNormBlock as ExtAdaptiveInstanceNormBlock, DemeanBlock as ExtDemeanBlock,
        GroupNormBlock, RMSNormBlock, PowerNormBlock, QuantileNormBlock
    )
except ImportError:
    # Fallback to basic implementations if detailed blocks not available
    GroupNormBlock = None
    RMSNormBlock = None
    PowerNormBlock = None
    QuantileNormBlock = None

try:
    from .blocks.feature_extraction_blocks import (
        PCABlock as ExtPCABlock, FourierFeatureBlock as ExtFourierFeatureBlock,
        WaveletFeatureBlock, StatisticalMomentsBlock,
        AutoEncoderFeatureBlock, PolynomialFeatureBlock
    )
except ImportError:
    WaveletFeatureBlock = None
    StatisticalMomentsBlock = None
    AutoEncoderFeatureBlock = None
    PolynomialFeatureBlock = None

try:
    from .blocks.mixing_blocks import (
        TimeMixingBlock as ExtTimeMixingBlock, ChannelMixingBlock as ExtChannelMixingBlock,
        CrossAttentionMixingBlock, GatedMixingBlock, FourierMixingBlock
    )
except ImportError:
    CrossAttentionMixingBlock = None
    GatedMixingBlock = None
    FourierMixingBlock = None

try:
    from .blocks.encoding_blocks import (
        LinearEncodingBlock, MLPEncodingBlock, ConvolutionalEncodingBlock,
        TransformerEncodingBlock, ResidualEncodingBlock, VariationalEncodingBlock
    )
except ImportError:
    LinearEncodingBlock = None
    MLPEncodingBlock = None
    ConvolutionalEncodingBlock = None
    TransformerEncodingBlock = None
    ResidualEncodingBlock = None
    VariationalEncodingBlock = None

try:
    from .blocks.financial_domain_blocks import (
        MultiTimeFrameBlock as ExtMultiTimeFrameBlock, LeadLagBlock as ExtLeadLagBlock,
        RegimeDetectionBlock as ExtRegimeDetectionBlock, FactorExposureBlock,
        VolatilityClusteringBlock, CrossSectionalBlock
    )
except ImportError:
    FactorExposureBlock = None
    VolatilityClusteringBlock = None
    CrossSectionalBlock = None

try:
    from .blocks.prediction_head_blocks import (
        RegressionHeadBlock as ExtRegressionHeadBlock, ClassificationHeadBlock as ExtClassificationHeadBlock,
        RankingHeadBlock, MultiTaskHeadBlock, DistributionHeadBlock
    )
except ImportError:
    RankingHeadBlock = None
    MultiTaskHeadBlock = None
    DistributionHeadBlock = None


class ExtendedDomainBlockRegistry(DomainBlockRegistry):
    """Extended registry with all implemented domain blocks."""
    
    def _register_default_blocks(self):
        """Register all available domain blocks."""
        blocks = []
        
        # Import original blocks from parent class
        from .domain_blocks import (
            BatchNormBlock, LayerNormBlock, AdaptiveInstanceNormBlock, DemeanBlock,
            PCABlock, FourierFeatureBlock, TimeMixingBlock, ChannelMixingBlock,
            MultiTimeFrameBlock, LeadLagBlock, RegimeDetectionBlock,
            LSTMBlock, TransformerBlock, RegressionHeadBlock, ClassificationHeadBlock
        )
        
        # Basic blocks (always available)
        blocks.extend([
            BatchNormBlock(),
            LayerNormBlock(),
            AdaptiveInstanceNormBlock(),
            DemeanBlock(),
            PCABlock(),
            FourierFeatureBlock(),
            TimeMixingBlock(),
            ChannelMixingBlock(),
            MultiTimeFrameBlock(),
            LeadLagBlock(),
            RegimeDetectionBlock(),
            LSTMBlock(),
            TransformerBlock(),
            RegressionHeadBlock(),
            ClassificationHeadBlock(),
        ])
        
        # Extended normalization blocks
        if GroupNormBlock is not None:
            blocks.append(GroupNormBlock())
        if RMSNormBlock is not None:
            blocks.append(RMSNormBlock())
        if PowerNormBlock is not None:
            blocks.append(PowerNormBlock())
        if QuantileNormBlock is not None:
            blocks.append(QuantileNormBlock())
        
        # Extended feature extraction blocks
        if WaveletFeatureBlock is not None:
            blocks.append(WaveletFeatureBlock())
        if StatisticalMomentsBlock is not None:
            blocks.append(StatisticalMomentsBlock())
        if AutoEncoderFeatureBlock is not None:
            blocks.append(AutoEncoderFeatureBlock())
        if PolynomialFeatureBlock is not None:
            blocks.append(PolynomialFeatureBlock())
        
        # Extended mixing blocks
        if CrossAttentionMixingBlock is not None:
            blocks.append(CrossAttentionMixingBlock())
        if GatedMixingBlock is not None:
            blocks.append(GatedMixingBlock())
        if FourierMixingBlock is not None:
            blocks.append(FourierMixingBlock())
        
        # Encoding blocks
        if LinearEncodingBlock is not None:
            blocks.append(LinearEncodingBlock())
        if MLPEncodingBlock is not None:
            blocks.append(MLPEncodingBlock())
        if ConvolutionalEncodingBlock is not None:
            blocks.append(ConvolutionalEncodingBlock())
        if TransformerEncodingBlock is not None:
            blocks.append(TransformerEncodingBlock())
        if ResidualEncodingBlock is not None:
            blocks.append(ResidualEncodingBlock())
        if VariationalEncodingBlock is not None:
            blocks.append(VariationalEncodingBlock())
        
        # Extended financial domain blocks
        if FactorExposureBlock is not None:
            blocks.append(FactorExposureBlock())
        if VolatilityClusteringBlock is not None:
            blocks.append(VolatilityClusteringBlock())
        if CrossSectionalBlock is not None:
            blocks.append(CrossSectionalBlock())
        
        # Extended prediction head blocks
        if RankingHeadBlock is not None:
            blocks.append(RankingHeadBlock())
        if MultiTaskHeadBlock is not None:
            blocks.append(MultiTaskHeadBlock())
        if DistributionHeadBlock is not None:
            blocks.append(DistributionHeadBlock())
        
        # Register all blocks
        for block in blocks:
            if block is not None:
                self.register_block(block)
    
    def get_block_count_by_category(self) -> Dict[str, int]:
        """Get count of blocks by category."""
        category_counts = {}
        for block in self._blocks.values():
            category = block.category
            category_counts[category] = category_counts.get(category, 0) + 1
        return category_counts
    
    def get_total_blocks(self) -> int:
        """Get total number of registered blocks."""
        return len(self._blocks)
    
    def print_registry_summary(self):
        """Print a summary of all registered blocks."""
        print("=" * 60)
        print("EXTENDED DOMAIN BLOCK REGISTRY SUMMARY")
        print("=" * 60)
        
        total_blocks = self.get_total_blocks()
        print(f"Total blocks: {total_blocks}")
        
        category_counts = self.get_block_count_by_category()
        print(f"\nBlocks by category:")
        for category, count in sorted(category_counts.items()):
            print(f"  {category}: {count} blocks")
        
        print(f"\nAll available blocks:")
        for category in sorted(self.get_categories()):
            blocks = self.get_blocks_by_category(category)
            print(f"\n{category.upper()}:")
            for block in blocks:
                hyperparams = block.get_hyperparameters()
                param_info = f" ({len(hyperparams)} hyperparams)" if hyperparams else ""
                print(f"  - {block.name}: {block.description}{param_info}")


# Create global extended registry instance
extended_registry = ExtendedDomainBlockRegistry()


def get_extended_domain_block_registry() -> ExtendedDomainBlockRegistry:
    """Get the global extended domain block registry."""
    return extended_registry


if __name__ == "__main__":
    # Print registry summary when run as script
    registry = get_extended_domain_block_registry()
    registry.print_registry_summary()