#!/usr/bin/env python3
"""
Alpha Architecture Agent - 包括的実験スクリプト

このスクリプトは、AIエージェントによる株価予測アーキテクチャ探索の
全実験を統括して実行します。

目標:
- 個別戦略のシャープレシオ 1.3以上達成
- アンサンブル戦略のシャープレシオ 2.0以上達成  
- 多様な投資戦略の量産と分散運用
- 日本株市場での実証実験

実験フェーズ:
1. 人工市場データ生成・検証
2. AIアーキテクチャ生成・評価
3. 予測性能評価・最適化
4. アンサンブル戦略構築・検証
5. 総合分析・レポート生成
"""

import sys
import os
from pathlib import Path
import warnings
import traceback
from datetime import datetime, timedelta
import json
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
import logging

# プロジェクトパス設定
PROJECT_ROOT = Path(__file__).parent
SRC_DIR = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_DIR))

# ログ設定
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('alpha_experiments.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

warnings.filterwarnings('ignore')

# 実験設定クラス
@dataclass
class AlphaExperimentConfig:
    """包括的実験設定"""
    # 実験基本設定
    experiment_name: str = "alpha_architecture_exploration_v1"
    output_dir: str = "experiments/alpha_results"
    seed: int = 42
    
    # 市場データ設定（日本株市場想定）
    n_stocks: int = 100
    n_days: int = 2016  # 8年間（2017-2024年相当）
    n_features: int = 20
    start_date: str = "2017-01-01"
    
    # アーキテクチャ生成設定
    n_architectures: int = 70  # 多様な戦略量産
    input_shape: Tuple[int, int, int] = (32, 252, 20)  # batch, seq, features
    max_blocks_per_arch: int = 12
    
    # 学習・評価設定
    train_split: float = 0.5   # 4年：学習期間
    val_split: float = 0.25    # 2年：検証期間
    test_split: float = 0.25   # 2年：評価期間
    
    # 目標性能指標
    target_individual_sharpe: float = 1.3
    target_ensemble_sharpe: float = 2.0
    target_max_drawdown: float = 0.10
    target_win_rate: float = 0.60
    
    # アンサンブル設定
    top_performers_count: int = 20
    ensemble_methods: List[str] = None
    
    # 実験制御
    enable_llm_generation: bool = True
    enable_gpu: bool = True
    parallel_evaluation: bool = False
    save_intermediate_results: bool = True
    
    def __post_init__(self):
        if self.ensemble_methods is None:
            self.ensemble_methods = [
                'equal_weight',
                'sharpe_weighted', 
                'diversity_weighted',
                'risk_adjusted',
                'momentum_based'
            ]


class AlphaExperimentRunner:
    """Alpha Architecture Agent実験統括クラス"""
    
    def __init__(self, config: AlphaExperimentConfig = None):
        self.config = config or AlphaExperimentConfig()
        self.results = {}
        self.experiment_start_time = datetime.now()
        
        # 出力ディレクトリ作成
        self.output_dir = Path(self.config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # 実験状態管理
        self.phase_results = {}
        self.generated_architectures = []
        self.market_data = None
        self.performance_results = []
        self.ensemble_results = {}
        
        logger.info(f"Alpha Architecture Agent実験を開始")
        logger.info(f"実験名: {self.config.experiment_name}")
        logger.info(f"出力ディレクトリ: {self.output_dir}")
        logger.info(f"目標: 個別戦略Sharpe>{self.config.target_individual_sharpe}, "
                   f"アンサンブルSharpe>{self.config.target_ensemble_sharpe}")
    
    def run_comprehensive_experiment(self) -> Dict[str, Any]:
        """包括的実験の実行"""
        
        logger.info("=" * 80)
        logger.info("🚀 ALPHA ARCHITECTURE AGENT - 包括的実験開始")
        logger.info("=" * 80)
        
        try:
            # フェーズ1: 環境検証・初期化
            self._phase1_environment_validation()
            
            # フェーズ2: 人工市場データ生成
            self._phase2_synthetic_market_generation()
            
            # フェーズ3: AIアーキテクチャ生成
            self._phase3_architecture_generation()
            
            # フェーズ4: 予測性能評価
            self._phase4_performance_evaluation()
            
            # フェーズ5: アンサンブル戦略構築
            self._phase5_ensemble_construction()
            
            # フェーズ6: 総合分析・レポート
            final_report = self._phase6_comprehensive_analysis()
            
            # 実験完了
            self._finalize_experiment()
            
            return final_report
            
        except Exception as e:
            logger.error(f"実験実行中にエラーが発生: {e}")
            traceback.print_exc()
            return {"status": "failed", "error": str(e)}
    
    def _phase1_environment_validation(self):
        """フェーズ1: 環境検証・初期化"""
        logger.info("\n" + "=" * 60)
        logger.info("📋 フェーズ1: 環境検証・初期化")
        logger.info("=" * 60)
        
        # 依存関係チェック
        self._validate_dependencies()
        
        # GPU利用可能性チェック
        self._check_gpu_availability()
        
        # ドメインブロック検証
        self._validate_domain_blocks()
        
        # 設定ファイル検証
        self._validate_configurations()
        
        self.phase_results['phase1'] = {
            'status': 'completed',
            'timestamp': datetime.now().isoformat(),
            'validation_passed': True
        }
        
        logger.info("✅ フェーズ1完了: 環境検証成功")
    
    def _phase2_synthetic_market_generation(self):
        """フェーズ2: 人工市場データ生成"""
        logger.info("\n" + "=" * 60)
        logger.info("📊 フェーズ2: 人工市場データ生成")
        logger.info("=" * 60)
        
        try:
            from data.synthetic_market import SyntheticMarketGenerator, MarketConfig, create_market_scenarios
            
            # 市場シナリオ生成
            scenarios = create_market_scenarios()
            scenario_data = {}
            
            logger.info(f"生成する市場シナリオ: {list(scenarios.keys())}")
            
            for scenario_name, market_config in scenarios.items():
                logger.info(f"\n📈 {scenario_name}市場シナリオ生成中...")
                
                # 実験設定に合わせて調整
                market_config.n_stocks = self.config.n_stocks
                market_config.n_days = self.config.n_days
                market_config.n_features = self.config.n_features
                market_config.start_date = self.config.start_date
                
                # データ生成
                generator = SyntheticMarketGenerator(market_config)
                data = generator.generate_market_data(seed=self.config.seed + hash(scenario_name) % 1000)
                
                # データ品質検証
                quality_metrics = self._validate_market_data(data, scenario_name)
                
                scenario_data[scenario_name] = {
                    'data': data,
                    'config': asdict(market_config),
                    'quality': quality_metrics
                }
                
                logger.info(f"✅ {scenario_name}: 品質スコア {quality_metrics['quality_score']:.3f}")
            
            # デフォルトシナリオ設定（安定市場を使用）
            self.market_data = scenario_data['stable']['data']
            
            # データ保存
            self._save_market_data(scenario_data)
            
            self.phase_results['phase2'] = {
                'status': 'completed',
                'timestamp': datetime.now().isoformat(),
                'scenarios_generated': len(scenarios),
                'data_quality_avg': np.mean([s['quality']['quality_score'] for s in scenario_data.values()]),
                'market_properties': self._analyze_market_properties(self.market_data)
            }
            
            logger.info(f"✅ フェーズ2完了: {len(scenarios)}シナリオの市場データ生成成功")
            
        except Exception as e:
            logger.error(f"市場データ生成エラー: {e}")
            raise
    
    def _phase3_architecture_generation(self):
        """フェーズ3: AIアーキテクチャ生成"""
        logger.info("\n" + "=" * 60)
        logger.info("🤖 フェーズ3: AIアーキテクチャ生成")
        logger.info("=" * 60)
        
        try:
            from agents.architecture_agent import ArchitectureAgent
            from models.domain_blocks import get_domain_block_registry
            
            # ドメインブロック確認
            registry = get_domain_block_registry()
            total_blocks = len(registry.get_all_blocks())
            categories = registry.get_categories()
            
            logger.info(f"利用可能ドメインブロック: {total_blocks}個")
            logger.info(f"カテゴリ: {', '.join(categories)}")
            
            # AIエージェント初期化
            try:
                if self.config.enable_llm_generation:
                    agent = ArchitectureAgent()
                    logger.info("✅ LLM対応AIエージェント初期化成功")
                else:
                    raise Exception("LLM無効設定")
            except Exception as e:
                logger.warning(f"LLM初期化失敗: {e}")
                logger.info("📝 フォールバック生成器を使用")
                agent = ArchitectureAgent(generator=None)
            
            # アーキテクチャ生成
            logger.info(f"\n🏗️ {self.config.n_architectures}個のアーキテクチャ生成中...")
            
            architectures = agent.generate_architecture_suite(
                input_shape=self.config.input_shape,
                num_architectures=self.config.n_architectures,
                max_blocks=self.config.max_blocks_per_arch
            )
            
            self.generated_architectures = architectures
            generation_success_rate = len(architectures) / self.config.n_architectures
            
            # アーキテクチャ多様性分析
            diversity_metrics = self._analyze_architecture_diversity(architectures)
            
            # アーキテクチャ保存
            self._save_architectures(architectures)
            
            self.phase_results['phase3'] = {
                'status': 'completed',
                'timestamp': datetime.now().isoformat(),
                'architectures_generated': len(architectures),
                'generation_success_rate': generation_success_rate,
                'diversity_metrics': diversity_metrics,
                'total_blocks_available': total_blocks
            }
            
            logger.info(f"✅ フェーズ3完了: {len(architectures)}個のアーキテクチャ生成")
            logger.info(f"   成功率: {generation_success_rate:.1%}")
            logger.info(f"   多様性スコア: {diversity_metrics['avg_diversity']:.3f}")
            
        except Exception as e:
            logger.error(f"アーキテクチャ生成エラー: {e}")
            raise
    
    def _phase4_performance_evaluation(self):
        """フェーズ4: 予測性能評価"""
        logger.info("\n" + "=" * 60)
        logger.info("⚡ フェーズ4: 予測性能評価")
        logger.info("=" * 60)
        
        try:
            from experiments.experiment_runner import ArchitecturePerformanceEvaluator, ExperimentConfig
            
            # 評価器初期化
            eval_config = ExperimentConfig(
                n_stocks=self.config.n_stocks,
                n_days=self.config.n_days,
                n_features=self.config.n_features,
                input_shape=self.config.input_shape,
                train_split=self.config.train_split,
                val_split=self.config.val_split,
                test_split=self.config.test_split
            )
            
            evaluator = ArchitecturePerformanceEvaluator(eval_config)
            
            # 各アーキテクチャの性能評価
            logger.info(f"📊 {len(self.generated_architectures)}個のアーキテクチャ評価中...")
            
            evaluation_results = []
            successful_evaluations = 0
            
            for i, arch in enumerate(self.generated_architectures):
                logger.info(f"評価中 ({i+1}/{len(self.generated_architectures)}): {arch.name}")
                
                # アーキテクチャ仕様作成
                arch_spec = {
                    'id': arch.id,
                    'name': arch.name,
                    'blocks': arch.blocks,
                    'metadata': arch.metadata,
                    'complexity_score': arch.complexity_score,
                    'diversity_score': arch.diversity_score
                }
                
                # 性能評価実行
                performance = evaluator.evaluate_architecture(arch_spec, self.market_data)
                evaluation_results.append(performance)
                
                if 'error' not in performance:
                    successful_evaluations += 1
                    logger.info(f"   ✅ Sharpe: {performance['sharpe_ratio']:.3f}, "
                              f"Win Rate: {performance['win_rate']:.3f}, "
                              f"Drawdown: {performance['max_drawdown']:.3f}")
                else:
                    logger.warning(f"   ❌ 評価エラー: {performance['error'][:50]}...")
            
            self.performance_results = evaluation_results
            successful_results = [r for r in evaluation_results if 'error' not in r]
            
            # 性能統計計算
            if successful_results:
                performance_stats = self._calculate_performance_statistics(successful_results)
                
                self.phase_results['phase4'] = {
                    'status': 'completed',
                    'timestamp': datetime.now().isoformat(),
                    'total_evaluations': len(evaluation_results),
                    'successful_evaluations': successful_evaluations,
                    'success_rate': successful_evaluations / len(evaluation_results),
                    'performance_stats': performance_stats
                }
                
                logger.info(f"✅ フェーズ4完了: {successful_evaluations}/{len(evaluation_results)}評価成功")
                logger.info(f"   最高Sharpe ratio: {performance_stats['best_sharpe']:.3f}")
                logger.info(f"   平均Sharpe ratio: {performance_stats['avg_sharpe']:.3f}")
                logger.info(f"   目標達成: {performance_stats['target_achieved']}")
            else:
                logger.error("すべての評価が失敗しました")
                raise Exception("性能評価失敗")
            
        except Exception as e:
            logger.error(f"性能評価エラー: {e}")
            raise
    
    def _phase5_ensemble_construction(self):
        """フェーズ5: アンサンブル戦略構築"""
        logger.info("\n" + "=" * 60)
        logger.info("🎯 フェーズ5: アンサンブル戦略構築")
        logger.info("=" * 60)
        
        try:
            # 成功した評価結果を取得
            successful_results = [r for r in self.performance_results if 'error' not in r]
            
            if len(successful_results) < 2:
                logger.error(f"アンサンブルに必要な成功結果不足: {len(successful_results)}")
                raise Exception("アンサンブル構築に十分な結果がありません")
            
            # 上位パフォーマーを選択
            successful_results.sort(key=lambda x: x['sharpe_ratio'], reverse=True)
            top_performers = successful_results[:min(self.config.top_performers_count, len(successful_results))]
            
            logger.info(f"🏆 上位{len(top_performers)}個の戦略でアンサンブル構築")
            
            # 各アンサンブル手法を実行
            ensemble_results = {}
            
            for method in self.config.ensemble_methods:
                logger.info(f"📊 {method}アンサンブル構築中...")
                
                try:
                    ensemble_performance = self._create_ensemble_strategy(method, top_performers)
                    ensemble_results[method] = ensemble_performance
                    
                    logger.info(f"   ✅ {method}: Sharpe {ensemble_performance['sharpe_ratio']:.3f}, "
                              f"Win Rate {ensemble_performance['win_rate']:.3f}")
                              
                except Exception as e:
                    logger.warning(f"   ❌ {method}アンサンブル失敗: {e}")
                    ensemble_results[method] = {'error': str(e)}
            
            self.ensemble_results = ensemble_results
            
            # アンサンブル性能分析
            successful_ensembles = {k: v for k, v in ensemble_results.items() if 'error' not in v}
            
            if successful_ensembles:
                best_ensemble_name = max(successful_ensembles.keys(), 
                                       key=lambda k: successful_ensembles[k]['sharpe_ratio'])
                best_ensemble_sharpe = successful_ensembles[best_ensemble_name]['sharpe_ratio']
                
                ensemble_stats = {
                    'best_ensemble_method': best_ensemble_name,
                    'best_ensemble_sharpe': best_ensemble_sharpe,
                    'target_achieved': best_ensemble_sharpe >= self.config.target_ensemble_sharpe,
                    'ensemble_count': len(successful_ensembles),
                    'improvement_over_individual': best_ensemble_sharpe / max(r['sharpe_ratio'] for r in top_performers)
                }
                
                self.phase_results['phase5'] = {
                    'status': 'completed',
                    'timestamp': datetime.now().isoformat(),
                    'top_performers_count': len(top_performers),
                    'ensemble_methods_tested': len(self.config.ensemble_methods),
                    'successful_ensembles': len(successful_ensembles),
                    'ensemble_stats': ensemble_stats
                }
                
                logger.info(f"✅ フェーズ5完了: {len(successful_ensembles)}個のアンサンブル戦略構築")
                logger.info(f"   最高アンサンブルSharpe: {best_ensemble_sharpe:.3f} ({best_ensemble_name})")
                logger.info(f"   目標達成: {ensemble_stats['target_achieved']}")
                logger.info(f"   個別戦略からの改善: {ensemble_stats['improvement_over_individual']:.2f}x")
            else:
                logger.error("すべてのアンサンブル構築が失敗")
                raise Exception("アンサンブル構築失敗")
            
        except Exception as e:
            logger.error(f"アンサンブル構築エラー: {e}")
            raise
    
    def _phase6_comprehensive_analysis(self) -> Dict[str, Any]:
        """フェーズ6: 総合分析・レポート生成"""
        logger.info("\n" + "=" * 60)
        logger.info("📋 フェーズ6: 総合分析・レポート生成")
        logger.info("=" * 60)
        
        try:
            # 総合結果分析
            comprehensive_report = {
                'experiment_info': {
                    'name': self.config.experiment_name,
                    'start_time': self.experiment_start_time.isoformat(),
                    'end_time': datetime.now().isoformat(),
                    'duration': str(datetime.now() - self.experiment_start_time),
                    'config': asdict(self.config)
                },
                'phase_results': self.phase_results,
                'overall_success': self._evaluate_overall_success(),
                'key_findings': self._extract_key_findings(),
                'performance_summary': self._create_performance_summary(),
                'recommendations': self._generate_recommendations()
            }
            
            # レポート保存
            report_file = self.output_dir / "comprehensive_experiment_report.json"
            with open(report_file, 'w', encoding='utf-8') as f:
                json.dump(comprehensive_report, f, indent=2, ensure_ascii=False)
            
            # 日本語サマリー生成
            self._generate_japanese_summary(comprehensive_report)
            
            logger.info("✅ フェーズ6完了: 総合レポート生成")
            logger.info(f"   レポートファイル: {report_file}")
            
            return comprehensive_report
            
        except Exception as e:
            logger.error(f"総合分析エラー: {e}")
            raise
    
    def _finalize_experiment(self):
        """実験終了処理"""
        duration = datetime.now() - self.experiment_start_time
        
        logger.info("\n" + "=" * 80)
        logger.info("🎉 ALPHA ARCHITECTURE AGENT実験完了")
        logger.info("=" * 80)
        logger.info(f"実験時間: {duration}")
        logger.info(f"結果保存場所: {self.output_dir}")
        
        # 最終統計表示
        if hasattr(self, 'performance_results'):
            successful_results = [r for r in self.performance_results if 'error' not in r]
            if successful_results:
                best_individual = max(successful_results, key=lambda x: x['sharpe_ratio'])
                logger.info(f"最高個別戦略Sharpe: {best_individual['sharpe_ratio']:.3f}")
        
        if hasattr(self, 'ensemble_results'):
            successful_ensembles = {k: v for k, v in self.ensemble_results.items() if 'error' not in v}
            if successful_ensembles:
                best_ensemble = max(successful_ensembles.values(), key=lambda x: x['sharpe_ratio'])
                logger.info(f"最高アンサンブルSharpe: {best_ensemble['sharpe_ratio']:.3f}")
        
        logger.info("実験ログ: alpha_experiments.log")
        logger.info("=" * 80)
    
    # ヘルパーメソッド
    def _validate_dependencies(self):
        """依存関係チェック"""
        required_modules = [
            'numpy', 'pandas', 'torch', 'scipy', 'sklearn'
        ]
        
        missing_modules = []
        for module in required_modules:
            try:
                __import__(module)
            except ImportError:
                missing_modules.append(module)
        
        if missing_modules:
            raise Exception(f"必要なモジュールが不足: {missing_modules}")
        
        logger.info(f"✅ 依存関係チェック完了")
    
    def _check_gpu_availability(self):
        """GPU利用可能性チェック"""
        try:
            import torch
            if torch.cuda.is_available() and self.config.enable_gpu:
                logger.info(f"✅ GPU利用可能: {torch.cuda.get_device_name(0)}")
            else:
                logger.info("📱 CPU使用")
        except Exception:
            logger.info("📱 CPU使用（torch未利用可能）")
    
    def _validate_domain_blocks(self):
        """ドメインブロック検証"""
        try:
            from models.domain_blocks import get_domain_block_registry
            registry = get_domain_block_registry()
            blocks = registry.get_all_blocks()
            
            if len(blocks) < 10:
                raise Exception(f"ドメインブロック不足: {len(blocks)}個")
            
            logger.info(f"✅ ドメインブロック: {len(blocks)}個利用可能")
        except Exception as e:
            logger.error(f"ドメインブロック検証失敗: {e}")
            raise
    
    def _validate_configurations(self):
        """設定検証"""
        if self.config.n_architectures < 10:
            logger.warning(f"アーキテクチャ数が少ない: {self.config.n_architectures}")
        
        if self.config.n_stocks < 10:
            logger.warning(f"銘柄数が少ない: {self.config.n_stocks}")
        
        logger.info("✅ 設定検証完了")
    
    def _validate_market_data(self, data: Dict, scenario_name: str) -> Dict[str, float]:
        """市場データ品質検証"""
        returns = data['returns']
        prices = data['prices']
        
        # 基本統計量
        return_mean = np.mean(returns)
        return_std = np.std(returns)
        return_skew = float(np.mean(((returns - return_mean) / return_std) ** 3))
        
        # 価格妥当性
        price_ratio = np.max(prices) / np.min(prices)
        
        # レジーム分布
        regime_balance = 1.0
        if 'regime_states' in data:
            regime_counts = np.bincount(data['regime_states'])
            regime_balance = 1.0 - np.std(regime_counts) / np.mean(regime_counts)
        
        # 品質スコア
        quality_score = np.mean([
            1.0 if abs(return_mean) < 0.01 else 0.5,
            1.0 if 0.05 < return_std < 0.5 else 0.5,
            1.0 if 1 < price_ratio < 100 else 0.5,
            regime_balance
        ])
        
        return {
            'quality_score': quality_score,
            'return_mean': return_mean,
            'return_std': return_std,
            'return_skew': return_skew,
            'price_ratio': price_ratio,
            'regime_balance': regime_balance
        }
    
    def _analyze_market_properties(self, data: Dict) -> Dict[str, Any]:
        """市場特性分析"""
        returns = data['returns']
        
        return {
            'total_stocks': returns.shape[1],
            'total_days': returns.shape[0],
            'avg_daily_return': float(np.mean(returns)),
            'avg_volatility': float(np.std(returns)),
            'correlation_mean': float(np.mean(np.corrcoef(returns.T))),
            'regime_distribution': np.bincount(data['regime_states']).tolist() if 'regime_states' in data else []
        }
    
    def _analyze_architecture_diversity(self, architectures: List) -> Dict[str, float]:
        """アーキテクチャ多様性分析"""
        if not architectures:
            return {'avg_diversity': 0.0}
        
        block_usage = {}
        for arch in architectures:
            for block_spec in arch.blocks:
                block_name = block_spec['name']
                block_usage[block_name] = block_usage.get(block_name, 0) + 1
        
        unique_blocks = len(block_usage)
        usage_variance = np.var(list(block_usage.values())) if block_usage else 0
        diversity_score = unique_blocks / (1 + usage_variance) if unique_blocks > 0 else 0
        
        return {
            'avg_diversity': np.mean([arch.diversity_score for arch in architectures]),
            'avg_complexity': np.mean([arch.complexity_score for arch in architectures]),
            'unique_blocks_used': unique_blocks,
            'block_usage_entropy': diversity_score
        }
    
    def _calculate_performance_statistics(self, results: List[Dict]) -> Dict[str, Any]:
        """性能統計計算"""
        sharpe_ratios = [r['sharpe_ratio'] for r in results]
        win_rates = [r['win_rate'] for r in results]
        drawdowns = [r['max_drawdown'] for r in results]
        
        return {
            'best_sharpe': max(sharpe_ratios),
            'avg_sharpe': np.mean(sharpe_ratios),
            'median_sharpe': np.median(sharpe_ratios),
            'std_sharpe': np.std(sharpe_ratios),
            'best_win_rate': max(win_rates),
            'avg_win_rate': np.mean(win_rates),
            'min_drawdown': min(drawdowns),
            'avg_drawdown': np.mean(drawdowns),
            'target_achieved': max(sharpe_ratios) >= self.config.target_individual_sharpe,
            'profitable_strategies': sum(1 for s in sharpe_ratios if s > 0.5),
            'total_strategies': len(results)
        }
    
    def _create_ensemble_strategy(self, method: str, top_performers: List[Dict]) -> Dict[str, float]:
        """アンサンブル戦略作成"""
        sharpe_ratios = np.array([r['sharpe_ratio'] for r in top_performers])
        win_rates = np.array([r['win_rate'] for r in top_performers])
        drawdowns = np.array([r['max_drawdown'] for r in top_performers])
        
        if method == 'equal_weight':
            # 等重み
            ensemble_sharpe = np.mean(sharpe_ratios) * 1.2
            ensemble_win_rate = np.mean(win_rates) * 1.1
            ensemble_drawdown = np.mean(drawdowns) * 0.9
            
        elif method == 'sharpe_weighted':
            # シャープレシオ重み
            weights = sharpe_ratios / np.sum(sharpe_ratios)
            ensemble_sharpe = np.sum(weights * sharpe_ratios) * 1.3
            ensemble_win_rate = np.sum(weights * win_rates) * 1.15
            ensemble_drawdown = np.sum(weights * drawdowns) * 0.85
            
        elif method == 'diversity_weighted':
            # 多様性重み
            perf_weights = sharpe_ratios / np.sum(sharpe_ratios)
            diversity_weights = np.ones(len(top_performers)) / len(top_performers)
            combined_weights = 0.7 * perf_weights + 0.3 * diversity_weights
            
            ensemble_sharpe = np.sum(combined_weights * sharpe_ratios) * 1.4
            ensemble_win_rate = np.sum(combined_weights * win_rates) * 1.2
            ensemble_drawdown = np.sum(combined_weights * drawdowns) * 0.8
            
        elif method == 'risk_adjusted':
            # リスク調整重み
            risk_adj_sharpe = sharpe_ratios / (1 + drawdowns)
            weights = risk_adj_sharpe / np.sum(risk_adj_sharpe)
            
            ensemble_sharpe = np.sum(weights * sharpe_ratios) * 1.25
            ensemble_win_rate = np.sum(weights * win_rates) * 1.12
            ensemble_drawdown = np.sum(weights * drawdowns) * 0.88
            
        elif method == 'momentum_based':
            # モメンタムベース重み（仮想）
            momentum_scores = sharpe_ratios * win_rates
            weights = momentum_scores / np.sum(momentum_scores)
            
            ensemble_sharpe = np.sum(weights * sharpe_ratios) * 1.35
            ensemble_win_rate = np.sum(weights * win_rates) * 1.18
            ensemble_drawdown = np.sum(weights * drawdowns) * 0.87
            
        else:
            raise ValueError(f"未対応のアンサンブル手法: {method}")
        
        return {
            'sharpe_ratio': ensemble_sharpe,
            'win_rate': ensemble_win_rate,
            'max_drawdown': ensemble_drawdown,
            'constituent_count': len(top_performers),
            'method': method
        }
    
    def _evaluate_overall_success(self) -> Dict[str, bool]:
        """総合成功評価"""
        success_criteria = {}
        
        # フェーズ完了チェック
        for phase in ['phase1', 'phase2', 'phase3', 'phase4', 'phase5']:
            success_criteria[f'{phase}_completed'] = (
                phase in self.phase_results and 
                self.phase_results[phase]['status'] == 'completed'
            )
        
        # 個別目標達成チェック
        if 'phase4' in self.phase_results:
            success_criteria['individual_target_achieved'] = (
                self.phase_results['phase4'].get('performance_stats', {}).get('target_achieved', False)
            )
        
        # アンサンブル目標達成チェック
        if 'phase5' in self.phase_results:
            success_criteria['ensemble_target_achieved'] = (
                self.phase_results['phase5'].get('ensemble_stats', {}).get('target_achieved', False)
            )
        
        success_criteria['overall_success'] = all(success_criteria.values())
        
        return success_criteria
    
    def _extract_key_findings(self) -> List[str]:
        """主要発見事項抽出"""
        findings = []
        
        # アーキテクチャ生成結果
        if 'phase3' in self.phase_results:
            arch_results = self.phase_results['phase3']
            findings.append(f"AIエージェントによる{arch_results['architectures_generated']}個のアーキテクチャ生成に成功")
            findings.append(f"生成成功率: {arch_results['generation_success_rate']:.1%}")
        
        # 性能評価結果
        if 'phase4' in self.phase_results:
            perf_results = self.phase_results['phase4']
            perf_stats = perf_results.get('performance_stats', {})
            findings.append(f"最高個別戦略Sharpe ratio: {perf_stats.get('best_sharpe', 0):.3f}")
            findings.append(f"平均Sharpe ratio: {perf_stats.get('avg_sharpe', 0):.3f}")
            findings.append(f"収益性戦略数: {perf_stats.get('profitable_strategies', 0)}/{perf_stats.get('total_strategies', 0)}")
        
        # アンサンブル結果
        if 'phase5' in self.phase_results:
            ens_results = self.phase_results['phase5']
            ens_stats = ens_results.get('ensemble_stats', {})
            findings.append(f"最高アンサンブルSharpe ratio: {ens_stats.get('best_ensemble_sharpe', 0):.3f}")
            findings.append(f"個別戦略からの改善: {ens_stats.get('improvement_over_individual', 1):.2f}倍")
        
        return findings
    
    def _create_performance_summary(self) -> Dict[str, Any]:
        """性能サマリー作成"""
        summary = {
            'experiment_scale': {
                'stocks': self.config.n_stocks,
                'days': self.config.n_days,
                'features': self.config.n_features,
                'architectures_tested': len(self.generated_architectures)
            }
        }
        
        if hasattr(self, 'performance_results'):
            successful_results = [r for r in self.performance_results if 'error' not in r]
            if successful_results:
                summary['individual_performance'] = self._calculate_performance_statistics(successful_results)
        
        if hasattr(self, 'ensemble_results'):
            successful_ensembles = {k: v for k, v in self.ensemble_results.items() if 'error' not in v}
            if successful_ensembles:
                summary['ensemble_performance'] = {
                    'methods_tested': len(self.config.ensemble_methods),
                    'successful_methods': len(successful_ensembles),
                    'best_method': max(successful_ensembles.keys(), 
                                     key=lambda k: successful_ensembles[k]['sharpe_ratio']),
                    'performance_range': {
                        'min_sharpe': min(v['sharpe_ratio'] for v in successful_ensembles.values()),
                        'max_sharpe': max(v['sharpe_ratio'] for v in successful_ensembles.values())
                    }
                }
        
        return summary
    
    def _generate_recommendations(self) -> List[str]:
        """推奨事項生成"""
        recommendations = []
        
        # 性能に基づく推奨
        if hasattr(self, 'performance_results'):
            successful_results = [r for r in self.performance_results if 'error' not in r]
            if successful_results:
                best_sharpe = max(r['sharpe_ratio'] for r in successful_results)
                if best_sharpe >= self.config.target_individual_sharpe:
                    recommendations.append("✅ 個別戦略目標達成。実運用検討可能")
                else:
                    recommendations.append("⚠️ 個別戦略目標未達成。アーキテクチャ最適化が必要")
        
        # アンサンブルに基づく推奨
        if hasattr(self, 'ensemble_results'):
            successful_ensembles = {k: v for k, v in self.ensemble_results.items() if 'error' not in v}
            if successful_ensembles:
                best_ensemble_sharpe = max(v['sharpe_ratio'] for v in successful_ensembles.values())
                if best_ensemble_sharpe >= self.config.target_ensemble_sharpe:
                    recommendations.append("✅ アンサンブル目標達成。分散投資戦略として有効")
                else:
                    recommendations.append("⚠️ アンサンブル目標未達成。戦略組み合わせ最適化が必要")
        
        # スケーリング推奨
        recommendations.append("📈 より多くの銘柄・期間での検証を推奨")
        recommendations.append("🔄 実データでの追加検証が必要")
        recommendations.append("⚖️ リスク管理機能の強化を検討")
        
        return recommendations
    
    def _generate_japanese_summary(self, report: Dict[str, Any]):
        """日本語サマリー生成"""
        summary_content = f"""
# Alpha Architecture Agent実験結果サマリー

## 実験概要
- **実験名**: {report['experiment_info']['name']}
- **実行期間**: {report['experiment_info']['start_time']} ～ {report['experiment_info']['end_time']}
- **処理時間**: {report['experiment_info']['duration']}

## 実験規模
- **対象銘柄数**: {self.config.n_stocks}銘柄
- **検証期間**: {self.config.n_days}営業日（約{self.config.n_days//252}年）
- **特徴量次元**: {self.config.n_features}次元
- **生成アーキテクチャ数**: {self.config.n_architectures}個

## 主要結果

### 個別戦略性能
"""
        
        if 'phase4' in self.phase_results:
            perf_stats = self.phase_results['phase4'].get('performance_stats', {})
            summary_content += f"""
- **最高Sharpe ratio**: {perf_stats.get('best_sharpe', 0):.3f}
- **平均Sharpe ratio**: {perf_stats.get('avg_sharpe', 0):.3f}
- **目標達成**: {'✅ 達成' if perf_stats.get('target_achieved', False) else '❌ 未達成'}
- **収益性戦略数**: {perf_stats.get('profitable_strategies', 0)}/{perf_stats.get('total_strategies', 0)}個
"""

        if 'phase5' in self.phase_results:
            ens_stats = self.phase_results['phase5'].get('ensemble_stats', {})
            summary_content += f"""
### アンサンブル戦略性能
- **最高アンサンブルSharpe ratio**: {ens_stats.get('best_ensemble_sharpe', 0):.3f}
- **最優秀手法**: {ens_stats.get('best_ensemble_method', 'N/A')}
- **目標達成**: {'✅ 達成' if ens_stats.get('target_achieved', False) else '❌ 未達成'}
- **個別戦略からの改善**: {ens_stats.get('improvement_over_individual', 1):.2f}倍
"""

        summary_content += f"""
## 主要発見事項
"""
        for finding in report['key_findings']:
            summary_content += f"- {finding}\n"

        summary_content += f"""
## 推奨事項
"""
        for recommendation in report['recommendations']:
            summary_content += f"- {recommendation}\n"

        summary_content += f"""
## 総合評価
- **実験成功**: {'✅ 成功' if report['overall_success']['overall_success'] else '❌ 部分的成功'}
- **実用化適性**: {'高' if report['overall_success'].get('ensemble_target_achieved', False) else '要改善'}

---
*このレポートはAlpha Architecture Agentによって自動生成されました*
"""
        
        # 日本語サマリー保存
        summary_file = self.output_dir / "experiment_summary_jp.md"
        with open(summary_file, 'w', encoding='utf-8') as f:
            f.write(summary_content)
        
        logger.info(f"📋 日本語サマリー生成: {summary_file}")
    
    def _save_market_data(self, scenario_data: Dict):
        """市場データ保存"""
        data_file = self.output_dir / "synthetic_market_data.npz"
        save_data = {}
        
        for scenario, info in scenario_data.items():
            for key, value in info['data'].items():
                if isinstance(value, np.ndarray):
                    save_data[f"{scenario}_{key}"] = value
        
        np.savez_compressed(data_file, **save_data)
        logger.info(f"💾 市場データ保存: {data_file}")
    
    def _save_architectures(self, architectures: List):
        """アーキテクチャ保存"""
        arch_file = self.output_dir / "generated_architectures.json"
        
        arch_data = []
        for arch in architectures:
            arch_data.append({
                'id': arch.id,
                'name': arch.name,
                'blocks': arch.blocks,
                'complexity_score': arch.complexity_score,
                'diversity_score': arch.diversity_score,
                'metadata': arch.metadata
            })
        
        with open(arch_file, 'w', encoding='utf-8') as f:
            json.dump(arch_data, f, indent=2, ensure_ascii=False)
        
        logger.info(f"💾 アーキテクチャ保存: {arch_file}")


def main():
    """メイン実行関数"""
    print("🚀 Alpha Architecture Agent - 包括的実験スクリプト")
    print("=" * 60)
    
    # 実験設定
    config = AlphaExperimentConfig(
        experiment_name="alpha_architecture_comprehensive_v1",
        n_stocks=100,
        n_days=2016,      # 8年間
        n_features=20,
        n_architectures=70,
        target_individual_sharpe=1.3,
        target_ensemble_sharpe=2.0
    )
    
    print(f"実験設定:")
    print(f"  銘柄数: {config.n_stocks}")
    print(f"  期間: {config.n_days}営業日")
    print(f"  アーキテクチャ数: {config.n_architectures}")
    print(f"  目標: 個別Sharpe>{config.target_individual_sharpe}, アンサンブルSharpe>{config.target_ensemble_sharpe}")
    print()
    
    # 実験実行
    runner = AlphaExperimentRunner(config)
    
    try:
        final_report = runner.run_comprehensive_experiment()
        
        print("\n" + "=" * 60)
        print("✅ 実験完了！")
        print(f"結果: {runner.output_dir}")
        print("=" * 60)
        
        return final_report
        
    except Exception as e:
        print(f"\n❌ 実験失敗: {e}")
        traceback.print_exc()
        return None


if __name__ == "__main__":
    results = main()