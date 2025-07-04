#!/usr/bin/env python3
"""
実市場データ収集スクリプト

yfinanceを使用して無料で株価データを取得・保存
- 日本株（Nikkei 225主要銘柄）
- 米国株（S&P 500主要銘柄）  
- 主要ETF、指数
- 為替レート

解析とは分離して、確実なデータ収集・保存を実行
"""

import sys
import os
from pathlib import Path
import time
from datetime import datetime, timedelta
import traceback
import json

# プロジェクトパス設定
PROJECT_ROOT = Path(__file__).parent.parent
SRC_DIR = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_DIR))

# データディレクトリ作成
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)

import yfinance as yf
import pandas as pd
import numpy as np

def get_nikkei_225_symbols():
    """日経225主要銘柄リスト（東証プライム）"""
    # 主要な日本株銘柄（.T は東京証券取引所）
    symbols = [
        # 自動車・輸送機器
        "7203.T",  # トヨタ自動車
        "7267.T",  # ホンダ
        "7201.T",  # 日産自動車
        "7269.T",  # スズキ
        
        # 電機・精密機器
        "6758.T",  # ソニーグループ
        "6861.T",  # キーエンス
        "7974.T",  # 任天堂
        "6954.T",  # ファナック
        "6981.T",  # 村田製作所
        
        # 情報・通信
        "9984.T",  # ソフトバンクグループ
        "4689.T",  # Zホールディングス
        "9613.T",  # エヌ・ティ・ティ・データ
        "9432.T",  # 日本電信電話
        
        # 銀行・証券・保険
        "8306.T",  # 三菱UFJフィナンシャル・グループ
        "8316.T",  # 三井住友フィナンシャルグループ
        "8411.T",  # みずほフィナンシャルグループ
        "8604.T",  # 野村ホールディングス
        
        # 小売・サービス
        "9983.T",  # ファーストリテイリング
        "3382.T",  # セブン&アイ・ホールディングス
        "8267.T",  # イオン
        "9843.T",  # ニトリホールディングス
        
        # 素材・化学
        "4063.T",  # 信越化学工業
        "4502.T",  # 武田薬品工業
        "4519.T",  # 中外製薬
        "5401.T",  # 新日鐵住金
        
        # エネルギー・商社
        "8058.T",  # 三菱商事
        "8031.T",  # 三井物産
        "8053.T",  # 住友商事
        "1605.T",  # 国際石油開発帝石
        
        # 不動産・建設
        "1925.T",  # 大和ハウス工業
        "1802.T",  # 大林組
        "8804.T",  # 東京建物
    ]
    
    return symbols

def get_sp500_symbols():
    """S&P500主要銘柄リスト"""
    symbols = [
        # テクノロジー
        "AAPL",   # Apple
        "MSFT",   # Microsoft
        "GOOGL",  # Alphabet
        "AMZN",   # Amazon
        "TSLA",   # Tesla
        "META",   # Meta
        "NVDA",   # NVIDIA
        "NFLX",   # Netflix
        
        # 金融
        "JPM",    # JPMorgan Chase
        "BAC",    # Bank of America
        "WFC",    # Wells Fargo
        "GS",     # Goldman Sachs
        
        # ヘルスケア
        "JNJ",    # Johnson & Johnson
        "PFE",    # Pfizer
        "UNH",    # UnitedHealth
        "ABBV",   # AbbVie
        
        # 消費財
        "PG",     # Procter & Gamble
        "KO",     # Coca-Cola
        "PEP",    # PepsiCo
        "WMT",    # Walmart
        
        # 工業
        "BA",     # Boeing
        "CAT",    # Caterpillar
        "GE",     # General Electric
        "MMM",    # 3M
        
        # エネルギー
        "XOM",    # Exxon Mobil
        "CVX",    # Chevron
    ]
    
    return symbols

def get_market_indices():
    """主要指数・ETFリスト"""
    indices = {
        # 日本
        "^N225": "Nikkei 225",
        "^TOPX": "TOPIX",
        "1306.T": "TOPIX連動型上場投資信託",
        
        # 米国
        "^GSPC": "S&P 500",
        "^DJI": "Dow Jones",
        "^IXIC": "NASDAQ",
        "SPY": "SPDR S&P 500 ETF",
        "QQQ": "Invesco QQQ Trust",
        "VTI": "Vanguard Total Stock Market ETF",
        
        # 世界
        "VT": "Vanguard Total World Stock ETF",
        "VEA": "Vanguard FTSE Developed Markets ETF",
        "VWO": "Vanguard FTSE Emerging Markets ETF",
        
        # 債券
        "TLT": "iShares 20+ Year Treasury Bond ETF",
        "IEF": "iShares 7-10 Year Treasury Bond ETF",
        
        # 商品
        "GLD": "SPDR Gold Shares",
        "SLV": "iShares Silver Trust",
        "USO": "United States Oil Fund",
    }
    
    return indices

def get_forex_pairs():
    """主要為替ペア"""
    pairs = {
        "USDJPY=X": "USD/JPY",
        "EURJPY=X": "EUR/JPY", 
        "GBPJPY=X": "GBP/JPY",
        "AUDJPY=X": "AUD/JPY",
        "EURUSD=X": "EUR/USD",
        "GBPUSD=X": "GBP/USD",
        "AUDUSD=X": "AUD/USD",
    }
    
    return pairs

def download_stock_data(symbols, period="2y", interval="1d", market_name="stocks"):
    """
    株価データ一括ダウンロード
    
    Args:
        symbols: シンボルリスト
        period: 期間 ("1d", "5d", "1mo", "3mo", "6mo", "1y", "2y", "5y", "10y", "ytd", "max")
        interval: 間隔 ("1m", "2m", "5m", "15m", "30m", "60m", "90m", "1h", "1d", "5d", "1wk", "1mo", "3mo")
        market_name: 市場名（保存ファイル用）
    """
    
    print(f"\n🔄 {market_name}データダウンロード開始")
    print(f"銘柄数: {len(symbols)}, 期間: {period}, 間隔: {interval}")
    
    all_data = {}
    success_count = 0
    error_count = 0
    errors = []
    
    for i, symbol in enumerate(symbols, 1):
        try:
            print(f"[{i}/{len(symbols)}] ダウンロード中: {symbol}")
            
            # yfinanceでデータ取得
            ticker = yf.Ticker(symbol)
            hist = ticker.history(period=period, interval=interval)
            
            if hist.empty:
                print(f"⚠️  {symbol}: データが空です")
                error_count += 1
                errors.append(f"{symbol}: Empty data")
                continue
            
            # 基本情報も取得
            info = {}
            try:
                ticker_info = ticker.info
                info = {
                    'symbol': symbol,
                    'shortName': ticker_info.get('shortName', 'N/A'),
                    'longName': ticker_info.get('longName', 'N/A'),
                    'currency': ticker_info.get('currency', 'N/A'),
                    'exchange': ticker_info.get('exchange', 'N/A'),
                    'country': ticker_info.get('country', 'N/A'),
                    'sector': ticker_info.get('sector', 'N/A'),
                    'industry': ticker_info.get('industry', 'N/A'),
                    'marketCap': ticker_info.get('marketCap', 'N/A'),
                    'download_date': datetime.now().isoformat(),
                    'data_start': hist.index.min().isoformat(),
                    'data_end': hist.index.max().isoformat(),
                    'data_points': len(hist)
                }
            except Exception as e:
                print(f"⚠️  {symbol}: 基本情報取得エラー - {e}")
                info = {
                    'symbol': symbol,
                    'download_date': datetime.now().isoformat(),
                    'data_start': hist.index.min().isoformat(),
                    'data_end': hist.index.max().isoformat(),
                    'data_points': len(hist),
                    'info_error': str(e)
                }
            
            all_data[symbol] = {
                'price_data': hist,
                'info': info
            }
            
            success_count += 1
            print(f"✅ {symbol}: {len(hist)}日分のデータ取得成功")
            
            # レート制限対策
            time.sleep(0.1)
            
        except Exception as e:
            print(f"❌ {symbol}: エラー - {e}")
            error_count += 1
            errors.append(f"{symbol}: {str(e)}")
            continue
    
    print(f"\n📊 {market_name}ダウンロード完了")
    print(f"✅ 成功: {success_count}銘柄")
    print(f"❌ エラー: {error_count}銘柄")
    
    if errors:
        print("\n⚠️ エラー詳細:")
        for error in errors[:5]:  # 最初の5個のエラーのみ表示
            print(f"  - {error}")
        if len(errors) > 5:
            print(f"  ... 他{len(errors)-5}個のエラー")
    
    return all_data, {'success': success_count, 'errors': error_count, 'error_list': errors}

def save_market_data(data_dict, market_name, period, interval):
    """市場データを保存"""
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # 各銘柄の価格データをCSVで保存
    price_data_dir = RAW_DATA_DIR / f"{market_name}_{period}_{interval}" / "price_data"
    price_data_dir.mkdir(parents=True, exist_ok=True)
    
    # 銘柄情報をJSONで保存
    info_file = RAW_DATA_DIR / f"{market_name}_{period}_{interval}" / f"symbols_info_{timestamp}.json"
    info_file.parent.mkdir(parents=True, exist_ok=True)
    
    # 統合価格データ（全銘柄の終値）
    all_closes = {}
    all_volumes = {}
    symbols_info = {}
    
    for symbol, data in data_dict.items():
        # 個別価格データ保存
        price_file = price_data_dir / f"{symbol.replace('.', '_').replace('=', '_').replace('^', '')}.csv"
        data['price_data'].to_csv(price_file)
        
        # 統合データ用
        if 'Close' in data['price_data'].columns:
            all_closes[symbol] = data['price_data']['Close']
        if 'Volume' in data['price_data'].columns:
            all_volumes[symbol] = data['price_data']['Volume']
        
        # 銘柄情報
        symbols_info[symbol] = data['info']
    
    # 統合終値データフレーム
    if all_closes:
        closes_df = pd.DataFrame(all_closes)
        closes_file = RAW_DATA_DIR / f"{market_name}_{period}_{interval}" / f"all_closes_{timestamp}.csv"
        closes_df.to_csv(closes_file)
        print(f"💾 統合終値データ保存: {closes_file}")
    
    # 統合出来高データフレーム
    if all_volumes:
        volumes_df = pd.DataFrame(all_volumes)
        volumes_file = RAW_DATA_DIR / f"{market_name}_{period}_{interval}" / f"all_volumes_{timestamp}.csv"
        volumes_df.to_csv(volumes_file)
        print(f"💾 統合出来高データ保存: {volumes_file}")
    
    # 銘柄情報JSON
    with open(info_file, 'w', encoding='utf-8') as f:
        json.dump(symbols_info, f, indent=2, ensure_ascii=False, default=str)
    print(f"💾 銘柄情報保存: {info_file}")
    
    return {
        'price_data_dir': str(price_data_dir),
        'closes_file': str(closes_file) if all_closes else None,
        'volumes_file': str(volumes_file) if all_volumes else None,
        'info_file': str(info_file)
    }

def create_data_summary():
    """収集したデータのサマリー作成"""
    
    summary = {
        'collection_date': datetime.now().isoformat(),
        'data_directories': [],
        'total_symbols': 0,
        'markets_covered': []
    }
    
    # data/rawディレクトリをスキャン
    for market_dir in RAW_DATA_DIR.iterdir():
        if market_dir.is_dir():
            price_data_dir = market_dir / "price_data"
            if price_data_dir.exists():
                csv_files = list(price_data_dir.glob("*.csv"))
                
                market_info = {
                    'market_name': market_dir.name,
                    'symbol_count': len(csv_files),
                    'csv_files': [f.name for f in csv_files],
                    'directory': str(market_dir)
                }
                
                summary['data_directories'].append(market_info)
                summary['total_symbols'] += len(csv_files)
                summary['markets_covered'].append(market_dir.name)
    
    # サマリー保存
    summary_file = DATA_DIR / f"data_collection_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(summary_file, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    
    print(f"\n📋 データ収集サマリー: {summary_file}")
    print(f"総銘柄数: {summary['total_symbols']}")
    print(f"市場数: {len(summary['markets_covered'])}")
    
    return summary

def main():
    """メイン実行関数"""
    
    print("🚀 実市場データ収集開始")
    print("=" * 60)
    
    collection_results = {}
    
    try:
        # 1. 日本株データ収集
        nikkei_symbols = get_nikkei_225_symbols()
        jp_data, jp_stats = download_stock_data(
            nikkei_symbols, 
            period="2y", 
            interval="1d", 
            market_name="japanese_stocks"
        )
        
        if jp_data:
            jp_files = save_market_data(jp_data, "japanese_stocks", "2y", "1d")
            collection_results['japanese_stocks'] = {
                'stats': jp_stats,
                'files': jp_files,
                'symbols': list(jp_data.keys())
            }
        
        # 2. 米国株データ収集
        sp500_symbols = get_sp500_symbols()
        us_data, us_stats = download_stock_data(
            sp500_symbols,
            period="2y",
            interval="1d", 
            market_name="us_stocks"
        )
        
        if us_data:
            us_files = save_market_data(us_data, "us_stocks", "2y", "1d")
            collection_results['us_stocks'] = {
                'stats': us_stats,
                'files': us_files,
                'symbols': list(us_data.keys())
            }
        
        # 3. 指数・ETFデータ収集
        indices = get_market_indices()
        index_data, index_stats = download_stock_data(
            list(indices.keys()),
            period="2y",
            interval="1d",
            market_name="indices_etfs"
        )
        
        if index_data:
            index_files = save_market_data(index_data, "indices_etfs", "2y", "1d")
            collection_results['indices_etfs'] = {
                'stats': index_stats,
                'files': index_files,
                'symbols': list(index_data.keys())
            }
        
        # 4. 為替データ収集
        forex_pairs = get_forex_pairs()
        forex_data, forex_stats = download_stock_data(
            list(forex_pairs.keys()),
            period="2y",
            interval="1d",
            market_name="forex"
        )
        
        if forex_data:
            forex_files = save_market_data(forex_data, "forex", "2y", "1d")
            collection_results['forex'] = {
                'stats': forex_stats,
                'files': forex_files,
                'symbols': list(forex_data.keys())
            }
        
        # 5. 全体サマリー作成
        summary = create_data_summary()
        
        # 6. 収集結果レポート保存
        results_file = DATA_DIR / f"collection_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(collection_results, f, indent=2, ensure_ascii=False, default=str)
        
        print("\n" + "=" * 60)
        print("🎉 実市場データ収集完了")
        print("=" * 60)
        
        total_success = sum(result['stats']['success'] for result in collection_results.values())
        total_errors = sum(result['stats']['errors'] for result in collection_results.values())
        
        print(f"✅ 総成功銘柄数: {total_success}")
        print(f"❌ 総エラー銘柄数: {total_errors}")
        print(f"📁 データ保存先: {DATA_DIR}")
        print(f"📊 収集結果: {results_file}")
        
        return True
        
    except Exception as e:
        print(f"\n💥 致命的エラー: {e}")
        print(f"詳細: {traceback.format_exc()}")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)