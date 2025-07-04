# API仕様書

## 概要
Alpha Architecture Agent のAPI仕様について詳細に記載いたします。本APIは、AIエージェントによるアーキテクチャ生成、モデル訓練、バックテストを自動化するRESTful APIです。

## 基本情報

### ベースURL
```
http://localhost:8000/api/v1
```

### 認証方式
```
Bearer Token認証
Authorization: Bearer <API_KEY>
```

### レスポンス形式
```json
{
  "success": true,
  "data": {},
  "message": "処理が正常に完了しました",
  "timestamp": "2024-07-04T10:30:00Z"
}
```

## エンドポイント一覧

### 1. アーキテクチャ管理

#### 1.1 アーキテクチャ生成
```
POST /architectures/generate
```

**リクエスト**:
```json
{
  "input_shape": [32, 252, 20],
  "num_architectures": 10,
  "complexity_range": [5, 15],
  "diversity_threshold": 0.8,
  "generation_mode": "ai_agent"
}
```

**レスポンス**:
```json
{
  "success": true,
  "data": {
    "architectures": [
      {
        "id": "arch_001",
        "name": "LSTM_PCA_Regime_Architecture",
        "blocks": [
          {
            "type": "DemeanBlock",
            "parameters": {}
          },
          {
            "type": "PCABlock", 
            "parameters": {
              "n_components": 15
            }
          },
          {
            "type": "LSTMBlock",
            "parameters": {
              "hidden_dim": 64,
              "num_layers": 2
            }
          }
        ],
        "complexity": 8,
        "estimated_params": 125000
      }
    ],
    "generation_stats": {
      "total_generated": 10,
      "diversity_score": 0.85,
      "generation_time": 45.2
    }
  }
}
```

#### 1.2 アーキテクチャ詳細取得
```
GET /architectures/{architecture_id}
```

**レスポンス**:
```json
{
  "success": true,
  "data": {
    "id": "arch_001",
    "name": "LSTM_PCA_Regime_Architecture",
    "description": "PCA特徴抽出とLSTMを組み合わせたレジーム検出アーキテクチャ",
    "blocks": [...],
    "performance_metrics": {
      "sharpe_ratio": 1.25,
      "max_drawdown": -0.08,
      "total_return": 0.45
    },
    "created_at": "2024-07-04T10:30:00Z",
    "updated_at": "2024-07-04T10:30:00Z"
  }
}
```

#### 1.3 アーキテクチャ一覧取得
```
GET /architectures
```

**クエリパラメータ**:
- `page`: ページ番号（デフォルト: 1）
- `limit`: 1ページあたりの件数（デフォルト: 20）
- `sort_by`: ソート基準（performance, created_at, complexity）
- `filter_by`: フィルタ条件（JSON形式）

**レスポンス**:
```json
{
  "success": true,
  "data": {
    "architectures": [...],
    "pagination": {
      "page": 1,
      "limit": 20,
      "total": 150,
      "total_pages": 8
    }
  }
}
```

### 2. モデル管理

#### 2.1 モデル訓練開始
```
POST /models/train
```

**リクエスト**:
```json
{
  "architecture_id": "arch_001",
  "training_config": {
    "epochs": 100,
    "batch_size": 32,
    "learning_rate": 0.001,
    "validation_split": 0.2
  },
  "data_config": {
    "symbols": ["7203.T", "9984.T", "6758.T"],
    "start_date": "2020-01-01",
    "end_date": "2024-06-30",
    "features": ["ohlcv", "technical_indicators"]
  }
}
```

**レスポンス**:
```json
{
  "success": true,
  "data": {
    "job_id": "train_job_001",
    "model_id": "model_001",
    "status": "started",
    "estimated_duration": 1800
  }
}
```

#### 2.2 訓練状況取得
```
GET /models/train/{job_id}/status
```

**レスポンス**:
```json
{
  "success": true,
  "data": {
    "job_id": "train_job_001",
    "status": "training",
    "progress": 0.65,
    "current_epoch": 65,
    "total_epochs": 100,
    "metrics": {
      "train_loss": 0.0234,
      "val_loss": 0.0289,
      "train_accuracy": 0.72,
      "val_accuracy": 0.68
    },
    "estimated_remaining": 630
  }
}
```

#### 2.3 モデル詳細取得
```
GET /models/{model_id}
```

**レスポンス**:
```json
{
  "success": true,
  "data": {
    "id": "model_001",
    "architecture_id": "arch_001",
    "name": "LSTM_PCA_Regime_Model_v1",
    "status": "trained",
    "training_metrics": {
      "final_train_loss": 0.0198,
      "final_val_loss": 0.0245,
      "training_time": 1654,
      "convergence_epoch": 87
    },
    "model_size": 2.4,
    "created_at": "2024-07-04T10:30:00Z"
  }
}
```

### 3. バックテスト

#### 3.1 バックテスト実行
```
POST /backtests/run
```

**リクエスト**:
```json
{
  "model_ids": ["model_001", "model_002"],
  "backtest_config": {
    "start_date": "2023-01-01",
    "end_date": "2024-06-30",
    "initial_capital": 10000000,
    "long_percentage": 0.05,
    "short_percentage": 0.05,
    "rebalance_frequency": "daily"
  },
  "risk_config": {
    "max_position_size": 0.02,
    "sector_limit": 0.3,
    "stop_loss": 0.05
  }
}
```

**レスポンス**:
```json
{
  "success": true,
  "data": {
    "backtest_id": "backtest_001",
    "status": "started",
    "estimated_duration": 300
  }
}
```

#### 3.2 バックテスト結果取得
```
GET /backtests/{backtest_id}/results
```

**レスポンス**:
```json
{
  "success": true,
  "data": {
    "backtest_id": "backtest_001",
    "status": "completed",
    "results": {
      "summary": {
        "total_return": 0.342,
        "sharpe_ratio": 1.45,
        "sortino_ratio": 1.78,
        "max_drawdown": -0.089,
        "calmar_ratio": 3.84,
        "volatility": 0.156
      },
      "daily_returns": [...],
      "positions": [...],
      "trades": [...]
    },
    "comparison": {
      "benchmark": {
        "name": "TOPIX",
        "total_return": 0.156,
        "sharpe_ratio": 0.78,
        "max_drawdown": -0.145
      },
      "excess_return": 0.186,
      "information_ratio": 1.23
    }
  }
}
```

### 4. データ管理

#### 4.1 データ収集開始
```
POST /data/collect
```

**リクエスト**:
```json
{
  "data_type": "japanese_stocks",
  "symbols": ["7203.T", "9984.T", "6758.T"],
  "start_date": "2020-01-01",
  "end_date": "2024-06-30",
  "frequency": "1d",
  "features": ["ohlcv", "technical_indicators", "fundamental"]
}
```

**レスポンス**:
```json
{
  "success": true,
  "data": {
    "collection_id": "collect_001",
    "status": "started",
    "estimated_duration": 120
  }
}
```

#### 4.2 データ状況確認
```
GET /data/status
```

**レスポンス**:
```json
{
  "success": true,
  "data": {
    "available_datasets": [
      {
        "name": "japanese_stocks_2y_1d",
        "symbols": 50,
        "start_date": "2022-01-01",
        "end_date": "2024-06-30",
        "frequency": "1d",
        "size": "125.6MB"
      }
    ],
    "storage_usage": {
      "total": "2.3GB",
      "available": "7.7GB"
    }
  }
}
```

### 5. アンサンブル管理

#### 5.1 アンサンブル作成
```
POST /ensembles/create
```

**リクエスト**:
```json
{
  "name": "Top20_Ensemble",
  "model_ids": ["model_001", "model_002", "model_003"],
  "weighting_method": "sharpe_weighted",
  "correlation_threshold": 0.8,
  "max_models": 20
}
```

**レスポンス**:
```json
{
  "success": true,
  "data": {
    "ensemble_id": "ensemble_001",
    "selected_models": 18,
    "average_correlation": 0.65,
    "ensemble_sharpe": 2.1
  }
}
```

#### 5.2 アンサンブル予測
```
POST /ensembles/{ensemble_id}/predict
```

**リクエスト**:
```json
{
  "date": "2024-07-04",
  "symbols": ["7203.T", "9984.T", "6758.T"]
}
```

**レスポンス**:
```json
{
  "success": true,
  "data": {
    "predictions": [
      {
        "symbol": "7203.T",
        "prediction": 0.023,
        "confidence": 0.78,
        "rank": 1
      },
      {
        "symbol": "9984.T", 
        "prediction": -0.012,
        "confidence": 0.65,
        "rank": 3
      }
    ],
    "portfolio_weights": {
      "long": {"7203.T": 0.05, "9984.T": 0.03},
      "short": {"6758.T": 0.02}
    }
  }
}
```

### 6. システム管理

#### 6.1 システム状況確認
```
GET /system/health
```

**レスポンス**:
```json
{
  "success": true,
  "data": {
    "status": "healthy",
    "services": {
      "database": "healthy",
      "cache": "healthy",
      "mlflow": "healthy",
      "worker_queue": "healthy"
    },
    "resources": {
      "cpu_usage": 45,
      "memory_usage": 67,
      "disk_usage": 23,
      "gpu_usage": 89
    },
    "active_jobs": {
      "training": 2,
      "backtesting": 1,
      "data_collection": 0
    }
  }
}
```

#### 6.2 システム設定取得
```
GET /system/config
```

**レスポンス**:
```json
{
  "success": true,
  "data": {
    "version": "0.1.0",
    "environment": "development",
    "limits": {
      "max_concurrent_training": 4,
      "max_architectures_per_request": 100,
      "max_backtest_duration": 30
    },
    "features": {
      "ai_agent_generation": true,
      "ensemble_optimization": true,
      "real_time_prediction": false
    }
  }
}
```

## エラーハンドリング

### エラーレスポンス形式
```json
{
  "success": false,
  "error": {
    "code": "INVALID_INPUT",
    "message": "入力パラメータが無効です",
    "details": {
      "field": "input_shape",
      "reason": "配列の次元数が正しくありません"
    }
  },
  "timestamp": "2024-07-04T10:30:00Z"
}
```

### エラーコード一覧
- `INVALID_INPUT`: 入力パラメータエラー
- `RESOURCE_NOT_FOUND`: リソースが見つかりません
- `INSUFFICIENT_RESOURCES`: リソース不足
- `TRAINING_FAILED`: 訓練失敗
- `INTERNAL_ERROR`: 内部エラー
- `RATE_LIMIT_EXCEEDED`: レート制限超過
- `AUTHENTICATION_FAILED`: 認証失敗

## 使用例

### Python SDK
```python
import requests
from alpha_agent_sdk import AlphaAgentClient

# クライアント初期化
client = AlphaAgentClient(
    base_url="http://localhost:8000/api/v1",
    api_key="your_api_key"
)

# アーキテクチャ生成
architectures = client.generate_architectures(
    input_shape=[32, 252, 20],
    num_architectures=10
)

# モデル訓練
training_job = client.train_model(
    architecture_id=architectures[0].id,
    training_config={
        "epochs": 100,
        "batch_size": 32
    }
)

# バックテスト実行
backtest = client.run_backtest(
    model_ids=[training_job.model_id],
    start_date="2023-01-01",
    end_date="2024-06-30"
)

print(f"シャープレシオ: {backtest.results.sharpe_ratio:.2f}")
```

### cURL例
```bash
# アーキテクチャ生成
curl -X POST "http://localhost:8000/api/v1/architectures/generate" \
  -H "Authorization: Bearer your_api_key" \
  -H "Content-Type: application/json" \
  -d '{
    "input_shape": [32, 252, 20],
    "num_architectures": 10,
    "generation_mode": "ai_agent"
  }'

# バックテスト実行
curl -X POST "http://localhost:8000/api/v1/backtests/run" \
  -H "Authorization: Bearer your_api_key" \
  -H "Content-Type: application/json" \
  -d '{
    "model_ids": ["model_001"],
    "backtest_config": {
      "start_date": "2023-01-01",
      "end_date": "2024-06-30",
      "initial_capital": 10000000
    }
  }'
```

## レート制限
- 一般API: 1000リクエスト/時間
- 重い処理（訓練、バックテスト）: 10リクエスト/時間
- データ収集: 50リクエスト/時間

## セキュリティ
- HTTPS必須（本番環境）
- API キーローテーション対応
- リクエストログ記録
- 入力値検証
- SQL インジェクション対策

この API を使用することで、Alpha Architecture Agent の全機能をプログラムから利用することができます。