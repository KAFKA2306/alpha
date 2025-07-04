from typing import Dict, Any, Optional, List
import os
import yaml
from pathlib import Path
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings


class DatabaseConfig(BaseModel):
    host: str = "localhost"
    port: int = 5432
    database: str = "stockprediction"
    username: str = "stock_user"
    password: str = "stock_pass"


class RedisConfig(BaseModel):
    host: str = "localhost"
    port: int = 6379
    db: int = 0


class MLConfig(BaseModel):
    tracking_uri: str = "http://localhost:5000"
    experiment_name: str = "alpha-architecture-search"
    max_blocks: int = 50
    max_combinations: int = 1000
    training_epochs: int = 100
    batch_size: int = 32
    learning_rate: float = 0.001
    domains: List[str] = Field(default_factory=lambda: [
        "normalization", "feature_extraction", "mixing", "encoding",
        "financial_domain", "feature_integration", "time_integration",
        "stock_features", "attention", "feedforward", "time_embedding",
        "sequence_models", "prediction_heads"
    ])


class LLMConfig(BaseModel):
    provider: str = "openai"
    model: str = "gpt-4"
    temperature: float = 0.7
    max_tokens: int = 4000
    openai_api_key: Optional[str] = None
    anthropic_api_key: Optional[str] = None


class AgentConfig(BaseModel):
    llm: LLMConfig = Field(default_factory=LLMConfig)
    max_architectures: int = 70
    diversity_threshold: float = 0.8
    complexity_range: tuple = (5, 15)


class DataConfig(BaseModel):
    source: str = "j-quants"
    api_key: Optional[str] = None
    symbols_file: str = "config/japanese_stocks.txt"
    timeframes: List[str] = Field(default_factory=lambda: ["1d", "1h", "15m"])
    sequence_length: int = 252
    prediction_horizon: int = 1
    returns_calculation: str = "log"
    outlier_threshold: float = 3.0
    missing_value_strategy: str = "forward_fill"


class StrategyConfig(BaseModel):
    long_percentage: float = 0.05
    short_percentage: float = 0.05
    rebalance_frequency: str = "daily"
    start_date: str = "2017-01-01"
    end_date: str = "2023-12-31"
    validation_start: str = "2024-01-01"
    validation_end: str = "2025-02-28"
    metrics: List[str] = Field(default_factory=lambda: [
        "sharpe_ratio", "sortino_ratio", "max_drawdown", "calmar_ratio",
        "total_return", "volatility"
    ])
    ensemble_top_n: int = 20
    ensemble_weighting: str = "equal"
    correlation_threshold: float = 0.8


class APIConfig(BaseModel):
    host: str = "0.0.0.0"
    port: int = 8000
    debug: bool = False


class MonitoringConfig(BaseModel):
    prometheus_enabled: bool = True
    prometheus_port: int = 9090
    grafana_enabled: bool = True
    grafana_port: int = 3001
    log_level: str = "INFO"
    log_format: str = "structured"


class MCPConfig(BaseModel):
    finance_enabled: bool = True
    finance_sources: List[str] = Field(default_factory=lambda: [
        "yahoo_finance", "alpha_vantage", "j_quants"
    ])
    macro_data_enabled: bool = True
    macro_data_sources: List[str] = Field(default_factory=lambda: [
        "fred", "world_bank", "oecd"
    ])
    alternative_data_enabled: bool = True
    alternative_data_sources: List[str] = Field(default_factory=lambda: [
        "news_sentiment", "social_media", "satellite_data"
    ])


class SecurityConfig(BaseModel):
    api_key_rotation: bool = True
    encryption_at_rest: bool = True
    audit_logging: bool = True


class DevelopmentConfig(BaseModel):
    jupyter_enabled: bool = True
    jupyter_port: int = 8888
    hot_reload: bool = True
    profiling: bool = False


class Config(BaseSettings):
    """Main configuration class for the Alpha Architecture Agent."""
    
    # Project metadata
    project_name: str = "alpha-architecture-agent"
    project_version: str = "0.1.0"
    project_description: str = "AI Agent-based Stock Prediction Architecture Explorer"
    
    # Component configurations
    database: DatabaseConfig = Field(default_factory=DatabaseConfig)
    redis: RedisConfig = Field(default_factory=RedisConfig)
    ml: MLConfig = Field(default_factory=MLConfig)
    agent: AgentConfig = Field(default_factory=AgentConfig)
    data: DataConfig = Field(default_factory=DataConfig)
    strategy: StrategyConfig = Field(default_factory=StrategyConfig)
    api: APIConfig = Field(default_factory=APIConfig)
    monitoring: MonitoringConfig = Field(default_factory=MonitoringConfig)
    mcp: MCPConfig = Field(default_factory=MCPConfig)
    security: SecurityConfig = Field(default_factory=SecurityConfig)
    development: DevelopmentConfig = Field(default_factory=DevelopmentConfig)
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        env_nested_delimiter = "__"
        case_sensitive = False


def load_config(config_path: Optional[str] = None) -> Config:
    """Load configuration from YAML file and environment variables."""
    
    if config_path is None:
        config_path = os.environ.get(
            "CONFIG_PATH", 
            str(Path(__file__).parent.parent.parent / "config" / "config.yaml")
        )
    
    config_dict = {}
    
    # Load from YAML file if it exists
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            config_dict = yaml.safe_load(f) or {}
    
    # Override with environment variables
    config = Config(**config_dict)
    
    # Update API keys from environment
    if os.getenv("OPENAI_API_KEY"):
        config.agent.llm.openai_api_key = os.getenv("OPENAI_API_KEY")
    if os.getenv("ANTHROPIC_API_KEY"):
        config.agent.llm.anthropic_api_key = os.getenv("ANTHROPIC_API_KEY")
    if os.getenv("JQUANTS_API_KEY"):
        config.data.api_key = os.getenv("JQUANTS_API_KEY")
    
    return config


# Global configuration instance
config = load_config()


def get_config() -> Config:
    """Get the global configuration instance."""
    return config


def reload_config(config_path: Optional[str] = None) -> Config:
    """Reload configuration from file."""
    global config
    config = load_config(config_path)
    return config