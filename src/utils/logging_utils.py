#!/usr/bin/env python3
"""
包括的ログシステム - データフロー追跡用
各段階での入出力、パス、DataFrameカラム、サイズ等を詳細ログ出力
"""

import logging
import json
import sys
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, Union, List
import traceback
import inspect
try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False

try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False

try:
    import pandas as pd
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


class DataFlowLogger:
    """データフロー追跡専用ロガー"""
    
    def __init__(self, 
                 name: str = "DataFlow",
                 log_file: Optional[str] = None,
                 log_level: int = logging.INFO,
                 enable_console: bool = True,
                 enable_file: bool = True,
                 log_dir: str = "logs"):
        """
        初期化
        
        Args:
            name: ロガー名
            log_file: ログファイル名 (Noneの場合は自動生成)
            log_level: ログレベル
            enable_console: コンソール出力有効
            enable_file: ファイル出力有効
            log_dir: ログディレクトリ
        """
        self.name = name
        self.logger = logging.getLogger(name)
        self.logger.setLevel(log_level)
        
        # 既存ハンドラーをクリア
        for handler in self.logger.handlers[:]:
            self.logger.removeHandler(handler)
        
        # フォーマッター設定
        formatter = logging.Formatter(
            '%(asctime)s | %(name)s | %(levelname)s | %(funcName)s:%(lineno)d | %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        
        # コンソールハンドラー
        if enable_console:
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setFormatter(formatter)
            console_handler.setLevel(log_level)
            self.logger.addHandler(console_handler)
        
        # ファイルハンドラー
        if enable_file:
            log_dir_path = Path(log_dir)
            log_dir_path.mkdir(exist_ok=True)
            
            if log_file is None:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                log_file = f"dataflow_{timestamp}.log"
            
            log_file_path = log_dir_path / log_file
            file_handler = logging.FileHandler(log_file_path, encoding='utf-8')
            file_handler.setFormatter(formatter)
            file_handler.setLevel(log_level)
            self.logger.addHandler(file_handler)
            
            self.log_file_path = str(log_file_path)
        else:
            self.log_file_path = None
        
        # セッション開始ログ
        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        self.log_session_start()
    
    def log_session_start(self):
        """セッション開始ログ"""
        system_info = {
            'session_id': self.session_id,
            'python_version': sys.version,
            'platform': sys.platform,
            'cwd': os.getcwd(),
            'pid': os.getpid(),
            'memory_usage_mb': psutil.Process().memory_info().rss / 1024 / 1024 if HAS_PSUTIL else 'N/A' if HAS_PSUTIL else 'N/A',
            'libraries': {
                'numpy': np.__version__ if HAS_NUMPY else 'Not Available',
                'pandas': pd.__version__ if HAS_PANDAS else 'Not Available', 
                'torch': torch.__version__ if HAS_TORCH else 'Not Available'
            }
        }
        
        self.logger.info("=" * 100)
        self.logger.info(f"DATA FLOW LOGGING SESSION START - {self.session_id}")
        self.logger.info("=" * 100)
        self.logger.info(f"SYSTEM_INFO: {json.dumps(system_info, indent=2, ensure_ascii=False)}")
    
    def get_caller_info(self) -> Dict[str, str]:
        """呼び出し元情報取得"""
        frame = inspect.currentframe()
        try:
            # 2つ上のフレーム（呼び出し元）を取得
            caller_frame = frame.f_back.f_back
            return {
                'file': caller_frame.f_code.co_filename,
                'function': caller_frame.f_code.co_name,
                'line': caller_frame.f_lineno
            }
        finally:
            del frame
    
    def analyze_data_object(self, obj: Any, name: str = "object") -> Dict[str, Any]:
        """データオブジェクト詳細分析"""
        analysis = {
            'name': name,
            'type': type(obj).__name__,
            'size_bytes': None,
            'shape': None,
            'columns': None,
            'dtype': None,
            'memory_usage': None,
            'null_count': None,
            'unique_count': None,
            'sample_values': None
        }
        
        try:
            # NumPy array
            if HAS_NUMPY and isinstance(obj, np.ndarray):
                analysis.update({
                    'shape': obj.shape,
                    'dtype': str(obj.dtype),
                    'size_bytes': obj.nbytes,
                    'memory_usage': f"{obj.nbytes / 1024 / 1024:.2f} MB",
                    'min_value': float(np.min(obj)) if obj.size > 0 else None,
                    'max_value': float(np.max(obj)) if obj.size > 0 else None,
                    'mean_value': float(np.mean(obj)) if obj.size > 0 else None,
                    'sample_values': obj.flatten()[:5].tolist() if obj.size > 0 else []
                })
            
            # Pandas DataFrame
            elif HAS_PANDAS and isinstance(obj, pd.DataFrame):
                analysis.update({
                    'shape': obj.shape,
                    'columns': obj.columns.tolist(),
                    'dtypes': obj.dtypes.to_dict(),
                    'size_bytes': obj.memory_usage(deep=True).sum(),
                    'memory_usage': f"{obj.memory_usage(deep=True).sum() / 1024 / 1024:.2f} MB",
                    'null_count': obj.isnull().sum().to_dict(),
                    'unique_count': obj.nunique().to_dict(),
                    'sample_rows': obj.head(3).to_dict('records') if len(obj) > 0 else []
                })
            
            # Pandas Series
            elif HAS_PANDAS and isinstance(obj, pd.Series):
                analysis.update({
                    'shape': obj.shape,
                    'dtype': str(obj.dtype),
                    'size_bytes': obj.memory_usage(deep=True),
                    'memory_usage': f"{obj.memory_usage(deep=True) / 1024 / 1024:.2f} MB",
                    'null_count': obj.isnull().sum(),
                    'unique_count': obj.nunique(),
                    'sample_values': obj.head(5).tolist() if len(obj) > 0 else []
                })
            
            # PyTorch Tensor
            elif HAS_TORCH and isinstance(obj, torch.Tensor):
                analysis.update({
                    'shape': list(obj.shape),
                    'dtype': str(obj.dtype),
                    'device': str(obj.device),
                    'requires_grad': obj.requires_grad,
                    'size_bytes': obj.element_size() * obj.nelement(),
                    'memory_usage': f"{obj.element_size() * obj.nelement() / 1024 / 1024:.2f} MB"
                })
            
            # リスト・タプル
            elif isinstance(obj, (list, tuple)):
                analysis.update({
                    'length': len(obj),
                    'sample_values': obj[:5] if len(obj) > 0 else [],
                    'element_types': list(set(type(item).__name__ for item in obj[:100]))
                })
            
            # 辞書
            elif isinstance(obj, dict):
                analysis.update({
                    'keys': list(obj.keys())[:10],  # 最初の10個のキー
                    'length': len(obj),
                    'value_types': list(set(type(value).__name__ for value in list(obj.values())[:100]))
                })
            
            # 文字列
            elif isinstance(obj, str):
                analysis.update({
                    'length': len(obj),
                    'sample_value': obj[:100] + "..." if len(obj) > 100 else obj
                })
            
            # その他
            else:
                analysis.update({
                    'value': str(obj)[:200] + "..." if len(str(obj)) > 200 else str(obj),
                    'size_bytes': sys.getsizeof(obj)
                })
                
        except Exception as e:
            analysis['analysis_error'] = str(e)
        
        return analysis
    
    def log_step(self, 
                 step_name: str,
                 inputs: Optional[Dict[str, Any]] = None,
                 outputs: Optional[Dict[str, Any]] = None,
                 parameters: Optional[Dict[str, Any]] = None,
                 file_paths: Optional[Dict[str, str]] = None,
                 metadata: Optional[Dict[str, Any]] = None,
                 level: int = logging.INFO):
        """
        処理ステップの詳細ログ
        
        Args:
            step_name: ステップ名
            inputs: 入力データ辞書
            outputs: 出力データ辞書  
            parameters: パラメータ辞書
            file_paths: ファイルパス辞書
            metadata: メタデータ
            level: ログレベル
        """
        caller_info = self.get_caller_info()
        
        step_log = {
            'session_id': self.session_id,
            'timestamp': datetime.now().isoformat(),
            'step_name': step_name,
            'caller': caller_info,
            'memory_usage_mb': psutil.Process().memory_info().rss / 1024 / 1024 if HAS_PSUTIL else 'N/A'
        }
        
        # 入力データ分析
        if inputs:
            step_log['inputs'] = {}
            for name, data in inputs.items():
                step_log['inputs'][name] = self.analyze_data_object(data, name)
        
        # 出力データ分析
        if outputs:
            step_log['outputs'] = {}
            for name, data in outputs.items():
                step_log['outputs'][name] = self.analyze_data_object(data, name)
        
        # パラメータ
        if parameters:
            step_log['parameters'] = parameters
        
        # ファイルパス
        if file_paths:
            step_log['file_paths'] = {}
            for name, path in file_paths.items():
                path_obj = Path(path)
                step_log['file_paths'][name] = {
                    'path': str(path),
                    'exists': path_obj.exists(),
                    'size_bytes': path_obj.stat().st_size if path_obj.exists() else None,
                    'size_mb': f"{path_obj.stat().st_size / 1024 / 1024:.2f} MB" if path_obj.exists() else None,
                    'modified': datetime.fromtimestamp(path_obj.stat().st_mtime).isoformat() if path_obj.exists() else None
                }
        
        # メタデータ
        if metadata:
            step_log['metadata'] = metadata
        
        # ログ出力
        self.logger.log(level, f"STEP: {step_name}")
        self.logger.log(level, f"STEP_DATA: {json.dumps(step_log, indent=2, ensure_ascii=False, default=str)}")
    
    def log_function_call(self, 
                         func_name: str,
                         args: Optional[tuple] = None,
                         kwargs: Optional[Dict[str, Any]] = None,
                         result: Any = None,
                         execution_time: Optional[float] = None,
                         error: Optional[Exception] = None):
        """関数呼び出しログ"""
        call_log = {
            'session_id': self.session_id,
            'timestamp': datetime.now().isoformat(),
            'function_name': func_name,
            'caller': self.get_caller_info(),
            'execution_time_seconds': execution_time,
            'memory_usage_mb': psutil.Process().memory_info().rss / 1024 / 1024 if HAS_PSUTIL else 'N/A'
        }
        
        # 引数分析
        if args:
            call_log['args'] = [self.analyze_data_object(arg, f"arg_{i}") for i, arg in enumerate(args)]
        
        if kwargs:
            call_log['kwargs'] = {k: self.analyze_data_object(v, k) for k, v in kwargs.items()}
        
        # 結果分析
        if result is not None:
            call_log['result'] = self.analyze_data_object(result, 'result')
        
        # エラー情報
        if error:
            call_log['error'] = {
                'type': type(error).__name__,
                'message': str(error),
                'traceback': traceback.format_exc()
            }
            level = logging.ERROR
        else:
            level = logging.INFO
        
        self.logger.log(level, f"FUNCTION_CALL: {func_name}")
        self.logger.log(level, f"CALL_DATA: {json.dumps(call_log, indent=2, ensure_ascii=False, default=str)}")
    
    def log_data_transformation(self,
                              transform_name: str,
                              input_data: Any,
                              output_data: Any,
                              transform_params: Optional[Dict[str, Any]] = None):
        """データ変換ログ"""
        transform_log = {
            'session_id': self.session_id,
            'timestamp': datetime.now().isoformat(),
            'transform_name': transform_name,
            'caller': self.get_caller_info(),
            'input_analysis': self.analyze_data_object(input_data, 'input'),
            'output_analysis': self.analyze_data_object(output_data, 'output'),
            'memory_usage_mb': psutil.Process().memory_info().rss / 1024 / 1024 if HAS_PSUTIL else 'N/A'
        }
        
        if transform_params:
            transform_log['parameters'] = transform_params
        
        # データ変化の分析
        try:
            if HAS_PANDAS and isinstance(input_data, pd.DataFrame) and isinstance(output_data, pd.DataFrame):
                transform_log['data_changes'] = {
                    'shape_change': f"{input_data.shape} -> {output_data.shape}",
                    'columns_added': list(set(output_data.columns) - set(input_data.columns)),
                    'columns_removed': list(set(input_data.columns) - set(output_data.columns)),
                    'size_change_mb': f"{(output_data.memory_usage(deep=True).sum() - input_data.memory_usage(deep=True).sum()) / 1024 / 1024:.2f} MB"
                }
        except Exception as e:
            transform_log['analysis_error'] = str(e)
        
        self.logger.info(f"DATA_TRANSFORM: {transform_name}")
        self.logger.info(f"TRANSFORM_DATA: {json.dumps(transform_log, indent=2, ensure_ascii=False, default=str)}")
    
    def log_file_operation(self,
                          operation: str,
                          file_path: str,
                          data: Optional[Any] = None,
                          success: bool = True,
                          error: Optional[Exception] = None):
        """ファイル操作ログ"""
        path_obj = Path(file_path)
        
        file_log = {
            'session_id': self.session_id,
            'timestamp': datetime.now().isoformat(),
            'operation': operation,
            'file_path': str(file_path),
            'file_info': {
                'exists': path_obj.exists(),
                'absolute_path': str(path_obj.absolute()),
                'parent_dir': str(path_obj.parent),
                'filename': path_obj.name,
                'extension': path_obj.suffix
            },
            'success': success,
            'caller': self.get_caller_info(),
            'memory_usage_mb': psutil.Process().memory_info().rss / 1024 / 1024 if HAS_PSUTIL else 'N/A'
        }
        
        if path_obj.exists():
            stat = path_obj.stat()
            file_log['file_info'].update({
                'size_bytes': stat.st_size,
                'size_mb': f"{stat.st_size / 1024 / 1024:.2f} MB",
                'modified': datetime.fromtimestamp(stat.st_mtime).isoformat(),
                'created': datetime.fromtimestamp(stat.st_ctime).isoformat()
            })
        
        if data is not None:
            file_log['data_analysis'] = self.analyze_data_object(data, 'file_data')
        
        if error:
            file_log['error'] = {
                'type': type(error).__name__,
                'message': str(error),
                'traceback': traceback.format_exc()
            }
            level = logging.ERROR
        else:
            level = logging.INFO
        
        self.logger.log(level, f"FILE_OPERATION: {operation} - {file_path}")
        self.logger.log(level, f"FILE_DATA: {json.dumps(file_log, indent=2, ensure_ascii=False, default=str)}")


# グローバルロガーインスタンス
_global_logger = None

def get_dataflow_logger(name: str = "DataFlow") -> DataFlowLogger:
    """グローバルデータフローロガー取得"""
    global _global_logger
    if _global_logger is None:
        _global_logger = DataFlowLogger(name)
    return _global_logger

def log_step(step_name: str, **kwargs):
    """ステップログのショートカット"""
    logger = get_dataflow_logger()
    logger.log_step(step_name, **kwargs)

def log_function_call(func_name: str, **kwargs):
    """関数呼び出しログのショートカット"""
    logger = get_dataflow_logger()
    logger.log_function_call(func_name, **kwargs)

def log_data_transformation(transform_name: str, input_data: Any, output_data: Any, **kwargs):
    """データ変換ログのショートカット"""
    logger = get_dataflow_logger()
    logger.log_data_transformation(transform_name, input_data, output_data, **kwargs)

def log_file_operation(operation: str, file_path: str, **kwargs):
    """ファイル操作ログのショートカット"""
    logger = get_dataflow_logger()
    logger.log_file_operation(operation, file_path, **kwargs)


# デコレータ
def log_dataflow(step_name: Optional[str] = None):
    """データフロー自動ログデコレータ"""
    def decorator(func):
        def wrapper(*args, **kwargs):
            import time
            
            func_step_name = step_name or f"{func.__module__}.{func.__name__}"
            start_time = time.time()
            
            try:
                result = func(*args, **kwargs)
                execution_time = time.time() - start_time
                
                log_function_call(
                    func_name=func_step_name,
                    args=args,
                    kwargs=kwargs,
                    result=result,
                    execution_time=execution_time
                )
                
                return result
            
            except Exception as e:
                execution_time = time.time() - start_time
                
                log_function_call(
                    func_name=func_step_name,
                    args=args,
                    kwargs=kwargs,
                    execution_time=execution_time,
                    error=e
                )
                
                raise
        
        return wrapper
    return decorator


if __name__ == "__main__":
    # テスト実行
    print("データフローロガーテスト開始")
    
    logger = DataFlowLogger("TestLogger")
    
    # テストデータ
    if HAS_NUMPY and HAS_PANDAS:
        test_array = np.random.randn(100, 10)
        test_df = pd.DataFrame(test_array, columns=[f"col_{i}" for i in range(10)])
        
        logger.log_step(
            "テストステップ",
            inputs={'test_array': test_array},
            outputs={'test_df': test_df},
            parameters={'shape': (100, 10), 'random_seed': 42},
            file_paths={'output': '/tmp/test.csv'},
            metadata={'test_run': True}
        )
    
    print("テスト完了")