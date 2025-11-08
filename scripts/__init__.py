# Scripts package initialization
from .evaluate import LyricEvaluator, LyricMetrics, compare_lyrics
from .sentiment_analysis import SentimentAnalyzer, EthosDataManager
from .database import Database

__all__ = [
    "LyricEvaluator",
    "LyricMetrics",
    "compare_lyrics",
    "SentimentAnalyzer",
    "EthosDataManager",
    "Database",
]
