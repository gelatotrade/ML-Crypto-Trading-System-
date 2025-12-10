"""
ML Engine - Core Machine Learning Trading Components

Optimiertes ML-Trading-System basierend auf der Adaptiven Markthypothese (AMH)
mit mehrschichtiger Pipeline-Architektur.
"""

from .config import Config
from .data_collector import DataCollector
from .market_data_aggregator import MarketDataAggregator
from .feature_engineering import FeatureEngineer
from .risk_models import RiskModels
from .portfolio_optimizer import PortfolioOptimizer
from .ml_models import MLModels
from .signal_generator import SignalGenerator
from .execution_engine import ExecutionEngine
from .position_manager import PositionManager
from .margin_manager import MarginManager
from .options_flow import OptionsFlowAnalyzer
from .options_data_collector import OptionsDataCollector
from .regime_detector import RegimeDetector
from .correlation_screener import CorrelationScreener

# Inflation/Deflation Strategy Components
from .token_unlock_pipeline import TokenUnlockPipeline, UnlockRisk, UnlockEvent
from .buyback_analyzer import BuybackAnalyzer, BuybackSentiment, BuybackEvent
from .orderbook_pipeline import OrderbookPipeline, Exchange, OrderbookSnapshot
from .inflation_strategy import InflationDeflationStrategy, InflationSignal

# Drift Protection
from .drift_protection import (
    DriftProtectionPipeline,
    DriftConfig,
    DriftSeverity,
    ModelAction,
    DriftReport
)

# ═══════════════════════════════════════════════════════════════════════════════
# ERWEITERTE PIPELINE-ARCHITEKTUR
# ═══════════════════════════════════════════════════════════════════════════════

# 0. META-PIPELINE: System-Steuerung & Adaptivität
from .meta_pipeline import (
    MetaPipeline,
    HMMRegimeDetector,
    HMMRegimeState,
    BayesianHyperparameterOptimizer,
    CapitalAllocationEngine,
    StrategyPerformance,
    CapitalAllocation,
    MetaPipelineState
)

# 1. VERFEINERTE DATEN-PIPELINE
from .advanced_data_pipeline import (
    AdvancedDataPipeline,
    MicrostructureAnalyzer,
    MicrostructureFeatures,
    OnlineOutlierDetector,
    NLPSentimentPipeline,
    SentimentAnalysis,
    SABRVolatilitySurface,
    VolatilitySurface,
    StationarityTransformer,
    TradeEvent,
    ProcessedFeatures
)

# 2. TIEFGEHENDE ALPHA-PIPELINE
from .alpha_pipeline import (
    AlphaPipeline,
    HierarchicalAlphaModel,
    MomentumFeatures,
    MeanReversionFeatures,
    CarryFeatures,
    GaussianProcessRegressor,
    QuantileRandomForest,
    CausalInferenceEngine,
    AlphaPrediction,
    GPPrediction,
    CausalEffect
)

# 3. FORTGESCHRITTENE RISIKO-PIPELINE
from .advanced_risk_pipeline import (
    AdvancedRiskPipeline,
    KalmanFilterBeta,
    MultiFactorRiskModel,
    CornishFisherVaR,
    CopulaRiskModel,
    LiquidityAdjustedRisk,
    FactorExposure,
    RiskDecomposition,
    VaREstimate,
    LiquidityRisk,
    CopulaMetrics,
    ComprehensiveRiskReport
)

# 4. OPTIMIERTE PORTFOLIO-KONSTRUKTION
from .advanced_portfolio_optimizer import (
    AdvancedPortfolioPipeline,
    BlackLittermanOptimizer,
    BlackLittermanView,
    OnlineConvexOptimizer,
    TransactionCostOptimizer,
    TransactionCostModel,
    AlmgrenChrissExecutor,
    RobustPortfolioOptimizer,
    PortfolioWeights,
    TradeSchedule
)

# 5. HOCHLEISTUNGS-AUSFÜHRUNG
from .execution_pipeline import (
    MultiAgentExecutionSystem,
    MarketMakerAgent,
    SmartOrderRouter,
    ExecutionManager,
    RLExecutionAgent,
    ExecutionState,
    ExecutionAction,
    ExecutionResult,
    AgentRecommendation,
    OrderType,
    ExecutionStyle,
    Venue
)

# 6. UMFASSENDE ÜBERWACHUNGS-PIPELINE
from .monitoring_pipeline import (
    MonitoringPipeline,
    PerformanceAttributor,
    PerformanceAttribution,
    PBOAnalyzer,
    PBOResult,
    StressTestingFramework,
    StressScenario,
    StressTestResult,
    ModelDiagnosticsEngine,
    ModelDiagnostics
)

# 7. FORTSCHRITTLICHE ML-OPTIMIERUNGEN
from .advanced_ml_optimizations import (
    AdvancedMLPipeline,
    MAMLOptimizer,
    MAMLTask,
    BayesianNeuralNetwork,
    EvidentialNeuralNetwork,
    UncertaintyEstimate,
    AdvancedCausalInference,
    CausalEstimate
)

__all__ = [
    # ═══════════════════════════════════════════════════════════════════════════
    # CORE COMPONENTS
    # ═══════════════════════════════════════════════════════════════════════════
    'Config',
    'DataCollector',
    'MarketDataAggregator',
    'FeatureEngineer',
    'RiskModels',
    'PortfolioOptimizer',
    'MLModels',
    'SignalGenerator',
    'ExecutionEngine',
    'PositionManager',
    'MarginManager',
    'OptionsFlowAnalyzer',
    'OptionsDataCollector',
    'RegimeDetector',
    'CorrelationScreener',

    # ═══════════════════════════════════════════════════════════════════════════
    # INFLATION/DEFLATION STRATEGY
    # ═══════════════════════════════════════════════════════════════════════════
    'TokenUnlockPipeline',
    'UnlockRisk',
    'UnlockEvent',
    'BuybackAnalyzer',
    'BuybackSentiment',
    'BuybackEvent',
    'OrderbookPipeline',
    'Exchange',
    'OrderbookSnapshot',
    'InflationDeflationStrategy',
    'InflationSignal',

    # ═══════════════════════════════════════════════════════════════════════════
    # DRIFT PROTECTION
    # ═══════════════════════════════════════════════════════════════════════════
    'DriftProtectionPipeline',
    'DriftConfig',
    'DriftSeverity',
    'ModelAction',
    'DriftReport',

    # ═══════════════════════════════════════════════════════════════════════════
    # 0. META-PIPELINE
    # ═══════════════════════════════════════════════════════════════════════════
    'MetaPipeline',
    'HMMRegimeDetector',
    'HMMRegimeState',
    'BayesianHyperparameterOptimizer',
    'CapitalAllocationEngine',
    'StrategyPerformance',
    'CapitalAllocation',
    'MetaPipelineState',

    # ═══════════════════════════════════════════════════════════════════════════
    # 1. ADVANCED DATA PIPELINE
    # ═══════════════════════════════════════════════════════════════════════════
    'AdvancedDataPipeline',
    'MicrostructureAnalyzer',
    'MicrostructureFeatures',
    'OnlineOutlierDetector',
    'NLPSentimentPipeline',
    'SentimentAnalysis',
    'SABRVolatilitySurface',
    'VolatilitySurface',
    'StationarityTransformer',
    'TradeEvent',
    'ProcessedFeatures',

    # ═══════════════════════════════════════════════════════════════════════════
    # 2. ALPHA PIPELINE
    # ═══════════════════════════════════════════════════════════════════════════
    'AlphaPipeline',
    'HierarchicalAlphaModel',
    'MomentumFeatures',
    'MeanReversionFeatures',
    'CarryFeatures',
    'GaussianProcessRegressor',
    'QuantileRandomForest',
    'CausalInferenceEngine',
    'AlphaPrediction',
    'GPPrediction',
    'CausalEffect',

    # ═══════════════════════════════════════════════════════════════════════════
    # 3. ADVANCED RISK PIPELINE
    # ═══════════════════════════════════════════════════════════════════════════
    'AdvancedRiskPipeline',
    'KalmanFilterBeta',
    'MultiFactorRiskModel',
    'CornishFisherVaR',
    'CopulaRiskModel',
    'LiquidityAdjustedRisk',
    'FactorExposure',
    'RiskDecomposition',
    'VaREstimate',
    'LiquidityRisk',
    'CopulaMetrics',
    'ComprehensiveRiskReport',

    # ═══════════════════════════════════════════════════════════════════════════
    # 4. ADVANCED PORTFOLIO OPTIMIZER
    # ═══════════════════════════════════════════════════════════════════════════
    'AdvancedPortfolioPipeline',
    'BlackLittermanOptimizer',
    'BlackLittermanView',
    'OnlineConvexOptimizer',
    'TransactionCostOptimizer',
    'TransactionCostModel',
    'AlmgrenChrissExecutor',
    'RobustPortfolioOptimizer',
    'PortfolioWeights',
    'TradeSchedule',

    # ═══════════════════════════════════════════════════════════════════════════
    # 5. EXECUTION PIPELINE
    # ═══════════════════════════════════════════════════════════════════════════
    'MultiAgentExecutionSystem',
    'MarketMakerAgent',
    'SmartOrderRouter',
    'ExecutionManager',
    'RLExecutionAgent',
    'ExecutionState',
    'ExecutionAction',
    'ExecutionResult',
    'AgentRecommendation',
    'OrderType',
    'ExecutionStyle',
    'Venue',

    # ═══════════════════════════════════════════════════════════════════════════
    # 6. MONITORING PIPELINE
    # ═══════════════════════════════════════════════════════════════════════════
    'MonitoringPipeline',
    'PerformanceAttributor',
    'PerformanceAttribution',
    'PBOAnalyzer',
    'PBOResult',
    'StressTestingFramework',
    'StressScenario',
    'StressTestResult',
    'ModelDiagnosticsEngine',
    'ModelDiagnostics',

    # ═══════════════════════════════════════════════════════════════════════════
    # 7. ADVANCED ML OPTIMIZATIONS
    # ═══════════════════════════════════════════════════════════════════════════
    'AdvancedMLPipeline',
    'MAMLOptimizer',
    'MAMLTask',
    'BayesianNeuralNetwork',
    'EvidentialNeuralNetwork',
    'UncertaintyEstimate',
    'AdvancedCausalInference',
    'CausalEstimate',
]
