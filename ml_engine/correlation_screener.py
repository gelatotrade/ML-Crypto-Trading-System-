"""
Correlation Screener - Prevents longing highly correlated assets
Ensures portfolio diversification by screening correlation clusters
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Set, Any
from dataclasses import dataclass
import logging
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform

logger = logging.getLogger(__name__)


@dataclass
class CorrelationCluster:
    """Container for correlation cluster information"""
    cluster_id: int
    assets: List[str]
    avg_correlation: float
    representative_asset: str  # Best asset to represent the cluster


@dataclass
class ScreeningResult:
    """Result of correlation screening"""
    allowed_positions: Dict[str, str]  # asset -> allowed direction ('long', 'short', 'both', 'blocked')
    blocked_pairs: List[Tuple[str, str, float]]  # (asset1, asset2, correlation)
    clusters: List[CorrelationCluster]
    correlation_matrix: pd.DataFrame
    warnings: List[str]


class CorrelationScreener:
    """
    Correlation screener to prevent taking same-direction positions
    in highly correlated assets.

    Key Features:
    - Real-time correlation monitoring
    - Cluster detection for asset groups
    - Position direction constraints
    - Rolling correlation analysis
    - Regime-aware correlation adjustments
    """

    def __init__(
        self,
        max_correlation: float = 0.7,
        lookback_periods: int = 168,  # 1 week of hourly data
        min_periods: int = 24,
        cluster_threshold: float = 0.6,
        config=None
    ):
        """
        Initialize correlation screener.

        Args:
            max_correlation: Maximum allowed correlation for same-direction positions
            lookback_periods: Number of periods for correlation calculation
            min_periods: Minimum periods required for correlation
            cluster_threshold: Threshold for clustering highly correlated assets
            config: Optional configuration object
        """
        self.max_correlation = max_correlation
        self.lookback_periods = lookback_periods
        self.min_periods = min_periods
        self.cluster_threshold = cluster_threshold
        self.config = config

        self._correlation_cache: Optional[pd.DataFrame] = None
        self._cache_timestamp: Optional[pd.Timestamp] = None

    def calculate_correlation_matrix(
        self,
        returns_df: pd.DataFrame,
        method: str = 'pearson'
    ) -> pd.DataFrame:
        """
        Calculate correlation matrix from returns.

        Args:
            returns_df: DataFrame with returns for each asset (columns)
            method: 'pearson', 'spearman', or 'kendall'

        Returns:
            Correlation matrix
        """
        # Use recent data based on lookback
        recent_returns = returns_df.iloc[-self.lookback_periods:]

        if len(recent_returns) < self.min_periods:
            logger.warning(f"Insufficient data for correlation: {len(recent_returns)} < {self.min_periods}")
            return pd.DataFrame()

        corr_matrix = recent_returns.corr(method=method, min_periods=self.min_periods)

        # Cache the result
        self._correlation_cache = corr_matrix
        self._cache_timestamp = pd.Timestamp.now()

        return corr_matrix

    def screen_positions(
        self,
        proposed_positions: Dict[str, float],
        returns_df: pd.DataFrame,
        existing_positions: Optional[Dict[str, float]] = None
    ) -> ScreeningResult:
        """
        Screen proposed positions for correlation conflicts.

        Args:
            proposed_positions: Dict of {asset: proposed_weight} (positive=long, negative=short)
            returns_df: Historical returns DataFrame
            existing_positions: Current portfolio positions

        Returns:
            ScreeningResult with allowed positions and blocked pairs
        """
        # Calculate correlation matrix
        corr_matrix = self.calculate_correlation_matrix(returns_df)

        if corr_matrix.empty:
            return ScreeningResult(
                allowed_positions={a: 'both' for a in proposed_positions},
                blocked_pairs=[],
                clusters=[],
                correlation_matrix=corr_matrix,
                warnings=["Insufficient data for correlation screening"]
            )

        # Identify correlation clusters
        clusters = self._identify_clusters(corr_matrix)

        # Screen for conflicts
        blocked_pairs = []
        warnings = []
        allowed_positions = {}

        # Combine existing and proposed positions
        all_positions = {}
        if existing_positions:
            all_positions.update(existing_positions)
        all_positions.update(proposed_positions)

        # Get long and short assets
        long_assets = [a for a, w in all_positions.items() if w > 0]
        short_assets = [a for a, w in all_positions.items() if w < 0]

        # Check for highly correlated longs
        blocked_longs = set()
        for i, asset1 in enumerate(long_assets):
            for asset2 in long_assets[i+1:]:
                if asset1 in corr_matrix.index and asset2 in corr_matrix.columns:
                    corr = corr_matrix.loc[asset1, asset2]
                    if abs(corr) > self.max_correlation:
                        blocked_pairs.append((asset1, asset2, corr))
                        # Block the asset with lower expected return or smaller position
                        if abs(all_positions.get(asset1, 0)) < abs(all_positions.get(asset2, 0)):
                            blocked_longs.add(asset1)
                        else:
                            blocked_longs.add(asset2)
                        warnings.append(
                            f"High correlation ({corr:.2f}) between long positions: {asset1} & {asset2}"
                        )

        # Check for highly correlated shorts
        blocked_shorts = set()
        for i, asset1 in enumerate(short_assets):
            for asset2 in short_assets[i+1:]:
                if asset1 in corr_matrix.index and asset2 in corr_matrix.columns:
                    corr = corr_matrix.loc[asset1, asset2]
                    if abs(corr) > self.max_correlation:
                        blocked_pairs.append((asset1, asset2, corr))
                        if abs(all_positions.get(asset1, 0)) < abs(all_positions.get(asset2, 0)):
                            blocked_shorts.add(asset1)
                        else:
                            blocked_shorts.add(asset2)
                        warnings.append(
                            f"High correlation ({corr:.2f}) between short positions: {asset1} & {asset2}"
                        )

        # Determine allowed directions for each asset
        for asset in proposed_positions:
            if asset in blocked_longs and asset in blocked_shorts:
                allowed_positions[asset] = 'blocked'
            elif asset in blocked_longs:
                allowed_positions[asset] = 'short'  # Can only short
            elif asset in blocked_shorts:
                allowed_positions[asset] = 'long'   # Can only long
            else:
                allowed_positions[asset] = 'both'

        return ScreeningResult(
            allowed_positions=allowed_positions,
            blocked_pairs=blocked_pairs,
            clusters=clusters,
            correlation_matrix=corr_matrix,
            warnings=warnings
        )

    def filter_correlated_signals(
        self,
        signals: Dict[str, float],
        returns_df: pd.DataFrame,
        keep_strongest: bool = True
    ) -> Dict[str, float]:
        """
        Filter trading signals to remove highly correlated same-direction positions.

        Args:
            signals: Dict of {asset: signal_strength} (positive=long, negative=short)
            returns_df: Historical returns DataFrame
            keep_strongest: If True, keep strongest signal in correlated group

        Returns:
            Filtered signals dictionary
        """
        if not signals:
            return signals

        corr_matrix = self.calculate_correlation_matrix(returns_df)

        if corr_matrix.empty:
            return signals

        # Separate long and short signals
        long_signals = {a: s for a, s in signals.items() if s > 0}
        short_signals = {a: s for a, s in signals.items() if s < 0}

        # Filter long signals
        filtered_longs = self._filter_correlated_group(
            long_signals, corr_matrix, keep_strongest
        )

        # Filter short signals
        filtered_shorts = self._filter_correlated_group(
            short_signals, corr_matrix, keep_strongest
        )

        # Combine filtered signals
        filtered = {}
        filtered.update(filtered_longs)
        filtered.update(filtered_shorts)

        removed = set(signals.keys()) - set(filtered.keys())
        if removed:
            logger.info(f"Correlation screener removed signals: {removed}")

        return filtered

    def _filter_correlated_group(
        self,
        signals: Dict[str, float],
        corr_matrix: pd.DataFrame,
        keep_strongest: bool
    ) -> Dict[str, float]:
        """Filter a group of same-direction signals for correlation"""
        if len(signals) <= 1:
            return signals

        assets = list(signals.keys())
        filtered = dict(signals)
        removed = set()

        # Check all pairs
        for i, asset1 in enumerate(assets):
            if asset1 in removed:
                continue
            for asset2 in assets[i+1:]:
                if asset2 in removed:
                    continue
                if asset1 in corr_matrix.index and asset2 in corr_matrix.columns:
                    corr = abs(corr_matrix.loc[asset1, asset2])
                    if corr > self.max_correlation:
                        # Remove the weaker signal
                        if keep_strongest:
                            if abs(signals[asset1]) >= abs(signals[asset2]):
                                removed.add(asset2)
                                filtered.pop(asset2, None)
                            else:
                                removed.add(asset1)
                                filtered.pop(asset1, None)
                                break
                        else:
                            # Remove both
                            removed.add(asset1)
                            removed.add(asset2)
                            filtered.pop(asset1, None)
                            filtered.pop(asset2, None)
                            break

        return filtered

    def _identify_clusters(
        self,
        corr_matrix: pd.DataFrame
    ) -> List[CorrelationCluster]:
        """
        Identify clusters of highly correlated assets using hierarchical clustering.
        """
        if corr_matrix.empty or len(corr_matrix) < 2:
            return []

        try:
            # Convert correlation to distance
            # Distance = 1 - |correlation|
            distance_matrix = 1 - corr_matrix.abs()
            np.fill_diagonal(distance_matrix.values, 0)

            # Hierarchical clustering
            condensed_dist = squareform(distance_matrix.values)
            linkage_matrix = linkage(condensed_dist, method='average')

            # Form clusters based on threshold
            cluster_labels = fcluster(
                linkage_matrix,
                t=1 - self.cluster_threshold,  # Convert correlation threshold to distance
                criterion='distance'
            )

            # Build cluster objects
            clusters = []
            assets = list(corr_matrix.index)

            for cluster_id in np.unique(cluster_labels):
                cluster_assets = [assets[i] for i, label in enumerate(cluster_labels) if label == cluster_id]

                if len(cluster_assets) > 1:  # Only include actual clusters
                    # Calculate average intra-cluster correlation
                    cluster_corr = corr_matrix.loc[cluster_assets, cluster_assets]
                    mask = np.triu(np.ones_like(cluster_corr, dtype=bool), k=1)
                    avg_corr = cluster_corr.where(mask).stack().mean()

                    # Find representative asset (highest average correlation with others)
                    avg_corrs = cluster_corr.mean()
                    representative = avg_corrs.idxmax()

                    clusters.append(CorrelationCluster(
                        cluster_id=int(cluster_id),
                        assets=cluster_assets,
                        avg_correlation=avg_corr,
                        representative_asset=representative
                    ))

            return clusters

        except Exception as e:
            logger.warning(f"Clustering failed: {e}")
            return []

    def get_correlation_pairs(
        self,
        corr_matrix: pd.DataFrame,
        threshold: Optional[float] = None
    ) -> List[Tuple[str, str, float]]:
        """
        Get all asset pairs with correlation above threshold.

        Args:
            corr_matrix: Correlation matrix
            threshold: Correlation threshold (defaults to max_correlation)

        Returns:
            List of (asset1, asset2, correlation) tuples
        """
        if threshold is None:
            threshold = self.max_correlation

        pairs = []
        assets = list(corr_matrix.index)

        for i, asset1 in enumerate(assets):
            for asset2 in assets[i+1:]:
                corr = corr_matrix.loc[asset1, asset2]
                if abs(corr) > threshold:
                    pairs.append((asset1, asset2, corr))

        return sorted(pairs, key=lambda x: abs(x[2]), reverse=True)

    def calculate_rolling_correlation(
        self,
        returns_df: pd.DataFrame,
        asset1: str,
        asset2: str,
        window: int = 24
    ) -> pd.Series:
        """
        Calculate rolling correlation between two assets.

        Args:
            returns_df: Returns DataFrame
            asset1: First asset
            asset2: Second asset
            window: Rolling window size

        Returns:
            Series of rolling correlations
        """
        if asset1 not in returns_df.columns or asset2 not in returns_df.columns:
            return pd.Series()

        return returns_df[asset1].rolling(window).corr(returns_df[asset2])

    def get_diversification_score(
        self,
        weights: Dict[str, float],
        corr_matrix: pd.DataFrame
    ) -> float:
        """
        Calculate portfolio diversification score (0-1, higher is better).

        Args:
            weights: Portfolio weights
            corr_matrix: Correlation matrix

        Returns:
            Diversification score
        """
        if not weights or corr_matrix.empty:
            return 0.0

        assets = [a for a in weights.keys() if a in corr_matrix.index]
        if len(assets) < 2:
            return 1.0

        # Calculate weighted average correlation
        total_weight = sum(abs(weights[a]) for a in assets)
        if total_weight == 0:
            return 1.0

        weighted_corr = 0.0
        weight_sum = 0.0

        for i, asset1 in enumerate(assets):
            for asset2 in assets[i+1:]:
                w1 = abs(weights[asset1]) / total_weight
                w2 = abs(weights[asset2]) / total_weight
                corr = abs(corr_matrix.loc[asset1, asset2])
                weighted_corr += w1 * w2 * corr
                weight_sum += w1 * w2

        if weight_sum == 0:
            return 1.0

        avg_corr = weighted_corr / weight_sum

        # Diversification score = 1 - average correlation
        return 1.0 - avg_corr

    def suggest_hedges(
        self,
        position: str,
        direction: str,
        corr_matrix: pd.DataFrame,
        min_negative_corr: float = -0.3
    ) -> List[Tuple[str, float]]:
        """
        Suggest hedge positions based on negative correlations.

        Args:
            position: Current position asset
            direction: 'long' or 'short'
            corr_matrix: Correlation matrix
            min_negative_corr: Minimum negative correlation for hedge

        Returns:
            List of (asset, correlation) tuples for potential hedges
        """
        if position not in corr_matrix.index:
            return []

        correlations = corr_matrix.loc[position]

        # Find negatively correlated assets
        hedges = []
        for asset, corr in correlations.items():
            if asset != position and corr < min_negative_corr:
                hedges.append((asset, corr))

        return sorted(hedges, key=lambda x: x[1])

    def get_correlation_report(
        self,
        returns_df: pd.DataFrame,
        positions: Optional[Dict[str, float]] = None
    ) -> Dict[str, Any]:
        """
        Generate a comprehensive correlation report.

        Args:
            returns_df: Returns DataFrame
            positions: Optional current positions

        Returns:
            Dictionary with correlation analysis
        """
        corr_matrix = self.calculate_correlation_matrix(returns_df)

        report = {
            'correlation_matrix': corr_matrix,
            'high_correlation_pairs': self.get_correlation_pairs(corr_matrix),
            'clusters': self._identify_clusters(corr_matrix),
            'average_correlation': corr_matrix.where(
                np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)
            ).stack().mean() if not corr_matrix.empty else 0,
        }

        if positions:
            report['diversification_score'] = self.get_diversification_score(positions, corr_matrix)
            screening = self.screen_positions(positions, returns_df)
            report['screening_result'] = screening

        return report
