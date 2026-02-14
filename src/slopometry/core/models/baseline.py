"""Baseline computation, impact assessment, and statistical models."""

from datetime import datetime
from enum import Enum

from pydantic import BaseModel, ConfigDict, Field, field_validator

from slopometry.core.models.complexity import ComplexityDelta, ExtendedComplexityMetrics
from slopometry.core.models.hook import AnalysisSource
from slopometry.core.models.smell import SmellCounts


class BaselineStrategy(str, Enum):
    """How to select commits for building the historic quality baseline.

    MERGE_ANCHORED: Follows first-parent (trunk) history, so each delta represents
    the net quality effect of one accepted merge/PR. Best for repos using merge
    workflows where merges are quality checkpoints (code review happened).

    TIME_SAMPLED: Samples commits at regular time intervals within a bounded
    lookback window. Prevents the 'N commits = 2 days' problem in active repos.
    Best for repos with linear history (squash merges, rebase workflows).

    AUTO: Examines recent commit history to compute merge ratio. If merges are
    frequent enough (above configurable threshold), uses MERGE_ANCHORED.
    Otherwise falls back to TIME_SAMPLED.
    """

    MERGE_ANCHORED = "merge_anchored"
    TIME_SAMPLED = "time_sampled"
    AUTO = "auto"


class ResolvedBaselineStrategy(BaseModel):
    """Records which baseline strategy was actually used after AUTO resolution.

    AUTO never appears as the resolved strategy -- it always resolves to one of
    the concrete strategies. This model is stored with the cached baseline so
    we can invalidate the cache when the user changes strategy settings.
    """

    model_config = ConfigDict(frozen=True)

    requested: BaselineStrategy = Field(description="Strategy requested via settings (may be AUTO)")
    resolved: BaselineStrategy = Field(
        description="Concrete strategy actually used (never AUTO). "
        "MERGE_ANCHORED uses first-parent trunk history at merge points. "
        "TIME_SAMPLED samples commits at regular time intervals within a bounded lookback window."
    )
    merge_ratio: float = Field(
        description="Fraction of merge commits in the detection sample (0.0-1.0). "
        "Used by AUTO to decide strategy: above threshold -> MERGE_ANCHORED, below -> TIME_SAMPLED."
    )
    total_commits_sampled: int = Field(description="Number of recent commits examined during strategy auto-detection")

    @field_validator("resolved")
    @classmethod
    def resolved_must_be_concrete(cls, v: BaselineStrategy) -> BaselineStrategy:
        if v == BaselineStrategy.AUTO:
            raise ValueError("resolved strategy cannot be AUTO")
        return v


class HistoricalMetricStats(BaseModel):
    """Statistical summary of a metric across repository history."""

    metric_name: str = Field(description="Name of the metric (e.g., 'cc_delta', 'effort_delta')")
    mean: float = Field(description="Mean value across all commits")
    std_dev: float = Field(description="Standard deviation")
    median: float = Field(description="Median value")
    min_value: float = Field(description="Minimum observed value")
    max_value: float = Field(description="Maximum observed value")
    sample_count: int = Field(description="Number of commits analyzed")
    trend_coefficient: float = Field(
        default=0.0, description="Linear regression slope indicating improvement/degradation trend"
    )


GALEN_TOKENS_PER_MONTH = 1_000_000
GALEN_TOKENS_PER_DAY = GALEN_TOKENS_PER_MONTH / 30  # ~33,333 tokens/day


class GalenMetrics(BaseModel):
    """Developer productivity metrics based on code token throughput.

    Named after Galen Hunt at Microsoft, who calculated that rewriting C++ to Rust
    requires approximately 1 million tokens per developer per month.

    1 Galen = 1 million code tokens per developer per month.
    Based on Microsoft's calculation for C++ to Rust migration effort.
    """

    tokens_changed: int = Field(description="Net tokens changed during the period")
    period_days: float = Field(description="Duration of the analysis period in days")
    tokens_per_day: float = Field(description="Average tokens changed per day")
    galen_rate: float = Field(description="Productivity rate (1.0 = on track for 1M tokens/month)")
    tokens_per_day_to_reach_one_galen: float | None = Field(
        default=None,
        description="Additional tokens/day needed to reach 1 Galen (None if already >= 1 Galen)",
    )

    @classmethod
    def calculate(cls, tokens_changed: int, period_days: float) -> "GalenMetrics":
        """Calculate Galen metrics from token change and time period."""
        if period_days <= 0:
            return cls(
                tokens_changed=tokens_changed,
                period_days=0.0,
                tokens_per_day=0.0,
                galen_rate=0.0,
                tokens_per_day_to_reach_one_galen=GALEN_TOKENS_PER_DAY,
            )

        tokens_per_day = abs(tokens_changed) / period_days
        galen_rate = tokens_per_day / GALEN_TOKENS_PER_DAY

        tokens_needed = None
        if galen_rate < 1.0:
            tokens_needed = GALEN_TOKENS_PER_DAY - tokens_per_day

        return cls(
            tokens_changed=tokens_changed,
            period_days=period_days,
            tokens_per_day=tokens_per_day,
            galen_rate=galen_rate,
            tokens_per_day_to_reach_one_galen=tokens_needed,
        )


class RepoBaseline(BaseModel):
    """Baseline statistics computed from entire repository history.

    Delta statistics use TOTAL metrics (sums across all files), not averages.
    This allows correct comparison even when baseline is computed from only
    changed files per commit (an optimization that works because sums are additive).
    """

    repository_path: str = Field(description="Absolute path to the repository")
    computed_at: datetime = Field(default_factory=datetime.now)
    head_commit_sha: str = Field(description="HEAD commit when baseline was computed")
    total_commits_analyzed: int = Field(description="Number of commits in baseline calculation")

    cc_delta_stats: HistoricalMetricStats = Field(description="Total CC delta statistics per commit")
    effort_delta_stats: HistoricalMetricStats = Field(description="Total Effort delta statistics per commit")
    mi_delta_stats: HistoricalMetricStats = Field(description="Total MI delta statistics per commit")

    current_metrics: "ExtendedComplexityMetrics" = Field(description="Metrics at HEAD commit")

    oldest_commit_date: datetime | None = Field(
        default=None, description="Timestamp of the oldest commit in the analysis"
    )
    newest_commit_date: datetime | None = Field(
        default=None, description="Timestamp of the newest commit in the analysis"
    )
    oldest_commit_tokens: int | None = Field(
        default=None, description="Total tokens in codebase at oldest analyzed commit"
    )

    qpe_stats: HistoricalMetricStats | None = Field(default=None, description="QPE statistics from commit history")
    current_qpe: "QPEScore | None" = Field(default=None, description="QPE score at HEAD")
    token_delta_stats: HistoricalMetricStats | None = Field(
        default=None, description="Token delta statistics from commit history"
    )

    strategy: ResolvedBaselineStrategy | None = Field(
        default=None,
        description="Which baseline computation strategy produced this baseline. "
        "None for legacy baselines computed before strategy support was added. "
        "Used for cache invalidation: strategy mismatch with current settings triggers recomputation.",
    )

    qpe_weight_version: str | None = Field(
        default=None,
        description="QPE_WEIGHT_VERSION at time of computation. None = pre-versioning entry.",
    )


class ImpactCategory(str, Enum):
    """Categories for staged changes impact assessment."""

    SIGNIFICANT_IMPROVEMENT = "significant_improvement"
    MINOR_IMPROVEMENT = "minor_improvement"
    NEUTRAL = "neutral"
    MINOR_DEGRADATION = "minor_degradation"
    SIGNIFICANT_DEGRADATION = "significant_degradation"


class ZScoreInterpretation(str, Enum):
    """Human-readable interpretation of Z-score values."""

    MUCH_BETTER = "much better than avg"
    BETTER = "better than avg"
    ABOUT_AVERAGE = "about avg"
    WORSE = "worse than avg"
    MUCH_WORSE = "much worse than avg"

    @classmethod
    def from_z_score(cls, normalized_z: float, verbose: bool = False) -> "ZScoreInterpretation":
        """Interpret a normalized Z-score (positive = good)."""
        if verbose:
            if normalized_z > 1.5:
                return cls.MUCH_BETTER
            elif normalized_z > 0.5:
                return cls.BETTER
            elif normalized_z > -0.5:
                return cls.ABOUT_AVERAGE
            elif normalized_z > -1.5:
                return cls.WORSE
            else:
                return cls.MUCH_WORSE
        else:
            if normalized_z > 1.0:
                return cls.MUCH_BETTER
            elif normalized_z > 0.3:
                return cls.BETTER
            elif normalized_z > -0.3:
                return cls.ABOUT_AVERAGE
            elif normalized_z > -1.0:
                return cls.WORSE
            else:
                return cls.MUCH_WORSE


class ImpactAssessment(BaseModel):
    """Assessment of staged changes impact against repo baseline."""

    cc_z_score: float = Field(description="Z-score for CC change (positive = above avg increase)")
    effort_z_score: float = Field(description="Z-score for Effort change (positive = above avg increase)")
    mi_z_score: float = Field(description="Z-score for MI change (positive = above avg increase, which is good)")

    impact_score: float = Field(
        description="Composite score: positive = above-average quality improvement, negative = below-average"
    )
    impact_category: ImpactCategory = Field(description="Categorical assessment of impact")

    cc_delta: float = Field(description="Total CC change (sum across all files)")
    effort_delta: float = Field(description="Total Effort change (sum across all files)")
    mi_delta: float = Field(description="Total MI change (sum across all files)")

    qpe_delta: float = Field(default=0.0, description="QPE change between baseline and current")
    qpe_z_score: float = Field(default=0.0, description="Z-score for QPE change (positive = above avg improvement)")

    token_delta: int = Field(default=0, description="Token change between baseline and current")
    token_z_score: float = Field(default=0.0, description="Z-score for token change (positive = larger than avg)")

    def interpret_cc(self, verbose: bool = False) -> ZScoreInterpretation:
        """Interpret CC z-score (lower CC is better, so invert)."""
        return ZScoreInterpretation.from_z_score(-self.cc_z_score, verbose)

    def interpret_effort(self, verbose: bool = False) -> ZScoreInterpretation:
        """Interpret Effort z-score (lower effort is better, so invert)."""
        return ZScoreInterpretation.from_z_score(-self.effort_z_score, verbose)

    def interpret_mi(self, verbose: bool = False) -> ZScoreInterpretation:
        """Interpret MI z-score (higher MI is better, no inversion)."""
        return ZScoreInterpretation.from_z_score(self.mi_z_score, verbose)

    def interpret_qpe(self, verbose: bool = False) -> ZScoreInterpretation:
        """Interpret QPE z-score (higher QPE is better, no inversion)."""
        return ZScoreInterpretation.from_z_score(self.qpe_z_score, verbose)

    def interpret_tokens(self, verbose: bool = False) -> ZScoreInterpretation:
        """Interpret token z-score - neutral interpretation since size isn't inherently good/bad."""
        return ZScoreInterpretation.from_z_score(self.token_z_score, verbose)


class CodeQualityCache(BaseModel):
    """Cached code quality metrics for a specific session/repository/commit combination."""

    id: int | None = None
    session_id: str = Field(description="Session ID this cache entry belongs to")
    repository_path: str = Field(description="Absolute path to the repository")
    commit_sha: str = Field(description="Git commit SHA when metrics were calculated")
    calculated_at: datetime = Field(default_factory=datetime.now, description="When metrics were calculated")
    complexity_metrics: "ExtendedComplexityMetrics" = Field(description="Cached complexity metrics")
    complexity_delta: "ComplexityDelta | None" = Field(
        default=None, description="Cached complexity delta from previous commit"
    )
    working_tree_hash: str | None = Field(
        default=None, description="Hash of working tree state for uncommitted changes. NULL for clean repos."
    )


class StagedChangesAnalysis(BaseModel):
    """Complete analysis of staged changes against repository baseline.

    Deprecated: Use CurrentChangesAnalysis instead for analyzing uncommitted changes.
    """

    repository_path: str
    analysis_timestamp: datetime = Field(default_factory=datetime.now)

    staged_files: list[str] = Field(description="List of staged Python files")
    staged_metrics: "ExtendedComplexityMetrics" = Field(description="Metrics if staged changes were applied")
    baseline_metrics: "ExtendedComplexityMetrics" = Field(description="Metrics at current HEAD")

    assessment: ImpactAssessment
    baseline: RepoBaseline


class CurrentChangesAnalysis(BaseModel):
    """Complete analysis of changes against repository baseline."""

    repository_path: str
    analysis_timestamp: datetime = Field(default_factory=datetime.now)

    source: AnalysisSource = Field(
        default=AnalysisSource.UNCOMMITTED_CHANGES,
        description="Whether analyzing uncommitted changes or previous commit",
    )
    analyzed_commit_sha: str | None = Field(
        default=None,
        description="SHA of the commit being analyzed (when source is PREVIOUS_COMMIT)",
    )
    base_commit_sha: str | None = Field(
        default=None,
        description="SHA of the base commit for comparison (when source is PREVIOUS_COMMIT)",
    )

    changed_files: list[str] = Field(description="List of changed Python files")
    current_metrics: "ExtendedComplexityMetrics" = Field(description="Metrics with uncommitted changes applied")
    baseline_metrics: "ExtendedComplexityMetrics" = Field(description="Metrics at current HEAD")

    assessment: ImpactAssessment
    baseline: RepoBaseline

    blind_spots: list[str] = Field(
        default_factory=list, description="Files dependent on changed files but not in changed set"
    )
    filtered_coverage: dict[str, float] | None = Field(default=None, description="Coverage % for changed files")

    blind_spot_tokens: int = Field(default=0, description="Total tokens in blind spot files")
    changed_files_tokens: int = Field(default=0, description="Total tokens in changed files")
    complete_picture_context_size: int = Field(default=0, description="Sum of tokens in changed files + blind spots")

    galen_metrics: GalenMetrics | None = Field(
        default=None, description="Developer productivity metrics based on token throughput"
    )

    smell_advantages: list["SmellAdvantage"] = Field(
        default_factory=list,
        description="Per-smell advantage breakdown between baseline and current QPE. "
        "Shows which specific smells changed and their weighted impact.",
    )


class CurrentImpactSummary(BaseModel):
    """Compact JSON output of current-impact analysis for CI consumption."""

    source: AnalysisSource = Field(description="Whether analyzing uncommitted changes or previous commit")
    analyzed_commit_sha: str | None = Field(
        default=None, description="SHA of the analyzed commit (when source is previous_commit)"
    )
    base_commit_sha: str | None = Field(
        default=None, description="SHA of the base commit (when source is previous_commit)"
    )
    impact_score: float = Field(description="Weighted composite impact score")
    impact_category: ImpactCategory = Field(description="Human-readable impact category")
    qpe_delta: float = Field(description="Change in QPE score")
    cc_delta: float = Field(description="Change in cyclomatic complexity")
    effort_delta: float = Field(description="Change in Halstead effort")
    mi_delta: float = Field(description="Change in maintainability index")
    token_delta: int = Field(default=0, description="Change in tokens")
    changed_files_count: int = Field(description="Number of changed code files")
    blind_spots_count: int = Field(description="Number of dependent files not in changed set")
    smell_advantages: list["SmellAdvantage"] = Field(default_factory=list, description="Per-smell advantage breakdown")

    @staticmethod
    def from_analysis(analysis: CurrentChangesAnalysis) -> "CurrentImpactSummary":
        """Create compact summary from full analysis."""
        return CurrentImpactSummary(
            source=analysis.source,
            analyzed_commit_sha=analysis.analyzed_commit_sha,
            base_commit_sha=analysis.base_commit_sha,
            impact_score=analysis.assessment.impact_score,
            impact_category=analysis.assessment.impact_category,
            qpe_delta=analysis.assessment.qpe_delta,
            cc_delta=analysis.assessment.cc_delta,
            effort_delta=analysis.assessment.effort_delta,
            mi_delta=analysis.assessment.mi_delta,
            token_delta=analysis.assessment.token_delta,
            changed_files_count=len(analysis.changed_files),
            blind_spots_count=len(analysis.blind_spots),
            smell_advantages=analysis.smell_advantages,
        )


class QPEScore(BaseModel):
    """Quality score for principled code quality comparison."""

    qpe: float = Field(description="Adjusted quality score (higher is better)")
    mi_normalized: float = Field(description="Maintainability Index normalized to 0-1")
    smell_penalty: float = Field(description="Penalty from code smells (sigmoid-saturated, 0-0.9 range)")
    adjusted_quality: float = Field(description="MI after smell penalty applied")

    smell_counts: SmellCounts = Field(
        default_factory=SmellCounts, description="Individual smell counts contributing to penalty"
    )


class SmellAdvantage(BaseModel):
    """Per-smell contribution to the GRPO advantage signal."""

    model_config = ConfigDict(frozen=True)

    smell_name: str = Field(description="Internal name from SMELL_REGISTRY")
    baseline_count: int = Field(description="Number of this smell in the baseline")
    candidate_count: int = Field(description="Number of this smell in the candidate")
    weight: float = Field(description="Smell weight from SMELL_REGISTRY")
    weighted_delta: float = Field(description="Weighted delta for this smell")


class ImplementationComparison(BaseModel):
    """Result of comparing two parallel implementations."""

    prefix_a: str = Field(description="Subtree path prefix for implementation A")
    prefix_b: str = Field(description="Subtree path prefix for implementation B")
    ref: str = Field(description="Git ref both subtrees were extracted from")
    qpe_a: QPEScore = Field(description="QPE score for implementation A")
    qpe_b: QPEScore = Field(description="QPE score for implementation B")
    aggregate_advantage: float = Field(description="GRPO advantage of B over A")
    smell_advantages: list[SmellAdvantage] = Field(default_factory=list)
    winner: str = Field(description="Which prefix produced better code")


class ProjectQPEResult(BaseModel):
    """QPE result for a single project."""

    project_path: str = Field(description="Path to the project")
    project_name: str = Field(description="Name of the project")
    qpe_score: QPEScore = Field(description="QPE score for this project")
    metrics: "ExtendedComplexityMetrics" = Field(description="Full metrics for this project")


class CrossProjectComparison(BaseModel):
    """Result of comparing multiple projects using QPE."""

    compared_at: datetime = Field(default_factory=datetime.now)
    total_projects: int = Field(description="Total number of projects compared")
    rankings: list[ProjectQPEResult] = Field(default_factory=list)
