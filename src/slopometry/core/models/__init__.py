"""Models package for slopometry.

Re-exports all model types from submodules for convenience.
This maintains backwards compatibility with the previous single-file models.py.
"""

# Core types (base types with no external dependencies)
# Baseline/impact types
from slopometry.core.models.baseline import (
    CodeQualityCache,
    CurrentChangesAnalysis,
    CurrentImpactSummary,
    GalenMetrics,
    HistoricalMetricStats,
    ImpactAssessment,
    ImpactCategory,
    ImplementationComparison,
    ProjectQPEResult,
    QPEScore,
    RepoBaseline,
    ResolvedBaselineStrategy,
    SmellAdvantage,
    StagedChangesAnalysis,
    ZScoreInterpretation,
)

# Complexity types (re-exports from core)
from slopometry.core.models.complexity import (
    CommitChain,
    CommitComplexitySnapshot,
    ComplexityEvolution,
    FileAnalysisResult,
)
from slopometry.core.models.core import (
    CacheUpdateError,
    ComplexityDelta,
    ComplexityMetrics,
    ExtendedComplexityMetrics,
    SmellCounts,
    TokenCountError,
)

# Display types
from slopometry.core.models.display import (
    ExperimentDisplayData,
    FeatureDisplayData,
    LeaderboardEntry,
    NFPObjectiveDisplayData,
    SessionDisplayData,
)

# Experiment types
from slopometry.core.models.experiment import (
    ExperimentProgress,
    ExperimentRun,
    ExperimentStatus,
    FeatureBoundary,
)

# Hook types
from slopometry.core.models.hook import (
    AgentTool,
    AnalysisSource,
    GitState,
    Project,
    ProjectLanguage,
)
from slopometry.core.models.memory import (
    MemoryCandidate,
    MemoryCreateRequest,
    MemoryEntry,
    MemoryType,
)
from slopometry.core.models.protocol import (
    AbstractEventSource,
    AbstractEventType,
    AbstractHookEvent,
    ToolCallPayload,
)

# Session types
from slopometry.core.models.session import (
    CompactEvent,
    ContextCoverage,
    FileCoverageStatus,
    PlanEvolution,
    PlanStep,
    SavedCompact,
    SessionMetadata,
    SessionStatistics,
    TodoItem,
    TokenUsage,
)

# User story types
from slopometry.core.models.user_story import (
    NextFeaturePrediction,
    UserStory,
    UserStoryDisplayData,
    UserStoryEntry,
    UserStoryStatistics,
)

__all__ = [
    # Core
    "CacheUpdateError",
    "ComplexityDelta",
    "ComplexityMetrics",
    "ExtendedComplexityMetrics",
    "SmellCounts",
    "TokenCountError",
    # Complexity
    "CommitChain",
    "CommitComplexitySnapshot",
    "ComplexityEvolution",
    "FileAnalysisResult",
    # Hook
    "AbstractEventSource",
    "AbstractEventType",
    "AbstractHookEvent",
    "ToolCallPayload",
    "AgentTool",
    "AnalysisSource",
    "GitState",
    "Project",
    "ProjectLanguage",
    # Session
    "CompactEvent",
    "ContextCoverage",
    "FileCoverageStatus",
    "PlanEvolution",
    "PlanStep",
    "SavedCompact",
    "SessionMetadata",
    "SessionStatistics",
    "TodoItem",
    "TokenUsage",
    # Baseline/impact
    "CodeQualityCache",
    "CurrentChangesAnalysis",
    "CurrentImpactSummary",
    "GalenMetrics",
    "HistoricalMetricStats",
    "ImpactAssessment",
    "ImpactCategory",
    "ImplementationComparison",
    "ProjectQPEResult",
    "QPEScore",
    "RepoBaseline",
    "ResolvedBaselineStrategy",
    "SmellAdvantage",
    "StagedChangesAnalysis",
    "ZScoreInterpretation",
    # Experiment
    "ExperimentProgress",
    "ExperimentRun",
    "ExperimentStatus",
    "FeatureBoundary",
    # Display
    "ExperimentDisplayData",
    "FeatureDisplayData",
    "LeaderboardEntry",
    "NFPObjectiveDisplayData",
    "SessionDisplayData",
    # User story
    "NextFeaturePrediction",
    "UserStory",
    "UserStoryDisplayData",
    "UserStoryEntry",
    "UserStoryStatistics",
    "MemoryCandidate",
    "MemoryCreateRequest",
    "MemoryEntry",
    "MemoryType",
]
