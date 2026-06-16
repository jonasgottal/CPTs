from typing import Dict, List, Optional, Union, Any, Literal
from pydantic import BaseModel, RootModel, Field, model_validator


class MeanStdStats(BaseModel):
    """Statistics for normally distributed continuous or ordinal variables."""

    mean: float = Field(description="The arithmetic mean value")
    std: float = Field(description="The standard deviation value")

    # Additional distribution parameters
    skew: Optional[float] = Field(
        0,
        description="Skewness parameter controlling distribution asymmetry (0=symmetric, positive=right tail, negative=left tail)",
    )
    min_val: Optional[float] = Field(
        None,
        description="Minimum allowed value for the distribution (truncates values below this)",
    )
    max_val: Optional[float] = Field(
        None,
        description="Maximum allowed value for the distribution (truncates values above this)",
    )
    tail_weight: Optional[float] = Field(
        0,
        description="Weight (0-1) given to the tail distribution for generating heavy-tailed distributions",
    )
    tail_std_factor: Optional[float] = Field(
        0,
        description="Factor to multiply std by for the tail distribution (>1 creates heavier tails)",
    )


class MedianIQRStats(BaseModel):
    """Statistics for non-normally distributed continuous or ordinal variables."""

    median: float = Field(description="The median value (50th percentile)")
    iqr: Optional[float] = Field(None, description="The interquartile range")
    q1: Optional[float] = Field(
        None, description="The first quartile (25th percentile)"
    )
    q3: Optional[float] = Field(
        None, description="The third quartile (75th percentile)"
    )
    ci95: Optional[List[float]] = Field(
        None, description="The 95% confidence interval for the median"
    )
    min_val: Optional[float] = Field(
        None,
        description="Minimum allowed value for the distribution (truncates values below this)",
    )
    max_val: Optional[float] = Field(
        None,
        description="Maximum allowed value for the distribution (truncates values above this)",
    )


class CategoricalStats(RootModel):
    """Counts for each category in a categorical variable."""

    root: Dict[str, int] = Field(
        description="Mapping of category names to counts"
    )


class BinaryGroupStats(BaseModel):
    count: int = Field(description="Count of positive outcomes")
    denominator: Optional[int] = Field(
        None,
        description="Total number of subjects for this variable in this group",
    )


class BinaryStats(RootModel):
    """
    For each group, maps to an object with count and denominator.
    Example: {"Intervention": {"count": 4, "denominator": 11}, ...}
    """

    root: Dict[str, BinaryGroupStats] = Field(
        description="Mapping of group names to count/denominator objects"
    )


# Variable definitions, using Union types to handle different stat formats
VariableStats = Union[
    MeanStdStats,
    MedianIQRStats,
    CategoricalStats,
    BinaryStats,
    Dict[str, float],
    Dict[str, int],
    int,
]


class GroupData(RootModel):
    """Statistical data for a variable across different groups."""

    root: Dict[str, VariableStats] = Field(
        description="Mapping of group names to statistical data"
    )


class Variable(RootModel):
    """Data structure for a single variable across groups."""

    root: Dict[str, GroupData] = Field(
        description="Mapping of variable names to group data"
    )


class Variables(BaseModel):
    """Container for all variables categorized by their data type."""

    continuous: Dict[str, Dict[str, Any]] = Field(
        default_factory=dict,
        description="Continuous variables with a potentially infinite range of values (e.g., weight, BMI)",
    )

    ordinal: Dict[str, Dict[str, Any]] = Field(
        default_factory=dict,
        description="ordinal/ordinal variables (e.g., age in years, counts)",
    )

    binary: Dict[str, Dict[str, Any]] = Field(
        default_factory=dict,
        description="Variables with exactly two possible outcomes (e.g., mortality, gender)",
    )

    categorical: Dict[str, Dict[str, Any]] = Field(
        default_factory=dict,
        description="Variables with multiple non-binary categories (e.g., race, education level)",
    )

    @model_validator(mode="after")
    def validate_structure(self) -> "Variables":
        """Validate the structure of variables data"""
        return self


TestType = Literal[
    "pearson",
    "spearman",
    "chi_square",
    "mcnemar",
    "wilcoxon_signed_rank",
    "paired_t_test",
    "wilcoxon_mann_whitney",
    "unpaired_t_test",
    "friedman",
    "kruskal_wallis",
    "one_way_anova",
    "ranova",
]


class StatisticalTest(BaseModel):
    """Information about a statistical test performed on variables."""

    variables: List[str] = Field(
        description="Names of variables involved in the test"
    )
    test_type: TestType = Field(
        ...,
        description="Type of statistical test performed (e.g., t-test, ANOVA, correlation)",
    )
    test_statistic: Optional[float] = Field(
        None,
        description="The calculated test statistic value (e.g., t-value, F-value)",
    )
    p_value: Optional[float] = Field(
        None,
        description="The probability value indicating statistical significance",
    )
    effect_size: Optional[float] = Field(
        None,
        description="Measure of the magnitude of the effect (e.g., Cohen's d, r value)",
    )
    group_means: Optional[List[float]] = Field(
        None, description="List of group means when comparing groups"
    )
    groups: List[str] = Field(
        description="Names of groups involved in the test"
    )


class ResearchData(BaseModel):
    """Complete structured representation of research data extracted from a publication."""

    study_size: Optional[int] = Field(
        None, description="Total number of participants in the study"
    )
    groups: List[str] = Field(
        default_factory=list,
        description="List of group names in the study (e.g., 'Control', 'Treatment')",
    )
    group_sizes: Dict[str, int] = Field(
        default_factory=dict,
        description="Number of participants in each group",
    )
    variables: Variables = Field(
        default_factory=Variables,
        description="All variables identified in the study, categorized by type",
    )
    statistical_tests: List[StatisticalTest] = Field(
        default_factory=list,
        description="All statistical tests performed in the study",
    )

    @model_validator(mode="after")
    def validate_group_sizes(self) -> "ResearchData":
        """Validate that group sizes sum to study size"""
        if self.study_size and self.group_sizes:
            total = sum(self.group_sizes.values())
            if total != self.study_size:
                # You could raise ValidationError here or just log a warning
                pass
        return self
