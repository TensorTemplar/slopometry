"""Type stubs for rust-code-analysis Python bindings."""

from collections.abc import Iterator

class CyclomaticMetrics:
    sum: float
    average: float
    max: float

class HalsteadMetrics:
    volume: float
    difficulty: float
    effort: float
    bugs: float
    length: int
    vocabulary: int
    n1: int
    n2: int
    N1: int
    N2: int

class MIMetrics:
    mi_original: float
    mi_sei: float
    mi_visual_studio: float

class Metrics:
    cyclomatic: CyclomaticMetrics
    halstead: HalsteadMetrics
    mi: MIMetrics
    sloc: int
    ploc: int
    lloc: int
    cloc: int
    blank: int

class FunctionInfo:
    name: str
    start_line: int
    end_line: int
    metrics: Metrics

class AnalysisResult:
    metrics: Metrics
    def get_functions(self) -> Iterator[FunctionInfo]: ...

def analyze_file(path: str) -> AnalysisResult: ...
def analyze_code(code: str, language: str) -> AnalysisResult: ...
