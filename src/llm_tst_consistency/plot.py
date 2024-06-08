from typing import Callable

import pandas as pd
import plotly.graph_objects as go
from plotly.basedatatypes import BaseTraceType
from plotly.graph_objs import Figure
from plotly.subplots import make_subplots

from llm_tst_consistency.stats import Stats


def _add_variance(
    df: pd.DataFrame,
    benchmark_stats: dict[str, Stats],
    features: list[str],
    name_extractor: Callable[[str], str],
) -> pd.DataFrame:
    for feature in features:
        df[f"{name_extractor(feature)}_varlo"] = (
            df[f"{name_extractor(feature)}"] - benchmark_stats[feature].variance
        )
        df[f"{name_extractor(feature)}_varhi"] = (
            df[f"{name_extractor(feature)}"] + benchmark_stats[feature].variance
        )
    return df


def _baseline_name(feature: str) -> str:
    return f"baseline_{feature}"


def _hlf_name(feature: str) -> str:
    return f"hlf_{feature}"


def _create_benchmark_trace(
    df: pd.DataFrame,
    feature: str,
    title: str,
    benchmark_value: float,
    color: str = "#575757",
    dash: str = "dot",
) -> BaseTraceType:
    return go.Scatter(
        x=df.index,
        y=[benchmark_value] * len(df),
        mode="lines",
        name=title,
        line={"color": color, "dash": dash},
        showlegend=False,
        legendgroup=feature,
    )


def _create_baseline(df: pd.DataFrame, feature: str) -> BaseTraceType:
    return go.Scatter(
        x=df.index,
        y=df[_baseline_name(feature)],
        mode="lines",
        name=f"baseline {feature}",
        legendgroup=feature,
        line={"color": "#575757"},
    )


def _create_hlf(df: pd.DataFrame, feature: str) -> BaseTraceType:
    return go.Scatter(
        x=df.index,
        y=df[_hlf_name(feature)],
        mode="lines",
        name=f"hlf-enhanced {feature}",
        legendgroup=feature,
        line={"color": "#6E0B14"},
    )


def _add_trace_to_plot(
    fig: Figure, trace: BaseTraceType, i: int, col_count: int
) -> None:
    fig.add_trace(trace, row=i // col_count + 1, col=i % col_count + 1)


def draw_plots(
    title: str,
    df: pd.DataFrame,
    features: list[str],
    benchmark_stats: dict[str, Stats],
    col_count: int = 2,
):
    df = _add_variance(df, benchmark_stats, features, _baseline_name)
    df = _add_variance(df, benchmark_stats, features, _hlf_name)
    fig = make_subplots(
        rows=len(features) // col_count + 1, subplot_titles=features, cols=col_count
    )

    for i, f in enumerate(features):
        for trace in [
            _create_hlf(df, f),
            _create_baseline(df, f),
            _create_benchmark_trace(
                df, f, f"{f} benchmark min", benchmark_stats[f].low
            ),
            _create_benchmark_trace(
                df, f, f"{f} benchmark max", benchmark_stats[f].high
            ),
            _create_benchmark_trace(
                df,
                f,
                f"{f} benchmark average",
                benchmark_stats[f].mean,
                "#000000",
                "solid",
            ),
        ]:
            _add_trace_to_plot(fig, trace, i, col_count)

    fig.update_layout(height=200 * len(features), width=900, title_text=title)
    fig.show()
