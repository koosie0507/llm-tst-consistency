from pathlib import Path
from typing import Callable, Iterable

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


def draw_one_figure(title, features, df, ds_stats, plot_col_count):
    fig = make_subplots(
        rows=len(features) // plot_col_count + 1,
        subplot_titles=features,
        cols=plot_col_count,
    )
    for i, f in enumerate(features):
        for trace in _make_feature_traces(f, df, ds_stats):
            fig.add_trace(
                trace, row=i // plot_col_count + 1, col=i % plot_col_count + 1
            )
    fig.update_layout(height=200 * len(features), width=900, title_text=title)
    return fig


def draw_multiple_figures(title, features, df, ds_stats):
    for f in features:
        fig = go.Figure()
        for trace in _make_feature_traces(f, df, ds_stats):
            fig.add_trace(trace)
            
        fig.update_layout(
            height=480,
            width=640,
            font_color="black",
            #title_text=f"{title} - {f}",
            showlegend=False,
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
        )
        yield f, fig


def _make_feature_traces(feature_name, df, ds_stats):
    return [
        _create_hlf(df, feature_name),
        _create_baseline(df, feature_name),
        _create_benchmark_trace(
            df,
            feature_name,
            f"{feature_name} benchmark min",
            ds_stats[feature_name].low,
        ),
        _create_benchmark_trace(
            df,
            feature_name,
            f"{feature_name} benchmark max",
            ds_stats[feature_name].high,
        ),
        _create_benchmark_trace(
            df,
            feature_name,
            f"{feature_name} benchmark average",
            ds_stats[feature_name].mean,
            "#000000",
            "solid",
        ),
    ]


def draw_plots(
    prompt_name: str,
    title: str,
    df: pd.DataFrame,
    features: list[str],
    benchmark_stats: dict[str, Stats],
    col_count: int = 2,
):
    df = _add_variance(df, benchmark_stats, features, _baseline_name)
    df = _add_variance(df, benchmark_stats, features, _hlf_name)
    img_path = (
        Path(__file__).parent.parent.parent / "article" / "plots" / prompt_name / title
    )
    if not (img_path.exists() and img_path.is_dir()):
        img_path.mkdir(exist_ok=True, parents=True)

    fig = draw_one_figure(title, features, df, benchmark_stats, col_count)
    fig.write_image(img_path / f"{title}.png")
    for feature, fig in draw_multiple_figures(title, features, df, benchmark_stats):
        fig.write_image(img_path / f"{title}_{feature}.png")
