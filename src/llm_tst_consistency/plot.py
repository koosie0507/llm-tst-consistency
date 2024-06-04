import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from llm_tst_consistency.stats import Stats


def draw_plots(
    title: str,
    df: pd.DataFrame,
    columns: list[str],
    benchmark_stats: dict[str, Stats]
):

    # Assuming df is your DataFrame
    for metric in columns:
        df[f"{metric}_varlo"] = df[metric] - benchmark_stats[metric].variance
        df[f"{metric}_varhi"] = df[metric] + benchmark_stats[metric].variance

    COL_COUNT = 2
    fig = make_subplots(rows=len(columns)//COL_COUNT+1, subplot_titles=columns, cols=COL_COUNT)

    for i, metric in enumerate(columns):
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=df[metric],
                mode='lines',
                name=metric,
                legendgroup=metric,
                line={"color": "#6E0B14"}
            ),
            row=i//COL_COUNT+1,
            col=i%COL_COUNT+1
        )
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=[benchmark_stats[metric].low] * len(df),
                mode='lines',
                name=f'corpus {metric} min',
                line=dict(dash='dot', color="#676767"),
                showlegend=False,
                legendgroup=metric
            ),
            row=i//COL_COUNT+1,
            col=i%COL_COUNT+1
        )

        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=[benchmark_stats[metric].high] * len(df),
                mode='lines',
                name=f'corpus {metric} max',
                line=dict(dash='dot', color="#676767"),
                showlegend=False,
                legendgroup=metric
            ),
            row=i//COL_COUNT+1,
            col=i%COL_COUNT+1
        )

        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=[benchmark_stats[metric].mean] * len(df),
                mode='lines',
                name=f'corpus {metric} average',
                line={"color": "#000000"},
                showlegend=False,
                legendgroup=metric,
            ),
            row=i//COL_COUNT+1,
            col=i%COL_COUNT+1
        )

    fig.update_layout(height=200*len(columns), width=900, title_text=title)
    fig.show()
