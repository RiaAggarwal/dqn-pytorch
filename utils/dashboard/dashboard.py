"""
Usage:
`python dashboard.py`
"""
import os.path
from typing import List

import dash
import dash_bootstrap_components as dbc
import dash_core_components as dcc
import dash_html_components as html
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from dash.dependencies import Input, Output

from data_loader import get_experiments, get_rewards_history_df, get_steps_history_df, get_parameters_df

# Initialize app
app = dash.Dash(__name__, meta_tags=[{"name": "viewport", "content": "width=device-width"}],
                external_stylesheets=[dbc.themes.BOOTSTRAP],
                suppress_callback_exceptions=False)


def fig_formatter(**kw):
    """
    Decorator for functions that produce figures. By default, all margins are stripped, but the margin sized can be
    set individually.

    :param t: top margin
    :param l: left margin
    :param r: right margin
    :param b: bottom margin
    :return:
    """
    t = kw.get('t', 0)
    l = kw.get('l', 0)
    r = kw.get('r', 0)
    b = kw.get('b', 0)

    def wrap(func):
        def wrapped(*args, **kwargs):
            fig = func(*args, **kwargs)
            fig.update_layout(margin=dict(t=t, l=l, r=r, b=b),
                              paper_bgcolor='rgba(0,0,0,0)',
                              plot_bgcolor='rgba(0,0,0,0)',
                              )
            return fig

        return wrapped

    return wrap


def load_markdown_text(file: str) -> str:
    """
    Load markdown text from assets

    :param file: name of markdown file (without extension or path)
    :return: contents of markdown file
    """
    with open(os.path.join(os.path.dirname(__file__), 'assets', f'{file}.md'), 'r') as f:
        md = f.read()
    return md


experiments = get_experiments()

# Create app layout
app.layout = html.Div(
    [
        # empty Div to trigger javascript file for graph resizing
        html.Div(id='output-clientside'),
        html.Div(
            [
                html.Div(
                    [
                        html.Div(
                            [
                                html.H3(
                                    "RL for Underwater Communication Experiments",
                                    style={"margin-bottom": "0px"},
                                ),
                            ]
                        )
                    ],
                    className="two-half column",
                    id="title",
                ),
            ],
            id="header",
            className="row flex-display",
            style={"margin-bottom": "25px"},
        ),

        # Training
        html.Div(
            [
                html.Div(
                    [
                        html.P(
                            [
                                "Select experiments:",
                                dcc.Dropdown(
                                    id='experiment-selector',
                                    options=experiments,
                                    multi=True
                                ),
                                html.Br(),
                                html.P("Moving average length:", id='moving-avg-slider-text'),
                                dcc.Slider(
                                    id='moving-avg-slider',
                                    min=1,
                                    max=100,
                                    step=1,
                                    value=10,
                                ),
                            ]
                        ),
                    ],
                    className='pretty_container three columns',
                    id='training-div',
                ),
                html.Div(
                    [
                        dcc.Graph(id='reward-plot')
                    ],
                    className='pretty_container five columns',
                ),
                html.Div(
                    [
                        dcc.Graph(id='step-plot')
                    ],
                    className='pretty_container five columns',
                ),
            ],
            className='flex-display',
            style={'margin-bottom': '25px'}
        ),

        # Parameters
        html.Div(
            [
                html.Div(
                    [
                        dbc.Table(id='experiment-table'),
                    ],
                    id='params-table-div',
                ),
            ],
            className='flex-display',
            style={'margin-bottom': '25px'}
        ),
    ],
    id='mainContainer',
    style={'display': 'flex', 'flex-direction': 'column'},
)


def get_empty_sunburst(text: str):
    """
    Generates an empty sunburst plot with `text` at its center

    :param text: informational text to display
    :return: `plotly` figure
    """
    return px.sunburst(
        {'x'    : [text],
         'value': [1]},
        path=['x'],
        hover_data=None
    )


@app.callback(
    Output('experiment-table', 'children'),
    [Input('experiment-selector', 'value')]
)
def make_experiment_table(experiments: List[str]) -> List:
    if experiments:
        df = get_parameters_df(experiments)
        table = dbc.Table.from_dataframe(df, striped=True, bordered=True, hover=True)
    else:
        table = dbc.Table.from_dataframe(pd.DataFrame())
    return table


@app.callback(
    Output('moving-avg-slider-text', 'children'),
    [Input('moving-avg-slider', 'value')]
)
def update_moving_avg_slider_text(value: int) -> List:
    return [f"Moving average length: {value}"]


@app.callback(
    Output('reward-plot', 'figure'),
    [Input('experiment-selector', 'value'),
     Input('moving-avg-slider', 'value')]
)
def make_rewards_plot(experiments: List[str], moving_avg_window: int) -> go.Figure:
    if not experiments:
        fig = get_empty_sunburst("Select an experiment")
    else:
        fig = get_reward_plot(experiments, moving_avg_window)
    return fig


@app.callback(
    Output('step-plot', 'figure'),
    [Input('experiment-selector', 'value'),
     Input('moving-avg-slider', 'value')]
)
def make_rewards_plot(experiments: List[str], moving_avg_window: int) -> go.Figure:
    if not experiments:
        fig = get_empty_sunburst("Select an experiment")
    else:
        fig = get_step_plot(experiments, moving_avg_window)
    return fig


@fig_formatter(t=50)
def get_reward_plot(experiments: List[str], moving_avg_len) -> go.Figure:
    df = get_rewards_history_df(experiments, moving_avg_len)
    return px.line(df, labels=dict(value='reward', index='episode', variable='experiment'))


@fig_formatter(t=50)
def get_step_plot(experiments: List[str], moving_avg_len) -> go.Figure:
    df = get_steps_history_df(experiments, moving_avg_len)
    return px.line(df, labels=dict(value='steps', index='episode', variable='experiment'))


if __name__ == '__main__':
    # noinspection PyTypeChecker
    app.run_server(debug=True,
                   dev_tools_hot_reload=False,
                   host=os.getenv("HOST", "127.0.0.1"),
                   # host=os.getenv("HOST", "192.168.1.10"),
                   port=os.getenv("PORT", "8050"),
                   )
