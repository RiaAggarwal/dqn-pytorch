# -*- coding: utf-8 -*-
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
import plotly.express as px
import plotly.graph_objects as go
from dash.dependencies import Input, Output

from utils.utils import get_experiments, get_rewards_history_df, get_steps_history_df



# Initialize app and cache
app = dash.Dash(__name__, meta_tags=[{"name": "viewport", "content": "width=device-width"}],
                external_stylesheets=[dbc.themes.BOOTSTRAP],
                suppress_callback_exceptions=True)


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


# Create app layout
app.layout = html.Div(
    [
        # empty Div to trigger javascript file for graph resizing
        html.Div(id='output-clientside'),
        html.Div(
            [
                html.Div(
                    [
                        html.Img(
                            src=app.get_asset_url("ucsd-logo.png"),
                            id="plotly-image",
                            style={
                                "height"       : "60px",
                                "width"        : "auto",
                                "margin-bottom": "25px",
                            },
                        )
                    ],
                    className="one-third column",
                ),
                html.Div(
                    [
                        html.Div(
                            [
                                html.H1(
                                    "Boosting Interest in STEM",
                                    style={"margin-bottom": "0px"},
                                ),
                                html.H5(
                                    """An analysis of ninth graders' feelings towards science""",
                                    style={"margin-top": "0px"}
                                ),
                                html.H6(
                                    """ECE229 - Team 5: Ian Pegg, Subrato Chakravorty, Yan Sun, Daniel You, 
                                    Heqian Lu, Kai Wang""",
                                    style={"margin-top": "0px"}
                                ),
                            ]
                        )
                    ],
                    className="two-half column",
                    id="title",
                ),
                html.Div(
                    [
                        html.A(
                            html.Button("github", id="learn-more-button"),
                            href="https://github.com/SubratoChakravorty/ECE-229-Group5-final-project",
                        ),
                    ],
                    className="three-third column",
                    id="github-button",
                ),
            ],
            id="header",
            className="row flex-display",
            style={"margin-bottom": "25px"},
        ),

        # Introduction
        html.Div(
            [
                dcc.Markdown(load_markdown_text('introduction'), dedent=False)
            ],
            className='pretty_container',
            id='introduction',
            style={'margin-bottom': '25px'}
        ),

        # Dashboard
        html.Div(
            [
                html.Div(
                    [
                        html.P(
                            [
                                "Select experiments:",
                                dcc.Dropdown(
                                    id='experiment-selector',
                                    options=get_experiments(),
                                    multi=True
                                ),
                            ]
                        ),
                    ],
                    className='pretty_container four columns',
                    id='ml_controls',
                ),
                html.Div(
                    [
                        dcc.Graph(id='reward-plot')
                    ],
                    className='pretty_container four columns',
                ),
                html.Div(
                    [
                        dcc.Graph(id='step-plot')
                    ],
                    className='pretty_container four columns',
                ),
            ],
            className='flex-display',
            style={'margin-bottom': '25px'}
        ),
    ],
    id='mainContainer',
    style={'display': 'flex', 'flex-direction': 'column'},
)


@fig_formatter()
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
    Output('reward-plot', 'figure'),
    [Input('experiment-selector', 'value')]
)
def make_rewards_plot(experiments: List[str]) -> go.Figure:
    if not experiments:
        fig = get_empty_sunburst("Select an experiment")
    else:
        fig = get_reward_plot(experiments)
    return fig


@app.callback(
    Output('step-plot', 'figure'),
    [Input('experiment-selector', 'value')]
)
def make_rewards_plot(experiments: List[str]) -> go.Figure:
    if not experiments:
        fig = get_empty_sunburst("Select an experiment")
    else:
        fig = get_step_plot(experiments)
    return fig


@fig_formatter()
def get_reward_plot(experiments: List[str]) -> go.Figure:
    df = get_rewards_history_df(experiments)
    return px.line(df)


@fig_formatter()
def get_step_plot(experiments: List[str]) -> go.Figure:
    df = get_steps_history_df(experiments)
    return px.line(df)


if __name__ == '__main__':
    app.run_server(debug=True, dev_tools_hot_reload=False)
