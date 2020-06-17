#!/usr/bin/env python
"""
Usage:
`python dashboard.py`
"""
import os.path
from typing import List, Union, Dict, Tuple

import dash
import dash_bootstrap_components as dbc
import dash_core_components as dcc
import dash_html_components as html
import plotly.express as px
import plotly.graph_objects as go
from dash.dependencies import Input, Output, State

from utils.utils import load_history, get_experiments

# Initialize app and cache
app = dash.Dash(__name__, meta_tags=[{'name': 'viewport', 'content': 'width=device-width'}],
                external_stylesheets=[dbc.themes.BOOTSTRAP],
                suppress_callback_exceptions=True)

# color for frontend
colors = {
    'background': '#111111',
    'text'      : '#7FDBFF'
}


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
                            src=app.get_asset_url('ucsd-logo.png'),
                            id='ucsd-logo',
                            style={
                                'height'       : '60px',
                                'width'        : 'auto',
                                'margin-bottom': '25px',
                            },
                        )
                    ],
                    className='one-third column',
                ),
                html.Div(
                    [
                        html.Div(
                            [
                                html.H1(
                                    "RL for Underwater Communication Experiment Dashboard",
                                    style={'margin-bottom': '0px'},
                                ),
                            ]
                        )
                    ],
                    className='two-half column',
                    id='title',
                ),
            ],
            id='header',
            className='row flex-display',
            style={'margin-bottom': '25px'},
        ),

        # Introduction
        html.Div(
            [
                dcc.Markdown(load_markdown_text('introduction'), dedent=False)
            ],
            className='pretty_container',
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
                                    value=[],
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
    [Input('experiment', 'value')]
)
def make_experiments_plot(experiments: List[str]) -> go.Figure:
    if not experiments:
        fig = get_empty_sunburst("Select an experiment")
    else:
        fig = get_experiments_plot(experiments)
    return fig


@fig_formatter()
def get_experiments_plot(experiments: List[str]) -> go.Figure:
    return px.line()


if __name__ == '__main__':
    app.run_server(debug=True, dev_tools_hot_reload=False)
