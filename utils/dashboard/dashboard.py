"""
Usage:
`python dashboard.py`
"""
import os.path
from typing import List, Dict, Union
import sys

import dash
import dash_bootstrap_components as dbc
import dash_core_components as dcc
import dash_html_components as html
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from dash.dependencies import Input, Output, State

try:
    from .utils import *
    from .data_loader import *
except ImportError:
    from utils import *
    from data_loader import *

# Initialize app
app = dash.Dash(__name__, meta_tags=[{"name": "viewport", "content": "width=device-width"}],
                external_stylesheets=[dbc.themes.BOOTSTRAP],
                suppress_callback_exceptions=False)

grid_searches = get_grid_searches()
grid_search_params = get_all_grid_search_params()


def get_sliders() -> html.P:
    """
    Return a div of hidden divs with slider text above the slider.
    """
    sliders = []
    for grid_search, param_dict in grid_search_params.items():
        for param, values in param_dict.items():
            prefix = grid_search + param
            s = html.Div([
                html.P(children=[param], id=prefix + '-slider-state'),
                dcc.Slider(
                    id=prefix + '-slider',
                    min=1,
                    max=len(values),
                    value=1,
                    step=1,
                    included=False,
                    marks={n + 1: v for n, v in enumerate(values)}
                ),
            ],
                style={'display': 'none'},
                id=prefix + '-slider-div',
            )
            sliders.append(s)
    return html.P(children=sliders, id='grid-search-sliders')


invisible_grid_search_dropdown_div = html.Div(children=[
    dcc.Dropdown(id='grid-search-params-selector',
                 options=[dict(label='', value='')],
                 style={'display': 'none'})],
    id='grid-search-params-selector-div')

# Create app layout
app.layout = html.Div(
    [
        # empty Div to trigger javascript file for graph resizing
        html.Div(id='output-clientside'),

        # Header
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

        # Reward and steps
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

        # Grid Search
        html.Div(
            [
                html.Div(
                    [
                        html.P(
                            [
                                "Select grid search:",
                                dcc.Dropdown(
                                    id='grid-search-selector',
                                    options=grid_searches,
                                    multi=False
                                ),
                                invisible_grid_search_dropdown_div,
                                html.Br(),
                                get_sliders()
                            ]
                        ),
                    ],
                    className='pretty_container four columns',
                    id='grid-search-selector-div',
                ),
                html.Div(
                    [
                        dcc.Graph(id='grid-search-plot')
                    ],
                    className='pretty_container twelve columns',
                ),
            ],
            className='flex-display',
            style={'margin-bottom': '25px'}
        ),
    ],
    id='mainContainer',
    style={'display': 'flex', 'flex-direction': 'column'},
)


@app.callback(Output('grid-search-sliders', 'children'),
              [Input('grid-search-selector', 'value'),
               Input('grid-search-params-selector', 'value')],
              [State('grid-search-sliders', 'children')],
              prevent_initial_call=False)
def show_grid_search_sliders(grid_search: str, params: List[str], state: List):
    """
    Show the sliders that were selected using the multiple dropdown. Hide the others.

    :param grid_search: name of the grid search
    :param params: list of parameters
    :param state: children of the grid-search-sliders Div
    :return: updated state
    """
    ids = []
    for gs, param_dict in grid_search_params.items():
        for p in param_dict.keys():
            ids.append(gs + p + '-slider-div')

    if params:
        selected_ids = {grid_search + p + '-slider-div' for p in params}
    else:
        selected_ids = set()

    for n, i in enumerate(ids):
        if grid_search and (i not in selected_ids):
            state[n]['props']['style'] = None
        else:
            state[n]['props']['style'] = dict(display='none')
    return state


def assign_slider_text_update_callback(grid_search: str, param: str) -> None:
    """
    Register a callback on the text above categorical sliders. It will then update that text according to the current
    selection.

    :param grid_search: the name of the grid search
    :param param: the name of the parameter the slider adjusts
    """
    prefix = grid_search + param

    def slider_text_update(value: int, marks):
        value = marks[str(value)]
        return [f"{param}: {value}"]

    app.callback(output=Output(prefix + '-slider-state', 'children'),
                 inputs=[Input(prefix + '-slider', 'value')],
                 state=[State(prefix + '-slider', 'marks')],
                 prevent_initial_call=False)(slider_text_update)


# Construct slider inputs and assign callbacks to slider labels
slider_inputs = []
for gs, param_dict in grid_search_params.items():
    for p in param_dict.keys():
        assign_slider_text_update_callback(gs, p)
        slider_inputs.append(Input(gs + p + '-slider', 'value'))


@app.callback(Output('grid-search-params-selector-div', 'children'),
              [Input('grid-search-selector', 'value')], )
def make_grid_search_param_selector(grid_search: str) -> List[dcc.Dropdown]:
    if grid_search:
        options = grid_search_params[grid_search].keys()
        options = [dict(label=o, value=o) for o in options]
        return [dcc.Dropdown(
            id='grid-search-params-selector',
            options=options,
            multi=True
        )]
    else:
        return invisible_grid_search_dropdown_div


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
