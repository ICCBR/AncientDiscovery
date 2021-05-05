import os

import dash
import dash_core_components as dcc
import dash_html_components as html
import numpy as np
import plotly.graph_objects as go
from collections import Counter

names_mapping = {"jia": "OBI", "jin": "BE", "chu": "CSC"}


def run_count(ws, included):
    counts = []
    for root, dirs, files in os.walk(f"{root_dir}/{ws}"):
        if not dirs:
            char = root.split(os.sep)[-1]
            if char in included:
                counts.append(len(files))
    return counts


def set_arrow(x_start, y_start, x_end, y_end):
    arrow = go.layout.Annotation(dict(
        x=x_end,
        y=y_end,
        xref="x", yref="y",
        text="",
        showarrow=True,
        axref="x", ayref='y',
        ax=x_start,
        ay=y_start,
        arrowhead=2,
        arrowwidth=1,
        arrowcolor='rgb(151,51,0)', )
    )
    return arrow


def category_instances():
    x_starts = list(range(5, 100, 20)) + list(range(300, 600, 50))
    y_starts = [280, 150, 100, 85, 70] + [30, 25, 20, 15, 10]
    y_ends = [50 for _ in range(10)]
    arrows = [set_arrow(xs, ys, xs, ye) for xs, ys, ye in zip(x_starts, y_starts, y_ends)]
    title = {
        'text': "Ancient Dataset Distribution",
        'y': 0.9,
        'x': 0.5,
        'xanchor': 'center',
        'yanchor': 'top'}
    fig.update_layout(annotations=arrows, title=title)
    fig.update_xaxes(title_text="Index of Category", zeroline=True, zerolinewidth=2, zerolinecolor='Black')
    fig.update_yaxes(title_text="Number of Images", zeroline=True, zerolinewidth=2, zerolinecolor='Black')
    for ws, count in ws_count.items():
        if len(count) < len(x):
            y = np.pad(count, (0, len(x) - len(count)), "constant", constant_values=(0, 0))
        elif len(count) > len(x):
            y = count[:len(x)]
        else:
            y = count
        fig.add_trace(go.Scatter(x=x, y=y, mode='lines', name=names_mapping[ws], x0=0, y0=0))
    fig.add_trace(go.Scatter(x=x, y=[50 for _ in range(len(x))], mode="lines", name="Threshold"))
    fig.add_trace(go.Scatter(
        x=[20, 450],
        y=[100, 30],
        mode="text",
        name="Text",
        text=["Under Sampling", "Over Sampling"],
        textposition="bottom center",
        textfont=dict(
            family="sans serif",
            size=8,
            color="Black"
        )
    ))


writing_systems = ["jia", "jin", "chu"]
root_dir = "ancient_5_ori"
chars = [os.listdir(f"{root_dir}/{ws}") for ws in writing_systems]
jia_included = set(chars[0]) & set(chars[1])
jin_included = jia_included | (set(chars[1]) & set(chars[2]))
chu_included = set(chars[1]) & set(chars[2])
ws_count = {ws: run_count(ws, char_included)
            for ws, char_included in zip(writing_systems, [jia_included, jin_included, chu_included])}
ws_count = {ws: sorted(counts, reverse=True) for ws, counts in ws_count.items()}
max_len = 600
# max_len = max([len(count) for count in ws_count.values()])
x = np.arange(max_len)
fig = go.Figure()
# category_instances()
"""
dark grey, light grey, white
50, 50 100, 100 200, 200 300, 300
the inscription number in each category
the number of categories
"""
# title = {
#     'text': "Original Ancient-3 Data-set Distribution",
#     'y': 0.8,
#     'x': 0.5,
#     'xanchor': 'center',
#     'yanchor': 'top'
# }
fig.update_layout(plot_bgcolor="white", legend_font_size=24, font_size=24, legend_x=0.9, font_color="black")
fig.update_xaxes(title_text="The Inscription Number in Each Category", showline=True, linewidth=2, linecolor='black',
                 mirror=True)
fig.update_yaxes(title_text="The Number of Categories", showgrid=True, gridcolor="lightgrey", showline=True, linewidth=2,
                 linecolor='black', mirror=True)

ws_count_2 = {ws: {"x": [], "y": []} for ws in ws_count.keys()}
color_mapping = {"jia": "darkgrey", "jin": "lightgrey", "chu": "lightslategrey"}
for ws, counts in ws_count.items():
    result = Counter(counts)
    for k, v in result.items():
        ws_count_2[ws]["x"].append(k)
        ws_count_2[ws]["y"].append(v)
    y = [sum([ws_count_2[ws]["y"][i] for i, x in enumerate(ws_count_2[ws]["x"]) if start < x <= end])
         for start, end in [(0, 50), (50, 100), (100, 200), (200, 300), (300, 600)]]
    fig.add_trace(go.Bar(x=["(0, 50]", "(50, 100]", "(100, 200]", "(200, 300]", "(300, 600]"], y=y, text=y,
                         textposition="outside", name=names_mapping[ws], marker_color=color_mapping[ws]))

fig.show()

app = dash.Dash(__name__)
# wide_df = px.data.medals_wide()
#
# fig = px.bar(wide_df, x="nation", y=["gold", "silver", "bronze"], title="Wide-Form Input")
# fig.show()
app.layout = html.Div([
    dcc.Graph(figure=fig),
])

app.run_server(debug=True)
