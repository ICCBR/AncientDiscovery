# -*- coding: utf-8 -*-

# Run this app with `python app.py` and
# visit http://127.0.0.1:8050/ in your web browser.
import base64
import os

import dash
import dash_html_components as html


def b64_image(image_filename):
    with open(image_filename, 'rb') as f:
        image = f.read()
    return 'data:image/png;base64,' + base64.b64encode(image).decode('utf-8')


def add_char_images(char, path, title):
    pages = [html.H2(children=f"{title}: {char}"),
             html.Div([html.Img(src=b64_image(p), width=96, height=96) for p in os.scandir(f"{path}/{char}")])]
    return pages


def run_visualize(ws, root, remove=None, included=None):
    if not included:
        included = os.listdir(f'{root}/{ws}')
    if remove:
        remove_chars = os.listdir(f'{remove}/{ws}')
    external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
    app = dash.Dash(__name__, external_stylesheets=external_stylesheets)
    pages = [html.H1(children=f'Images In dataset {ws}')]
    for char in os.listdir(f"{root}/{ws}"):
        if char not in included:
            continue
        pages.extend(add_char_images(char, f"{root}/{ws}", "Keep"))
        if remove and char in remove_chars:
            pages.extend(add_char_images(char, f"{remove}/{ws}", "Remove"))
    app.layout = html.Div(children=pages)
    app.run_server(debug=True)


if __name__ == '__main__':
    dataset_path = "ancient_5_ds"
    remove_path = "ancient_5_remove"
    chars_included = ["丁", "七"]
    run_visualize("OBI", dataset_path, remove_path)
