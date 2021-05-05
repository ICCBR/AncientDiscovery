import torch
import os
import shutil
import models as module_arch
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from tqdm import tqdm
from utils import cluster, tools
from utils.dimension_reduction import get_reduction_result
from datasets.data_loader import WSDataLoader
from utils.tools import ensure_dir
import pathlib

temp = pathlib.PosixPath
pathlib.PosixPath = pathlib.WindowsPath
root_data_path = "../datasets/ancient_5_ori/"


def get_shared_dataloader(target, source):
    """
    Return dataloader with shared characters between target and source
    Args:
        target: target name
        source: source name

    Returns: target loader and source loader

    """
    target_path = os.path.join(root_data_path, target)
    source_path = os.path.join(root_data_path, source)
    chars_shared = [c for c in os.listdir(target_path) if c in os.listdir(source_path)]
    # chars_shared = ["一", "万", "上", "不", "丑", "且", "丘", "东", "义", "乎"]
    target_loader = WSDataLoader(target_path, chars_include=chars_shared, batch_size=64, return_path=True)
    source_loader = WSDataLoader(source_path, chars_include=chars_shared, batch_size=64, return_path=True)
    return target_loader, source_loader


def run_model(data_loader, model):
    """
    Run data with defined model
    Args:
        data_loader: data loader object
        model: VAE model to output z

    Returns: latent vectors, labels, and file paths

    """
    zs, ys, paths = [], [], []
    with torch.no_grad():
        for batch in tqdm(data_loader, total=len(data_loader)):
            image, y, path = batch
            recon, z, mu, log_var = model(image.to(device))
            zs.extend(z.cpu().tolist())
            ys.extend(y.cpu().tolist())
            paths.extend(path)
    return zs, ys, paths


def run_reduction(zs, ys, paths):
    red_df = pd.DataFrame({"feature": zs, "label": ys, "path": paths})
    red_df = get_reduction_result(red_df)
    red_df["x"] = red_df.feature.apply(lambda i: i[0])
    red_df["y"] = red_df.feature.apply(lambda i: i[1])
    return red_df


def run_split_cluster(cluster_df, keep_df=None, remove_df=None):
    """
    Split cluster dataframe into keep and remove dataframe
    Args:
        cluster_df: dataframe after cluster
        keep_df: dataframe keep for the rest operations
        remove_df: dataframe that will be removed

    Returns:

    """
    if keep_df is None:
        keep_df = pd.DataFrame(columns=cluster_df.columns)
    if remove_df is None:
        remove_df = pd.DataFrame(columns=cluster_df.columns)
    for label, group_df in cluster_df.groupby(["label"]):
        # only keep the maximum cluster
        center_count = {center: len(group_df[group_df.center == center]) for center in group_df.center.unique()}
        max_center = sorted(center_count.items(), key=lambda i: i[1], reverse=True)[0][0]
        keep_df = keep_df.append(group_df[group_df.center == max_center], ignore_index=True)
        remove_df = remove_df.append(group_df[group_df.center != max_center], ignore_index=True)
    return keep_df, remove_df


def run_clean_cluster(red_df, remove_df=None):
    """
    Run first time cluster for cleaning the data
    Args:
        remove_df: The dataframe store remove images
        red_df: The reduction dataframe

    Returns: the dataframes that tend to keep or remove after cluster

    """
    cluster_df = cluster.get_cluster_output_df(red_df, False, quantile=0.8)
    return run_split_cluster(cluster_df, remove_df)


def run_ds_cluster(keep_df, remove_df, iter_num=1):
    ds_df = pd.DataFrame(columns=keep_df.columns)
    for label, group_df in keep_df.groupby(["label"]):
        df_size = len(group_df)
        if df_size <= 50 or (iter_num > 10 and df_size < 80):
            ds_df = ds_df.append(group_df, ignore_index=True)
        else:
            if 50 < df_size <= 100:
                quantile = 0.8 - 0.0001 * (df_size - 50) * iter_num
            elif 100 < df_size <= 200:
                quantile = 0.8 - 0.0005 * (df_size - 100) * iter_num
            else:
                quantile = 0.8 - 0.001 * (df_size - 200) * iter_num
            threshold = 1 / (iter_num + 1)
            quantile = max(quantile, threshold)
            label_df = cluster.run_cluster(label, group_df, add_center=False, quantile=quantile)
            keep, remove_df = run_split_cluster(label_df, remove_df=remove_df)
            ds_df = ds_df.append(keep, ignore_index=True)
    tools.print_log("shape of output after down sampling cluster: %s" % str(ds_df.shape))
    return ds_df, remove_df


def plot_df(df, color, name=""):
    f = px.scatter(df, x="x", y="y", color=color)
    f.show()
    ensure_dir("saved/html/", exist_ok=True)
    f.write_html(f"saved/html/{name}.html")


def run_statistic(df):
    # statistic about the distribution of label
    return sorted(df.label.value_counts(), reverse=True)


def compare_plot(ori_count, ds_count):
    fig = make_subplots(rows=1, cols=2, subplot_titles=["Original Dataset Distribution", "After Down Sampling"])
    trace0 = go.Bar(x=np.arange(len(ori_count)), y=ori_count)
    trace1 = go.Bar(x=np.arange(len(ds_count)), y=ds_count)
    fig.append_trace(trace0, 1, 1)
    fig.append_trace(trace1, 1, 2)
    fig.show()
    ensure_dir("saved/html/", exist_ok=True)
    fig.write_html("saved/html/down_sampling.html")


def move_path(df, move_dir):
    ensure_dir(move_dir)
    for p in df.path.values:
        move_p = f"{move_dir}/{p.split(os.sep)[-2]}/"
        ensure_dir(move_p, exist_ok=True)
        shutil.copy(p, f"{move_p}/{p.split(os.sep)[-1]}")


def run_down_sampling(loader, model, ws):
    # run target dataframe
    red_df = run_reduction(*run_model(loader, model))
    keep_df, remove_df = run_clean_cluster(red_df)
    remove_df = pd.DataFrame(columns=keep_df.columns)
    iter_num = 1
    while max(keep_df.label.value_counts()) > 80:
        # run cluster here
        keep_df, remove_df = run_ds_cluster(keep_df, remove_df, iter_num)
        iter_num += 1
    compare_plot(run_statistic(red_df), run_statistic(keep_df))
    plot_df(red_df, "label", "original")
    plot_df(keep_df, "label", "after_ds")
    down_sample_dir = "../datasets/ancient_5_ds_2"
    remove_dir = "../datasets/ancient_5_remove_2"
    move_path(keep_df, f"{down_sample_dir}/{ws}")
    move_path(remove_df, f"{remove_dir}/{ws}")
    return keep_df, remove_df


if __name__ == "__main__":
    check_path = "checkpoint/vae/best_model.pth"
    checkpoint = torch.load(check_path)
    vanilla_model = checkpoint["config"].init_obj("arch", module_arch)
    vanilla_model.load_state_dict(checkpoint["state_dict"])
    # setup device and model here
    device = torch.device("cuda")
    vanilla_model = vanilla_model.to(device)
    vanilla_model.eval()
    # load data and run model
    obi_loader, be_loader = get_shared_dataloader("jia", "jin")
    # run down sampling method
    run_down_sampling(obi_loader, vanilla_model, "OBI")
    # run_down_sampling(be_loader, vanilla_model, "BE")
