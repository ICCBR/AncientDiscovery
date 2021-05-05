import argparse
import os

import numpy as np
import pandas as pd
import torch

from datasets.ancient_dataset import AncientDataset
from helper import ModelConfiguration
from prediction import Prediction
from utils import tools


def cal_top_ns_acc(correct, ns):
    return np.array([sum([len(correct[p]) for p in correct.keys() if p < n]) for n in ns])


def save_accuracy(acc_df, t, m, fss, source_mode, correct, acc_dir="statistic/accuracy", columns=None):
    top_ns = [1, 10, 20, 50, 100, 200, 400, 600]
    columns = ["Task", "Model", "FSS", "source mode"] + ["Top-" + str(n) for n in top_ns] if not columns else columns
    acc_path = os.path.join(acc_dir, "%s_%s_%s.csv" % (t.replace("->", "_"), m, fss))
    acc = cal_top_ns_acc(correct, top_ns) / len(dataset.target_dataset) * 100
    line = [t, m, fss, source_mode] + ["%.2f" % a for a in acc]
    acc_df = acc_df.append(pd.DataFrame([line], columns=columns), ignore_index=True)
    acc_df[acc_df["Task"] == t].to_csv(acc_path)
    return acc_df


def run_predict(acc_df, fss, outputs):
    result_dic = pred.predict(pred.get_classifier(with_cluster=False, source_outputs=outputs))
    acc_df = save_accuracy(acc_df, task, core, fss, "category", result_dic[f"{pred.set_type}_correct"])
    tools.print_log(result_dic, file=open("log.txt", "a"))
    result_dic = pred.predict(pred.get_classifier(with_cluster=False, source_mode="instance", source_outputs=outputs))
    acc_df = save_accuracy(acc_df, task, core, fss, "instance", result_dic[f"{pred.set_type}_correct"])
    tools.print_log(result_dic, file=open("log.txt", "a"))
    return acc_df


if __name__ == "__main__":
    # add argument here
    parser = argparse.ArgumentParser(description='get prediction result after 10-fold cross validation')
    root, space_size = "../datasets/", 600  # space size should larger than 600, or you can choose other top-n value
    parser.add_argument('--root', '-r', dest="root", metavar='TEXT', help='dataset root directory', default=root)
    parser.add_argument('--size', '-s', dest="size", metavar='TEXT', help='remaining space size', default=space_size)
    args = parser.parse_args()

    # initial statistic directory and statistic data structure
    accuracy_dir = "statistic/accuracy"
    accuracy_df = pd.DataFrame()

    # convert some terms
    convert_map = {"paired": "P", "single": "S", "jia": "OBI", "jin": "BE", "chu": "CSC"}

    # create directory
    tools.make_dir(accuracy_dir)
    base = "log/"

    for root, dirs, files in os.walk(base, topdown=False):
        if "config.yaml" not in files:  # check if there is configuration file
            continue
        # set configuration
        config = tools.load_config(os.path.join(root, "config.yaml"))
        saved_path = os.path.join("_".join(config["paired_chars"]), config["core"] + "_" + config["level"])
        conf = ModelConfiguration(**config, saved_path=saved_path)

        # set checkpoint and load model
        if not os.path.exists(conf.best_model_path):  # check if the best model exist
            print(conf.best_model_path)
            continue
        checkpoint = torch.load(conf.best_model_path, map_location=conf.device)
        model = tools.get_model_class(conf.core, **conf.model_params)
        model.load_state_dict(checkpoint["model"])
        model = model.to(conf.device)
        tools.print_log("Load Model success")

        # set test dataset and get source data
        test_path = "cross_dataset/chars_%s_test.csv" % ("_".join(conf.paired_chars))
        test_char = pd.read_csv(test_path)["test"].tolist()
        dataset = AncientDataset(val_chars=test_char, conf=conf, root_dir=args.root)
        tools.print_log("Load Dataset")

        # set terms
        task = "%s->%s" % (convert_map[conf.paired_chars[0]], convert_map[conf.paired_chars[1]])
        core = convert_map[conf.strategy]
        fss_nor, fss_full = len(dataset.shared_chars), len(dataset.source_full_chars)
        tools.print_log(f"Run {task}: {core} with space sizes are {fss_nor} and {fss_full}")

        # set prediction class and make prediction
        pred = Prediction(dataset, model, core=conf.core, set_type="full", top_k=args.size, batch_size=128)
        tools.print_log("Run Prediction")
        source_outputs = pred.run_model(dataset.source_full_loader, input_source=True)
        # make prediction on normal size and save statistic data
        accuracy_df = run_predict(accuracy_df, fss_full, source_outputs)
        pred.set_type = "shared"
        # make prediction on expansion size and save statistic data
        shared_outputs = source_outputs[[char in set(dataset.shared_chars) for char in source_outputs["label"].values]]
        accuracy_df = run_predict(accuracy_df, fss_nor, shared_outputs)
        accuracy_df.to_csv(f"{accuracy_dir}/accuracy.csv", index=False)
