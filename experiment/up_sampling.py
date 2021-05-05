import os
import pathlib
import imgaug.augmenters as iaa


temp = pathlib.PosixPath
pathlib.PosixPath = pathlib.WindowsPath


if __name__ == "__main__":
    root_data_path = "../datasets/ancient_5_ds"
    ws = "OBI"
    up_chars = {}
    aug = iaa.GaussianBlur(sigma=(0.0, 3.0))
    for root, dirs, files in os.walk(f"{root_data_path}/{ws}"):
        if dirs:
            continue
        if len(files) > 50:
            up_chars[root.split(os.sep)[-1]] = len(files)
    print(sum(up_chars.values()))
    print(up_chars)
    # obi_loader = get_dataloader(f"{root_data_path}/{ws}")
    # check_path = "checkpoint/vae/best_model.pth"
    # checkpoint = torch.load(check_path)
    # vanilla_model = checkpoint["config"].init_obj("arch", module_arch)
    # vanilla_model.load_state_dict(checkpoint["state_dict"])

    # # setup device and model here
    # device = torch.device("cuda")
    # vanilla_model = vanilla_model.to(device)
    # vanilla_model.eval()
