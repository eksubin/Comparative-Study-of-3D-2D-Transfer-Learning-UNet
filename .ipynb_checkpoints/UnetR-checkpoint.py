from monai.utils import first, set_determinism
from monai.transforms import (
    AsDiscrete,
    AsDiscreted,
    EnsureChannelFirstd,
    Compose,
    CropForegroundd,
    LoadImaged,
    Orientationd,
    RandCropByPosNegLabeld,
    SaveImaged,
    ScaleIntensityRanged,
    Spacingd,
    Invertd,
    Lambdad,
    Activations,
    ScaleIntensityRange,
    Lambda,
    Transposed
)
from monai.handlers.utils import from_engine
from monai.networks.nets import UNet, UNETR
from monai.networks.layers import Norm
from monai.metrics import DiceMetric
from monai.losses import DiceLoss
from monai.inferers import sliding_window_inference
from monai.data import CacheDataset, DataLoader, Dataset, decollate_batch
from monai.config import print_config
from monai.apps import download_and_extract
import numpy as np
import torch
import matplotlib.pyplot as plt
import tempfile
import shutil
import os
import glob
import nrrd


def parse_args():
    parser = argparse.ArgumentParser(
        description="Training script with argparse")

    # Add argument for num_of_epochs
    parser.add_argument(
        "--epochs",
        type=int,
        required=True,
        help="Number of training epochs"
    )

    # Add argument for train_location
    parser.add_argument(
        "--train_location",
        type=str,
        required=True,
        help="Location of training data"
    )

    return parser.parse_args()


def binarize(label, threshold=0.1):
    binary_mask = (label > threshold)
    binary_mask[binary_mask > 0] = 1  # Set all non-zero pixels to 1
    return binary_mask


def unetR3D(number_of_epoch=6, train_location="./Train"):

    max_epochs = number_of_epoch
    val_interval = 2
    best_metric = -1
    best_metric_epoch = -1
    epoch_loss_values = []
    metric_values = []
    post_pred = Compose([AsDiscrete(argmax=True, to_onehot=2)])
    post_label = Compose([AsDiscrete(to_onehot=2)])

    directory = os.environ.get("DATA_DIRECTORY")
    root_dir = train_location
    print(root_dir)

    train_images = sorted(
        glob.glob(os.path.join(root_dir, "RTrainVolumes", "*.nrrd")))
    train_labels = sorted(
        glob.glob(os.path.join(root_dir, "RTrainLabels", "*.nrrd")))
    data_dicts = [{"image": image_name, "label": label_name}
                  for image_name, label_name in zip(train_images, train_labels)]
    train_files, val_files = data_dicts[:-9], data_dicts[-9:]

    train_transforms = Compose(
        [
            LoadImaged(keys=["image", "label"],
                       reader="NrrdReader", image_only=True),
            Transposed(keys=['image', 'label'], indices=[2, 1, 0]),
            EnsureChannelFirstd(keys=["image", "label"]),
            Lambdad(("label"), binarize),
            ScaleIntensityRanged(
                keys=["image"],
                a_min=-57,
                a_max=1000,
                b_min=0.0,
                b_max=1.0,
                clip=True,
            ),
            CropForegroundd(keys=["image", "label"], source_key="image"),
            RandCropByPosNegLabeld(
                keys=["image", "label"],
                label_key="label",
                spatial_size=(128, 128, 16),
                pos=1,
                neg=1,
                num_samples=4,
                image_key="image",
                image_threshold=0,
            ),
        ]
    )
    val_transforms = Compose(
        [
            LoadImaged(keys=["image", "label"],
                       reader="NrrdReader", image_only=True),
            Transposed(keys=['image', 'label'], indices=[2, 1, 0]),
            EnsureChannelFirstd(keys=["image", "label"]),
            Lambdad(("label"), binarize),
            ScaleIntensityRanged(
                keys=["image"],
                a_min=-57,
                a_max=1000,
                b_min=0.0,
                b_max=1.0,
                clip=True,
            ),
            CropForegroundd(keys=["image", "label"], source_key="image"),
        ]
    )

    train_ds = CacheDataset(
        data=train_files, transform=train_transforms, cache_rate=1.0, num_workers=4)

    train_loader = DataLoader(train_ds, batch_size=2,
                              shuffle=True, num_workers=4)

    val_ds = CacheDataset(
        data=val_files, transform=val_transforms, cache_rate=1.0, num_workers=4)

    val_loader = DataLoader(val_ds, batch_size=1, num_workers=4)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Using device:', device)

    # Additional Info when using cuda
    if device.type == 'cuda':
        print(torch.cuda.get_device_name(0))
        print('Memory Usage:')
        print('Allocated:', round(torch.cuda.memory_allocated(0)/1024**3, 1), 'GB')
        print('Cached:   ', round(torch.cuda.memory_reserved(0)/1024**3, 1), 'GB')


    model = UNETR(
        in_channels=1,
        out_channels=2,
        img_size=(128, 128, 16),
        feature_size=16,
        hidden_size=768,
        mlp_dim=3072,
        num_heads=12,
        pos_embed="perceptron",
        norm_name="instance",
        res_block=True,
        dropout_rate=0.0,
    ).to(device)

    loss_function = DiceLoss(to_onehot_y=True, softmax=True)
    optimizer = torch.optim.Adam(model.parameters(), 1e-4)
    dice_metric = DiceMetric(include_background=False, reduction="mean")

    for epoch in range(max_epochs):
        print("-" * 10)
        print(f"epoch {epoch + 1}/{max_epochs}")
        model.train()
        epoch_loss = 0
        step = 0
        for batch_data in train_loader:
            step += 1
            inputs, labels = (
                batch_data["image"].to(device),
                batch_data["label"].to(device),
            )
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = loss_function(outputs, labels)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            # print(f"{step}/{len(train_ds) // train_loader.batch_size}, " f"train_loss: {loss.item():.4f}")
        epoch_loss /= step
        epoch_loss_values.append(epoch_loss)
        print(f"epoch {epoch + 1} average loss: {epoch_loss:.4f}")

        if (epoch + 1) % val_interval == 0:
            model.eval()
            with torch.no_grad():
                for val_data in val_loader:
                    val_inputs, val_labels = (
                        val_data["image"].to(device),
                        val_data["label"].to(device),
                    )
                    roi_size = (128, 128, 16)
                    sw_batch_size = 4
                    val_outputs = sliding_window_inference(
                        val_inputs, roi_size, sw_batch_size, model)
                    val_outputs = [post_pred(i)
                                   for i in decollate_batch(val_outputs)]
                    val_labels = [post_label(i)
                                  for i in decollate_batch(val_labels)]
                    # compute metric for current iteration
                    dice_metric(y_pred=val_outputs, y=val_labels)

                # aggregate the final mean dice result
                metric = dice_metric.aggregate().item()
                # reset the status for next validation round
                dice_metric.reset()

                metric_values.append(metric)
                if metric > best_metric:
                    best_metric = metric
                    best_metric_epoch = epoch + 1
                    torch.save(model.state_dict(), os.path.join(
                        "./UnetrOutput", "best_metric_model.pth"))
                    print("saved new best metric model")
                print(
                    f"current epoch: {epoch + 1} current mean dice: {metric:.4f}"
                    f"\nbest mean dice: {best_metric:.4f} "
                    f"at epoch: {best_metric_epoch}"
                )

    print(
        f"train completed, best_metric: {best_metric:.4f} " f"at epoch: {best_metric_epoch}")

    plt.figure("train", (12, 6))
    plt.subplot(1, 2, 1)
    plt.title("Epoch Average Loss")
    x = [i + 1 for i in range(len(epoch_loss_values))]
    y = epoch_loss_values
    plt.xlabel("epoch")
    plt.plot(x, y)
    plt.subplot(1, 2, 2)
    plt.title("Val Mean Dice")
    x = [val_interval * (i + 1) for i in range(len(metric_values))]
    y = metric_values
    plt.xlabel("epoch")
    plt.plot(x, y)
    # Saving the figure.
    plt.savefig(os.path.join("./UnetrOutput", "modelPerformance.jpg"))

    model.load_state_dict(torch.load(
        os.path.join("./UnetrOutput", "best_metric_model.pth")))

    model.eval()
    with torch.no_grad():
        for i, val_data in enumerate(val_loader):
            roi_size = (128, 128, 16)
            sw_batch_size = 4
            val_outputs = sliding_window_inference(
                val_data["image"].to(device), roi_size, sw_batch_size, model)
            # plot the slice [:, :, 80]
            plt.figure("check", (18, 6))
            plt.subplot(1, 3, 1)
            plt.title(f"image {i}")
            plt.imshow(val_data["image"][0, 0, :, :, 10], cmap="gray")
            plt.subplot(1, 3, 2)
            plt.title(f"label {i}")
            plt.imshow(val_data["label"][0, 0, :, :, 10])
            plt.subplot(1, 3, 3)
            plt.title(f"output {i}")
            plt.imshow(torch.argmax(
                val_outputs, dim=1).detach().cpu()[0, :, :, 10])
            plt.savefig(os.path.join("./UnetrOutput", f"modelOutput{i}.jpg"))


if __name__ == "__main__":
    import argparse
    args = parse_args()

    epochs = args.epochs
    train_location = args.train_location

    print("Number of Epochs:", epochs)
    print("Training Location:", train_location)

    # run the aimodel
    unetR3D(epochs, train_location)