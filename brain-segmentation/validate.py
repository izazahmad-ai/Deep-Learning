import os
import argparse

import torch
import numpy as np
from omegaconf import OmegaConf
from monai.inferers import sliding_window_inference

from utils.data_utils import get_loader
from utils.model_utils import get_model
from utils.utils import dice, ImageSaver
from visualize import create_image_visual


def main(model_config: dict, data_config: dict, title: str = ''):
    device = torch.device(model_config['device']) if torch.cuda.is_available() else torch.device('cpu')
    loader = get_loader(
        data_dir=os.path.expanduser(data_config['data_dir']),
        batch_size=1,
        roi_size=data_config['inf_size'],
        spacing=data_config['spacing'],
        modality=data_config['modality'],
        a_min=data_config['a_min'],
        a_max=data_config['a_max'],
        b_min=data_config['b_min'],
        b_max=data_config['b_max'],
        RandFlipd_prob=data_config['RandFlipd_prob'],
        RandRotate90d_prob=data_config['RandRotate90d_prob'],
        RandScaleIntensityd_prob=data_config['RandScaleIntensityd_prob'],
        RandShiftIntensityd_prob=data_config['RandShiftIntensityd_prob'],
        gauss_noise_prob=data_config['gauss_noise_prob'],
        gauss_noise_std=data_config['gauss_noise_std'],
        gauss_smooth_prob=data_config['gauss_smooth_prob'],
        gauss_smooth_std=data_config['gauss_smooth_std'],
        device=device,
        n_workers=0,
        cache_num=0,
    )[1]
    model = get_model(
        model_name=model_config['model_name'],
        in_channels=model_config['in_channels'],
        out_channels=model_config['out_channels'],
        inf_size=model_config['inf_size'],
        feature_size=model_config['feature_size'],
        hidden_size=model_config['hidden_size'],
        mlp_dim=model_config['mlp_dim'],
        num_heads=model_config['num_heads'],
        pos_embed=model_config['pos_embed'],
        norm_name=model_config['norm_name'],
        conv_block=True,
        res_block=True,
        dropout_rate=model_config['dropout_rate'],
        device=device
    )
    model_dict = torch.load(model_config['path_to_checkpoint'], map_location=device)['state_dict']
    model.load_state_dict(model_dict)
    model.to(device)
    model.eval()

    labels = list(range(model_config['out_channels']))
    image_saver = ImageSaver(
        os.path.join(model_config['log_dir'], 'images'),
        save_name=title
    )

    with torch.inference_mode():
        dice_scores = []
        for i, batch in enumerate(loader):
            val_inputs, val_labels = batch["image"].to(device), batch["label"].to(device)
            val_outputs = sliding_window_inference(
                val_inputs,
                roi_size=(data_config['inf_size'], data_config['inf_size'], data_config['inf_size']),
                sw_batch_size=data_config['sw_batch_size'],
                predictor=model,
                overlap=data_config['infer_overlap']
            )
            val_outputs = torch.softmax(val_outputs, 1).cpu().numpy()
            val_outputs = np.argmax(val_outputs, axis=1).astype(np.uint8)
            val_labels = val_labels.cpu().numpy().squeeze()
            dice_score_sample = [
                dice(val_outputs == int(label), val_labels == int(label)) for label in labels
            ]
            dice_scores.append(dice_score_sample)

            dice_scores_by_class = np.round(dice_score_sample, 3).tolist()
            dice_sample = np.mean(dice_score_sample[1:])
            sample_name = os.path.basename(batch['image_meta_dict']["filename_or_obj"][0])
            title = f'Dice: {dice_sample:.3f} | {dice_scores_by_class} | {sample_name}'
            image_visualization = create_image_visual(
                val_inputs.cpu().numpy()[0, 0],
                val_labels,
                val_outputs,
                title
            )
            image_saver.save_image(image_visualization, sample_name)

            print(f"Class Dice: {dice_score_sample}")

        dice_scores = np.array(dice_scores)
        overall_dice_by_class = np.mean(dice_scores, axis=0).round(3)
        overall_dice = np.mean(dice_scores[:, 1:]).round(3)
        print(f"Overall Mean Dice By Class: {overall_dice_by_class}")
        print(f"Overall Mean Dice: {overall_dice}")

        image_saver.save_results(overall_dice_by_class=overall_dice_by_class, overall_dice=overall_dice)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_config', type=str)
    parser.add_argument('--data_config', type=str)
    parser.add_argument('--path_to_checkpoint', type=str, default=None)
    parser.add_argument('--title', type=str, default='')
    args = parser.parse_args()

    path_to_model_confid = OmegaConf.load(args.model_config)
    model_config = OmegaConf.to_container(path_to_model_confid, resolve=True)

    path_to_data_config = OmegaConf.load(args.data_config)
    data_config = OmegaConf.to_container(path_to_data_config, resolve=True)

    if args.path_to_checkpoint is not None:
        model_config['path_to_checkpoint'] = args.path_to_checkpoint

    main(model_config, data_config, args.title)
