import os
from argparse import ArgumentParser

import cv2
import mmcv
from matplotlib import pyplot as plt

from mmseg.apis import inference_segmentor, init_segmentor, show_result_pyplot
from mmseg.core.evaluation import get_palette

def show_result_pyplot(model,
                       img,
                       result,
                       palette=None,
                       fig_size=(15, 10),
                       opacity=0.2,
                       title='',
                       block=True):

    if hasattr(model, 'module'):
        model = model.module
    img = model.show_result(
        img, result, palette=palette, show=False, opacity=opacity)
    plt.figure(figsize=fig_size)
    img= mmcv.bgr2rgb(img)
    return img
def main():
    parser = ArgumentParser()
    parser.add_argument('imgs-folder', help='Images folder')
    parser.add_argument('config', help='test config file path')
    parser.add_argument('checkpoint',help='checkpoint file')

    parser.add_argument(
        '--device', default='cuda:0', help='Device used for inference')
    # parser.add_argument(
    #     '--palette',
    #     help='Color palette used for segmentation map')
    parser.add_argument(
        '--opacity',
        type=float,
        default=0.2,
        help='Opacity of painted segmentation map. In (0, 1] range.')
    args = parser.parse_args()

    # build the model from a config file and a checkpoint file

    model = init_segmentor(args.config, args.checkpoint, device=args.device)

    results_folder = 'seg_img'
    if not os.path.exists(results_folder):
        os.mkdir(results_folder)

    for img_name in os.listdir(args.imgs_folder):
        img_path = os.path.join(args.imgs_folder, img_name)

        result = inference_segmentor(model, img_path)

        # 显示并保存图片
        result_img = show_result_pyplot(model, img_path, result, opacity=args.opacity)
        cv2.imwrite(os.path.join(results_folder, img_name), result_img)


if __name__ == '__main__':
    main()