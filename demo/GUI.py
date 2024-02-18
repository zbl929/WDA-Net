import os
from tkinter import *
from tkinter import filedialog

import mmcv
from PIL import Image, ImageTk
from matplotlib import pyplot as plt

from mmseg.apis import init_segmentor, inference_segmentor,show_result_pyplot
import numpy as np
import cv2
config_path = ''
checkpoint_path = ''

# 创建GUI窗口
root = Tk()

# 定义选择文件或文件夹的函数
def select_file(var):
    global file_path
    file_path = filedialog.askopenfilename()
    if file_path:
        var.set(file_path)
        show_image(file_path)

# 定义展示图片的函数
def show_image(image_path):
    image = Image.open(image_path)
    image = image.resize((image.width // 4, image.height // 4))
    image_tk = ImageTk.PhotoImage(image)
    image_canvas.image = image_tk
    image_canvas.config(width=image.width, height=image.height)
    image_canvas.create_image(0, 0, anchor=NW, image=image_tk)
    result_canvas.delete("all")
    predict_button.config(state=NORMAL)
    global input_image
    input_image = np.array(image)  # 将 PIL Image 对象转换为 NumPy 数组

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
# 定义预测函数
# def predict():
#     result = inference_segmentor(model, file_path)
#     vis_image = show_result_pyplot(model, file_path, result)
#
#     # vis_iamge = show_result_pyplot(model, file_path, result, out_file='work_dirs/result.png')
#
#     vis_image = vis_image.resize((vis_image.width//4 , vis_image.height//4 ))
#     vis_image_tk = ImageTk.PhotoImage(vis_image)
#     result_canvas.image = vis_image_tk
#     result_canvas.config(width=vis_image.width, height=vis_image.height)
#     result_canvas.create_image(0, 0, anchor=NW, image=vis_image_tk)


def predict():
    result = inference_segmentor(model, file_path)
    vis_image = show_result_pyplot(model, file_path, result)

    # resize image
    vis_image = cv2.resize(vis_image, (vis_image.shape[1]//4, vis_image.shape[0]//4))

    vis_image_tk = ImageTk.PhotoImage(Image.fromarray(vis_image))
    result_canvas.image = vis_image_tk
    result_canvas.config(width=vis_image.shape[1], height=vis_image.shape[0])
    result_canvas.create_image(0, 0, anchor=NW, image=vis_image_tk)

# 创建文件输入框和按钮
input_path_var = StringVar()
input_path_entry = Entry(root, textvariable=input_path_var)
input_path_entry.pack(side=TOP, padx=5, pady=5)
input_path_button = Button(root, text="选择文件", command=lambda: select_file(input_path_var))
input_path_button.pack(side=TOP, padx=5, pady=5)

# 创建预测按钮
predict_button = Button(root, text="预测", command=predict, state=DISABLED)
predict_button.pack(side=TOP, padx=5, pady=5)

# 创建图片展示区域
image_canvas = Canvas(root)
image_canvas.pack(side=LEFT, padx=5, pady=5)

# 创建结果展示区域
result_canvas = Canvas(root)
result_canvas.pack(side=RIGHT, padx=5, pady=5)

# 从配置文件和权重文件构建模型
model = init_segmentor(config_path, checkpoint_path, device='cuda:0')

# 运行GUI窗口
root.mainloop()