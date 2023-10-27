import random

import cv2
from PIL import Image
import numpy as np
import os
import time

import torch
from torch import nn
import torchvision.transforms as T
from main import get_args_parser as get_main_args_parser
from models import build_model
from PIL.JpegImagePlugin import JpegImageFile

torch.set_grad_enabled(False)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("[INFO] 当前使用{}做推断".format(device))

# 图像数据处理
transform = T.Compose([
    # 按照w缩放
    T.Resize(800),
    T.ToTensor(),
    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])


# plot box by opencv
def plot_result(pil_img, prob, boxes, save_name=None, imshow=False, imwrite=False):
    opencvImage = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
    LABEL = ['N/A', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
             'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A',
             'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse',
             'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack',
             'umbrella', 'N/A', 'N/A', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis',
             'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove',
             'skateboard', 'surfboard', 'tennis racket', 'bottle', 'N/A', 'wine glass',
             'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich',
             'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake',
             'chair', 'couch', 'potted plant', 'bed', 'N/A', 'dining table', 'N/A',
             'N/A', 'toilet', 'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard',
             'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'N/A',
             'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush']
    COLOR = [(random.randint(1, 255), random.randint(1, 255), random.randint(1, 255)) for _ in LABEL]
    for p, (xmin, ymin, xmax, ymax) in zip(prob, boxes):
        cl = p.argmax()
        label_text = '{}: {}%'.format(LABEL[cl], round(p[cl] * 100, 2))

        cv2.rectangle(opencvImage, (int(xmin), int(ymin)), (int(xmax), int(ymax)), COLOR[cl], 2)
        # cv2.putText(opencvImage, label_text, (int(xmin) + 10, int(ymin) + 30), cv2.FONT_HERSHEY_SIMPLEX, 1,
        #             COLOR[cl], 2)

    if imshow:
        cv2.imshow('detect', opencvImage)
        cv2.waitKey(0)

    if imwrite:
        if not os.path.exists("./result/pred"):
            os.makedirs('./result/pred')
        cv2.imwrite('./result/pred/{}'.format(save_name), opencvImage)


# 将xywh转xyxy
def box_cxcywh_to_xyxy(x):
    # 对图像归一化后的中心点, 长和宽，相对于整个图像做了归一化
    x_c, y_c, w, h = x.unbind(1)
    b = [(x_c - 0.5 * w), (y_c - 0.5 * h),
         (x_c + 0.5 * w), (y_c + 0.5 * h)]
    return torch.stack(b, dim=1)


def rescale_bboxes(out_bbox, size):
    img_w, img_h = size
    b = box_cxcywh_to_xyxy(out_bbox)
    b = b.cpu().numpy()
    b = b * np.array([img_w, img_h, img_w, img_h], dtype=np.float32)
    return b


def load_model(model_path, args):
    model, _, _ = build_model(args)
    model.cuda()
    model.eval()
    state_dict = torch.load(model_path)  # <-----------修改加载模型的路径
    model.load_state_dict(state_dict["model"])
    model.to(device)
    print("load model sucess")
    return model


def detect(
        im: JpegImageFile,
        model: nn.Module,
        transform: T ,
        prob_threshold: float = 0.5
):
    # mean-std normalize the input image (batch-size: 1)
    img = transform(im).unsqueeze(0)

    # propagate through the model
    img = img.to(device)
    start = time.time()
    outputs = model(img)

    # keep only predictions with 0.7+ confidence
    # print(outputs['pred_logits'].softmax(-1)[0, :, :-1])
    probas = outputs['pred_logits'].softmax(-1)[0, :, :-1]
    keep = probas.max(-1).values > prob_threshold

    probas = probas.cpu().detach().numpy()
    keep = keep.cpu().detach().numpy()

    # convert boxes from [0; 1] to image scales
    bboxes_scaled = rescale_bboxes(outputs['pred_boxes'][0, keep], im.size)
    end = time.time()
    return probas[keep], bboxes_scaled, end - start


if __name__ == "__main__":

    main_args = get_main_args_parser().parse_args()
    # 加载模型
    dfdetr = load_model('./r50_deformable_detr-checkpoint.pth', main_args)  # <--修改为自己加载模型的路径

    files = os.listdir("data/coco/mytestdata")  # <--修改为待预测图片所在文件夹路径

    cn = 0
    waste = 0
    for file in files:
        img_path = os.path.join("data/coco/mytestdata/", file)  # <--修改为待预测图片所在文件夹路径
        im = Image.open(img_path)
        print(type(im))

        scores, boxes, waste_time = detect(im, dfdetr, transform)
        plot_result(im, scores, boxes, save_name=file, imshow=False, imwrite=True)
        # print("{} [INFO] {} time: {} done!!!".format(cn, file, waste_time))

        cn += 1
        waste += waste_time
        waste_avg = waste / cn
