import torch
import requests

from PIL import Image
from model import DETRdemo
from utils import rescale_bboxes, plot_results, transform

def detect(im, model, transform):
    # mean-std normalize the input image (batch-size: 1)
    img = transform(im).unsqueeze(0)

    # demo model only support by default images with aspect ratio between 0.5 and 2
    # if you want to use images with an aspect ratio outside this range
    # rescale your image so that the maximum size is at most 1333 for best results
    assert img.shape[-2] <= 1600 and img.shape[-1] <= 1600, 'demo model only supports images up to 1600 pixels on each side'

    # propagate through the model
    outputs = model(img)

    # keep only predictions with 0.7+ confidence
    probas = outputs['pred_logits'].softmax(-1)[0, :, :-1]
    keep = probas.max(-1).values > 0.7

    # convert boxes from [0; 1] to image scales
    bboxes_scaled = rescale_bboxes(outputs['pred_boxes'][0, keep], im.size)
    return probas[keep], bboxes_scaled


if __name__=="__main__":
    detr = DETRdemo(num_classes=91)

    # Load the state dict
    state_dict = torch.hub.load_state_dict_from_url(
        url='https://dl.fbaipublicfiles.com/detr/detr_demo-da2a99e9.pth',
        map_location='cpu', check_hash=True
    )

    # Rename keys in the state_dict to match the model
    state_dict["linear_box.weight"] = state_dict.pop("linear_bbox.weight")
    state_dict["linear_box.bias"] = state_dict.pop("linear_bbox.bias")

    # Load the updated state_dict
    detr.load_state_dict(state_dict)
    detr.eval()

    url = 'http://images.cocodataset.org/val2017/000000039769.jpg'
    im = Image.open(requests.get(url, stream=True).raw)

    scores, boxes = detect(im, detr, transform)

    plot_results(im, scores, boxes)