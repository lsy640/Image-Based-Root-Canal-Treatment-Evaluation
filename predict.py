# predict.py

import torch
from model import resnet101, resnet34, resnet50
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt
import json
import os


def main(model_weight_path):
    data_transform = transforms.Compose(
        [
            transforms.Resize(224),
            # transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.2369732, 0.2066169, 0.043554377], [0.24172652, 0.37390167, 0.16018319])
        ])
    # load image
    img_file = "./rgb/rgb.jpg"
    img = Image.open(img_file)
    img = data_transform(img)
    # expand batch dimension
    img = torch.unsqueeze(img, dim=0)

    # read class_indict
    try:
        json_file = open('./class_indices.json', 'r')
        class_indict = json.load(json_file)
    except Exception as e:
        print(e)
        exit(-1)

    # create model
    model = resnet50(num_classes=4)
    # load model weights
    model.load_state_dict(torch.load(model_weight_path, map_location='cpu'))
    model.eval()
    with torch.no_grad():
        # predict class
        output = torch.squeeze(model(img))
        predict = torch.softmax(output, dim=0)
        predict_cla = torch.argmax(predict).numpy()
        predict_top4 = torch.topk(predict, k=4)
    print("top1:", class_indict[str(predict_cla)],
          "%.8f" % predict[predict_cla].numpy(),
          "top2:", class_indict[str(predict_top4[1].numpy()[1])],
          "%.8f" % predict[predict_top4[1]].numpy()[1],
          "top3:", class_indict[str(predict_top4[1].numpy()[2])],
          "%.8f" % predict[predict_top4[1]].numpy()[2],
          "top4:", class_indict[str(predict_top4[1].numpy()[3])],
          "%.8f" % predict[predict_top4[1]].numpy()[3],
          )
    return class_indict[str(predict_cla)], predict[predict_cla].numpy(), \
           class_indict[str(predict_top4[1].numpy()[1])], predict[predict_top4[1]].numpy()[1],\
           class_indict[str(predict_top4[1].numpy()[2])], predict[predict_top4[1]].numpy()[2],\
           class_indict[str(predict_top4[1].numpy()[3])], predict[predict_top4[1]].numpy()[3]


if __name__ == '__main__':
    main()
