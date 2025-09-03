import os
import csv
import json

import matplotlib.pyplot as plt
import torch
from PIL import Image
from torchvision import transforms
from tqdm import tqdm
import numpy as np

from config.test_config import test_cfg
from dataloader.coco_dataset import coco
from utils.draw_box_utils import draw_box
from utils.train_utils import create_model



def test():
    model = create_model(num_classes=test_cfg.num_classes)

    model.cuda()
    weights = test_cfg.model_weights

    checkpoint = torch.load(weights, map_location='cpu')
    model.load_state_dict(checkpoint['model'])
    model.eval()

    # read class_indict
    data_transform = transforms.Compose([transforms.ToTensor()])
    test_data_set = coco(test_cfg.test_anno_path, test_cfg.test_image_dir, data_transform)
    category_index = test_data_set.class_to_coco_cat_id
    index_category = dict(zip(category_index.values(), category_index.keys()))

    images = test_data_set.anno['images']
    predictions_data = []
    for image in tqdm(images):
        image_idx = image['id']
        image_file_name = image['file_name']
        img_path = os.path.join(test_cfg.test_image_dir, image_file_name)
        original_img = Image.open(img_path)
        # original_img = Image.open(test_cfg.image_path)
        img = data_transform(original_img)
        img = torch.unsqueeze(img, dim=0)

        with torch.no_grad():
            predictions = model(img.cuda())[0]
            predict_boxes = predictions["boxes"].to("cpu").numpy()
            predict_classes = predictions["labels"].to("cpu").numpy()
            predict_scores = predictions["scores"].to("cpu").numpy()
            
            if len(predict_boxes) == 0:
                print("No target detected!")
                continue
            
            predict_results = {
                "boxes": predict_boxes.tolist(),
                "labels": predict_classes.tolist(),
                "scores": predict_scores.tolist()
            }
            predict_results = json.dumps(predict_results)
            predictions_data.append([image_idx, image_file_name, predict_results])
            # print(image_idx, image_file_name, predict_boxes)
            
            # 可视化
            # draw_box(original_img,
            #          predict_boxes,
            #          predict_classes,
            #          predict_scores,
            #          index_category,
            #          thresh=0.3,
            #          line_thickness=3)
            # original_img.save('vis.png')
            # plt.imshow(original_img)
            # plt.show()
    
    # 输出csv文件
    with open("submission.csv", mode='w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow(['image_idx', 'image_file_name', 'predict_results'])
        for image_idx, image_file_name, predict_results in predictions_data:
            writer.writerow([image_idx, image_file_name, predict_results])


if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = test_cfg.gpu_id
    test()
