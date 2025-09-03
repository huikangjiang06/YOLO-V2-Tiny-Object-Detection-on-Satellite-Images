import time
import torch
import csv
import json
import argparse

from utils.train_utils import MetricLogger
from utils.coco_utils import get_coco_api_from_dataset, CocoEvaluator
from utils.im_utils import Compose, ToTensor, RandomHorizontalFlip
from config.test_config import test_cfg
from dataloader.coco_dataset import coco


@torch.no_grad()
def evaluate(result_csv_file, data_loader, device, mAP_list=None, world_size=1):
    n_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    cpu_device = torch.device("cpu")
    
    metric_logger = MetricLogger(delimiter="  ")
    header = "Test: "

    coco = get_coco_api_from_dataset(data_loader.dataset)
    iou_types = ["bbox"]
    coco_evaluator = CocoEvaluator(coco, iou_types)
    
    with open(result_csv_file, mode='r', newline='', encoding='utf-8') as file:
        reader = csv.reader(file)
        header = next(reader, None)
        for row in reader:
            image_idx, image_file_name, predict_results = row
            predict_results = json.loads(predict_results)
            predict_boxes = torch.tensor(predict_results["boxes"])
            predict_classes = torch.tensor(predict_results["labels"])
            predict_scores = torch.tensor(predict_results["scores"])

            output = {
                "boxes": predict_boxes,
                "labels": predict_classes,
                "scores": predict_scores
            }

            res = {int(image_idx): output}
            # print(res)
            
            evaluator_time = time.time()
            coco_evaluator.update(res)
            evaluator_time = time.time() - evaluator_time

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    coco_evaluator.synchronize_between_processes()

    # accumulate predictions from all images
    coco_evaluator.accumulate()
    coco_evaluator.summarize()
    torch.set_num_threads(n_threads)

    print_txt = coco_evaluator.coco_eval[iou_types[0]].stats
    coco_mAP = print_txt[0]
    voc_mAP = print_txt[1]
    
    if isinstance(mAP_list, list):
        mAP_list.append(voc_mAP)
        
    print('best mAp is {}'.format(voc_mAP))
    return coco_evaluator, voc_mAP



def eval_file(submission_csv_file):
    data_transform = Compose([ToTensor()])
    test_data_set = coco(test_cfg.test_anno_path, test_cfg.test_image_dir, data_transform)
    test_data_set_loader = torch.utils.data.DataLoader(test_data_set,
                                                      batch_size=8,
                                                      shuffle=False,
                                                      num_workers=8,
                                                      collate_fn=test_data_set.collate_fn)

    _, mAP = evaluate(submission_csv_file, test_data_set_loader, device='cuda:1', mAP_list=[])

    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Evaluate a submission CSV file.")
    parser.add_argument('--submission_file_path', type=str, default="submission.csv", help="Path to the submission CSV file ")
    args = parser.parse_args()
    
    # submission_file_path = "submission.csv"
    submission_file_path = args.submission_file_path
    eval_file(submission_file_path)
    
    # python metrics.py --submission_file_path submission.csv
