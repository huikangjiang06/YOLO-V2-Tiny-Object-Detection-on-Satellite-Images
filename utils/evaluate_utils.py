import time
import torch
from utils.train_utils import MetricLogger
from utils.coco_utils import get_coco_api_from_dataset, CocoEvaluator


@torch.no_grad()
def evaluate(model, data_loader, device, mAP_list=None, world_size=1):
    n_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    cpu_device = torch.device("cpu")
    model.eval()
    
    is_distributed = world_size > 1 and dist.is_initialized()
    if is_distributed:
        dist.barrier()
    
    metric_logger = MetricLogger(delimiter="  ")
    header = "Test: "

    coco = get_coco_api_from_dataset(data_loader.dataset)
    iou_types = ["bbox"]
    coco_evaluator = CocoEvaluator(coco, iou_types)

    for image, targets in metric_logger.log_every(data_loader, 100, header):
        image = list(img.to(device) for img in image)
        
        if device != torch.device("cpu"):
            torch.cuda.synchronize(device)

        model_time = time.time()
        outputs = model(image)

        # outputs = [{k: v.to(cpu_device) for k, v in t.items()} for t in outputs]
        outputs = [{k: v.cpu() for k, v in t.items()} for t in outputs]
        model_time = time.time() - model_time

        res = {target["image_id"].item(): output for target, output in zip(targets, outputs)}

        evaluator_time = time.time()
        coco_evaluator.update(res)
        evaluator_time = time.time() - evaluator_time
        metric_logger.update(model_time=model_time, evaluator_time=evaluator_time)

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
    
    if is_distributed:
        voc_mAP_tensor = torch.tensor([voc_mAP], dtype=torch.float32, device=device)
        dist.all_reduce(voc_mAP_tensor, op=dist.ReduceOp.SUM)
        voc_mAP = voc_mAP_tensor.item() / world_size 
    
    if isinstance(mAP_list, list):
        mAP_list.append(voc_mAP)

    return coco_evaluator, voc_mAP

