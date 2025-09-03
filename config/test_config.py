class Config:
    model_weights = "output/mobilenet-model-45-mAp-0.30963427013682265.pth"
    image_path = "data/SkyFusion/test/3c027498a_png_jpg.rf.da3db5f98f9637aa4c5a2e0100855ea6.jpg"
    gpu_id = '1'
    num_classes = 4
    data_root_dir = " "
    
    # dataset
    test_anno_path = "data/SkyFusion/test/_annotations.coco.json"
    test_image_dir = "data/SkyFusion/test/"


test_cfg = Config()
