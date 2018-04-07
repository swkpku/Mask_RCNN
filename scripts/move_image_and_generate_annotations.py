import os
import sys
from shutil import copyfile
from tqdm import tqdm
import json
import numpy as np
from skimage.io import imread
from skimage.measure import regionprops, approximate_polygon, find_contours
from sklearn.model_selection import KFold

TRAIN_ROOT = "/home/swk/dsb2018/stage1_train/"
TEST_ROOT = "/home/swk/dsb2018/stage1_test/"

DST_ROOT = "/home/swk/Documents/kaggle/"#nuclei-1/nuclei_train2018/"

# Get train IDs
all_ids = next(os.walk(TRAIN_ROOT))[1]
all_ids_arr = np.array(all_ids)

kf = KFold(n_splits=11)
split_idx = 0

for train_index, val_index in kf.split(all_ids_arr):
    print("split_idx = %d" % split_idx)
    
    DST_TRAIN_ROOT = DST_ROOT + "nuclei-" + str(split_idx) + "/nuclei_train2018/"
    DST_VAL_ROOT = DST_ROOT + "nuclei-" + str(split_idx) + "/nuclei_val2018/"
    DST_ANNO_ROOT = DST_ROOT + "nuclei-" + str(split_idx) + "/annotations/"
    
    split_idx = split_idx + 1
    
    if not os.path.exists(DST_TRAIN_ROOT):
        os.makedirs(DST_TRAIN_ROOT)
    
    if not os.path.exists(DST_VAL_ROOT):
        os.makedirs(DST_VAL_ROOT)
        
    if not os.path.exists(DST_ANNO_ROOT):
        os.makedirs(DST_ANNO_ROOT)
    
    train_annotation_file = DST_ANNO_ROOT + "instances_dsb2018_train.json"
    val_annotation_file = DST_ANNO_ROOT + "instances_dsb2018_val.json"

    train_outfile = open(train_annotation_file, 'w')
    val_outfile = open(val_annotation_file, 'w')

    nuclei_train = dict()
    nuclei_val = dict()

    nuclei_train["info"] = {"description": "Data Science Bowl 2018 Train Dataset"}
    nuclei_train["licenses"] = []
    nuclei_train["images"] = []
    nuclei_train["annotations"] = []
    nuclei_train["categories"] = [{"supercategory":"nuclei", "id":1, "name":"nuclei"}]

    nuclei_val["info"] = {"description": "Data Science Bowl 2018 Val Dataset"}
    nuclei_val["licenses"] = []
    nuclei_val["images"] = []
    nuclei_val["annotations"] = []
    nuclei_val["categories"] = [{"supercategory":"nuclei", "id":1, "name":"nuclei"}]

    IMG_CHANNELS = 3
    train_annotation_id = 1
    val_annotation_id = 1

    train_ids, val_ids = all_ids_arr[train_index], all_ids_arr[val_index]

    print("Getting train images")
    sys.stdout.flush()
    for n, id_ in tqdm(enumerate(train_ids), total=len(train_ids)):
        path = TRAIN_ROOT + id_
        src_img_path = path + '/images/' + id_ + '.png'
        dst_img_path = DST_TRAIN_ROOT + '%012d' % (n + 1) + '.png'

        copyfile(src_img_path, dst_img_path)

        img_dict = dict()
        img_dict["license"] = 0
        img_dict["file_name"] = '%012d' % (n + 1) + '.png'
        img_dict["id"] = n+1

        img = imread(path + '/images/' + id_ + '.png')[:, :, :IMG_CHANNELS]

        img_dict["height"], img_dict["width"] = img.shape[0], img.shape[1]

        nuclei_train["images"].append(img_dict)

        for mask_file in next(os.walk(path + '/masks/'))[2]:
            mask_ = imread(path + '/masks/' + mask_file)

            props = regionprops(mask_)

            if len(props) != 1:
                exit(-1)

            annotation = dict()
            annotation["area"] = int(props[0].area)
            annotation["bbox"] = [props[0].bbox[1], props[0].bbox[0], (props[0].bbox[2]-props[0].bbox[0]), (props[0].bbox[3]-props[0].bbox[1])]
            annotation["iscrowd"] = 0
            annotation["image_id"] = n+1
            annotation["category_id"] = 1
            annotation["id"] = train_annotation_id
            train_annotation_id = train_annotation_id + 1

            #coordinates = props[0].coords
            #polygon = approximate_polygon(coordinates, 0)
            #segmentation = []
            #for line in polygon:
            #    segmentation.append(line[1])
            #    segmentation.append(line[0])
            
            segmentation = []
            contours = find_contours(mask_, 0)
            
            if len(contours[0]) < 9:
                print(path + '/masks/' + mask_file)
                continue
            
            for point in contours[0]:
                segmentation.append(int(point[1]))
                segmentation.append(int(point[0]))
            
            annotation["segmentation"] = [segmentation]

            nuclei_train["annotations"].append(annotation)

    print("Getting val images")
    sys.stdout.flush()
    for n, id_ in tqdm(enumerate(val_ids), total=len(val_ids)):
        path = TRAIN_ROOT + id_
        src_img_path = path + '/images/' + id_ + '.png'
        dst_img_path = DST_VAL_ROOT + '%012d' % (n + 1) + '.png'

        copyfile(src_img_path, dst_img_path)

        img_dict = dict()
        img_dict["license"] = 0
        img_dict["file_name"] = '%012d' % (n + 1) + '.png'
        img_dict["id"] = n+1

        img = imread(path + '/images/' + id_ + '.png')[:, :, :IMG_CHANNELS]

        img_dict["height"], img_dict["width"] = img.shape[0], img.shape[1]

        nuclei_val["images"].append(img_dict)

        for mask_file in next(os.walk(path + '/masks/'))[2]:
            mask_ = imread(path + '/masks/' + mask_file)

            props = regionprops(mask_)

            if len(props) != 1:
                exit(-1)

            annotation = dict()
            annotation["area"] = int(props[0].area)
            annotation["bbox"] = [props[0].bbox[1], props[0].bbox[0], (props[0].bbox[2]-props[0].bbox[0]), (props[0].bbox[3]-props[0].bbox[1])]
            annotation["iscrowd"] = 0
            annotation["image_id"] = n+1
            annotation["category_id"] = 1
            annotation["id"] = val_annotation_id
            val_annotation_id = val_annotation_id + 1

            #coordinates = props[0].coords
            #polygon = approximate_polygon(coordinates, 0)
            #segmentation = []
            #for line in polygon:
            #    segmentation.append(line[1])
            #    segmentation.append(line[0])
            
            segmentation = []
            contours = find_contours(mask_, 0)
            
            if len(contours[0]) < 9:
                print(path + '/masks/' + mask_file)
                continue
            
            for point in contours[0]:
                segmentation.append(int(point[1]))
                segmentation.append(int(point[0]))
            
            annotation["segmentation"] = [segmentation]

            nuclei_val["annotations"].append(annotation)


    json.dump(nuclei_train, train_outfile)
    json.dump(nuclei_val, val_outfile)
