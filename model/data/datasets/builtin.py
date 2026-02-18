import os
import tempfile

from detectron2.data import DatasetCatalog
from detectron2.data import MetadataCatalog
from detectron2.data.datasets.coco import load_coco_json
import json
from detectron2.data.datasets import register_coco_instances
from detectron2.structures import BoxMode
from fsdetection import load_fs_dataset


def hf_to_detectron2(dataset, split="train"):
    records = []

    for idx, sample in enumerate(dataset):
        width, height = sample["image"].size

        record = {
            "file_name": None,
            "image_id": idx,
            "height": height,
            "width": width,
            "annotations": [],
        }

        for bbox, cat_id in zip(
                sample["objects"]["bbox"],
                sample["objects"]["category"]
        ):
            record["annotations"].append({
                "bbox": bbox,
                "bbox_mode": BoxMode.XYWH_ABS,
                "category_id": cat_id,
            })

        record["image"] = sample["image"]
        records.append(record)

    return records

import json

def hf_to_coco_dict(dataset, categories):
    coco = {
        "images": [],
        "annotations": [],
        "categories": categories,
    }
    images_dict = {}

    ann_id = 1

    for img_id, sample in enumerate(dataset):
        width, height = sample["image"].size

        coco["images"].append({
            "id": img_id,
            "width": width,
            "height": height,
            "file_name": f"{img_id}.jpg",
        })
        images_dict[img_id] = sample["image"]

        for bbox, cat_id in zip(
            sample["objects"]["bbox"],
            sample["objects"]["category"]
        ):
            coco["annotations"].append({
                "id": ann_id,
                "image_id": img_id,
                "category_id": cat_id,
                "bbox": bbox,
                "area": bbox[2] * bbox[3],
                "iscrowd": 0,
            })
            ann_id += 1

    return coco, images_dict

def write_temp_coco(coco_dict):
    tmp = tempfile.NamedTemporaryFile(
        suffix=".json", mode='w', delete=False
    )
    json.dump(coco_dict, tmp)
    tmp.close()
    return tmp.name

def register_hf_data():
    seed = os.getenv("REPEAT_ID", 2026)
    dataset_name = os.getenv("DATASET")

    dataset = load_fs_dataset(f"HichTala/{dataset_name}")
    classes = dataset["train"].features["objects"]["category"].feature.names

    id2label = dict(enumerate(classes))
    categories = [{"id": i, "name": name} for i, name in id2label.items()]

    coco_dict, images_dict = hf_to_coco_dict(dataset["test"], categories=categories)
    coco_path = write_temp_coco(coco_dict)

    register_coco_instances(f"{dataset_name}_test", {}, coco_path, image_root=".")
    DatasetCatalog.register(f"{dataset_name}_test_images", lambda: images_dict)
    MetadataCatalog.get(f"{dataset_name}_test").set(thing_classes=classes, evaluator_type="coco")

    for shot in [1, 5, 10]:
        name = f"{dataset_name}_{shot}shot"
        dataset["train"].sampling(shots=shot, seed=int(seed))
        records = hf_to_detectron2(dataset["train"])
        DatasetCatalog.register(name, lambda: records)
        MetadataCatalog.get(name).set(thing_classes=classes)


_root = os.getenv("DETECTRON2_DATASETS", "datasets")
register_hf_data()
