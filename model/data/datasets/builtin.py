import os

from detectron2.data import DatasetCatalog
from detectron2.data import MetadataCatalog
from detectron2.data.datasets.coco import load_coco_json
import json
from detectron2.data.datasets import register_coco_instances
from detectron2.structures import BoxMode
from fsdetection import load_fs_dataset


def hf_to_detectron2(dataset):
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


def register_hf_data():
    seed = os.getenv("REPEAT_ID", 2026)
    dataset_name = os.getenv("REPEAT_ID", 2026)

    dataset = load_fs_dataset(f"/lustre/fsn1/projects/rech/mvq/ubc18yy/datasets/{dataset_name}")
    classes = dataset["train"].features["objects"]["category"].feature.names

    records_test = hf_to_detectron2(dataset["test"])
    DatasetCatalog.register(f"{dataset_name}_test", lambda: records_test)
    MetadataCatalog.get(f"{dataset_name}_test").set(thing_classes=classes)

    for shot in [1, 5, 10]:
        name = f"{dataset_name}_{shot}shot"
        dataset["train"].sampling(shots=shot, seed=int(seed))
        records = hf_to_detectron2(dataset["train"])
        DatasetCatalog.register(name, lambda: records)
        MetadataCatalog.get(name).set(thing_classes=classes)


_root = os.getenv("DETECTRON2_DATASETS", "datasets")
register_hf_data()
