import os

from detectron2.data import DatasetCatalog
from detectron2.data import MetadataCatalog
from detectron2.data.datasets.coco import load_coco_json
import json
from detectron2.data.datasets import register_coco_instances
from detectron2.structures import BoxMode
from fsdetection import load_fs_dataset

_PREDEFINED = [
    ("dior_train", "DIOR/train", "DIOR/annotations/train.json"),
    ("dior_test", "DIOR/test", "DIOR/annotations/test.json"),
    # ("ArTaxOr_train", "ArTaxOr/train", "ArTaxOr/annotations/train.json"),
    # ("ArTaxOr_test", "ArTaxOr/test", "ArTaxOr/annotations/test.json"),
    # ("UODD_train", "UODD/train", "UODD/annotations/train.json"),
    # ("UODD_test", "UODD/test", "UODD/annotations/test.json")
]


# for shot in [1]:#,5,10]:
#     new_anns =  ("dior_{}shot".format(shot),
#                "DIOR/train",
#                "DIOR/{}_shot.json".format( shot))
#     _PREDEFINED.append(new_anns)
  
  
    # new_anns =  ("ArTaxOr_{}shot".format(shot),
    # "ArTaxOr/train",
    # "ArTaxOr/annotations/{}_shot.json".format(shot))
    # _PREDEFINED.append(new_anns)
    #
    #
    # new_anns =  ("UODD_{}shot".format(shot),
    # "UODD/train",
    # "UODD/annotations/{}_shot.json".format(shot))
    # _PREDEFINED.append(new_anns)


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

def register_data(root):
    for name, image_dir, json_file in _PREDEFINED:
        with open(os.path.join(root, json_file), "r", encoding="utf-8") as f:
            data = json.load(f)
        classes = [i["name"] for i in data["categories"]]
        register_coco_instances(name, {}, os.path.join(root, json_file), os.path.join(root, image_dir))
        MetadataCatalog.get(name).set(thing_classes=classes)

def register_hf_data():
    dataset = load_fs_dataset("HichTala/dota")
    classes = dataset["train"].features["objects"]["category"].feature.names

    records_val = hf_to_detectron2(dataset["validation"])
    DatasetCatalog.register("dota_val", lambda: records_val)
    MetadataCatalog.get("dota_val").set(thing_classes=classes)

    records_test = hf_to_detectron2(dataset["test"])
    DatasetCatalog.register("dota_test", lambda: records_test)
    MetadataCatalog.get("dota_test").set(thing_classes=classes)

    for shot in [1,5,10]:
        name = "dota_{}shot".format(shot)
        dataset["train"].sampling(shots=shot)
        records = hf_to_detectron2(dataset["train"])
        DatasetCatalog.register(name, lambda: records)
        MetadataCatalog.get(name).set(thing_classes=classes)


# Register them all under "./datasets"
_root = os.getenv("DETECTRON2_DATASETS", "datasets")
# register_data(_root)
register_hf_data()
