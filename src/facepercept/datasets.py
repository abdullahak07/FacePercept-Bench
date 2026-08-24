DATASET_ID = "TheKernel01/140k-Real-and-Fake-Faces"


def dataset_spec():
    return {
        "repo_id": DATASET_ID,
        "split": "train",
        "image_column": "image",
        "label_column": "label",
        "real_label": 0,
        "synthetic_label": 1,
    }
