import pandas as pd
import numpy as np
import os
import argparse
from datasets.features import ClassLabel
from transformers import AutoProcessor
from sklearn.model_selection import train_test_split
from datasets import Features, Sequence, ClassLabel, Value, Array2D, Array3D, Dataset
from datasets import Image as Img
from PIL import Image
import traceback


import warnings
warnings.filterwarnings('ignore')


def read_text_file(file_path):
    with open(file_path, 'r') as f:
        return (f.readlines())


def prepare_examples(examples):
  images = examples[image_column_name]
  words = examples[text_column_name]
  boxes = examples[boxes_column_name]
  word_labels = examples[label_column_name]

  encoding = processor(images, words, boxes=boxes, word_labels=word_labels,
                       truncation=True, padding="max_length")

  return encoding

def get_zip_dir_name():
    try:
        os.chdir('/content')
        for file_name in os.listdir():
            if file_name.endswith('.zip'):
                return file_name[:file_name.find('.zip')]
        return False
    finally:
        os.chdir('./../')


def filter_out_unannotated(example):
    tags = example['ner_tags']
    return not all([tag == label2id['O'] for tag in tags])



if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--valid_size')
    parser.add_argument('--output_path')
    args = parser.parse_args()
    TEST_SIZE = float(args.valid_size)
    OUTPUT_PATH = args.output_path

    os.makedirs(args.output_path, exist_ok=True)
    files = {}
    zip_dir_name = get_zip_dir_name()
    data_folder_path = '/content/data'
    print(f"zip_dir_name: {zip_dir_name}")

    if zip_dir_name:
        box_file_path = os.path.join(data_folder_path, f'{zip_dir_name}_box.txt')
        image_file_path = os.path.join(data_folder_path, f'{zip_dir_name}_image.txt')
        train_file_path = os.path.join(data_folder_path, f'{zip_dir_name}.txt')
        print(f"Reading files: {box_file_path}, {image_file_path}, {train_file_path}")

        files['train_box'] = read_text_file(box_file_path)
        files['train_image'] = read_text_file(image_file_path)
        files['train'] = read_text_file(train_file_path)
    else:
        print("Zip directory name not found. Reading files from current directory.")

    # Debugging: Print keys present in the 'files' dictionary
    print("Keys present in 'files':", list(files.keys()))

    # Debugging: Print keys and lengths
    for key in files:
        print(f"Key: {key}, Length: {len(files[key])}")

    # Check to ensure all necessary keys are present
    assert all(key in files for key in ['train', 'train_box', 'train_image']), "Missing one or more required keys in 'files'"

    # Existing assertions
    assert(len(files['train']) == len(files['train_box']))
    assert(len(files['train_box']) == len(files['train_image']))
    assert(len(files['train_image']) == len(files['train']))


    images = {}
    for i, row in enumerate(files['train_image']):
        if row != '\n':
            image_name = row.split('\t')[-1]
            images.setdefault(image_name.replace('\n', ''), []).append(i)

    words, bboxes, ner_tags, image_path = [], [], [], []
    for image, rows in images.items():
        words_aux, ner_tags_aux, bboxes_aux = [], [], []

        for row in files['train'][rows[0]:rows[-1]+1]:
            try:
                word, ner_tag = row.strip().split('\t')
                words_aux.append(word)
                ner_tags_aux.append(ner_tag)
            except IndexError:
                # Log error and continue to the next row
                traceback.print_exc()
                continue

        for box in files['train_box'][rows[0]:rows[-1]+1]:
            try:
                bbox = box.strip().split('\t')[1]
                bboxes_aux.append(bbox)
            except IndexError:
                # Log error and continue to the next box
                traceback.print_exc()
                continue

        words.append(words_aux)
        ner_tags.append(ner_tags_aux)
        bboxes.append(bboxes_aux)

        if zip_dir_name:
            image_path.append(f"/content/data/{image}")
        else:
            image_path.append(f"/content/data/{image}")

    labels = list(set([tag for doc_tag in ner_tags for tag in doc_tag]))
    id2label = {v: k for v, k in enumerate(labels)}
    label2id = {k: v for v, k in enumerate(labels)}

    dataset_dict = {
        'id': range(len(words)),
        'tokens': words,
        'bboxes': [[list(map(int, bbox.split())) for bbox in doc] for doc in bboxes],
        'ner_tags': [[label2id[tag] for tag in ner_tag] for ner_tag in ner_tags],
        'image': [Image.open(path).convert("RGB") for path in image_path]
    }

    #raw features
    features = Features({
        'id': Value(dtype='string', id=None),
        'tokens': Sequence(feature=Value(dtype='string', id=None), length=-1, id=None),
        'bboxes': Sequence(feature=Sequence(feature=Value(dtype='int64', id=None), length=-1, id=None), length=-1, id=None),
        'ner_tags': Sequence(feature=ClassLabel(num_classes=len(labels), names=labels, names_file=None, id=None), length=-1, id=None),
        'image': Img(decode=True, id=None)
    })

    full_data_set = Dataset.from_dict(dataset_dict, features=features)
    dataset = full_data_set.train_test_split(test_size=TEST_SIZE)
    dataset["train"] = dataset["train"].filter(filter_out_unannotated)
    processor = AutoProcessor.from_pretrained(
        "microsoft/layoutlmv3-base", apply_ocr=False)

    features = dataset["train"].features
    column_names = dataset["train"].column_names
    image_column_name = "image"
    text_column_name = "tokens"
    boxes_column_name = "bboxes"
    label_column_name = "ner_tags"

    # In the event the labels are not a `Sequence[ClassLabel]`, we will need to go through the dataset to get the
    # unique labels.


#     def get_label_list(labels):
#         unique_labels = set()
#         for label in labels:
#             unique_labels = unique_labels | set(label)
#         label_list = list(unique_labels)
#         label_list.sort()
#         return label_list


#     if isinstance(features[label_column_name].feature, ClassLabel):
#         label_list = features[label_column_name].feature.names
#         # No need to convert the labels since they are already ints.
#         id2label = {k: v for k, v in enumerate(label_list)}
#         label2id = {v: k for k, v in enumerate(label_list)}
#     else:
#         label_list = get_label_list(dataset["train"][label_column_name])
#         id2label = {k: v for k, v in enumerate(label_list)}
#         label2id = {v: k for k, v in enumerate(label_list)}
#     num_labels = len(label_list)

    

    # we need to define custom features for `set_format` (used later on) to work properly
    features = Features({
        'pixel_values': Array3D(dtype="float32", shape=(3, 224, 224)),
        'input_ids': Sequence(feature=Value(dtype='int64')),
        'attention_mask': Sequence(Value(dtype='int64')),
        'bbox': Array2D(dtype="int64", shape=(512, 4)),
        'labels': Sequence(ClassLabel(names=labels)),
    })

    train_dataset = dataset["train"].map(
        prepare_examples,
        batched=True,
        remove_columns=column_names,
        features=features,
    )
    eval_dataset = dataset["test"].map(
        prepare_examples,
        batched=True,
        remove_columns=column_names,
        features=features,
    )
    train_dataset.set_format("torch")
    if not OUTPUT_PATH.endswith('/'):
        OUTPUT_PATH += '/'
    train_dataset.save_to_disk(f'/content/output/train_split')
    eval_dataset.save_to_disk(f'/content/output/test_split')
    dataset.save_to_disk(f'/content/output/raw_data')
