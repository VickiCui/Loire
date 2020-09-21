#!/usr/bin/python
#
# Copyright 2018 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse, json, os
from collections import Counter, defaultdict

import numpy as np
from scipy.misc import imread, imresize
import gc
from tqdm import tqdm

"""
vocab for objects contains a special entry "__image__" intended to be used for
dummy nodes encompassing the entire image; vocab for predicates includes a
special entry "__in_image__" to be used for dummy relationships making the graph
fully-connected.
"""


VG_DIR = '../data/vg'

parser = argparse.ArgumentParser()

# Input data
parser.add_argument('--splits_json', default='./data/vg_splits.json')
parser.add_argument('--images_json',
    default=os.path.join(VG_DIR, 'image_data.json'))
parser.add_argument('--objects_json',
    default=os.path.join(VG_DIR, 'objects.json'))
parser.add_argument('--regions_json',
    default=os.path.join(VG_DIR, 'region_graphs.json'))
parser.add_argument('--object_aliases',
    default=os.path.join(VG_DIR, 'object_alias.txt'))

# Arguments for images
parser.add_argument('--min_image_size', default=100, type=int)
parser.add_argument('--train_split', default='train')

# Arguments for objects
parser.add_argument('--min_object_instances', default=300, type=int)
parser.add_argument('--min_object_size', default=0.0016, type=float)
parser.add_argument('--min_objects_per_image', default=1, type=int)
parser.add_argument('--max_objects_per_image', default=20, type=int)

# Arguments for relationships
parser.add_argument('--min_regions_per_image', default=1, type=int)
parser.add_argument('--max_regions_per_image', default=25, type=int)

# Output
parser.add_argument('--output_vocab_json',
    default=os.path.join(VG_DIR, 'vocab.json'))
parser.add_argument('--output_dir', default=VG_DIR)


def main(args):
    # Load all images
    print('Loading image info from "%s"' % args.images_json)
    with open(args.images_json, 'r') as f:
        images = json.load(f)
    image_id_to_image = {i['image_id']: i for i in images}

    with open(args.splits_json, 'r') as f:
        splits = json.load(f)

    # Filter images for being too small
    splits = remove_small_images(args, image_id_to_image, splits)

    obj_aliases = load_aliases(args.object_aliases)

    print('Loading objects from "%s"' % args.objects_json)
    with open(args.objects_json, 'r') as f:
        objects = json.load(f)
    image_id_to_objs = {i['image_id']: i for i in objects}

    # Vocab for objects and relationships
    vocab = {}
    train_ids = splits[args.train_split]
    create_object_vocab(args, train_ids, objects, obj_aliases, vocab) # Return vocab['object_name_to_idx'], vocab['object_idx_to_name']
    print('object vocab length: {}'.format(len(vocab['object_idx_to_name'])))

    # filter too small objs
    object_id_to_obj = filter_objects(args, image_id_to_image, objects, obj_aliases, vocab, splits)
    print('After filtering there are %d object instances'
            % len(object_id_to_obj))

    # Filter images with too many/few objects
    splits = remove_invalide_obj_images(args, image_id_to_image, image_id_to_objs, splits)
    splits, image_ids_to_obj_ids = remove_many_obj_images(args, image_id_to_objs, object_id_to_obj, splits, obj_aliases)



    print('Loading regions from "%s"' % args.regions_json)
    with open(args.regions_json, 'r') as f:
        regions = json.load(f)
    image_id_to_regions = {i['image_id']: i for i in regions}

    # Filter regions don't have specific objects, return {image_id1: [regions], image_id2: [regions] ...}
    image_id_to_regions = filter_regions_wo_objs(args, image_id_to_regions, image_ids_to_obj_ids, splits, obj_aliases)

    # Filter images with too many/few regions
    splits = remove_many_region_images(args, image_id_to_regions, splits)

    splits_file = os.path.join(args.output_dir, 'vg_splits.json')
    with open(splits_file,'w',encoding='utf-8') as json_file:
       json.dump(splits, json_file, ensure_ascii=False)

    obj_cat = set()
    for k,v in image_ids_to_obj_ids.items():
        for obj_id in v:
            obj_cat.add(object_id_to_obj[obj_id]['name'])
    save_file = os.path.join(args.output_dir, 'vg_obj_categories.txt')
    with open(save_file,'w',encoding='utf-8') as f:
        for cat in obj_cat:
            f.write(cat)
            f.write('\n')
    
    del image_id_to_objs
    del obj_aliases
    gc.collect()

    creat_new_data(args, splits, image_id_to_image, image_ids_to_obj_ids, object_id_to_obj, image_id_to_regions)
    
    print(len(obj_cat))
    print(obj_cat)

    print('Writing vocab to "%s"' % args.output_vocab_json)
    with open(args.output_vocab_json, 'w') as f:
        json.dump(vocab, f)

def remove_small_images(args, image_id_to_image, splits):
    new_splits = {}
    for split_name, image_ids in splits.items():
        new_image_ids = []
        num_skipped = 0
        for image_id in image_ids:
            image = image_id_to_image[image_id]
            height, width = image['height'], image['width']
            if min(height, width) < args.min_image_size:
                num_skipped += 1
                continue
            new_image_ids.append(image_id)
        new_splits[split_name] = new_image_ids
        print('Removed %d images from split "%s" for being too small' %
            (num_skipped, split_name))

    for k,v in new_splits.items():
        print('leave {} {} data'.format(len(v), k))

    return new_splits

def remove_invalide_obj_images(args, image_id_to_image, image_id_to_objs, splits):
    new_splits = {}

    for split_name, image_ids in splits.items():
        new_image_ids = []
        num_skipped = 0
        for image_id in image_ids:
            flag = True
            W = image_id_to_image[image_id]['width']
            H = image_id_to_image[image_id]['height']
            objs = image_id_to_objs[image_id]['objects']
            for obj in objs:
                x = obj['x']
                y = obj['y']
                w = obj['w']
                h = obj['h']
                if x>W or y>H or (w-1)>W or (h-1)>H:
                    flag = False
                    break
            if flag == False:
                num_skipped += 1
                continue
            new_image_ids.append(image_id)
        new_splits[split_name] = new_image_ids
        print('Removed %d images from split "%s" for having invalide coordinate ' %
            (num_skipped, split_name))
    for k,v in new_splits.items():
        print('leave {} {} data'.format(len(v), k))

    return new_splits

def remove_many_obj_images(args, image_id_to_objs, object_id_to_obj, splits, aliases):
    new_splits = {}
    image_ids_to_obj_ids = {}

    for split_name, image_ids in splits.items():
        new_image_ids = []
        num_skipped = 0
        for image_id in image_ids:
            objs_in_this_img = []
            objs = image_id_to_objs[image_id]['objects']
            for obj in objs:
                if obj['object_id'] not in object_id_to_obj:
                    continue
                else:
                  objs_in_this_img.append(obj['object_id'])
            objs_num = len(objs_in_this_img)
            if objs_num < args.min_objects_per_image or objs_num > args.max_objects_per_image:
                num_skipped += 1
                continue
            new_image_ids.append(image_id)
            image_ids_to_obj_ids[image_id] = objs_in_this_img
        new_splits[split_name] = new_image_ids
        print('Removed %d images from split "%s" for having too many/few objs' %
            (num_skipped, split_name))
    for k,v in new_splits.items():
        print('leave {} {} data'.format(len(v), k))

    return new_splits, image_ids_to_obj_ids

def filter_regions_wo_objs(args, image_id_to_regions, image_ids_to_obj_ids, splits, aliases):
    """
    {
      'relationships': [], 'region_id': 1382, 'width': 82, 
      'synsets': [
        {'entity_idx_start': 4, 'entity_idx_end': 9, 'entity_name': 'clock', 'synset_name': 'clock.n.01'}, 
        {'entity_idx_start': 22, 'entity_idx_end': 28, 'entity_name': 'colour', 'synset_name': 'color.n.04'}
      ], 
      'height': 139, 'image_id': 1, 
      'objects': [
        {'name': 'clock', 'h': 339, 'object_id': 1058498, 'synsets': ['clock.n.01'], 'w': 79, 'y': 91, 'x': 421}
      ], 
      'phrase': 'the clock is green in colour', 'y': 57, 'x': 421
    }
    """
    new_image_ids_to_regions = dict()

    for _, image_ids in splits.items():
        for image_id in image_ids:
            regions_in_this_img = []
            regions = image_id_to_regions[image_id]['regions']
            obj_id_to_regions = defaultdict(list)
            for region in regions:
                for obj in region['objects']:
                    obj_id = obj['object_id']
                    if obj_id in image_ids_to_obj_ids[image_id]:
                        obj_id_to_regions[obj_id].append(region)
            for i in range(5):
                regions_ = []
                for obj_id in image_ids_to_obj_ids[image_id]:
                    region_list = obj_id_to_regions[obj_id]                    
                    rand_idx = np.random.permutation(len(region_list))[:1]
                    for idx in rand_idx:
                        if region_list[idx] not in regions_:
                            regions_.append(region_list[idx])
                regions_in_this_img.append(regions_)
                    
            new_image_ids_to_regions[image_id] = regions_in_this_img

    return new_image_ids_to_regions

def remove_many_region_images(args, image_id_to_regions, splits):
    new_splits = {}
    for split_name, image_ids in splits.items():
        new_image_ids = []
        num_skipped = 0
        for image_id in image_ids:
            flag = True
            for i in range(5):
                regions = image_id_to_regions[image_id][i]
                regions_num = len(regions)
                if regions_num < args.min_regions_per_image or regions_num > args.max_regions_per_image:
                    num_skipped += 1
                    flag = False
                    break
            if flag:
                new_image_ids.append(image_id)
        new_splits[split_name] = new_image_ids
        print('Removed %d images from split "%s" for having too many/few regions' %
            (num_skipped, split_name))
    for k,v in new_splits.items():
        print('leave {} {} data'.format(len(v), k))

    return new_splits

def creat_new_data(args, splits, image_id_to_image, image_ids_to_obj_ids, object_id_to_obj, image_id_to_regions):
    print('creating new dataset ...')
    l = 0
    cnt = 0
    data_list = []
    for _, image_ids in splits.items():
        for image_id in tqdm(image_ids):
            objs, obj_ids, boxes,captions = [], [], [], []
            image_data = image_id_to_image[image_id]
            image_w = image_data['width']
            image_h = image_data['height']
            image_path = image_data['url']
            obj_ids_in_this_image = image_ids_to_obj_ids[image_id]

            for obj_id in obj_ids_in_this_image:
                obj = object_id_to_obj[obj_id]
                objs.append(obj['name'])
                obj_ids.append(obj_id)
                boxes.append(obj['box']) # [obj['x'], obj['y'], obj['w'], obj['h']]

            for i in range(5):
                regions = image_id_to_regions[image_id][i]
                caption = ''
                for region in regions:
                    phrase = region['phrase'].strip()
                    if phrase[-1] not in ['.',',',';','!','?']:
                        phrase += '.'
                    caption += phrase
                    caption += ' '
                captions.append(caption)
                l = max(l, len(caption.split()))
                if len(caption.split()) > 300:
                    cnt+=1

            data = {
              'image_id': image_id,
              'width': image_w,
              'height': image_h,
              'image_path': image_path,
              'captions': captions,
              'boxes': boxes,
              'clses': objs,
              'objIds': obj_ids
            }

            data_list.append(data)
    save_file = os.path.join(args.output_dir, 'vg_for_layout.json')
    with open(save_file,'w',encoding='utf-8') as json_file:
       json.dump(data_list, json_file, ensure_ascii=False)

    return data_list


def get_image_paths(image_id_to_image, image_ids):
    paths = []
    for image_id in image_ids:
        image = image_id_to_image[image_id]
        base, filename = os.path.split(image['url'])
        path = os.path.join(os.path.basename(base), filename)
        paths.append(path)
    return paths


def handle_images(args, image_ids, h5_file):
    with open(args.images_json, 'r') as f:
        images = json.load(f)
    if image_ids:
        image_ids = set(image_ids)

    image_heights, image_widths = [], []
    image_ids_out, image_paths = [], []
    for image in images:
        image_id = image['image_id']
        if image_ids and image_id not in image_ids:
            continue
        height, width = image['height'], image['width']

        base, filename = os.path.split(image['url'])
        path = os.path.join(os.path.basename(base), filename)
        image_paths.append(path)
        image_heights.append(height)
        image_widths.append(width)
        image_ids_out.append(image_id)

    image_ids_np = np.asarray(image_ids_out, dtype=int)
    h5_file.create_dataset('image_ids', data=image_ids_np)

    image_heights = np.asarray(image_heights, dtype=int)
    h5_file.create_dataset('image_heights', data=image_heights)

    image_widths = np.asarray(image_widths, dtype=int)
    h5_file.create_dataset('image_widths', data=image_widths)

    return image_paths


def load_aliases(alias_path):
    aliases = {}
    print('Loading aliases from "%s"' % alias_path)
    with open(alias_path, 'r') as f:
        for line in f:
            line = [s.strip() for s in line.split(',')]
            for s in line:
                aliases[s] = line[0]
    return aliases


def create_object_vocab(args, image_ids, objects, aliases, vocab):
    strange_list = ['a','the','this','he']
    image_ids = set(image_ids)

    print('Making object vocab from %d training images' % len(image_ids))
    object_name_counter = Counter()
    for image in objects:
        if image['image_id'] not in image_ids:
            continue
        for obj in image['objects']:
            names = set()
            for name in obj['names']:
                std_name = aliases.get(name, name)
                if std_name not in strange_list:
                    names.add(std_name)
            object_name_counter.update(names)

    object_names = []
    for name, count in object_name_counter.most_common():
        if count >= args.min_object_instances:
            object_names.append(name)
    print('Found %d object categories with >= %d training instances' %
            (len(object_names), args.min_object_instances))

    object_name_to_idx = {}
    object_idx_to_name = []
    for idx, name in enumerate(object_names):
        object_name_to_idx[name] = idx
        object_idx_to_name.append(name)

    vocab['object_name_to_idx'] = object_name_to_idx
    vocab['object_idx_to_name'] = object_idx_to_name

def filter_objects(args, image_id_to_image, objects, aliases, vocab, splits):
    all_image_ids = set()
    for image_ids in splits.values():
        all_image_ids |= set(image_ids)

    object_name_to_idx = vocab['object_name_to_idx']
    object_id_to_obj = {}

    num_too_small = 0
    for image in objects:
        image_id = image['image_id']
        image_data = image_id_to_image[image_id]
        image_area = float(image_data['width'] * image_data['height'])
        if image_id not in all_image_ids:
            continue
        for obj in image['objects']:
            object_id = obj['object_id']
            final_name = None
            final_name_idx = None
            for name in obj['names']:
                name = aliases.get(name, name)
                if name in object_name_to_idx:
                    final_name = name
                    final_name_idx = object_name_to_idx[final_name]
                    break
            w, h = obj['w'], obj['h']
            obj_area = float(w * h)
            too_small = w < 0 or h < 0 or obj_area/image_area < args.min_object_size
            if too_small:
                num_too_small += 1
            if final_name is not None and not too_small:
                object_id_to_obj[object_id] = {
                'name': final_name,
                'name_idx': final_name_idx,
                'box': [obj['x'], obj['y'], obj['w'], obj['h']],
                }
    print('Skipped %d objects with size < %f' % (num_too_small, args.min_object_size))
    return object_id_to_obj


def create_rel_vocab(args, image_ids, relationships, object_id_to_obj,
                     rel_aliases, vocab):
    pred_counter = defaultdict(int)
    image_ids_set = set(image_ids)
    for image in relationships:
        image_id = image['image_id']
        if image_id not in image_ids_set:
            continue
        for rel in image['relationships']:
            sid = rel['subject']['object_id']
            oid = rel['object']['object_id']
            found_subject = sid in object_id_to_obj
            found_object = oid in object_id_to_obj
            if not found_subject or not found_object:
                continue
            pred = rel['predicate'].lower().strip()
            pred = rel_aliases.get(pred, pred)
            rel['predicate'] = pred
            pred_counter[pred] += 1

    pred_names = ['__in_image__']
    for pred, count in pred_counter.items():
        if count >= args.min_relationship_instances:
            pred_names.append(pred)
    print('Found %d relationship types with >= %d training instances'
            % (len(pred_names), args.min_relationship_instances))

    pred_name_to_idx = {}
    pred_idx_to_name = []
    for idx, name in enumerate(pred_names):
        pred_name_to_idx[name] = idx
        pred_idx_to_name.append(name)

    vocab['pred_name_to_idx'] = pred_name_to_idx
    vocab['pred_idx_to_name'] = pred_idx_to_name

if __name__ == '__main__':
    args = parser.parse_args()
    main(args)
