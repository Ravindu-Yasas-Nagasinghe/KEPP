import os
import numpy as np
import torch
from torch.utils.data import Dataset
import json
import math
from collections import namedtuple

def get_vids_from_json(path):
    task_vids = {}
    with open(path, 'r') as f:
        json_data = json.load(f)

    for i in json_data:
        task = i['task']
        vid = i['vid']
        if task not in task_vids:
            task_vids[task] = []
        task_vids[task].append(vid)
    return task_vids


def get_vids(path):
    task_vids = {}
    with open(path, 'r') as f:
        for line in f:
            task, vid, url = line.strip().split(',')
            if task not in task_vids:
                task_vids[task] = []
            task_vids[task].append(vid)
    return task_vids


def read_task_info(path):
    titles = {}
    urls = {}
    n_steps = {}
    steps = {}
    with open(path, 'r') as f:
        idx = f.readline()
        while idx != '':
            idx = idx.strip()
            titles[idx] = f.readline().strip()
            urls[idx] = f.readline().strip()
            n_steps[idx] = int(f.readline().strip())
            steps[idx] = f.readline().strip().split(',')
            next(f)
            idx = f.readline()
    return {'title': titles, 'url': urls, 'n_steps': n_steps, 'steps': steps}


class PlanningDataset(Dataset):
    def __init__(self,
                 root,
                 args=None,
                 is_val=False,
                 model=None,
                 crosstask_use_feature_how=True,
                 ):
        self.is_val = is_val
        self.data_root = root
        self.args = args
        self.max_traj_len = args.horizon
        self.vid_names = None
        self.frame_cnts = None
        self.images = None
        self.last_vid = ''
        self.crosstask_use_feature_how = crosstask_use_feature_how
        if args.dataset == 'crosstask':
            """
            .
            └── crosstask
                ├── crosstask_features
                └── crosstask_release
                    ├── tasks_primary.txt
                    ├── videos.csv or json
                    └── videos_val.csv or json
            """
            val_csv_path = os.path.join(
                root, 'crosstask_release', 'test_list.json') 
            video_csv_path = os.path.join(
                root, 'crosstask_release', 'train_list.json') 

            if crosstask_use_feature_how:
                self.features_path = os.path.join(root, 'processed_data')
            else:
                self.features_path = os.path.join(root, 'crosstask_features')

            self.constraints_path = os.path.join(
                root, 'crosstask_release', 'annotations')

            self.action_one_hot = np.load(
                os.path.join(root, 'crosstask_release', 'actions_one_hot.npy'),
                allow_pickle=True).item()

            if is_val:
                cross_task_data_name = args.json_path_val
            else:
                cross_task_data_name = args.json_path_train

            if os.path.exists(cross_task_data_name):
                with open(cross_task_data_name, 'r') as f:
                    self.json_data = json.load(f)
                print('Loaded {}'.format(cross_task_data_name))
            else:
                file_type = val_csv_path.split('.')[-1]
                if file_type == 'json':
                    all_task_vids = get_vids_from_json(video_csv_path)
                    val_vids = get_vids_from_json(val_csv_path)
                else:
                    all_task_vids = get_vids(video_csv_path)
                    val_vids = get_vids(val_csv_path)

                if is_val:
                    task_vids = val_vids
                else:
                    task_vids = {task: [vid for vid in vids if task not in val_vids or vid not in val_vids[task]] for
                                 task, vids in
                                 all_task_vids.items()}
                
                primary_info = read_task_info(os.path.join(
                    root, 'crosstask_release', 'tasks_primary.txt'))

                self.n_steps = primary_info['n_steps']
                all_tasks = set(self.n_steps.keys())

                task_vids = {task: vids for task,
                                            vids in task_vids.items() if task in all_tasks}
                
                all_vids = []
                for task, vids in task_vids.items():
                    all_vids.extend([(task, vid) for vid in vids])
                json_data = []
                for idx in range(len(all_vids)):
                    task, vid = all_vids[idx]
                    if self.crosstask_use_feature_how:
                        video_path = os.path.join(
                            self.features_path, str(task) + '_' + str(vid) + '.npy')
                    else:
                        video_path = os.path.join(
                            self.features_path, str(vid) + '.npy')
                    legal_range = self.process_single(task, vid)
                    if not legal_range:
                        continue

                    temp_len = len(legal_range)
                    temp = []
                    while temp_len < self.max_traj_len:
                        temp.append(legal_range[0])
                        temp_len += 1

                    temp.extend(legal_range)
                    legal_range = temp

                    for i in range(len(legal_range) - self.max_traj_len + 1):
                        legal_range_current = legal_range[i:i + self.max_traj_len]
                        json_data.append({'id': {'vid': vid, 'task': task, 'feature': video_path,
                                                 'legal_range': legal_range_current, },'instruction_len': self.n_steps[task]})

                self.json_data = json_data
                with open(cross_task_data_name, 'w') as f:
                    json.dump(json_data, f)

        elif args.dataset == 'coin':
            print(root)
            coin_path = os.path.join(root, 'coin', 'full_npy/')
            val_csv_path = os.path.join(
                root, 'coin', 'coin_test_30.json')
            video_csv_path = os.path.join(
                root, 'coin', 'coin_train_70.json')

            # coin_data_name = "/data1/wanghanlin/diffusion_planning/jsons_coin/sliding_window_cross_task_data_{}_{}_new_task_id_73.json".format(
            #         is_val, self.max_traj_len)
            if is_val:
                coin_data_name = args.json_path_val
            else:
                coin_data_name = args.json_path_train

            if os.path.exists(coin_data_name):
                with open(coin_data_name, 'r') as f:
                    self.json_data = json.load(f)
                print('Loaded {}'.format(coin_data_name))
            else:
                json_data = []
                num = 0
                if is_val:
                    with open(val_csv_path, 'r') as f:
                        coin_data = json.load(f)
                else:
                    with open(video_csv_path, 'r') as f:
                        coin_data = json.load(f)
                for i in coin_data:
                    for (k, v) in i.items():
                        file_name = v['class'] + '_' + str(v['recipe_type']) + '_' + k + '.npy'
                        file_path = coin_path + file_name
                        images_ = np.load(file_path, allow_pickle=True)
                        images = images_['frames_features']
                        legal_range = []

                        last_action = v['annotation'][-1]['segment'][1]
                        last_action = math.ceil(last_action)
                        if last_action > len(images):
                            print(k, last_action, len(images))
                            num += 1
                            continue

                        for annotation in v['annotation']:
                            action_label = int(annotation['id']) - 1
                            start_idx, end_idx = annotation['segment']
                            start_idx = math.floor(start_idx)
                            end_idx = math.ceil(end_idx)

                            if end_idx < images.shape[0]:
                                legal_range.append((start_idx, end_idx, action_label))
                            else:
                                legal_range.append((start_idx, images.shape[0] - 1, action_label))

                        temp_len = len(legal_range)
                        temp = []
                        while temp_len < self.max_traj_len:
                            temp.append(legal_range[0])
                            temp_len += 1

                        temp.extend(legal_range)
                        legal_range = temp

                        for i in range(len(legal_range) - self.max_traj_len + 1):
                            legal_range_current = legal_range[i:i + self.max_traj_len]
                            json_data.append({'id': {'vid': k, 'feature': file_path,
                                                     'legal_range': legal_range_current, 'task_id': v['recipe_type']},
                                              'instruction_len': 0})
                print(num)
                self.json_data = json_data
                with open(coin_data_name, 'w') as f:
                    json.dump(json_data, f)

        elif args.dataset == 'NIV':
            val_csv_path = os.path.join(
                root, '../NIV', 'test30.json')
            video_csv_path = os.path.join(
                root, '../NIV', 'train70.json')

            if is_val:
                niv_data_name = args.json_path_val
            else:
                niv_data_name = args.json_path_train


            if os.path.exists(niv_data_name):
                with open(niv_data_name, 'r') as f:
                    self.json_data = json.load(f)
                print('Loaded {}'.format(niv_data_name))
            else:
                json_data = []
                if is_val:
                    with open(val_csv_path, 'r') as f:
                        niv_data = json.load(f)
                else:
                    with open(video_csv_path, 'r') as f:
                        niv_data = json.load(f)
                for d in niv_data:
                    legal_range = []
                    path = d['feature']
                    info = np.load(path, allow_pickle=True)
                    num_steps = int(info['num_steps'])
                    assert num_steps == len(info['steps_ids'])
                    assert info['num_steps'] == len(info['steps_starts'])
                    assert info['num_steps'] == len(info['steps_ends'])
                    starts = info['steps_starts']
                    ends = info['steps_ends']
                    action_labels = info['steps_ids']
                    images = info['frames_features']

                    for i in range(num_steps):
                        action_label = int(action_labels[i])
                        start_idx = math.floor(float(starts[i]))
                        end_idx = math.ceil(float(ends[i]))

                        if end_idx < images.shape[0]:
                            legal_range.append((start_idx, end_idx, action_label))
                        else:
                            legal_range.append((start_idx, images.shape[0] - 1, action_label))

                    temp_len = len(legal_range)
                    temp = []
                    while temp_len < self.max_traj_len:
                        temp.append(legal_range[0])
                        temp_len += 1

                    temp.extend(legal_range)
                    legal_range = temp

                    for i in range(len(legal_range) - self.max_traj_len + 1):
                        legal_range_current = legal_range[i:i + self.max_traj_len]
                        json_data.append({'id': {'feature': path, 'legal_range': legal_range_current, 'task_id': d['task_id']}, 'instruction_len': 0})
                self.json_data = json_data
                with open(niv_data_name, 'w') as f:
                    json.dump(json_data, f)
                    print(len(json_data))
        else:
            raise NotImplementedError(
                'Dataset {} is not implemented'.format(args.dataset))

        self.model = model
        self.prepare_data()
        self.M = 3

    def process_single(self, task, vid):
        if self.crosstask_use_feature_how:
            if not os.path.exists(os.path.join(self.features_path, str(task) + '_' + str(vid) + '.npy')):
                return False
            images_ = np.load(os.path.join(self.features_path, str(task) + '_' + str(vid) + '.npy'), allow_pickle=True)
            images = images_['frames_features']
        else:
            if not os.path.exists(os.path.join(self.features_path, vid + '.npy')):
                return False
            images = np.load(os.path.join(self.features_path, vid + '.npy'))

        cnst_path = os.path.join(
            self.constraints_path, task + '_' + vid + '.csv')
        legal_range = self.read_assignment(task, cnst_path)
        legal_range_ret = []
        for (start_idx, end_idx, action_label) in legal_range:
            if not start_idx < images.shape[0]:
                print(task, vid, end_idx, images.shape[0])
                return False
            if end_idx < images.shape[0]:
                legal_range_ret.append((start_idx, end_idx, action_label))
            else:
                legal_range_ret.append((start_idx, images.shape[0] - 1, action_label))

        return legal_range_ret

    def read_assignment(self, task_id, path):
        legal_range = []
        with open(path, 'r') as f:
            for line in f:
                step, start, end = line.strip().split(',')
                start = int(math.floor(float(start)))
                end = int(math.ceil(float(end)))
                action_label_ind = self.action_one_hot[task_id + '_' + step]
                legal_range.append((start, end, action_label_ind))

        return legal_range

    def prepare_data(self):
        vid_names = []
        frame_cnts = []
        for listdata in self.json_data:
            vid_names.append(listdata['id'])
            frame_cnts.append(listdata['instruction_len'])
        self.vid_names = vid_names
        self.frame_cnts = frame_cnts

    def __len__(self):
        return len(self.json_data)
