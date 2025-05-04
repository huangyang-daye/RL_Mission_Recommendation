import csv
import json
import os
import random
from typing import Dict, List, Tuple

import numpy as npy
from dateutil.parser import parse

class Data:
    worker_quality: Dict[int, float]
    worker_category: Dict[Tuple[int, int], int]
    worker_project_cnt: Dict[int, int]
    project_info: Dict[int, Dict[str, int]]
    entry_info: Dict[int, Dict[int, Dict[str, int]]]
    industry_list: Dict[str, int]
    n_state: int
    _n_cat: int = 10
    _project_by_time: List[Dict[str, int]]
    _dev_project_by_time: List[Dict[str, int]]
    _worker_id_rmap: List[int]

    def get_data(self):
        project_path = os.path.abspath(os.path.dirname(__file__))
        data_path = project_path[:project_path.rindex('data')] + "resource"
        worker_quality = {}
        worker_id_rmap = []
        # read worker_quality
        csv_file = open(data_path + "worker_quality.csv", "r")
        csv_reader = csv.reader(csv_file)
        next(csv_reader)
        for line in csv_reader:
            if float(line[1])>0.0:
                worker_id = int(line[0])
                worker_quality[worker_id] = float(line[1]) / 100.0
                worker_id_rmap.append(worker_id)
        csv_file.close()

        # read project_list
        file = open(data_path + "project_list.csv", "r")
        project_list = file.readlines()
        file.close()
        project_dir = data_path + "project/"
        entry_dir = data_path + "entry/"
        project_info = {}
        entry_info = {}
        limit = 24
        worker_category = {}
        worker_project_count = {}
        
        for line in project_list:
            line = line.strip('\n').split(',')
            project_id = int(line[0])
            entry_count = int(line[1])
            file = open(project_dir + "project_" + str(project_id) + ".txt","r")
            htmlcode = file.read()
            file.close()
            text = json.loads(htmlcode)

            project_info[project_id] = {"id": project_id}
            project_info[project_id]["sub_category"] = int(text["sub_category"]) 
            project_info[project_id]["category"] = int(text["category"]) 
            project_info[project_id]["entry_count"] = int(text["entry_count"]) 
            project_info[project_id]["start_date"] = int(parse(text["start_date"]).timestamp()) 
            project_info[project_id]["deadline"] = int(parse(text["deadline"]).timestamp()) 

            if text["industry"] is None:
                text["industry"] = "none"
            if text["industry"] not in industry_list:
                industry_list[text["industry"]] = len(industry_list)
            project_info[project_id]["industry"] = industry_list[text["industry"]] 

            entry_info[project_id] = {}
            k = 0
            while (k < entry_count):
                file = open(entry_dir + "entry_" + str(project_id) + "_" + str(k) + ".txt", "r")
                htmlcode = file.read()
                file.close()
                text = json.loads(htmlcode)

                for item in text["results"]:
                    entry_number = int(item["entry_number"])
                    entry_info[project_id][entry_number] = {}
                    entry_info[project_id][entry_number]["entry_created_at"] = int(parse(item["entry_created_at"]).timestamp())
                    worker_id = int(item["author"])
                    entry_info[project_id][entry_number]["worker"] = int(item["worker"]) 
                    tp = (worker_id, project_info[project_id]["category"])
                    if tp not in worker_category:
                        worker_category[tp] = 0
                    worker_category[tp] += 1
                    if worker_id not in worker_project_count:
                        worker_project_count[worker_id] = 0
                    worker_project_count[worker_id] += 1
                    
                k += limit
        print("data read finished...")
        self.worker_quality = worker_quality
        self.worker_category = worker_category
        self.worker_project_cnt = worker_project_count
        self.project_info = project_info
        self.entry_info = entry_info
        self.industry_list = industry_list
        self.n_state = self._n_cat + len(self.industry_list)
        pbt: List[Dict[str, int]] = []
        dpbt: List[Dict[str, int]] = []
        for pid in project_info.keys():
            if random.randint(0, 5) == 0:
                pbt.append(project_info[pid])
            else:
                dpbt.append(project_info[pid])
        self._project_by_time = sorted(pbt, key=lambda a: a["start_date"])
        self._dev_project_by_time = sorted(dpbt, key=lambda a: a["start_date"])
        self._worker_id_rmap = worker_id_rmap

    def get_state_array(self, index, is_testing)-> np.ndarray:
        if is_testing:
            project = self._dev_project_by_time[index]
        else:
            project = self._project_by_time[index]
        ret = np.zeros((self.n_state))
        ret[project["category"] - 1] = 1
        ret[self._n_cat + project["industry"]] = 1
        return ret

    def get_standard_reward(self, worker_id, project_id) -> float:
        if (worker_id, self.project_info[project_id]["category"]) not in self.worker_category.keys():
            self.worker_category[(worker_id, self.project_info[project_id]["category"])] = 0
        return self.worker_category[(worker_id, self.project_info[project_id]["category"])] / self.worker_project_cnt[worker_id]

    def get_quality_reward(self, worker_id)-> float:
        return self.worker_quality[worker_id]

    def get_project_id_by_index(self, index, is_testing)->int:
        if is_testing:
            return self._dev_project_by_time[index]["id"]
        else:
            return self._project_by_time[index]["id"]
    
    def get_worker_id_by_index(self, index)-> int:
        return self,_worker_id_rmap[index]

    def get_project_length(self, is_testing)-> int:
        if is_testing:
            return len(self._dev_project_by_time)
        else:
            return len(self._project_by_time)