import os
from scipy.io import loadmat

from .oxford_pets import OxfordPets
from .utils import Datum, DatasetBase
import math
import torchfile
import json

template = ['a photo of a {} bird']


class CUB(DatasetBase):

    dataset_dir = "CUB_200_2011/CUB_200_2011"

    def __init__(self, root, num_shots, gpt_path=None, gpt_path_location=None):
        self.dataset_dir = os.path.join(root, self.dataset_dir, "images_extracted")
        self.template = template
        self.gpt_path = gpt_path
        self.gpt_path_location = gpt_path_location
        datfile_train = torchfile.load('./extras/anno/train.dat')
        datfile_val = torchfile.load('./extras/anno/val.dat')
        with open('./extras/cub_classes.json', 'r') as f:
            self.class_list = json.load(f)
        with open('./extras/class_names_cub.txt') as f:
            all_classes = f.readlines()
        self.all_classes = [line.rstrip('\n') for line in all_classes]

        train = self.read_split(datfile_train, self.dataset_dir)
        val = self.read_split(datfile_val, self.dataset_dir)
        test = self.read_split(datfile_val, self.dataset_dir)
        n_shots_val = min(num_shots, 4)
        val = self.generate_fewshot_dataset(val, num_shots=n_shots_val)
        train = self.generate_fewshot_dataset(train, num_shots=num_shots)
        # train, val, test = self.subsample_classes(train, val, test, subsample="new")
        super().__init__(train_x=train, val=val, test=test)
    
    def read_split(self, datfile, data_dir):
        out = []
        for name, _ in datfile.items():
            if name.decode() == 'Black_Tern_0079_143998.jpg':
                continue
            impath = os.path.join(data_dir, name.decode())
            label = self.class_list[name.decode()]-1
            classname = self.all_classes[label]
            item = Datum(
                    impath=impath,
                    label=int(label),
                    classname=classname
                )
            out.append(item)
        return out    
    
    @staticmethod
    def subsample_classes(*args, subsample="all"):
        """Divide classes into two groups. The first group
        represents base classes while the second group represents
        new classes.

        Args:
            args: a list of datasets, e.g. train, val and test.
            subsample (str): what classes to subsample.
        """
        assert subsample in ["all", "base", "new"]

        # if subsample == "all":
        #     return args
        
        dataset = args[0]
        labels = set()
        for item in dataset:
            labels.add(item.label)
        labels = list(labels)
        labels.sort()
        n = len(labels)
        # Divide classes into two halves
        m = math.ceil(n / 2)

        print(f"SUBSAMPLE {subsample.upper()} CLASSES!")
        if subsample == "base":
            selected = labels[:m]  # take the first half
        elif subsample == "new":
            selected = labels[m:]  # take the second half
        elif subsample == "all":
            selected = labels
        relabeler = {y: y_new for y_new, y in enumerate(selected)}
        
        output = []
        for dataset in args:
            dataset_new = []
            for item in dataset:
                if item.label not in selected:
                    continue
                item_new = Datum(
                    impath=item.impath,
                    label=relabeler[item.label],
                    classname=item.classname
                )
                dataset_new.append(item_new)
            output.append(dataset_new)
        
        return output