import os
import random
import argparse
import numpy as np
from datasets import get_all_dataloaders
from utils import *
import tqdm
import torch
import torch.optim as optim
import random
from torchvision import transforms
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm as tqdmfunc
from transformers import pipeline
import re
from TransCLIP_solver.TransCLIP import TransCLIP_solver


transform_train = transforms.Compose([
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ColorJitter(
                    brightness=0.4, contrast=0.4, saturation=0.4, hue=0),
            ])

def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset',
                        default='dtd',
                        help='Dataset name',
                        type=str,
                        choices=[
                            'oxford_pets', 'eurosat', 'ucf101', 'sun397', 'caltech101',
                            'dtd', 'fgvc_aircraft', 'food101', 'oxford_flowers',
                            'stanford_cars', 'imagenet', 'cub'
                        ]
                        )
    parser.add_argument('--root_path', default='./datasets', type=str)
    parser.add_argument('--shots', default=0, type=int)
    parser.add_argument('--seed', default=1, type=int)
    parser.add_argument('--backbone', default='vit_b16', type=str)
    parser.add_argument('--gpt_path', default=None)
    parser.add_argument('--gpt_path_location', default=None)

    args = parser.parse_args()

    return args

class PseudoLabelTuningDataset(Dataset):
    def __init__(
            self,
            image_names,
            texts,
            image_to_text_indices,
            preprocess):
        self.image_names = image_names
        self.texts = texts
        self.image_to_text_indices = image_to_text_indices
        self.preprocess = preprocess

    def __len__(self):
        return len(self.image_to_text_indices.keys())
    
    def __getitem__(self, index):
        image_index = list(self.image_to_text_indices.keys())[index]
        text_index = random.choice(self.image_to_text_indices[image_index])
        im = Image.open(self.image_names[image_index]).convert('RGB')
        im = transform_train(im)
        im = self.preprocess(im)

        return im, self.texts[text_index], image_index, text_index


def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def main():
    backbones = {'rn50': 'RN50',
                 'rn101': 'RN101',
                 'vit_b16': 'ViT-B/16',
                 'vit_b32': 'ViT-B/32',
                 'vit_l14': 'ViT-L/14'}

    # Load config file
    args = get_arguments()

    cfg = {'dataset': args.dataset}

    cache_dir = os.path.join('./caches', cfg['dataset'])
    attr_dir = os.path.join('./attributes_llama3', cfg['dataset'])
    os.makedirs(cache_dir, exist_ok=True)
    os.makedirs(attr_dir, exist_ok=True)
    cfg['cache_dir'] = cache_dir
    cfg['load_cache'] = False
    cfg['load_pre_feat'] = False

    print("\nRunning configs.")

    cfg['backbone'] = backbones[args.backbone]

    cfg['seed'] = args.seed
    cfg['root_path'] = args.root_path
    cfg['shots'] = args.shots

    cfg['gpt_path'] = args.gpt_path
    cfg['gpt_path_location'] = args.gpt_path_location

    print(cfg, "\n")
    EPOCHS = 30
    acc = 0.0
    # CLIP
    clip_model, preprocess = clip.load(cfg['backbone'])
    pipe = pipeline("text-generation", model="meta-llama/Llama-3.1-8B-Instruct",max_new_tokens=1000,device='cuda')

    # Prepare dataset
    set_random_seed(args.seed)

    print("Preparing dataset.")

    train_loader, val_loader, test_loader, dataset = get_all_dataloaders(cfg, preprocess)

    loss_fn = torch.nn.CrossEntropyLoss()

    texts=None
    classes_ever_confused = []
    count_thresh = None

    for epoch in range(EPOCHS):
        clip_model.eval()

        shot_features, shot_labels, val_features, val_labels, test_features, test_labels, clip_prototypes, texts, image_names = get_all_features(
            cfg, train_loader, val_loader, test_loader, dataset, clip_model, texts, args.dataset)
        
        z,_,acc,_ = TransCLIP_solver(shot_features, shot_labels, val_features, val_labels, test_features, test_labels,
                                clip_prototypes, acc)

        ## Dynamic attribute generation
        top_values_confused, indices_confused = torch.topk(z, 2, dim=1)
        differences_confused = torch.abs(top_values_confused[:,0] - top_values_confused[:,1])
        im_indices_high_confusion = torch.nonzero(differences_confused < 0.1, as_tuple=True)[0]
        classes_confused = indices_confused[im_indices_high_confusion]
        unique_rows, counts = torch.unique(classes_confused.sort(-1)[0],dim=0,return_counts=True)
        if count_thresh is None:
            target = len(unique_rows) * 0.05
            min_difference = float('inf')
            for k in range(max(counts).item() + 1):  
                current_len = len(unique_rows[counts > k])
                difference = abs(current_len - target)
                if difference < min_difference:
                    min_difference = difference
                    count_thresh = k

        confused_pairs = unique_rows[counts > count_thresh]
        confused_pairs = [row for row in confused_pairs if row.tolist() not in classes_ever_confused]

        classes_ever_confused.extend([row.tolist() for row in confused_pairs])
        confused_classnames = [[dataset.classnames[pair[0]], dataset.classnames[pair[1]]] for pair in confused_pairs]
        for i_c, pairs_c in enumerate(confused_classnames):
            if not os.path.exists(os.path.join(attr_dir, pairs_c[0]+"_"+pairs_c[1]+".txt")) and not os.path.exists(os.path.join(attr_dir, pairs_c[1]+"_"+pairs_c[0]+".txt")):
                c_i_attr_0 = texts[confused_pairs[i_c][0]]
                attr_list_0 = "\n".join([f"• {attribute}" for attribute in c_i_attr_0])
                c_i_attr_1 = texts[confused_pairs[i_c][1]]
                attr_list_1 = "\n".join([f"• {attribute}" for attribute in c_i_attr_1])
                texts_gen = []
                for c_i in pairs_c:
                    other_c = pairs_c[1] if c_i == pairs_c[0] else pairs_c[0]
                    messages = [{"role": "user", "content": "I have the following attributes for " + pairs_c[0] + " as:\n" + attr_list_0 + "\n\nI have the following attributes for " + pairs_c[1] + " as:\n" + attr_list_1 + "\n\n Provide an exhaustive list of the most distinguishable attributes for " + c_i + " which are useful to differentiate it from " + other_c + ". Provide attributes in the same format as given above. While the new attributes should help to differentiate between these two classes, the texts themselves should only contain " + c_i + " and not " + other_c + ". Please provide the attributes as a list."},]
                    response = pipe(messages)
                    texts_c_i = re.findall(r"• (.+)", response[0]['generated_text'][1]['content'])
                    texts_gen.append(texts_c_i)
                texts[confused_pairs[i_c][0]].extend(texts_gen[0])
                texts[confused_pairs[i_c][1]].extend(texts_gen[1])
                with open(os.path.join(attr_dir, pairs_c[0]+"_"+pairs_c[1]+".txt"), "w") as file:
                    for line in texts_gen[0]:
                        file.write(line + "\n")
                with open(os.path.join(attr_dir, pairs_c[1]+"_"+pairs_c[0]+".txt"), "w") as file:
                    for line in texts_gen[1]:
                        file.write(line + "\n")
            else:
                with open(os.path.join(attr_dir, pairs_c[0]+"_"+pairs_c[1]+".txt"), "r") as file:
                    lines1 = [line.strip() for line in file]
                texts[confused_pairs[i_c][0]].extend(lines1)
                with open(os.path.join(attr_dir, pairs_c[1]+"_"+pairs_c[0]+".txt"), "r") as file:
                    lines2 = [line.strip() for line in file]
                texts[confused_pairs[i_c][1]].extend(lines2)


        shot_features, shot_labels, val_features, val_labels, test_features, test_labels, clip_prototypes, texts, image_names = get_all_features(cfg, train_loader, val_loader, test_loader, dataset, clip_model, texts, args.dataset)
        
        z,_,acc,_ = TransCLIP_solver(shot_features, shot_labels, val_features, val_labels, test_features, test_labels,clip_prototypes, acc)  

        ## Fine-tuning        
        k = 8

        top_k_indices = {}

        for class_idx in range(z.shape[1]):  
            class_probs = z[:, class_idx]
            top_k = torch.topk(class_probs, k).indices
            top_k_indices[class_idx] = top_k
        
        text_indices = {}
        start_idx = 0
        i = 0
        for sublist in texts:
            indices = list(range(start_idx, start_idx + len(sublist)))
            text_indices[i] = indices
            start_idx += len(sublist)
            i += 1

        image_to_text_indices = {}

        for class_index in top_k_indices:
            image_indices = top_k_indices[class_index]
            text_indices_list = text_indices[class_index]
            
            for image_index in image_indices:
                image_index_int = int(image_index.item())  
                
                if image_index_int not in image_to_text_indices:
                    image_to_text_indices[image_index_int] = set()
                image_to_text_indices[image_index_int].update(text_indices_list)

        for image_index in image_to_text_indices:
            image_to_text_indices[image_index] = list(image_to_text_indices[image_index])

        text_to_image_indices = {}

        for image_index, text_indices in image_to_text_indices.items():
            for text_index in text_indices:
                if text_index not in text_to_image_indices:
                    text_to_image_indices[text_index] = set()
                text_to_image_indices[text_index].add(image_index)

        for text_index in text_to_image_indices:
            text_to_image_indices[text_index] = list(text_to_image_indices[text_index])

        all_text_indices = set()
        for indices in text_to_image_indices.values():
            all_text_indices.update(indices)

        texts_all = [item for sublist in texts for item in sublist]

        dataset_pseudo = PseudoLabelTuningDataset(image_names = image_names,
                                            texts = texts_all,
                                            image_to_text_indices = image_to_text_indices,
                                            preprocess = preprocess)

        clip_model.train()

        main_parameters = [param for name, param in clip_model.named_parameters() if "proj" not in name and "text_projection" not in name]
        optimizer = optim.AdamW([
        {'params': main_parameters, 'lr': 2e-7, 'weight_decay' : 1e-4},
        {'params': [clip_model.text_projection, clip_model.visual.proj], 'lr': 1e-6, 'weight_decay' : 1e-4},
        ], betas=(0.9,0.98),eps=1e-6) 

        
        dataloader = DataLoader(dataset_pseudo, batch_size = 32, shuffle=True, num_workers=8, pin_memory=False)
        
        
        total_train_loss = 0

        train_bar = tqdmfunc(enumerate(dataloader), total=len(dataloader), desc=f"Training Epoch {epoch + 1}/{EPOCHS}")
        for num_b, (images, texts, image_indices, text_indices) in train_bar:
            optimizer.zero_grad()
            texts = clip.tokenize(texts)
            images = images.cuda()
            texts = texts.cuda()
            logits_per_image, logits_per_text = clip_model(images, texts)

            list_probs_image = []
            for i in range(images.shape[0]):
                image_index = image_indices[i]
                text_indices_i = image_to_text_indices[image_index.item()]
                text_indices_i_set = set(text_indices_i)
                overlapping_indices = torch.tensor([i for i, val in enumerate(text_indices.tolist()) if val in text_indices_i_set]).cuda()
                this_sm = logits_per_image[i,:].log_softmax(-1)
                list_probs_image.append(-this_sm.index_select(-1, overlapping_indices).mean())

            list_probs_text = []
            for i in range(texts.shape[0]):
                text_index = text_indices[i]
                image_indices_i = text_to_image_indices[text_index.item()]
                image_indices_i_set = set(image_indices_i)
                overlapping_indices = torch.tensor([i for i, val in enumerate(image_indices.tolist()) if val in image_indices_i_set]).cuda()
                this_sm = logits_per_text[i,:].log_softmax(-1)
                list_probs_text.append(-this_sm.index_select(-1, overlapping_indices).mean())
        
            loss = torch.stack(list_probs_image).mean() + torch.stack(list_probs_text).mean()
            total_train_loss += loss.item()
            loss.backward()
            optimizer.step()
            train_bar.set_postfix(train_loss=(total_train_loss / (num_b + 1)))
    gta_clip_acc = acc

    cfg['gpt_path'] = None
    cfg['gpt_path_location'] = None
    clip_model, preprocess = clip.load(cfg['backbone'])
    set_random_seed(args.seed)
    train_loader, val_loader, test_loader, dataset = get_all_dataloaders(cfg, preprocess)
    clip_model.eval()
    shot_features, shot_labels, val_features, val_labels, test_features, test_labels, clip_prototypes, texts, image_names = get_all_features(
        cfg, train_loader, val_loader, test_loader, dataset, clip_model, None)
    z,_,acc,clip_acc = TransCLIP_solver(shot_features, shot_labels, val_features, val_labels, test_features, test_labels,
                            clip_prototypes)
    transclip_acc = acc

    print("{:<20} {:>10}".format("Model", "Accuracy"))
    print("-" * 30)
    print("{:<20} {:>10.2f}".format("CLIP", clip_acc))
    print("{:<20} {:>10.2f}".format("TransCLIP", transclip_acc))
    print("{:<20} {:>10.2f}".format("GTA-CLIP", gta_clip_acc))

    


if __name__ == '__main__':
    main()
