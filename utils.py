from tqdm import tqdm

import torch
import torch.nn.functional as F

import clip

import os 
import pandas as pd

EUROSAT_CNAMES = {
    "Annual Crop Land" : "Annual Crop",
    "Forest": "Forest",
    "Herbaceous Vegetation Land" : "Herbaceous Vegetation",
    "Highway or Road" : "Highway",
    "Industrial Buildings" : "Industrial",
    "Pasture Land" : "Pasture",
    "Permanent Crop Land" : "Permanent Crop",
    "Residential Buildings" : "Residential",
    "River" : "River",
    "Sea or Lake" : "SeaLake",
}


def load_class_ids(filepath="./extras/classnames.txt"):
    class_dict = {}
    with open(filepath, 'r') as file:
        for line in file:
            class_id, name = line.strip().split(maxsplit=1)
            class_dict[name.strip()] = class_id
    return class_dict
imgnet_class_dict = class_dict = load_class_ids()


with open('./extras/class_names_cub.txt') as f:
    all_classes = f.readlines()
cub_all_classes = [line.rstrip('\n') for line in all_classes]


def cls_acc(output, target, topk=1):
    pred = output.topk(topk, 1, True, True)[1].t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    acc = float(correct[: topk].reshape(-1).float().sum(0, keepdim=True).cpu().numpy())
    acc = 100 * acc / target.shape[0]
    return acc


def clip_classifier(classnames, template, gpt_path, gpt_path_location, clip_model, reduce='mean', gpt=False, wordnet_dict=None, texts_in=None, dataset_name="cub"):
    with torch.no_grad():
        clip_weights = []
        texts_all = []
        if wordnet_dict is not None:
            indices = []
            i = 0
            for classname in classnames:
                allnames = [classname] + wordnet_dict[classname]
                for name in allnames:
                   
                    # Tokenize the prompts
                    name = name.replace('_', ' ')
                    
                    texts = [t.format(name) for t in template]
                    texts = clip.tokenize(texts).cuda()
        
                    class_embeddings = clip_model.encode_text(texts)
                    class_embeddings /= class_embeddings.norm(dim=-1, keepdim=True)
                    if reduce=='mean':
                        class_embedding = class_embeddings.mean(dim=0)
                        class_embedding /= class_embedding.norm()
                        clip_weights.append(class_embedding)
                    if reduce is None:
                        class_embeddings /= class_embeddings.norm(dim=1, keepdim=True)
                        clip_weights.append(class_embeddings)
                    i+=1
                indices.append(i)
                
            return clip_weights, indices
        else:
            for ci, classname in enumerate(classnames):
                if isinstance(texts_in, list):
                    texts = texts_in[ci]

                # Tokenize the prompts
                else:
                    classname_org = classname
                    classname = classname.replace('_', ' ')

                    if gpt_path is not None:
                        if "satellite" in template[0]:
                            with open(os.path.join(gpt_path, EUROSAT_CNAMES[classname.replace('/','SLASH')]+".txt")) as f:
                                texts_class = f.readlines()
                        elif "imagenet" in gpt_path:
                            with open(os.path.join(gpt_path, imgnet_class_dict.get(classname, None)+".txt")) as f:
                                texts_class = f.readlines()
                        elif "caltech" in gpt_path or "sun" in gpt_path or "ucf" in gpt_path:
                            with open(os.path.join(gpt_path, classname_org +".txt")) as f:
                                texts_class = f.readlines()
                        else:
                            with open(os.path.join(gpt_path, classname.replace('/','SLASH')+".txt")) as f:
                                texts_class = f.readlines()

                        if dataset_name == "stanford_cars":
                            texts = ["a photo of a " + classname + line.rstrip('\n')[5:] for line in texts_class if line.strip()]
                        elif dataset_name == "cub":
                            texts = ["a photo of a " + classname + line.rstrip('\n')[1:] for line in texts_class if line.strip()]
                        elif dataset_name == "food101":
                            texts = ["a photo of " + classname + ", a type of" + line.rstrip('\n')[1:] for line in texts_class if line.strip()]
                        elif dataset_name == "oxford_flowers":
                            texts = ["a photo of a " + classname + " flower" + line.rstrip('\n')[7:] for line in texts_class if line.strip()]
                        elif dataset_name == "fgvc_aircraft":
                            texts = ["a photo of a " + classname + line.rstrip('\n')[2:] for line in texts_class if line.strip()]
                        elif dataset_name == "eurosat":
                            texts = ["a centered satellite photo of " + classname + line.rstrip('\n')[17:] for line in texts_class if line.strip()]
                        elif dataset_name in ["imagenet", "caltech101"]:
                            texts = ["a photo of a " + classname + line.rstrip('\n')[9:] for line in texts_class if line.strip()]
                        elif dataset_name == "dtd":
                            texts = [classname + " texture" + line.rstrip('\n')[9:] for line in texts_class if line.strip()]
                        elif dataset_name == "oxford_pets":
                            texts = ["a photo of a " + classname + ", a type of pet" + line.rstrip('\n')[5:] for line in texts_class if line.strip()]
                        elif dataset_name == "sun397":
                            texts = ["a photo of a " + classname + line.rstrip('\n')[7:] for line in texts_class if line.strip()]
                        elif dataset_name == "ucf101":
                            texts = ["a photo of a person doing " + classname + line.rstrip('\n')[9:] for line in texts_class if line.strip()]
                        else:
                            raise ValueError(f"Unsupported dataset: {dataset_name}")
                        if gpt_path_location is not None:
                            with open(os.path.join(gpt_path_location, classname.replace('/','SLASH')+".txt")) as f:
                                texts_class_loc = f.readlines()
                            texts_class_loc = [line.replace('"', '').replace("'", '') for line in texts_class_loc]
                            if dataset_name == "oxford_flowers":
                                texts_class_loc = ["a photo of " + classname + " flower" + line.rstrip('\n')[7:] for line in texts_class_loc if line.strip()] 
                            elif dataset_name == "cub":
                                texts_class_loc = ["a photo of a " + classname + line.rstrip('\n')[1:] for line in texts_class_loc if line.strip()]
                            texts.extend(texts_class_loc)
                    else:
                        texts = [t.format(classname)  for t in template]
                
                texts_all.append(texts)
                texts = clip.tokenize(texts).cuda()
    
                class_embeddings = clip_model.encode_text(texts)
                
                class_embeddings /= class_embeddings.norm(dim=-1, keepdim=True)
                if gpt_path is not None:
                    # class_embedding = class_embeddings.mean(dim=0)
                    # class_embedding /= class_embedding.norm()
                    # clip_weights.append(class_embedding)
                    clip_weights.append(class_embeddings.transpose(0,1))
                else:
                    class_embeddings /= class_embeddings.norm(dim=1, keepdim=True)
                    clip_weights.append(class_embeddings)
            if gpt_path is None:
                clip_weights = torch.stack(clip_weights, dim=-1).cuda()
            
    return clip_weights, texts_all


def get_all_features(cfg, train_loader, val_loader, test_loader, dataset, clip_model, texts_in=None, dataset_name="cub"):
    clip_prototypes, texts = clip_classifier(dataset.classnames, dataset.template, dataset.gpt_path, dataset.gpt_path_location, clip_model, reduce=None, texts_in=texts_in, dataset_name=dataset_name)
    if isinstance(texts_in, list):
        texts = texts_in
    test_features, test_labels, image_names = pre_load_features(cfg, "test", clip_model, test_loader)
    shot_features = None
    shot_labels = None
    val_features = None
    val_labels = None

    if cfg['shots'] > 0:
        shot_features, shot_labels = build_cache_model(cfg, clip_model, train_loader, n_views=0,
                                                     reduce=None)
        val_features, val_labels = pre_load_features(cfg, "val", clip_model, val_loader)
    return shot_features, shot_labels, val_features, val_labels, test_features, test_labels, clip_prototypes, texts, image_names


def build_cache_model(cfg, clip_model, train_loader_cache, n_views=0, reduce=None):
    if cfg['load_cache'] == False:    
        cache_keys = []
        cache_values = []
        if n_views == 0:
            n_epochs =1
        else:
            n_epochs = n_views
        with torch.no_grad():
            # Data augmentation for the cache model
            for augment_idx in range(n_epochs):
                train_features = []
                train_labels = []
                for i, (images, target) in enumerate(tqdm(train_loader_cache)):
                    images = images.cuda()
                    image_features = clip_model.encode_image(images)
                    train_features.append(image_features)
                    
                    if augment_idx == 0:
                        target = target.cuda()
                        cache_values.append(target)
                        
                cache_keys.append(torch.cat(train_features, dim=0).unsqueeze(0))


        
        if n_views == 1:
            cache_keys = torch.cat(cache_keys, dim=0).mean(dim=0)
            cache_keys /= cache_keys.norm(dim=-1, keepdim=True)
            #cache_keys = cache_keys.permute(1, 0)
        else:
            cache_keys = torch.cat(cache_keys, dim=0) # [n_views, n_classes, n_features]
            if reduce == 'mean':
                cache_keys = cache_keys.mean(0, keepdim=True)
                
            cache_keys /= cache_keys.norm(dim=-1, keepdim=True)
            cache_keys.permute(0, 2, 1)
            
        cache_values = F.one_hot(torch.cat(cache_values, dim=0)).half()

        torch.save(cache_keys, cfg['cache_dir'] + '/keys_' + str(cfg['shots']) + "shots.pt")
        torch.save(cache_values, cfg['cache_dir'] + '/values_' + str(cfg['shots']) + "shots.pt")

    else:
        cache_keys = torch.load(cfg['cache_dir'] + '/keys_' + str(cfg['shots']) + "shots.pt")
        cache_values = torch.load(cfg['cache_dir'] + '/values_' + str(cfg['shots']) + "shots.pt")

    return cache_keys, cache_values






def pre_load_features(cfg, split, clip_model, loader, n_views=1):
    if cfg['load_pre_feat'] == False:
        features, labels, image_names = [], [], []
        
        with torch.no_grad():
          
            for view in range(n_views):
                length = 0
                for i, (images, target, image_name) in enumerate(tqdm(loader)):
                    if n_views == 1:
                        
                        images, target = images.cuda(), target.cuda()
                        
                        
                        image_features = clip_model.encode_image(images)
                        
                        image_features /= image_features.norm(dim=-1, keepdim=True)
                        
                        
                        features.append(image_features.cpu())
                        labels.append(target.cpu())
                        image_names.extend(image_name)
                    else:
                        images, target = images.cuda(), target.cuda()
                        image_features = clip_model.encode_image(images)
                        image_features /= image_features.norm(dim=-1, keepdim=True)
                        if view == 0:
                            labels.append(target.cpu())
                            if i ==0:
                                mean_features = image_features
                            else:
                                mean_features = torch.cat((mean_features, image_features))
                        else:
                            mean_features[length:length+image_features.size(0)] += image_features
                            length += image_features.size(0)
                            
        if n_views > 1:
            mean_features = mean_features / n_views
            features = mean_features / mean_features.norm(dim=-1, keepdim=True)
            labels = torch.cat(labels)
        
        elif n_views==1:
            features, labels = torch.cat(features), torch.cat(labels)
        
        torch.save(features, cfg['cache_dir'] + "/" + split + "_f.pt")
        torch.save(labels, cfg['cache_dir'] + "/" + split + "_l.pt")
   
    else:
        
        features = torch.load(cfg['cache_dir'] + "/" + split + "_f.pt")
        labels = torch.load(cfg['cache_dir'] + "/" + split + "_l.pt")
    
    return features, labels, image_names

