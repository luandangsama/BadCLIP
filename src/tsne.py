import os
import sys
import torch
import pickle
import warnings
import argparse
import torchvision
import numpy as np
import pandas as pd
import seaborn as sns
from tqdm import tqdm
import matplotlib.pyplot as plt
from PIL import Image, ImageFile
from sklearn.manifold import TSNE ## Install sklearn: pip install -U scikit-learn
from sklearn.decomposition import PCA
from torch.utils.data import Dataset, DataLoader
current_directory = os.getcwd()
sys.path.insert(1,current_directory)
from pkgs.openai.clip import load as load_model
from backdoor.utils import apply_trigger

ImageFile.LOAD_TRUNCATED_IMAGES = True
warnings.filterwarnings("ignore")

def get_model(args, checkpoint):
    model, processor = load_model(name = args.model_name, pretrained = args.pretrained)

    if(args.device == "cpu"): model.float()
    else: model.to(args.device)

    if not args.pretrained:
        print("Loading model from checkpoint...")
        checkpoint = torch.load(checkpoint, map_location = args.device, weights_only=False)
        state_dict = checkpoint["state_dict"]
        if(next(iter(state_dict.items()))[0].startswith("module")):
            state_dict = {key[len("module."):]: value for key, value in state_dict.items()}
        model.load_state_dict(state_dict)

    model.eval()  
    return model, processor

class ImageCaptionDataset(Dataset):
    def __init__(self, path, images, captions, processor, add_backdoor=False, options=None):
        self.root = path
        self.processor = processor
        self.images = images
        self.captions = self.processor.process_text(captions)
        self.options = options
        self.add_backdoor = add_backdoor

    def __len__(self):
        return len(self.images)
    
    def add_trigger(self, image, patch_size = 16, patch_type = 'blended', patch_location = 'blended', tigger_pth=None, args=None):
        return apply_trigger(image, patch_size, patch_type, patch_location, tigger_pth, args=self.options)

    def __getitem__(self, idx):
        item = {}
        image = Image.open(os.path.join(self.root, self.images[idx])).convert("RGB")
        if self.add_backdoor:
            image = self.add_trigger(image, patch_size = self.options.patch_size, patch_type = self.options.patch_type, patch_location = self.options.patch_location, tigger_pth=self.options.tigger_pth, args=self.options)

        item["input_ids"] = self.captions["input_ids"][idx]
        item["attention_mask"] = self.captions["attention_mask"][idx]
        item["pixel_values"] = self.processor.process_image(image)
        return item

def get_embeddings(model, dataloader, processor, args):
    device = args.device
    list_embeddings = []
    with torch.no_grad():
        for batch in tqdm(dataloader):
            input_ids, attention_mask, pixel_values = batch["input_ids"].to(device, non_blocking = True), batch["attention_mask"].to(device, non_blocking = True), batch["pixel_values"].to(device, non_blocking = True)
            outputs = model(input_ids = input_ids, attention_mask = attention_mask, pixel_values = pixel_values)
            list_embeddings.append(outputs.image_embeds)
    return torch.cat(list_embeddings, dim = 0).cpu().detach().numpy()

def plot_embeddings(args):

    if not os.path.exists(args.save_data):    
        checkpoint = f'epoch_{args.epoch}.pt'
        model, processor = get_model(args, os.path.join(args.checkpoints_dir, checkpoint))
        df = pd.read_csv(args.original_csv)

        # to consider the top-k samples that were detected as backdoored
        if args.plot_detected_only:
            df = df[df['is_backdoor'] == 1]
            images, captions = df['image'].tolist(), df['caption'].tolist()
        else:
            images, captions = df['image'].tolist()[:10000], df['caption'].tolist()[:10000]

        backdoor_indices = list(filter(lambda x: 'backdoor' in images[x], range(len(images))))
        backdoor_images, backdoor_captions = [images[x] for x in backdoor_indices], [captions[x] for x in backdoor_indices]
        clean_indices = list(filter(lambda x: 'backdoor' not in images[x], range(len(images))))
        clean_images, clean_captions = [images[x] for x in clean_indices], [captions[x] for x in clean_indices]
        dataset_original = ImageCaptionDataset(args.original_csv, clean_images, clean_captions, processor)
        dataset_backdoor = ImageCaptionDataset(args.original_csv, backdoor_images, backdoor_captions, processor)
        dataloader_original = DataLoader(dataset_original, batch_size = args.batch_size, shuffle = False, pin_memory = True, drop_last = False)
        dataloader_backdoor = DataLoader(dataset_backdoor, batch_size = args.batch_size, shuffle = False, pin_memory = True, drop_last = False)

        original_images_embeddings = get_embeddings(model, dataloader_original, processor, args)
        backdoor_images_embeddings = get_embeddings(model, dataloader_backdoor, processor, args)
        len_original = len(original_images_embeddings)
        all_embeddings = np.concatenate([original_images_embeddings, backdoor_images_embeddings], axis = 0)
        print(len_original)
        with open(args.save_data, 'wb') as f:
            pickle.dump((all_embeddings, len_original), f)
    
    with open(args.save_data, 'rb') as f:
        all_embeddings, len_original = pickle.load(f)

    fig = plt.figure()
    # ax = fig.add_subplot(projection='2d')

    tsne = TSNE(n_components=2, verbose=1, perplexity=10, n_iter=1000)
    results = tsne.fit_transform(all_embeddings)

    # pca = PCA(n_components = 2)
    # results = pca.fit_transform(all_embeddings)
    # print(pca.explained_variance_ratio_)

    plt.scatter(results[:len_original, 0], results[:len_original, 1], label = 'Original')
    plt.scatter(results[len_original:, 0], results[len_original:, 1], label = 'Backdoor')

    plt.grid()
    plt.legend()
    plt.title(args.title)
    plt.tight_layout()

    os.makedirs(os.path.dirname(args.save_fig), exist_ok = True)
    plt.savefig(args.save_fig)

def plot_embeddings_custom(args):
    os.makedirs(f"{args.result_dir}/{args.name}", exist_ok = True)

    model, processor = get_model(args, args.checkpoints_path)
    df_target = pd.read_csv(args.target_csv)
    df_non_target = pd.read_csv(args.non_target_csv)

    if args.data_dir is None:
        args.data_dir = os.path.dirname(args.target_csv)

    dataset_target = ImageCaptionDataset(args.data_dir, df_target['image'].tolist(), df_target['caption'].tolist(), processor)
    dataset_non_target = ImageCaptionDataset(args.data_dir, df_non_target['image'].tolist(), df_non_target['caption'].tolist(), processor)
    dataset_backdoor = ImageCaptionDataset(args.data_dir, df_non_target['image'].tolist(), df_non_target['caption'].tolist(), processor, add_backdoor=True, options=args)

    dataloader_target = DataLoader(dataset_target, batch_size = args.batch_size, shuffle = False, pin_memory = True, drop_last = False)
    dataloader_non_target = DataLoader(dataset_non_target, batch_size = args.batch_size, shuffle = False, pin_memory = True, drop_last = False)
    dataloader_backdoor = DataLoader(dataset_backdoor, batch_size = args.batch_size, shuffle = False, pin_memory = True, drop_last = False)

    target_images_embeddings = get_embeddings(model, dataloader_target, processor, args)
    non_target_images_embeddings = get_embeddings(model, dataloader_non_target, processor, args)
    backdoor_images_embeddings = get_embeddings(model, dataloader_backdoor, processor, args)

    all_embeddings = np.concatenate([target_images_embeddings, non_target_images_embeddings, backdoor_images_embeddings], axis = 0)
    # all_embeddings = np.concatenate([target_images_embeddings, backdoor_images_embeddings], axis = 0)


    # with open(f"{args.result_dir}/{args.name}/data.pkl", 'wb') as f:
    #     pickle.dump(all_embeddings, f)
    
    fig = plt.figure()
    # ax = fig.add_subplot(projection='2d')

    tsne = TSNE(n_components=2, verbose=1, perplexity=10, n_iter=1000)
    results = tsne.fit_transform(all_embeddings)

    # pca = PCA(n_components = 2)
    # results = pca.fit_transform(all_embeddings)
    # print(pca.explained_variance_ratio_)

    plt.scatter(results[:len(dataset_target), 0], results[:len(dataset_target), 1], label = 'Target')
    plt.scatter(results[len(dataset_target):len(dataset_non_target) + len(dataset_target), 0], results[len(dataset_target):len(dataset_non_target) + len(dataset_target), 1], label = 'Non-target')
    plt.scatter(results[len(dataset_non_target) + len(dataset_target):, 0], results[len(dataset_non_target) + len(dataset_target):, 1], label = 'Backdoor')

    plt.grid()
    plt.legend()
    plt.title(args.title)
    plt.tight_layout()

    plt.savefig(f"{args.result_dir}/{args.name}/fig.png")


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("--original_csv", type = str, default = None, help = "original csv with captions and images")
    parser.add_argument("--device", type = str, default = 'cuda', help = "device")
    parser.add_argument("--model_name", type = str, default = "RN50", choices = ["RN50", "RN101", "RN50x4", "ViT-B/32"], help = "Model Name")
    parser.add_argument("--checkpoints_dir", type = str, default = "checkpoints/clip/", help = "Path to checkpoint directories")
    parser.add_argument("--save_data", type = str, default = None, help = "Save data")
    parser.add_argument("--save_fig", type = str, default = None, help = "Save fig png")
    parser.add_argument("--batch_size", type = int, default = 128, help = "Batch Size")
    parser.add_argument("--epoch", type=int, default=64, help="Epoch")
    parser.add_argument("--title", type=str, default=None, help="Title for plot")
    parser.add_argument("--plot_detected_only", action="store_true", default=False,
                        help="if True, we only plot the embeddings of images that were detected as backdoored (is_backdoor = 1)")
    
    parser.add_argument("--name", default = None, type = str, help = "name")
    parser.add_argument("--result_dir", default = '/home/necphy/luan/BadCLIP/analysis/embeddings_pca', type = str, help = "result dir")
    parser.add_argument("--checkpoints_path", type = str, default = None, help = "Path to checkpoint")
    parser.add_argument("--data_dir", type = str, default = None, help = "Path to data")
    parser.add_argument("--non_target_csv", type = str, default = None, help = "original csv with captions and images")
    parser.add_argument("--target_csv", type = str, default = None, help = "Path to target csv with captions and images")
    parser.add_argument("--add_backdoor", default = False, action = "store_true", help = "add backdoor or not")
    parser.add_argument("--pretrained", default = False, action = "store_true", help = "using pretrained or not")
    parser.add_argument("--patch_type", default = None, type = str, help = "patch type of backdoor")
    parser.add_argument("--blended_alpha", type = float, default = None, help = "Random crop size")
    parser.add_argument("--patch_location", default = None, type = str, help = "patch location of backdoor")
    parser.add_argument("--patch_size", default = None, type = int, help = "patch size of backdoor")
    parser.add_argument("--tigger_pth", default = None, type = str, help = "path of the trigger for backdoor")  # corrected parameter name
    parser.add_argument("--label", type = str, default = "banana", help = "Target label of the backdoor attack")
    parser.add_argument("--patch_name", type=str, default='../opti_patches/semdev_op0.jpg')

    parser.add_argument("--scale", type = str, default = None, help = "placeholder")



    args = parser.parse_args()

    plot_embeddings_custom(args)