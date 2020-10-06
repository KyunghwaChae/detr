from pathlib import Path
from PIL import Image

import torch
import torch.utils.data as data
import torchvision
import tqdm
import os
import numpy as np
import xml.etree.ElementTree as ET

import datasets.transforms as T
from datasets.coco import ConvertCocoPolysToMask

# fmt: off
CLASS_NAMES = ['3m', 'abus', 'accenture', 'adidas', 'adidas1', 'adidas_text', 'airhawk', 'airness', 'aldi', 'aldi_text',
               'alfaromeo', 'allett', 'allianz', 'allianz_text', 'aluratek', 'aluratek_text', 'amazon', 'amcrest',
               'amcrest_text', 'americanexpress', 'americanexpress_text', 'android', 'anz', 'anz_text', 'apc',
               'apecase', 'apple', 'aquapac_text', 'aral', 'armani', 'armitron', 'aspirin', 'asus', 'at_and_t',
               'athalon', 'audi', 'audi_text', 'axa', 'bacardi', 'bankofamerica', 'bankofamerica_text', 'barbie',
               'barclays', 'base', 'basf', 'batman', 'bayer', 'bbc', 'bbva', 'becks', 'bellataylor', 'bellodigital',
               'bellodigital_text', 'bem', 'benrus', 'bershka', 'bfgoodrich', 'bik', 'bionade', 'blackmores',
               'blizzardentertainment', 'bmw', 'boeing', 'boeing_text', 'bosch', 'bosch_text', 'bottegaveneta',
               'bridgestone', 'bridgestone_text', 'budweiser', 'budweiser_text', 'bulgari', 'burgerking',
               'burgerking_text', 'calvinklein', 'canon', 'carglass', 'carlsberg', 'carters', 'cartier', 'caterpillar',
               'chanel', 'chanel_text', 'cheetos', 'chevrolet', 'chevrolet_text', 'chevron', 'chickfila', 'chimay',
               'chiquita', 'cisco', 'citi', 'citroen', 'citroen_text', 'coach', 'cocacola', 'coke', 'colgate',
               'comedycentral', 'converse', 'corona', 'corona_text', 'costa', 'costco', 'cpa_australia', 'cvs',
               'cvspharmacy', 'danone', 'dexia', 'dhl', 'disney', 'doritos', 'drpepper', 'dunkindonuts', 'ebay', 'ec',
               'erdinger', 'espn', 'esso', 'esso_text', 'evernote', 'facebook', 'fedex', 'ferrari', 'firefox',
               'firelli', 'fly_emirates', 'ford', 'fosters', 'fritolay', 'fritos', 'gap', 'generalelectric', 'gildan',
               'gillette', 'goodyear', 'google', 'gucci', 'guinness', 'hanes', 'head', 'head_text', 'heineken',
               'heineken_text', 'heraldsun', 'hermes', 'hersheys', 'hh', 'hisense', 'hm', 'homedepot', 'homedepot_text',
               'honda', 'honda_text', 'hp', 'hsbc', 'hsbc_text', 'huawei', 'huawei_text', 'hyundai', 'hyundai_text',
               'ibm', 'ikea', 'infiniti', 'infiniti_text', 'intel', 'internetexplorer', 'jackinthebox', 'jacobscreek',
               'jagermeister', 'jcrew', 'jello', 'johnnywalker', 'jurlique', 'kelloggs', 'kfc', 'kia', 'kitkat',
               'kodak', 'kraft', 'lacoste', 'lacoste_text', 'lamborghini', 'lays', 'lego', 'levis', 'lexus',
               'lexus_text', 'lg', 'londonunderground', 'loreal', 'lotto', 'luxottica', 'lv', 'marlboro',
               'marlboro_fig', 'marlboro_text', 'maserati', 'mastercard', 'maxwellhouse', 'maxxis', 'mccafe',
               'mcdonalds', 'mcdonalds_text', 'medibank', 'mercedesbenz', 'mercedesbenz_text', 'michelin', 'microsoft',
               'milka', 'millerhighlife', 'mini', 'miraclewhip', 'mitsubishi', 'mk', 'mobil', 'motorola', 'mtv', 'nasa',
               'nb', 'nbc', 'nescafe', 'netflix', 'nike', 'nike_text', 'nintendo', 'nissan', 'nissan_text', 'nivea',
               'northface', 'nvidia', 'obey', 'olympics', 'opel', 'optus', 'optus_yes', 'oracle', 'pampers',
               'panasonic', 'paulaner', 'pepsi', 'pepsi_text', 'pepsi_text1', 'philadelphia', 'philips', 'pizzahut',
               'pizzahut_hut', 'planters', 'playstation', 'poloralphlauren', 'porsche', 'porsche_text', 'prada', 'puma',
               'puma_text', 'quick', 'rbc', 'recycling', 'redbull', 'redbull_text', 'reebok', 'reebok1', 'reebok_text',
               'reeses', 'renault', 'republican', 'rittersport', 'rolex', 'rolex_text', 'ruffles', 'samsung',
               'santander', 'santander_text', 'sap', 'schwinn', 'scion_text', 'sega', 'select', 'shell', 'shell_text',
               'shell_text1', 'siemens', 'singha', 'skechers', 'sony', 'soundcloud', 'soundrop', 'spar', 'spar_text',
               'spiderman', 'sprite', 'standard_liege', 'starbucks', 'stellaartois', 'subaru', 'subway', 'sunchips',
               'superman', 'supreme', 'suzuki', 't-mobile', 'tacobell', 'target', 'target_text', 'teslamotors',
               'texaco', 'thomsonreuters', 'tigerwash', 'timberland', 'tissot', 'tnt', 'tommyhilfiger', 'tostitos',
               'total', 'toyota', 'toyota_text', 'tsingtao', 'twitter', 'umbro', 'underarmour', 'unicef', 'uniqlo',
               'uniqlo1', 'unitednations', 'ups', 'us_president', 'vaio', 'velveeta', 'venus', 'verizon',
               'verizon_text', 'visa', 'vodafone', 'volkswagen', 'volkswagen_text', 'volvo', 'walmart', 'walmart_text',
               'warnerbros', 'wellsfargo', 'wellsfargo_text', 'wii', 'williamhill', 'windows', 'wordpress', 'xbox',
               'yahoo', 'yamaha', 'yonex', 'yonex_text', 'youtube', 'zara']
# fmt: on


class LogoDetection(torchvision.datasets.CocoDetection):
    def __init__(self, img_folder, ann_folder, split_set, transforms):

        super(LogoDetection, self).__init__(img_folder, None)
        self._transforms = transforms
        self.class_names = CLASS_NAMES
        self.root = img_folder
        self.prepare = ConvertCocoPolysToMask(False)
        self.coco.dataset["annotations"] = []

        fileids = np.loadtxt(split_set, dtype=np.str)

        instances = []
        ids = []
        files = []
        n_anns = 0
        for i, fileid in enumerate(tqdm.tqdm(fileids, desc='Reading annotations')):
            anno_file = os.path.join(ann_folder, fileid + ".xml")
            jpeg_file = os.path.join(img_folder, fileid + ".jpg")

            tree = ET.parse(anno_file)

            ids.append(i)
            files.append(jpeg_file)

            annotations = []
            for obj in tree.findall("object"):
                cls = obj.find("name").text

                bbox = obj.find("bndbox")
                bbox = [float(bbox.find(x).text) for x in ["xmin", "ymin", "xmax", "ymax"]]
                bbox[0] -= 1.0
                bbox[1] -= 1.0

                category_id = self.class_names.index(cls)

                bbox[2], bbox[3] = bbox[2] - bbox[0], bbox[3] - bbox[1]
                meta = {
                    "bbox": bbox,
                    "category_id": category_id,
                    "area": bbox[2] * bbox[3],
                    "id": n_anns,
                    "iscrowd": 0,
                }
                n_anns += 1
                self.coco.dataset["annotations"].append({**meta, "image_id": i})
                annotations.append(meta)
            instances.append(annotations)

        self.ids = ids

        list_categories = []
        for i, cat in enumerate(self.class_names):
            categories = {}
            categories["id"] = i
            categories["name"] = cat
            categories["supercategory"] = "logo"
            list_categories.append(categories)

        self.coco.dataset["categories"] = list_categories

        list_images = []
        for name, img_id in zip(files, ids):
            images = {}
            images["file_name"] = name.split("/")[-1]
            images["id"] = img_id
            list_images.append(images)

        self.coco.dataset["images"] = list_images
        self.coco.createIndex()


    def __getitem__(self, idx):

        img, target = super(LogoDetection, self).__getitem__(idx)
        image_id = self.ids[idx]
        target = {'image_id': image_id, 'annotations': target}
        img, target = self.prepare(img, target)
        if self._transforms is not None:
            img, target = self._transforms(img, target)

        return img, target

def build(image_set, args):
    root = Path(args.data_path)
    assert root.exists(), f'provided Dataset path {root} does not exist'
    PATHS = {
        "train": (root / "JPEGImages", root / "Annotations", root / "ImageSets/supervision_type/supervised_imageset/trainval.txt"),
        "val": (root / "JPEGImages", root / "Annotations", root / "ImageSets/supervision_type/supervised_imageset/test.txt"),
    }

    img_folder, ann_file, split_set = PATHS[image_set]
    dataset = LogoDetection(img_folder, ann_file, split_set, make_transforms(image_set, args))
    return dataset

def make_transforms(image_set, args):

    normalize = T.Compose([
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # avoid OOM error
    if args.dilation:
        scales = [480, 512, 544, 576, 608, 640]
        max_size = 800
    else:
        scales = [480, 512, 544, 576, 608, 640, 672, 704, 736, 768, 800]
        max_size = 1333

    if image_set == 'train':
        return T.Compose([
            T.RandomHorizontalFlip(),
            T.RandomSelect(
                T.RandomResize(scales, max_size=max_size),
                T.Compose([
                    T.RandomResize([400, 500, 600]),
                    T.RandomSizeCrop(384, 600),
                    T.RandomResize(scales, max_size=max_size),
                ])
            ),
            normalize,
        ])

    if image_set == 'val':
        return T.Compose([
            T.RandomResize([800], max_size=1333),
            normalize,
        ])

    raise ValueError(f'unknown {image_set}')

