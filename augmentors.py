import albumentations as A
import matplotlib.pyplot as plt
import numpy as np
import os, shutil, yaml, tqdm, time, cv2, gc

class ObjectAugmentor():
    def __init__(self, maximum_size = 16, maximum_process_second = 60 * 60 * 24, transform_option = [A.Blur(p=0.8)], ignore_classes = []):
        self.maximum_size = maximum_size
        self.maximum_process_second = maximum_process_second
        self.ignore_classes = ignore_classes
        self.transform_option = transform_option
        #, A.CoarseDropout(p=0.8,hole_height_range=(64, 256)

    def summary(self, dataset_path, new_dataset_path):
        if os.path.exists(new_dataset_path):
            shutil.rmtree(new_dataset_path)
        shutil.copytree(dataset_path, new_dataset_path)
        with open(f'{new_dataset_path}/data.yaml', 'r') as f:
            config = yaml.safe_load(f)
            config['folder'] = new_dataset_path
            for subset in ['train', 'val', 'test']:
                config[subset] = config[subset].replace(dataset_path, new_dataset_path)
        with open(f'{new_dataset_path}/data.yaml', 'w') as f:
            yaml.dump(config, f)
        self.classes = config['names']

        print('summary the samples...')
        self.paired_sample = []
        for image_name in os.listdir(new_dataset_path + '/train/images'):
            for label_name in os.listdir(new_dataset_path + '/train/labels'):
                if image_name.rstrip('.jpg') + '.txt' == label_name:
                    image_path, label_path = f"{new_dataset_path}/train/images/{image_name}", f"{new_dataset_path}/train/labels/{label_name}"
                    self.paired_sample.append([image_path, label_path])

    def custom_crop(self, crop_limit = 0.2):
        i = np.random.choice(len(self.paired_sample), 1)[0]
        image_path, label_path =  self.paired_sample[i][0], self.paired_sample[i][1]
        image = cv2.imread(image_path)
        with open(label_path, 'r') as f:
          annots = np.array([[float(x) for x in line.strip().split()] for line in f.readlines()])
            
        if len(annots) != 0:
            classes, bboxes = annots[:, 0], np.clip(annots[:, 1:], 0, 1)
            for i, (cls1, box1) in enumerate(zip(classes, bboxes)):
                x1, y1, x2, y2 = cxcywh_to_xyxy(box1, size = 100)
                box1_width = x2 - x1
                box1_height = y2 - y1
                crop_zone = (box1_width * box1_height) / (100 * 100)
                if crop_zone < crop_limit:
                    cropped_classes, cropped_bboxes  = [cls1], [cxcywh_to_xyxy(box1, 1)]
                    for j, (cls2, box2) in enumerate(zip(classes, bboxes)):
                        if (True in [RoI(cxcywh_to_xyxy(box2, 1), box) for box in cropped_bboxes]) and (i != j) and (self.classes[int(cls2)] not in self.ignore_classes):
                            cropped_classes.append(cls2)
                            cropped_bboxes.append(cxcywh_to_xyxy(box2, 1))
                    cropped_bboxes = np.array(cropped_bboxes)
                    x1, y1, x2, y2 = np.min(cropped_bboxes[:, 0]), np.min(cropped_bboxes[:, 1]), np.max(cropped_bboxes[:, 2]), np.max(cropped_bboxes[:, 3])
                    cropped_image = image[int(y1*image.shape[0]): int(y2*image.shape[0]), int(x1 * image.shape[1]): int(x2 * image.shape[1])]  
                    cropped_bboxes[:, 0] = (cropped_bboxes[:, 0] - x1) / (x2 - x1)
                    cropped_bboxes[:, 1] = (cropped_bboxes[:, 1] - y1) / (y2 - y1)
                    cropped_bboxes[:, 2] = (cropped_bboxes[:, 2] - x1) / (x2 - x1)
                    cropped_bboxes[:, 3] = (cropped_bboxes[:, 3] - y1) / (y2 - y1)
                    return cropped_image, cropped_classes, cropped_bboxes         
        else:
            return None, [], []

    def draw_heatmap(self, image, annots):
        search_map = np.zeros(image.shape[: 2])
        if len(annots) != 0:
            classes, bboxes = annots[:, 0], np.clip(annots[:, 1:], 0, 1)
            for i, (cls, box) in enumerate(zip(classes, bboxes)):
                if self.classes[int(cls)] != 'battery':
                    x1, y1, x2, y2 = cxcywh_to_xyxy(box, 1)
                    search_map[int(y1 *image.shape[0]): int(y2 *image.shape[0]), int(x1 *image.shape[1]): int(x2 *image.shape[1])] += 1
        search_map[:, -1], search_map[-1, :] = 1, 1
        return search_map
        
    def startpoint(self, search_map, x_min = 64, y_min = 64):
        Y, X = np.where(search_map == 0)
        for _ in range(16):
            i = np.random.choice(len(X), 1)[0]
            if (Y[i] + y_min) < search_map.shape[0] and (X[i] + x_min) < search_map.shape[1]:
                if np.sum(search_map[Y[i]: Y[i] + y_min, X[i]: X[i] + x_min]) == 0:
                    x1, y1 = X[i], Y[i]
                    x2 = x1 + 32
                    try:
                        x2 += np.random.choice(np.min([256, np.where(search_map[Y[i]][X[i] + 31: ] > 0)[0][0]]), 1)[0]
                    except:
                        pass
                    y2 = y1 + 32
                    try:
                        y2 += np.random.choice(np.min([256, np.where(np.sum(search_map[y1 + 31: ,x1: x2], axis = 1) > 0)[0][0] + 64]), 1)[0]
                    except:
                        pass
                    return x1, y1, x2, y2
        return 0, 0, search_map.shape[1], search_map.shape[0]

    def fit(self, dataset_path, period_day = 1, save=False, verbose = True):
        new_dataset_path = dataset_path + '_augmented'
        self.summary(dataset_path, new_dataset_path)

        images = os.listdir(new_dataset_path + '/train/images')
        limit_second = self.maximum_process_second / len(images)
        print('process each samples limited to', limit_second, 'seconds.')
        for image_name in tqdm.tqdm(images):
            for label_name in os.listdir(new_dataset_path + '/train/labels'):
                if image_name.rstrip('.jpg') + '.txt' == label_name:
                    start_time = time.time()
                    image_path, label_path = f"{new_dataset_path}/train/images/{image_name}", f"{new_dataset_path}/train/labels/{label_name}"
                    image = cv2.resize(cv2.imread(image_path), (1440, 1080), interpolation=cv2.INTER_AREA)
                    with open(label_path, 'r') as f:
                        annots = np.array([[float(x) for x in line.strip().split()] for line in f.readlines()])
                        if len(annots) != 0:
                            classes, bboxes = annots[:, 0].astype(int), annots[:, 1:]
                            xyxy_bboxes = np.array([cxcywh_to_xyxy(box, size = 1) for box in bboxes])
                            bboxes = np.array([xyxy_to_cxcywh(box) for box in np.clip(xyxy_bboxes, 0, 1)])
                            label = np.concatenate((bboxes, np.expand_dims(classes, axis=1)), axis=1)
                        else:
                            label = []
                    search_map = self.draw_heatmap(image, annots)
                    if verbose == True:
                        fig = plt.figure(figsize=[12, 5])
                        num_of_plot = 0
                    for i in range(self.maximum_size):
                        new_image, new_label = image.copy(), list(label.copy())
                        if abs(np.random.rand(1)[0]) >= 0.0:
                            try:
                                cropped_objects, cropped_classes, cropped_bboxes = self.custom_crop()
                                if type(cropped_objects) != type(None):
                                    width, height = cropped_objects.shape[1], cropped_objects.shape[0]
                                    x1, y1, _, _ = self.startpoint(search_map, x_min = width, y_min = height)
                                    new_image[y1: y1 + height, x1: x1 + width] = cropped_objects
                                    if len(cropped_classes) != 0:
                                        cropped_bboxes[:, 0] = (cropped_bboxes[:, 0] * cropped_objects.shape[1] + x1) / new_image.shape[1]
                                        cropped_bboxes[:, 1] = (cropped_bboxes[:, 1] * cropped_objects.shape[0] + y1) / new_image.shape[0]
                                        cropped_bboxes[:, 2] = (cropped_bboxes[:, 2] * cropped_objects.shape[1] + x1) / new_image.shape[1]
                                        cropped_bboxes[:, 3] = (cropped_bboxes[:, 3] * cropped_objects.shape[0] + y1) / new_image.shape[0]
                                        cropped_bboxes = xyxy_to_cxcywh(np.clip(cropped_bboxes, 0, 1))
                                        extra_label = np.concatenate((cropped_bboxes, np.expand_dims(cropped_classes, axis=1)), axis=1)
                                        for row in extra_label:
                                            new_label.append(list(row))
                            except:
                                continue
                            

                        new_label = np.array(new_label)
                        new_label = np.array(np.unique(new_label, axis=0))
                        transformer = A.Compose(self.transform_option, bbox_params=A.BboxParams(format='yolo'))
                        transformed = transformer(image=new_image, bboxes=new_label)
                        if len(transformed['bboxes']) > 0:
                            new_image, new_classes, new_bboxes = transformed['image'], np.array(transformed['bboxes'])[:, 4], np.array(transformed['bboxes'])[:, :4]
                            new_label = np.concatenate((np.expand_dims(new_classes, axis=1).astype(int), new_bboxes), axis=1)
                            if save == True:
                                new_image_path, new_label_path = image_path.rstrip('.jpg') + f'_{i}.jpg', label_path.rstrip('.txt') + f'_{i}.txt'
                                cv2.imwrite(new_image_path, new_image)
                                with open(new_label_path, 'w') as f:
                                    for row in new_label:
                                        row = [int(row[0])] + list(row[1:])
                                        f.write(f"{' '.join(str(i) for i in row)}\n")
                                if verbose == True and num_of_plot < 15:
                                    plt.subplot(3, 5, num_of_plot + 1)
                                    plt.imshow(drawer(cv2.cvtColor(new_image, cv2.COLOR_RGB2BGR), new_bboxes))
                                    plt.tight_layout(); plt.xticks([]); plt.yticks([])
                                    num_of_plot += 1
                            if (time.time() - start_time) > limit_second:
                                break
                    if verbose == True:
                        plt.savefig(image_path.split('/train/images/')[0] + '/log/' + image_path.split('/train/images/')[-1])
                        del new_image, new_label, image, label, transformer
                        plt.clf(); plt.cla(); gc.collect()
        return new_dataset_path
        
def RoI(box1, box2):
    if min(box1[2], box2[2]) < max(box1[0], box2[0]) or min(box1[3], box2[3]) < max(box1[1], box2[1]):
        return False
    return True
    
def xyxy_to_cxcywh(x):
  x_c = (x[..., 0] + x[..., 2]) / 2
  y_c = (x[..., 1] + x[..., 3]) / 2
  w = x[..., 2] - x[..., 0]
  h = x[..., 3] - x[..., 1]
  return np.stack([x_c, y_c, w, h], axis=-1)

def cxcywh_to_xyxy(x, size):
      x_c, y_c, w, h = np.array(x)
      x1 = np.clip((x_c - w / 2) * size, 0, size)
      y1 = np.clip((y_c - h / 2) * size, 0, size)
      x2 = np.clip((x_c + w / 2) * size, 0, size)
      y2 = np.clip((y_c + h / 2) * size, 0, size)
      return x1, y1, x2, y2
      
def drawer(image, bboxes):
    for box in bboxes:
        x1, y1, x2, y2 = cxcywh_to_xyxy(box, 640)
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        x1, x2 = int(x1/ 640 * image.shape[1]), int(x2/ 640 * image.shape[1])
        y1, y2 = int(y1/ 640 * image.shape[0]), int(y2/ 640 * image.shape[0])
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 15)
    return image
