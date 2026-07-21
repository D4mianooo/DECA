import os
import glob
import torch
import numpy as np
from skimage.io import imread
from skimage.transform import estimate_transform, warp
from torch.utils.data import Dataset

class CustomFaceDataset(Dataset):
    def __init__(self, dataset_dir, K, image_size, scale, trans_scale=0, isSingle=False):
        '''
        dataset_dir: ścieżka do Twojego głównego folderu (np. 'moj_dataset/')
        K: liczba zdjęć TEJ SAMEJ OSOBY ładowanych naraz
        '''
        self.dataset_dir = dataset_dir
        self.K = K
        self.image_size = image_size
        self.scale = scale #[scale_min, scale_max]
        self.trans_scale = trans_scale #[dx, dy]
        self.isSingle = isSingle

        if isSingle:
            self.K = 1

        # Szukamy wszystkich folderów (tożsamości) w Twoim datasecie
        self.identities = [d for d in glob.glob(os.path.join(dataset_dir, '*')) if os.path.isdir(d)]
        print(f"Znaleziono {len(self.identities)} różnych tożsamości do treningu.")

    def __len__(self):
        return len(self.identities)

    def __getitem__(self, idx):
        images_list = []; kpt_list = []; mask_list = []

        identity_folder = self.identities[idx]

        # Znajdź wszystkie zdjęcia (.jpg) dla danej osoby
        all_images = glob.glob(os.path.join(identity_folder, '*.jpg'))

        if len(all_images) == 0:
            raise ValueError(f"Brak zdjęć w folderze {identity_folder}")

        # Wybieramy K losowych zdjęć tej samej osoby.
        # Jeśli osoba ma mniej zdjęć niż K, losujemy ze zwracaniem (replace=True)
        selected_images = np.random.choice(all_images, self.K, replace=(len(all_images) < self.K))

        for img_path in selected_images:
            # Zakładamy nazywnictwo: plik "nazwa.jpg" ma odpowiadające "nazwa_kpt.npy" i "nazwa_mask.npy"
            base_name = os.path.splitext(img_path)[0]
            kpt_path = base_name + '_kpt.npy'
            seg_path = base_name + '_mask.npy'

            image = imread(img_path) / 255.

            # Zabezpieczenie na wypadek braku landmarków - w PoC musisz je mieć!
            if os.path.exists(kpt_path):
                kpt = np.load(kpt_path)[:, :2]
            else:
                raise FileNotFoundError(f"Brakuje pliku landmarków: {kpt_path}")

            mask = self.load_mask(seg_path, image.shape[0], image.shape[1])

            ### Zastosowanie oryginalnej logiki kadrowania twórców DECA
            tform = self.crop(image, kpt)
            cropped_image = warp(image, tform.inverse, output_shape=(self.image_size, self.image_size))
            cropped_mask = warp(mask, tform.inverse, output_shape=(self.image_size, self.image_size))
            cropped_kpt = np.dot(tform.params, np.hstack([kpt, np.ones([kpt.shape[0],1])]).T).T

            # Normalizacja landmarków do zakresu [-1, 1]
            cropped_kpt[:, :2] = cropped_kpt[:, :2] / self.image_size * 2 - 1

            images_list.append(cropped_image.transpose(2,0,1))
            kpt_list.append(cropped_kpt)
            mask_list.append(cropped_mask)

        ### Konwersja do tensorów PyTorch
        images_array = torch.from_numpy(np.array(images_list)).type(dtype = torch.float32)
        kpt_array = torch.from_numpy(np.array(kpt_list)).type(dtype = torch.float32)
        mask_array = torch.from_numpy(np.array(mask_list)).type(dtype = torch.float32)

        if self.isSingle:
            images_array = images_array.squeeze(0)
            kpt_array = kpt_array.squeeze(0)
            mask_array = mask_array.squeeze(0)

        return {
            'image': images_array,
            'landmark': kpt_array,
            'mask': mask_array
        }

    def crop(self, image, kpt):
        # [Zostaw oryginalną funkcję crop() z kodu twórców]
        left = np.min(kpt[:,0]); right = np.max(kpt[:,0]);
        top = np.min(kpt[:,1]); bottom = np.max(kpt[:,1])
        h, w, _ = image.shape
        old_size = (right - left + bottom - top)/2
        center = np.array([right - (right - left) / 2.0, bottom - (bottom - top) / 2.0 ])
        trans_scale = (np.random.rand(2)*2 -1) * self.trans_scale
        center = center + trans_scale*old_size
        scale = np.random.rand() * (self.scale[1] - self.scale[0]) + self.scale[0]
        size = int(old_size*scale)
        src_pts = np.array([[center[0]-size/2, center[1]-size/2], [center[0] - size/2, center[1]+size/2], [center[0]+size/2, center[1]-size/2]])
        DST_PTS = np.array([[0,0], [0,self.image_size - 1], [self.image_size - 1, 0]])
        tform = estimate_transform('similarity', src_pts, DST_PTS)
        return tform

    def load_mask(self, maskpath, h, w):
        # [Zostaw oryginalną funkcję load_mask() z kodu twórców]
        if os.path.isfile(maskpath):
            vis_parsing_anno = np.load(maskpath)
            mask = np.zeros_like(vis_parsing_anno)
            mask[vis_parsing_anno>0.5] = 1.
        else:
            # Jeśli maska nie istnieje, twórzy domyślną (same 1)
            mask = np.ones((h, w))
        return mask