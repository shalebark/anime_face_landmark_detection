import numpy as np
import torch
from torchvision import transforms
from PIL import Image
from CFA import CFA

from pathlib import Path
default_checkpoint_path = str(Path(__file__).resolve().parent / 'model/checkpoint_landmark_191116.pth.tar')

class LandmarkDetector():

    def __init__(self, checkpoint_path=default_checkpoint_path):
        # params
        self._num_landmarks = 24
        self._landmark_detector = CFA(output_channel_num=self._num_landmarks + 1, checkpoint_name=checkpoint_path).cuda()

        # transform
        normalize   = transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                        std=[0.5, 0.5, 0.5])
        train_transform = [transforms.ToTensor(), normalize]
        self._transformer = transforms.Compose(train_transform)

    def detect_landmarks(self, image, facebox):
        """
            Returns facial landmarks. The landmarks will be in the form of a dictionary. dtype=np.int32
                'face-box': (x,y,w,h)
                'face-right-pos': (x,y)
                'chin-pos': (x,y)
                'face-left-pos': (x,y)
                'right-brow-right-pos': (x,y)
                'right-brow-middle-pos': (x,y)
                'right-brow-left-pos': (x,y)
                'left-brow-right-pos': (x,y)
                'left-brow-middle-pos': (x,y)
                'left-brow-left-pos': (x,y)
                'nose-pos': (x,y)
                'right-eye-top-right-pos': (x,y)
                'right-eye-top-middle-pos': (x,y)
                'right-eye-top-left-pos': (x,y)
                'right-eye-bottom-pos': (x,y)
                'right-eye-center-pos': (x,y)
                'left-eye-top-right-pos': (x,y)
                'left-eye-top-middle-pos': (x,y)
                'left-eye-top-left-pos': (x,y)
                'left-eye-bottom-pos': (x,y)
                'left-eye-center-pos': (x,y)
                'mouth-top-right-pos': (x,y)
                'mouth-top-middle-pos': (x,y)
                'mouth-top-left-pos': (x,y)
                'mouth-bottom-pos': (x,y)
                'eye-center-pos': (x,y)
                'mouth-center-pos': (x,y)

            Parameters:
                image: numpy rgb array, dtype=np.uint8
                facebox: where the face is located in the image, tuple (x,y,w,h), dtype=np.int32
        """
        img = image.copy()

        x_, y_, w_, h_ = facebox

        # expand the face selection of the image (horizontally by 1/8 of original on both sides), (vertically, 1/4 of original going up)
        x = int(max(x_- w_ / 8, 0))
        rx = min(x_ + w_ * 9 / 8, img.shape[1])
        y = int(max(y_ - h_ / 4, 0))
        by = y_ + h_
        w = int(rx - x)
        h = int(by - y)

        # set image width (this should be the same size as the expected input data, which is 128x128x3)
        img_width = 128

        # crop and resize image
        cropped_img = img[y:y+h, x:x+w]
        facial_img = np.asarray(Image.fromarray(cropped_img).resize((img_width, img_width), Image.BICUBIC))
        # facial_img = cv2.resize(cropped_img, (img_width, img_width), interpolation = cv2.INTER_CUBIC)

        # normalize and convert to tensors
        process_img = self._transformer(facial_img.copy())
        process_img = process_img.unsqueeze(0).cuda()

        # get landmark classification heatmap
        heatmaps = self._landmark_detector(process_img)
        heatmaps = heatmaps[-1].cpu().detach().numpy()[0]

        landmarks = []
        # calculate landmark position
        for i in range(self._num_landmarks):
            heatmaps_tmp = np.asarray(Image.fromarray(heatmaps[i]).resize((img_width, img_width), Image.BICUBIC))
            # heatmaps_tmp = cv2.resize(heatmaps[i], (img_width, img_width), interpolation=cv2.INTER_CUBIC)
            landmark = np.unravel_index(np.argmax(heatmaps_tmp), heatmaps_tmp.shape)
            landmark_y = int(landmark[0] * h / img_width)
            landmark_x = int(landmark[1] * w / img_width)

            landmark = (x + landmark_x, y + landmark_y)
            landmarks.append(landmark)

        named_landmarks = {
            'face-box': facebox,
            'face-right-pos': landmarks[0],
            'chin-pos': landmarks[1],
            'face-left-pos': landmarks[2],
            'right-brow-right-pos': landmarks[3],
            'right-brow-middle-pos': landmarks[4],
            'right-brow-left-pos': landmarks[5],
            'left-brow-right-pos': landmarks[6],
            'left-brow-middle-pos': landmarks[7],
            'left-brow-left-pos': landmarks[8],
            'nose-pos': landmarks[9],
            'right-eye-top-right-pos': landmarks[10],
            'right-eye-top-middle-pos': landmarks[11],
            'right-eye-top-left-pos': landmarks[12],
            'right-eye-bottom-pos': landmarks[13],
            'right-eye-center-pos': landmarks[14],
            'left-eye-top-right-pos': landmarks[15],
            'left-eye-top-middle-pos': landmarks[16],
            'left-eye-top-left-pos': landmarks[17],
            'left-eye-bottom-pos': landmarks[18],
            'left-eye-center-pos': landmarks[19],
            'mouth-top-right-pos': landmarks[20],
            'mouth-top-middle-pos': landmarks[21],
            'mouth-top-left-pos': landmarks[22],
            'mouth-bottom-pos': landmarks[23],
        }

        return self._process_landmarks(image, named_landmarks)

    def _process_landmarks(self, image, landmarks):
        eye_center_pos = tuple(np.average([landmarks['left-eye-center-pos'], landmarks['right-eye-center-pos']], axis=0).astype(np.int32))

        mouth_center_pos = tuple(np.average([
            landmarks['mouth-top-right-pos'],
            landmarks['mouth-top-middle-pos'],
            landmarks['mouth-top-left-pos'],
            landmarks['mouth-bottom-pos'],
        ], axis=0).astype(np.int32))

        landmarks.update({
            'eye-center-pos': eye_center_pos,
            'mouth-center-pos': mouth_center_pos,
        })

        return landmarks


