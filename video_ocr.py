import cv2 as cv
import os
import re
import scipy.fft
from itertools import tee
import numpy as np
import pytesseract
from PIL import Image
from multiprocessing.pool import ThreadPool
import multiprocessing
import tqdm
import logging


logging.basicConfig(level=logging.INFO)


class Frame:
    def __init__(self, frame_number, image, ts_second):
        self.frame_number = frame_number
        self.image = image
        self.ts_second = ts_second
        self.debug_dir = None


class OCRHelper:
    def __init__(self, filepath, sample_rate):
        self.filepath = filepath
        self.sample_rate = sample_rate

    @staticmethod
    def _phash(image, hash_size=8, highfreq_factor=4):
        img_size = hash_size * highfreq_factor
        image = cv.resize(image, (img_size, img_size), interpolation=cv.INTER_LINEAR)
        dct = scipy.fft.dct(scipy.fft.dct(image, axis=0), axis=1)
        dctlowfreq = dct[:hash_size, :hash_size]
        med = np.median(dctlowfreq)
        diff = dctlowfreq > med
        return diff

    @staticmethod
    def _pairwise(iterable):
        a, b = tee(iterable)
        next(b, None)
        return zip(a, b)

    @staticmethod
    def _are_similar_frame(f1, f2):
        diff = np.count_nonzero(OCRHelper._phash(f1.image) != OCRHelper._phash(f2.image))
        return diff <= 15

    @staticmethod
    def _filter_redundant_frames(frames):
        for f1, f2 in OCRHelper._pairwise(frames):
            if not OCRHelper._are_similar_frame(f1, f2):
                yield f1

    @staticmethod
    def _ocr(frame):
        pil_image = Image.fromarray(frame.image)
        text = pytesseract.image_to_string(pil_image)
        frame.text = text
        return frame

    def _get_frames(self, video_capture):
        fps = int(video_capture.get(cv.CAP_PROP_FPS))
        frame_number = 0
        while video_capture.isOpened():
            ret, frame = video_capture.read()
            if not ret:
                break
            frame_number += 1
            if frame_number % (fps // self.sample_rate) != 0:
                continue
            frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
            yield Frame(frame_number, frame, frame_number // fps)

    def perform_video_ocr(self):
        logging.info(f"Starting OCR on video: {self.filepath}")
        frames = []
        cap = cv.VideoCapture(self.filepath)
        with ThreadPool(multiprocessing.cpu_count()) as pool:
            frames = pool.map(OCRHelper._ocr,
                              self._filter_redundant_frames(self._get_frames(cap)),
                              chunksize=multiprocessing.cpu_count())

        frames.sort(key=lambda frame: frame.frame_number)
        non_empty_frames = [frame for frame in frames if frame.text.strip()]
        all_text = '  '.join([frame.text for frame in non_empty_frames])
        clean_text = re.sub(r'\W+', ' ', all_text)  # substitute all non-alphanumeric characters with a space
        clean_text = re.sub(r'\s+', ' ', clean_text)  # substitute all multiple whitespace with a single space
        clean_text = clean_text.strip()  # remove leading and trailing whitespace

        if self.debug_dir:
            with open(os.path.join(self.debug_dir, f"{self.filepath}.txt"), "w") as f:
                f.write(clean_text.text)
                    
        logging.info(f"OCR completed on video: {self.filepath}")
        return clean_text