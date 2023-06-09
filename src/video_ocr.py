import cv2 as cv
import imageio.v3 as iio
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


class OCRHelper:
    def __init__(self, sample_rate):
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

    def _get_frames(self, video_bytes):
        for i, frame in enumerate(iio.imiter(video_bytes, format_hint=".webm")):
            if i % self.sample_rate != 0:
                continue
            frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
            yield Frame(i, frame, i // self.sample_rate)

    def perform_video_ocr(self, video_bytes):
        logging.info(f"Starting OCR on video")
        frames = []
        with ThreadPool(multiprocessing.cpu_count()) as pool:
            frames = pool.map(OCRHelper._ocr,
                              self._filter_redundant_frames(self._get_frames(video_bytes)),
                              chunksize=multiprocessing.cpu_count())

        frames.sort(key=lambda frame: frame.frame_number)
        non_empty_frames = [frame for frame in frames if frame.text.strip()]
        all_text = '  '.join([frame.text for frame in non_empty_frames])
        clean_text = re.sub(r'\W+', ' ', all_text)  
        clean_text = re.sub(r'\s+', ' ', clean_text) 
        clean_text = clean_text.strip() 
                    
        logging.info(f"OCR completed on video")
        return clean_text