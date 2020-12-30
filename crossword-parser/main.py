#!/usr/bin/python3

import sys
import os
import argparse

import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt

from job import Job
import file_utils

parser = argparse.ArgumentParser(description="Process crossword images")
parser.add_argument('files', metavar='path', nargs='+',
        help='A crossword image to process')
args = parser.parse_args()

for filename in args.files:
#    sample_filename = file_utils.get_pkg_path("sample-images/crossword5.webp")
    job = Job(filename)
    job.run()

#job = Job('simple-crossword1.jpg')

