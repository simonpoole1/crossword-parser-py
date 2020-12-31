import errno
import sys
import os
import cv2 as cv

import file_utils
import img_utils
import img_proc

class Job:
    def __init__(self, img_filename):
        self.dir = file_utils.get_pkg_path()
        self.img_fullpath = img_filename
        self.img_dirname = os.path.dirname(img_filename)
        self.img_filename = os.path.basename(img_filename)
        self.job_name, _ = os.path.splitext(self.img_filename)
        self.job_dir = file_utils.get_pkg_path("output/%s" % self.job_name)

        self.dim_cropped = [ 1024, 1024 ]

        print("Job [%s]:\n dir: %s\n img: %s\n img_dir: %s\n img_file: %s\n" % \
                (self.job_name, self.dir, self.img_fullpath, self.img_dirname, self.img_filename))

        if not os.path.exists(self.img_fullpath):
            raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), self.img_fullpath)

    def run(self):
        try:
            os.mkdir(self.job_dir)
        except (FileExistsError):
            pass

        self.load_image()
        self.crop_image()
        self.detect_grid()
        self.slice_grid()
        self.print_grid()


    def load_image(self):
        self.img_raw = cv.imread(self.img_fullpath, 0)
        self.dim_raw = (self.img_raw.shape[1], self.img_raw.shape[0])
        print("Loaded image %d x %d\n" % self.dim_raw)

    def crop_image(self):
        print("Detecting and cropping crossword...")
        self.img_raw_thresh, self.img_raw_dilated, self.contours = img_proc.find_edges(self.img_raw)
        self.img_raw_contoured = cv.cvtColor(self.img_raw, cv.COLOR_GRAY2BGR)
        cv.drawContours(self.img_raw_contoured, self.contours, -1, (0, 0, 255), 3)

        img_utils.contact_sheet("%s/a1.png" % self.job_dir, 2, 2, \
            [ self.img_raw, self.img_raw_thresh, self.img_raw_dilated, self.img_raw_contoured ], \
            self.dim_raw[0], self.dim_raw[1])

        self.img_cropped, matrixTransform = img_proc.crop_and_deskew_main_shape(self.contours, \
                [ self.img_raw, self.img_raw_thresh, self.img_raw_dilated ], \
                self.dim_cropped[0], self.dim_cropped[1])

    def detect_grid(self):
        print("Detecting crossword grid...")
        raw_lines, self.rows, self.cols = img_proc.detect_lines(self.img_cropped[2])
        print("Found %d rows and %d cols" % (len(self.rows) - 1, len(self.cols) - 1))

        self.img_lines_raw = cv.cvtColor(self.img_cropped[0], cv.COLOR_GRAY2BGR)
        self.img_lines_clean = cv.cvtColor(self.img_cropped[0], cv.COLOR_GRAY2BGR)
        img_proc.paint_lines(self.img_lines_raw, raw_lines, (0,255,0), 2)
        img_proc.paint_grid(self.img_lines_clean, self.rows, self.cols, (0,255,0), (255,0,0), 2)

        img_utils.contact_sheet("%s/a2.png" % self.job_dir, 3, 2, \
                [ self.img_cropped[0], self.img_cropped[1], self.img_cropped[2], self.img_lines_raw, self.img_lines_clean ], \
                self.dim_cropped[0], self.dim_cropped[1])

    def slice_grid(self):
        print("Slicing up crossword grid...")
        self.grid = img_proc.slice_image_into_cells(self.img_cropped[0], self.rows, self.cols, "%s/cell-%%dx%%d.png" % (self.job_dir))

    def print_grid(self):
        print("\nGrid:")
        for row in self.grid:
            print("  %s" % (''.join(row)))
        print()
