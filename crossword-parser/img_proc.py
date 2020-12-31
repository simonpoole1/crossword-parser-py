#!/usr/bin/python3
import re
import numpy as np
import cv2 as cv
import pytesseract

def find_edges(img):
    img_thresh = cv.GaussianBlur(img, (9, 9), 0)
    #img_thresh = cv.medianBlur(imgRaw, 5)

    #ret, thresh = cv.threshold(img_thresh, 127, 255, cv.THRESH_BINARY_INV)
    img_thresh = cv.adaptiveThreshold(img_thresh, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, 11, 2)
    img_thresh = cv.bitwise_not(img_thresh)

    dilate_kernel = np.array([[0., 1., 0.], [1., 1., 1.], [0., 1., 0.]], np.uint8)
    img_dilated = cv.dilate(img_thresh, dilate_kernel)

    contours, hierarchy = cv.findContours(img_dilated, cv.RETR_EXTERNAL, 1)

    return img_thresh, img_dilated, contours;

def crop_and_deskew_main_shape(contours, img_list, target_w, target_h):
    max_area = -1
    for cnt in contours:
        approx = cv.approxPolyDP(cnt, 0.02 * cv.arcLength(cnt, True), True)
        if len(approx) == 4:
            if cv.contourArea(cnt) > max_area:
                max_area = cv.contourArea(cnt)
                max_cnt = cnt
                max_approx = approx

    # Cut out and straighten / deskew
    new_pts = np.float32([[0,0], [0,target_h-1], [target_w-1,target_h-1], [target_w-1,0]])
    old_pts = max_approx.reshape(4,2).astype('float32')
    matrix_transform = cv.getPerspectiveTransform(old_pts,new_pts)

    img_out = []
    for img in img_list:
        img2 = cv.warpPerspective(img, matrix_transform, (target_w,target_h))
        img_out.append(img2)

    return img_out, matrix_transform

def detect_lines(img):
    lines = cv.HoughLinesP(img, 1, np.pi/180, 100, 200, 50)

    rows = []
    cols = []
    if lines is not None:
        for i in range(0, len(lines)):
            l = lines[i][0]
            # Cope with slightly wonky lines
            if abs(l[0]-l[2]) < 5:
                cols.append(l[0])
            if abs(l[1]-l[3]) < 5:
                rows.append(l[1])

    rows = dedupe_lines(rows, 20)
    cols = dedupe_lines(cols, 20)

    return lines, rows, cols

def paint_lines(img, lines, colour, thickness):
    for line in lines:
        l1 = (line[0][0], line[0][1])
        l2 = (line[0][2], line[0][3])
        cv.line(img, l1, l2, colour, thickness, cv.LINE_AA)

def paint_grid(img, rows, cols, row_colour, col_colour, thickness):
    h, w = img.shape[:2]
    for x in cols:
        cv.line(img, (x, 0), (x, h), row_colour, thickness)
    for y in rows:
        cv.line(img, (0, y), (w, y), col_colour, thickness)

def dedupe_lines(lines, threshold):
    result = []
    prev = -100
    curr_min = -100
    prev_result = -100
    gaps = []
    for i in sorted(lines):
        if i > prev + threshold:
            if curr_min >= 0:
                pos = int((curr_min + prev) / 2)
                result.append(pos)
                if prev_result >= 0:
                    gaps.append(pos - prev_result)
                prev_result = pos

            curr_min = i
        prev = i
    pos = int((curr_min + prev) / 2)
    result.append(pos)
    if prev_result >= 0:
        gaps.append(pos - prev_result)

    return result

def slice_image_into_cells(img, rows, cols, filename_template):
    img = cv.adaptiveThreshold(img, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, 201, 2)
    prev_x = -1
    prev_y = -1
    index_x = 0
    index_y = 0

    grid = []

    for y in rows:
        grid_row = []
        index_x = 0
        for x in cols:
            if x < prev_x:
                prev_x = -1
#            print(" processing cell %d, %d" % (index_x, index_y))
            print('∙ ', end='', flush=True)
            if prev_x >= 0 and prev_y >= 0:
                snippet = img[prev_y:y+1, prev_x:x+1]

                filename = filename_template % (index_x, index_y)
                
                if is_white_cell(snippet):
                    snippet, content = parse_cell_content(snippet)
                    if content is not None:
#                        print("  - cell content [%s]" % content)
                        grid_row.append("%-2d" % int(content))
                    else:
                        grid_row.append('∙ ')
                else:
                    grid_row.append('# ')
                cv.imwrite(filename, snippet)

            index_x += 1
            prev_x = x
        if prev_y >= 0:
            grid.append(grid_row)
        prev_y = y
        index_y += 1
        print()

    return grid

def white_pixel_proportion(img):
    white_count = cv.countNonZero(img)
    w, h = img.shape[:2]
    img_size = w * h
    return white_count / img_size

def is_white_cell(img):
    return white_pixel_proportion(img) > 0.5

def parse_cell_content(img):
    img_not = cv.bitwise_not(img)
    contours, hierarchy = cv.findContours(img_not, cv.RETR_CCOMP, cv.CHAIN_APPROX_SIMPLE)

    img_w, img_h = img.shape[:2]
    character_contours = []

    min_x = 999999
    min_y = 999999
    max_x = -1
    max_y = -1
    for i, h in enumerate(hierarchy[0]):
        cnt = contours[i]
        if h[3] > -1:
            continue

        x,y,w,h = cv.boundingRect(cnt)
        if w / img_w > 0.9 or h / img_h > 0.9:
            continue
        if h < 10:
            continue
        if h / w > 15:
            continue
        if w < 3:
            continue

        white_pixel_ratio = white_pixel_proportion(img_not[y:y+h, x:x+w])
        if white_pixel_ratio < 0.05 or white_pixel_ratio > 0.95:
            continue

        if x < min_x:
            min_x = x
        if y < min_y:
            min_y = y
        if (x + w) > max_x:
            max_x = x + w
        if (y + h) > max_y:
            max_y = y + h

        character_contours.append(cnt)

    if max_x < 0:
        return img, None

    pad_y = int((max_y - min_y) / 2)
    pad_x = int(pad_y * 1.2)
    cropped_img = cv.bitwise_not(cv.copyMakeBorder(img_not[min_y:max_y, min_x:max_x], pad_y, pad_y, pad_x, pad_x, cv.BORDER_CONSTANT))

    # Use OCR to read the number
    content = pytesseract.image_to_string(cropped_img, config='--psm 13 --dpi 100 digits')
    content = re.sub("[^0-9]", "", content)
    if len(content) == 0:
        content = None

    return cropped_img, content
