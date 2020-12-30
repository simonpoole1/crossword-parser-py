#!/usr/bin/python3

import numpy as np
import cv2 as cv

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

    # Cut the crossword region without deskewing
    #xx, yy, ww, hh = cv.boundingRect(max_cnt)
    #imgCroppedDilated = thresh2[yy:yy+hh, xx:xx+ww]

    # Cut out and straighten / deskew
    new_pts = np.float32([[0,0], [0,target_h-1], [target_w-1,target_h-1], [target_w-1,0]])
    old_pts = max_approx.reshape(4,2).astype('float32')
    matrix_transform = cv.getPerspectiveTransform(old_pts,new_pts)

    img_out = []
    for img in img_list:
        img_out.append(cv.warpPerspective(img, matrix_transform, (target_w,target_h)))

    return img_out, matrix_transform

def detect_lines(img):
    lines = cv.HoughLinesP(img, 1, np.pi/180, 100, 200, 50)
    #paintLines(imgCroppedDilated, lines)

    rows = []
    cols = []
    if lines is not None:
#        with open("contours/lines.csv", "w") as csv_file:
        for i in range(0, len(lines)):
            l = lines[i][0]
    #for x1,y1,x2,y2 in lines[0]:
#                cv.line(imgLinesRaw,(l[0],l[1]),(l[2],l[3]),(0,255,0),2)
#            csv_file.write("%d,%d,%d,%d\n" % (l[0], l[1], l[2], l[3]))
            # Cope with slightly wonky lines
            if abs(l[0]-l[2]) < 5:
                rows.append(l[0])
            if abs(l[1]-l[3]) < 5:
                cols.append(l[1])

    rows = dedupe_lines(rows, 20)
    cols = dedupe_lines(cols, 20)

    return lines, rows, cols

def paint_lines(img, lines, colour, thickness):
    for line in lines:
        l1 = (line[0][0], line[0][1])
        l2 = (line[0][2], line[0][3])
#        cv.line(img, line[0][:2], line[0][2:4], colour, thickness, cv.LINE_AA)
        cv.line(img, l1, l2, colour, thickness, cv.LINE_AA)

def paint_grid(img, rows, cols, row_colour, col_colour, thickness):
    for x in rows:
        cv.line(img, (x, 0), (x, 1023), row_colour, thickness)
    for y in cols:
        cv.line(img, (0, y), (1023, y), col_colour, thickness)

def dedupe_lines(lines, threshold):
    result = []
    prev = -100
    curr_min = -100
    prev_result = -100
    gaps = []
#    with open(filename, "w") as csv_file:
    for i in sorted(lines):
        if i > prev + threshold:
            if curr_min >= 0:
                pos = int((curr_min + prev) / 2)
                result.append(pos)
                if prev_result >= 0:
                    gaps.append(pos - prev_result)
#                    csv_file.write("%d\n" % pos)
                prev_result = pos

            curr_min = i
        prev = i
    pos = int((curr_min + prev) / 2)
    result.append(pos)
    if prev_result >= 0:
        gaps.append(pos - prev_result)
#        csv_file.write("%d\n" % pos)


    return result

def slice_image_into_cells(img, cols, rows, filename_template):
    img = cv.adaptiveThreshold(img, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, 201, 2)
    prev_x = -1
    prev_y = -1
    index_x = 0
    index_y = 0

    grid = []

    for y in rows:
        grid_row = []
        for x in cols:
            if x < prev_x:
                prev_x = -1
    #        print("%s,%s, %s,%s " % (prev_x, prev_y, x, y))
            if prev_x >= 0 and prev_y >= 0:
                snippet = img[prev_x:x+1, prev_y:y+1]
                white_count = cv.countNonZero(snippet)
                snippet_size = (x - prev_x + 1) * (y - prev_y + 1)
                is_black = (snippet_size - white_count > white_count)
                if is_black:
                    grid_row.append('#')
                else:
                    grid_row.append('∙')
#                snippet = cv.adaptiveThreshold(snippet, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, 201, 2)
                cv.imwrite(filename_template % (index_x, index_y), snippet)

            index_x += 1
            prev_x = x
        if prev_y >= 0:
            grid.append(grid_row)
        prev_y = y
        index_y += 1

    return grid


