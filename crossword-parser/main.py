#!/usr/bin/python3

import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt

def paintLines(img, lines):
    for line in lines:
        rho,theta = line[0]
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a*rho
        y0 = b*rho
        x1 = int(x0 + 1000*(-b))
        y1 = int(y0 + 1000*(a))
        x2 = int(x0 - 1000*(-b))
        y2 = int(y0 - 1000*(a))
        cv.line(img,(x1,y1),(x2,y2),(0,0,255),2)

def dedupeLines(lines, threshold, expect_gap, filename):
    result = []
    prev = -100
    currMin = -100
    prevResult = -100
    gaps = []
    with open(filename, "w") as csv_file:
        for i in sorted(lines):
            if i > prev + threshold:
                if currMin >= 0:
                    pos = int((currMin + prev) / 2)
                    result.append(pos)
                    if prevResult >= 0:
                        gaps.append(pos - prevResult)
                    csv_file.write("%d\n" % pos)
                    prevResult = pos

                currMin = i
            prev = i
        pos = int((currMin + prev) / 2)
        result.append(pos)
        if prevResult >= 0:
            gaps.append(pos - prevResult)
        csv_file.write("%d\n" % pos)


    return result

#filename = 'simple-crossword1'
#ext = 'jpg'
filename = "crossword5"
ext = "webp"
imgRaw = cv.imread('%s.%s' % (filename, ext), 0)
h, w = imgRaw.shape[:2]
#imgRaw = cv.cvtColor(imgRaw, cv.COLOR_BGR2GRAY)

imgThresholded = cv.GaussianBlur(imgRaw, (9, 9), 0)
#imgThresholded = cv.medianBlur(imgRaw, 5)
#ret, thresh = cv.threshold(imgThresholded, 127, 255, cv.THRESH_BINARY_INV)
imgThresholded = cv.adaptiveThreshold(imgThresholded, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, 11, 2)
imgThresholded = cv.bitwise_not(imgThresholded)

dilate_kernel = np.array([[0., 1., 0.], [1., 1., 1.], [0., 1., 0.]], np.uint8)
imgDilated = cv.dilate(imgThresholded, dilate_kernel)

plt.figure(figsize=(w * 2/100, h * 2/100), dpi=100)
plt.subplot(221)
plt.imshow(imgRaw, 'gray')
plt.subplot(222)
plt.imshow(imgThresholded, 'gray')
plt.subplot(223)
plt.imshow(imgDilated, 'gray')

contours, hierarchy = cv.findContours(imgDilated, cv.RETR_EXTERNAL, 1)

imgRawC = cv.cvtColor(imgRaw, cv.COLOR_GRAY2BGR)
cv.drawContours(imgRawC, contours, -1, (0, 0, 255), 3)
plt.subplot(224)
plt.imshow(imgRawC)
plt.show()
plt.savefig("contours/%s.png" % filename, dpi=100)

# Find the biggest contour - bounding box of the crossword
max_area = -1
for cnt in contours:
    approx = cv.approxPolyDP(cnt, 0.02 * cv.arcLength(cnt, True), True)
    if len(approx) == 4:
        if cv.contourArea(cnt) > max_area:
            max_area = cv.contourArea(cnt)
            max_cnt = cnt
            max_approx = approx

# Cut the crossword region
#xx, yy, ww, hh = cv.boundingRect(max_cnt)
#imgCroppedDilated = thresh2[yy:yy+hh, xx:xx+ww]

# you need to uncomment these lines if your image is rotated
new_pts = np.float32([[0,0], [0,1023],[1023,1023],[1023,0]])
old_pts = max_approx.reshape(4,2).astype('float32')
M = cv.getPerspectiveTransform(old_pts,new_pts)
imgCroppedDilated = cv.warpPerspective(imgDilated,M,(1024,1024))
imgCroppedThresholded = cv.warpPerspective(imgThresholded,M,(1024,1024))
imgCroppedRaw = cv.warpPerspective(imgRaw,M,(1024,1024))

plt.figure(figsize=(1024 * 3/100, 1024 * 2/100), dpi=100)
plt.subplot(231)
plt.imshow(imgCroppedRaw, 'gray')
plt.subplot(232)
plt.imshow(imgCroppedThresholded, 'gray')
plt.subplot(233)
plt.imshow(imgCroppedDilated, 'gray')

#imgCroppedDilated = cv.bitwise_not(imgCroppedDilated)

lines = cv.HoughLinesP(imgCroppedDilated, 1, np.pi/180, 100, 200, 50)
#paintLines(imgCroppedDilated, lines)

imgLinesRaw = cv.cvtColor(imgCroppedRaw, cv.COLOR_GRAY2BGR)
rows = []
cols = []
if lines is not None:
    with open("contours/lines.csv", "w") as csv_file:
        for i in range(0, len(lines)):
            l = lines[i][0]
    #for x1,y1,x2,y2 in lines[0]:
            cv.line(imgLinesRaw,(l[0],l[1]),(l[2],l[3]),(0,255,0),2)
            csv_file.write("%d,%d,%d,%d\n" % (l[0], l[1], l[2], l[3]))
            # Cope with slightly wonky lines
            if abs(l[0]-l[2]) < 5:
                rows.append(l[0])
            if abs(l[1]-l[3]) < 5:
                cols.append(l[1])

plt.subplot(234)
plt.imshow(imgLinesRaw)

rows2 = dedupeLines(rows, 3, 1, "contours/rows.csv")
cols2 = dedupeLines(cols, 3, 1, "contours/cols.csv")

expect_h = 1024 / (len(rows2) - 1)
expect_w = 1024 / (len(cols2) - 1)

rows = dedupeLines(rows, 20, expect_h, "contours/rows.csv")
cols = dedupeLines(cols, 20, expect_w, "contours/cols.csv")
print("Found %d rows and %d cols" % (len(rows) - 1, len(cols) - 1))

imgLinesClean = cv.cvtColor(imgCroppedRaw, cv.COLOR_GRAY2BGR)
for x in rows:
    cv.line(imgLinesClean, (x, 0), (x, 1023), (0,255,0), 2)
for y in cols:
    cv.line(imgLinesClean, (0, y), (1023, y), (255,0,0), 2)
plt.subplot(235)
plt.imshow(imgLinesClean)

plt.show()
plt.savefig("contours/%s-cropped.png" % filename, dpi=100)

prev_x = -1
prev_y = -1
index_x = 0
index_y = 0
for y in cols:
    if y < prev_y:
        prev_y = -1
    for x in rows:
        if x < prev_x:
            prev_x = -1
#        print("%s,%s, %s,%s " % (prev_x, prev_y, x, y))
        if prev_x >= 0 and prev_y >= 0:
            snippet = imgCroppedDilated[prev_x+1:x, prev_y+1:y]
            cv.imwrite("cells/%s-cell-%dx%d.png" % (filename, x, y), snippet)
        index_x += 1
        prev_x = x
    prev_y = y
    index_y += 1

#        snippet = imgCroppedDilated
#cv.imwrite("contours/%s-cropped.png" % filename, imgCroppedDilated)


