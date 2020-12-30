
from matplotlib import pyplot as plt

def contact_sheet(filename, sheetCols, sheetRows, imgArray, imgW, imgH):
    plt.figure(figsize=(imgW * sheetCols / 100, imgH * sheetRows / 100), dpi=100)
    i = 0
    for img in imgArray:
        i += 1
        plt.subplot(sheetRows, sheetCols, i)
        plt.imshow(img)

    plt.show()
    plt.savefig(filename, dpi=100)
