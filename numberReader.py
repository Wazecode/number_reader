import cv2 as cv
import numpy as np
import torch as t
from torch.nn.functional import normalize
from torch.utils.data import Dataset, DataLoader
import recognizer as rr

BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
IMG_DIM = (28,28)


class rect:
    def __init__(self, x, y, w, h):
        self.x = x
        self.y = y
        self.w = w
        self.h = h

    def slice(self, img):
        return img[self.y : self.y + self.h , self.x : self.x + self.w ]

    def _repr_(self):
        return str((self.x, self.y, self.w, self.h))

class NumberReader:
    def __init__(self, path):
        self.img = cv.imread(path, cv.IMREAD_GRAYSCALE)
        ret, self.img_bin = cv.threshold(self.img, 0, 255,cv.THRESH_OTSU)
        self.img_bin = cv.bitwise_not(self.img_bin)

    def imload(self, path):
        self.img = cv.imread(path, cv.IMREAD_GRAYSCALE)
        ret, self.img_bin = cv.threshold(self.img, 0, 255,cv.THRESH_OTSU)
        self.img_bin = cv.bitwise_not(self.img_bin)

    #calculates the cordinates of the parts of images with numbers
    #and stores in the varible stats
    def calc_parts(self):
        self.num_pos = []
        height, width = self.img_bin.shape

        lined_img = self.img_bin.copy()
        for i in range(height//20, height, height//20):
            str_pt = (2, i)     # Start point of the line
            end_pt = (width, i-2) # End point of the line
            cv.line(lined_img, str_pt, end_pt, WHITE, 2)
        cv.imshow('lined image',lined_img)
        cv.imwrite('lined_image.png', lined_img)

        block_output = cv.connectedComponentsWithStats(lined_img, 8, cv.CV_32S)
        (block_numlabels, block_labels, block_stats, block_centroids) = block_output


        self.blocks = []
        self.rects = []
        for i in range(1, block_numlabels):

            # calculate cordinates
            x_b = block_stats[i, cv.CC_STAT_LEFT]
            y_b = block_stats[i, cv.CC_STAT_TOP]
            w_b = block_stats[i, cv.CC_STAT_WIDTH]
            h_b = block_stats[i, cv.CC_STAT_HEIGHT]

            # skip any artifacts
            if (h_b < height//10):
                continue

            block = rect(x_b,y_b,w_b,h_b)
            self.blocks.append(block)
            #---------------------------------------------------------------------
            cv.imshow('block',block.slice(self.img_bin))
            cv.imwrite('block_image.png', block.slice(self.img_bin))
            cv.waitKey(0)
            output = cv.connectedComponentsWithStats(block.slice(self.img_bin), 8, cv.CV_32S)
            (numlabels, labels, stats, centroids) = output

            rects = []
            for i in range(1, numlabels):

                # calculate cordinates
                x = stats[i, cv.CC_STAT_LEFT]
                y = stats[i, cv.CC_STAT_TOP]
                w = stats[i, cv.CC_STAT_WIDTH]
                h = stats[i, cv.CC_STAT_HEIGHT]

                # skip any artifacts
                if (h < height/10) and (w < width/10):
                    continue

                rects.append(rect(x,y,w,h))

            #sort the coordinates from left to right
            rects.sort(key=lambda x:x.x)
            #-----------------------------------------------------------------------
            self.rects.append(rects)


    def output(self):
        self.calc_parts()
        for i in range(len(self.blocks)):
            n = []
            for rect in self.rects[i]:
                temp = rect.slice(self.blocks[i].slice(self.img_bin))
                temp = similarize(temp)
                cv.imshow('digit',temp)
                cv.waitKey(0)
                n.append(rr.rec(temp))
            print(n)

    def debug_output(self):
        cv.imshow('Original Image', self.img)
        cv.imwrite('original image.png', self.img)
        self.output()
        cv.waitKey(0)

    #         temp = self.img_bin[y : y + h , x : x + w ]

    #         temp = similarize(temp)

    #         cv.imshow('image',temp)

    #         self.num.append(rr.rec(temp))
    #         cv.waitKey(0)

    #     return self.num

    # cv.destroyAllWindows()

def similarize(img):
    # Make the image into a square
    h, w = img.shape
    if h > w:
        pad = (h-w)//2
        img = cv.copyMakeBorder(img, 0, 0, pad, pad, cv.BORDER_CONSTANT, value=BLACK)
    else:
        pad = (w-h)//2
        img = cv.copyMakeBorder(img, pad, pad, 0, 0, cv.BORDER_CONSTANT, value=BLACK)

    # Resize the image so the changes done below is constant over all the images
    img = cv.resize(img, (IMG_DIM), interpolation = cv.INTER_AREA)

    # Add padding to the image
    pad = 8
    img = cv.copyMakeBorder(img, pad, pad, pad, pad, cv.BORDER_CONSTANT, value=BLACK)

    # make the lines thicker
    kernel = np.ones((3,3), np.uint8)
    img = cv.dilate(img, kernel, iterations=1)

    # resize the image again
    img = cv.resize(img, IMG_DIM, interpolation = cv.INTER_AREA)

    return img


