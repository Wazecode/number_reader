img = cv.imread('numbers.jpg', cv.IMREAD_GRAYSCALE)

ret, img_bin = cv.threshold(img, 0, 255,cv.THRESH_OTSU)
img_bin = cv.bitwise_not(img_bin)
pcs = cv.connectedComponentsWithStats(img_bin, 8, cv.CV_32S)

BLUE = (255, 255, 255)

(totalLabels, label_ids, values, centroid) = pcs

cv.imshow('tehuno',label_ids)
cv.waitKey(0)

for i in range(1, totalLabels):
    x = values[i, cv.CC_STAT_LEFT]
    y = values[i, cv.CC_STAT_TOP]
    w = values[i, cv.CC_STAT_WIDTH]
    h = values[i, cv.CC_STAT_HEIGHT]

    pt1 = (x,y)
    pt2 = (x+ w, y+ h)

    img_tem = img_bin.copy()
    cv.rectangle(img_tem, pt1, pt2, BLUE, 4)
    cv.imshow('hethou',img_tem)
    cv.waitKey(0)
print(pcs)

cv.rectangle(img_bin, (12,33), (99,44), BLUE, 4)
cv.imshow('holle', img_bin)
cv.waitKey(0)
cv.destroyAllWindows()
