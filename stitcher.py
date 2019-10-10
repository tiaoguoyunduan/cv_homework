import imutils
import cv2
import numpy as np
 
 
 
# stitch the images together to create a panorama
def detectAndDescribe(image):
        # convert the image to grayscale
     #   gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
        # check to see if we are using OpenCV 3.X
        
            # detect and extract features from the image
        descriptor = cv2.xfeatures2d.SIFT_create()
        (kps, features) = descriptor.detectAndCompute(image, None)
        # otherwise, we are using OpenCV 2.4.X
        kps = np.float32([kp.pt for kp in kps])
        # return a tuple of keypoints and features
        return (kps, features)
def drawMatches(imageA, imageB, kpsA, kpsB, matches, status):
        # initialize the output visualization image
        (hA, wA) = imageA.shape[:2]
        (hB, wB) = imageB.shape[:2]
        vis = np.zeros((max(hA, hB), wA + wB, 3), dtype="uint8")
        vis[0:hA, 0:wA] = imageA
        vis[0:hB, wA:] = imageB
        # loop over the matches
        for ((trainIdx, queryIdx), s) in zip(matches, status):
            # only process the match if the keypoint was successfully
            # matched
            if s == 1:
                # draw the match
                ptA = (int(kpsA[queryIdx][0]), int(kpsA[queryIdx][1]))
                ptB = (int(kpsB[trainIdx][0]) + wA, int(kpsB[trainIdx][1]))
                cv2.line(vis, ptA, ptB, (0, 255, 0), 1)
        # return the visualization
        return vis
def matchKeypoints(kpsA, kpsB, featuresA, featuresB,
        ratio, reprojThresh):
        # compute the raw matches and initialize the list of actual
        # matches
        matcher = cv2.DescriptorMatcher_create("BruteForce")
        rawMatches = matcher.knnMatch(featuresA, featuresB, 2)
        matches = []
        # loop over the raw matches
        for m in rawMatches:
            # ensure the distance is within a certain ratio of each
            # other (i.e. Lowe's ratio test)
            if len(m) == 2 and m[0].distance < m[1].distance * ratio:
                matches.append((m[0].trainIdx, m[0].queryIdx))
        # computing a homography requires at least 4 matches
        if len(matches) > 4:
            # construct the two sets of points
            ptsA = np.float32([kpsA[i] for (_, i) in matches])
            ptsB = np.float32([kpsB[i] for (i, _) in matches])
            # compute the homography between the two sets of points
            (H, status) = cv2.findHomography(ptsA, ptsB, cv2.RANSAC,
                reprojThresh)
            # return the matches along with the homograpy matrix
            # and status of each matched point
            return (matches, H, status)
        # otherwise, no homograpy could be computed
        return None
def stitch(images, ratio=0.7, reprojThresh=4.0,showMatches=False):
        # unpack the images, then detect keypoints and extract
        # local invariant descriptors from them
        (imageB, imageA) = images
        top, bot, left, right = 0, 0, 0, 0
        srcImg = cv2.copyMakeBorder(imageA, top, bot, left, right, cv2.BORDER_CONSTANT, value=(0, 0, 0))
        testImg = cv2.copyMakeBorder(imageB, top, bot, left, right, cv2.BORDER_CONSTANT, value=(0, 0, 0))
        
        img1gray = cv2.cvtColor(srcImg, cv2.COLOR_BGR2GRAY)
        img2gray = cv2.cvtColor(testImg, cv2.COLOR_BGR2GRAY)
        
        (kpsA, featuresA) = detectAndDescribe(img1gray)
        (kpsB, featuresB) = detectAndDescribe(img2gray)
        # match features between the two images
        M = matchKeypoints(kpsA, kpsB,featuresA, featuresB, ratio, reprojThresh)
        # if the match is None, then there aren't enough matched
        # keypoints to create a panorama
        if M is None:
            return None
        # otherwise, apply a perspective warp to stitch the images
        # together
        
        (matches, H, status) = M
        result = cv2.warpPerspective(srcImg, H,
            (srcImg.shape[1] + testImg.shape[1], srcImg.shape[0]))
        result[0:imageB.shape[0], 0:imageB.shape[1]] = imageB
        # check to see if the keypoint matches should be visualized
        if showMatches:
            vis = drawMatches(imageA, imageB, kpsA, kpsB, matches,
                status)
            # return a tuple of the stitched image and the
            # visualization
            return (result, vis)
        # return the stitched image
        return result
 
imageA = cv2.imread("in/pic2.jpg")
imageB = cv2.imread("in/pic1.jpg")
imageA = imutils.resize(imageA, width=5400)
imageB = imutils.resize(imageB, width=5400)
 
(result, vis) = stitch([imageA, imageB], showMatches=True)
 
cv2.imwrite("out/imageA.jpg",imageA)
cv2.imwrite("out/imageB.jpg",imageB)
cv2.imwrite("out/match.jpg",vis)
cv2.imwrite("out/result.jpg",result)