#!/usr/bin/env python-real

import concurrent.futures as cf
import glob
import os
import sys

import numpy as np
import SimpleITK as sitk


def calcDID(whiteRadiographFName):
    MEAN_COMPARISON = 185
    # Read in the white radiograph
    whiteRadiograph = sitk.ReadImage(whiteRadiographFName)

    # Superpixel Segmentation
    slicImageFilter = sitk.SLICImageFilter()
    slicImageFilter.SetSuperGridSize([85, 85, 85])
    labelImage = slicImageFilter.Execute(whiteRadiograph)

    # Get the mean pixel value for each label
    labelStatsFilter = sitk.LabelStatisticsImageFilter()
    labelStatsFilter.Execute(whiteRadiograph, labelImage)
    N = labelStatsFilter.GetNumberOfLabels()
    labelMeanColors = np.zeros((N, 1))
    labelWidth, labelHeight = labelImage.GetSize()
    labels = list(labelStatsFilter.GetLabels())
    labels.sort()
    for labelIdx, labelValue in enumerate(labels):
        labelMeanColors[labelIdx, 0] = labelStatsFilter.GetMean(labelValue)

    # Create a binary label from the labelImage where all '1' are labels whose meanColor are < MEAN_COMPARISON
    labelShapeFilter = sitk.LabelShapeStatisticsImageFilter()
    labelShapeFilter.Execute(labelImage)
    binaryLabels = np.zeros((labelWidth, labelHeight))
    for labelIdx, labelValue in enumerate(labels):
        if labelValue == 0:
            continue
        if labelMeanColors[labelIdx, 0] < MEAN_COMPARISON:
            pixels = list(labelShapeFilter.GetIndexes(labelValue))
            for j in range(0, len(pixels), 2):
                y = pixels[j]
                x = pixels[j + 1]
                binaryLabels[x, y] = 1

    # Calculate the Data Intensity Density
    # Largest Region based off of https://discourse.itk.org/t/simpleitk-extract-largest-connected-component-from-binary-image/4958/2
    binaryImage = sitk.Cast(sitk.GetImageFromArray(binaryLabels), sitk.sitkUInt8)
    componentImage = sitk.ConnectedComponent(binaryImage)
    sortedComponentImage = sitk.RelabelComponent(componentImage, sortByObjectSize=True)
    largest = sortedComponentImage == 1

    return np.sum(sitk.GetArrayFromImage(largest))


def main(whiteRadiographDirName: str) -> float:
    """
    Calculates the data intensity density of the given camera on its corresponding white radiograph.
    Internal function used by :func:`optimizeCameras`.

    :param whiteRadiographFName: White radiograph file name

    return Data intensity density
    """
    whiteRadiographFiles = glob.glob(os.path.join(whiteRadiographDirName, "*.tif"))

    if not isinstance(whiteRadiographDirName, str):
        raise TypeError(f"whiteRadiographDirName must be a string, not {type(whiteRadiographDirName)}")
    if not os.path.isdir(whiteRadiographDirName):
        raise FileNotFoundError(f"Directory {whiteRadiographDirName} not found.")
    if len(whiteRadiographFiles) == 0:
        raise FileNotFoundError(f"No white radiographs found in {whiteRadiographDirName}")

    dids = []
    with cf.ThreadPoolExecutor() as executor:
        futures = [executor.submit(calcDID, wrFName) for wrFName in whiteRadiographFiles]
        for future in cf.as_completed(futures):
            dids.append(future.result())
    return np.mean(dids)


if __name__ == "__main__":
    expected_args = [
        "whiteRadiographFileName",
        # Value reported on standard output
        "DID",
    ]
    expected_args = [f"<{arg}>" for arg in expected_args]
    if len(sys.argv) < len(expected_args):
        print(f"Usage: {sys.argv[0]} {' '.join(expected_args)}")
        sys.exit(1)
    print(main(sys.argv[1]))
