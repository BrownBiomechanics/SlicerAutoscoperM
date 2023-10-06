#!/usr/bin/env python-real

import glob
import os
import sys


def main(whiteRadiographDirName: str) -> float:
    """
    Calculates the data intensity density of the given camera on its corresponding white radiograph.
    Internal function used by :func:`optimizeCameras`.

    :param whiteRadiographFName: White radiograph file name

    return Data intensity density
    """

    import numpy as np
    import SimpleITK as sitk

    MEAN_COMPARISON = 185

    whiteRadiographFiles = glob.glob(os.path.join(whiteRadiographDirName, "*.tif"))

    if not isinstance(whiteRadiographDirName, str):
        raise TypeError(f"whiteRadiographDirName must be a string, not {type(whiteRadiographDirName)}")
    if not os.path.isdir(whiteRadiographDirName):
        raise FileNotFoundError(f"Directory {whiteRadiographDirName} not found.")
    if len(whiteRadiographFiles) == 0:
        raise FileNotFoundError(f"No white radiographs found in {whiteRadiographDirName}")

    dids = []
    for whiteRadiographFName in whiteRadiographFiles:
        # Read in the white radiograph
        whiteRadiograph = sitk.ReadImage(whiteRadiographFName)

        # Superpixel Segmentation
        slicImageFilter = sitk.SLICImageFilter()
        slicImageFilter.SetSuperGridSize([15, 15, 15])  # smaller grid size = finer grid overall default is [50,50,50]
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

        dids.append(np.sum(sitk.GetArrayFromImage(largest)))

    # Calculate the average DID and check for any statistical outliers using interquartile range
    dids = np.array(dids)
    dids.sort()
    q1, q3 = np.percentile(dids, [25, 75])
    iqr = q3 - q1
    lower_bound = q1 - (1.5 * iqr)
    upper_bound = q3 + (1.5 * iqr)
    # flag any outliers by saving the indices of the outliers
    outliers = np.where((dids < lower_bound) | (dids > upper_bound))
    return np.mean(dids), outliers


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
    print(main(sys.argv[1])[0])
