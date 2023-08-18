#!/usr/bin/env python-real

import os
import sys


def main(whiteRadiographFName: str) -> float:
    """
    Calculates the data intensity density of the given camera on its corresponding white radiograph.
    Internal function used by :func:`optimizeCameras`.

    :param whiteRadiographFName: White radiograph file name
    :type whiteRadiographFName: str

    return Data intensity density
    :rtype: float
    """

    import numpy as np
    import SimpleITK as sitk

    MEAN_COMPARISON = 170  # 255 / 3 * 2

    # Read in the white radiograph
    if not isinstance(whiteRadiographFName, str):
        raise TypeError(f"whiteRadiographFName must be a string, not {type(whiteRadiographFName)}")
    if not os.path.isfile(whiteRadiographFName):
        raise FileNotFoundError(f"File {whiteRadiographFName} not found.")
    whiteRadiograph = sitk.ReadImage(whiteRadiographFName)

    # Superpixel Segmentation
    slicImageFilter = sitk.SLICImageFilter()
    slicImageFilter.SetSuperGridSize([15, 15, 15])  # smaller grid size = finer grid overall default is [50,50,50]
    labelImage = slicImageFilter.Execute(whiteRadiograph)

    # Get the mean pixel value for each label
    labelStatsFilter = sitk.LabelStatisticsImageFilter()
    labelStatsFilter.Execute(whiteRadiograph, labelImage)
    N = labelStatsFilter.GetNumberOfLabels()
    meanColor = np.zeros((N, 1))
    m, n = labelImage.GetSize()
    labels = list(labelStatsFilter.GetLabels())
    labels.sort()
    for i, label in enumerate(labels):
        meanColor[i, 0] = labelStatsFilter.GetMean(label)

    # Create a binary label from the labelImage where all '1' are labels whose meanColor are < MEAN_COMPARISON
    labelShapeFilter = sitk.LabelShapeStatisticsImageFilter()
    labelShapeFilter.Execute(labelImage)
    binaryLabels = np.zeros((m, n))
    for i, label in enumerate(labels):
        if label == 0:
            continue
        if meanColor[i, 0] < MEAN_COMPARISON:
            pixels = list(labelShapeFilter.GetIndexes(label))
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


if __name__ == "__main__":
    EXPECTED_ARGS = 2
    if len(sys.argv) < EXPECTED_ARGS:
        print("Usage: calculateDataIntensityDensity <input FName>")
        sys.exit(1)
    print(main(sys.argv[1]))
