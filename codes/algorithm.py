from collections import defaultdict
import cv2
import matplotlib.pyplot as plt
import numpy as np
from skimage.segmentation import slic
from skimage.segmentation import felzenszwalb
from skimage.segmentation import mark_boundaries
from skimage.segmentation import quickshift
import numpy as np

def average_each_segment(image, labels):
    """ Average each segment of the image
    :param image: the image to segment 3 x H x W
    :param labels: the labels from the superpixel segmentation
    :return: a 3D array of the averaged image
    """
    segments = np.zeros_like(image)
    for i in range(np.max(labels) + 1):
        if(len(image[labels == i]) == 0):
            continue
        segments[labels == i] = np.mean(image[labels == i], axis=0)
    return segments

def GHS(image, dsu_threshold, n_segments):
    def average_each_segment(image, labels):
        """ Average each segment of the image
        :param image: the image to segment 3 x H x W
        :param labels: the labels from the superpixel segmentation
        :return: a 3D array of the averaged image
        """
        segments = np.zeros_like(image)
        for i in range(np.max(labels) + 1):
            if(len(image[labels == i]) == 0):
                continue
            segments[labels == i] = np.mean(image[labels == i], axis=0)
        return segments

    segments = slic(image, n_segments=n_segments, compactness=10, sigma=1)

    segment_dict = {}
    for i in range(np.max(segments) + 1):
        if len(image[segments == i]) == 0:
            continue
        segment_dict[i] = {'rgb':np.mean(image[segments == i], axis=0), 'xy':np.mean(np.where(segments == i), axis=1), 'size':len(image[segments == i])}
    
    neighbours = defaultdict(list)


    n, m = segments.shape
    for i in range(n):
        for j in range(m):
            if i > 0:
                if segments[i, j] != segments[i-1, j]:
                    neighbours[segments[i, j]].append((segments[i-1, j], np.linalg.norm(segment_dict[segments[i, j]]['rgb'] - segment_dict[segments[i-1, j]]['rgb'])))
            if i < n-1:
                if segments[i, j] != segments[i+1, j]:
                    neighbours[segments[i, j]].append((segments[i+1, j], np.linalg.norm(segment_dict[segments[i, j]]['rgb'] - segment_dict[segments[i+1, j]]['rgb'])))
            if j > 0:
                if segments[i, j] != segments[i, j-1]:
                    neighbours[segments[i, j]].append((segments[i, j-1], np.linalg.norm(segment_dict[segments[i, j]]['rgb'] - segment_dict[segments[i, j-1]]['rgb'])))
            if j < m-1:
                if segments[i, j] != segments[i, j+1]:
                    neighbours[segments[i, j]].append((segments[i, j+1], np.linalg.norm(segment_dict[segments[i, j]]['rgb'] - segment_dict[segments[i, j+1]]['rgb'])))
    
    edges = []
    for i in neighbours:
        for j in neighbours[i]:
            edges.append((i, j[0], j[1]))
        
    #sort by weight
    edges = sorted(edges, key=lambda x: x[2])

    
    segment_sets = []
    for i in range(1, np.max(segments) + 1):
        segment_sets.append(set([i]))

    id_to_set = {}
    for i in range(1, np.max(segments) + 1):
        id_to_set[i] = i - 1

    for edge in edges:
        if edge[2] > dsu_threshold:
            break
        if id_to_set[edge[0]] != id_to_set[edge[1]]:
            temp = id_to_set[edge[1]]
            segment_sets[id_to_set[edge[0]]] = segment_sets[id_to_set[edge[0]]] | segment_sets[id_to_set[edge[1]]]
            for i in segment_sets[temp]:
                id_to_set[i] = id_to_set[edge[0]]
            segment_sets[temp] = set()

    #remove empty sets
    segment_sets = [i for i in segment_sets if len(i) != 0]

    segment_to_set = {}
    for i in range(len(segment_sets)):
        for j in segment_sets[i]:
            segment_to_set[j] = i + 1

    segments_copy = segments.copy()
    for i in range(n):
        for j in range(m):
            segments_copy[i, j] = segment_to_set[segments[i, j]]

    out_final = average_each_segment(image, segments_copy)
    return out_final, segments_copy



