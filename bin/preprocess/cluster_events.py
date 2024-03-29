#!/usr/bin/env python
# -------------------------------------------------------------------
# File Name : cluster_events
# Creation Date : 05-12-2016
# Last Modified : Fri Dec  9 12:26:58 2016
# Author: Thibaut Perol <tperol@g.harvard.edu>
# -------------------------------------------------------------------
"""Clustering of events from a catalog.

We want to be able to predict the origin of an earthquake from its trace only.
This script labels events using an off-the-shelf clustering algorithm on their
geographic coordinates.

The classification we will solve seek to retrieve this label from the traces.

e.g.,
./bin/preprocess/cluster_events --src data/catalogs/OK_2014-2015-2016.csv\
--dst data/6_clusters --n_components 6 --model KMeans
"""

import collections
import os
import sys

import gflags
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from obspy.core.utcdatetime import UTCDateTime
from sklearn.cluster import KMeans, DBSCAN
from sklearn.mixture import GaussianMixture

from openquake.hazardlib.geo.geodetic import distance
from quakenet import data_io
from quakenet.data_io import write_catalog_with_clusters

gflags.DEFINE_string(
    'src', None, 'path to the events catalog to use as labels.')
gflags.DEFINE_string('dst', None, 'path to the output catalog with cluster id')
gflags.MarkFlagAsRequired('src')
gflags.DEFINE_string('model', 'KMeans', 'DBSCAN or KMeans')
gflags.DEFINE_integer('n_components', 4, 'number of mixture component')
FLAGS = gflags.FLAGS

OK029_LAT = 35.796570
OK029_LONG = -97.454860
OK027_LAT = 35.72
OK027_LONG = -97.29


def filter_catalog(cat):
    """ Filter the events in cat
    to keep the events near Guthrie
    """
    cat = cat[(cat.latitude > 35.7) & (cat.latitude < 36)
              & (cat.longitude > -97.6) & (cat.longitude < -97.2)]
    # Filter event after 15th February 2014
    begin_february = UTCDateTime(2014, 2, 15, 00, 00, 00, 500000)
    cat = cat[cat.utc_timestamp > begin_february]
    return cat


def get_test_coordinates(cat):
    """
    Return coordinates of the test set
    (July 2014)
    """
    cat = cat[(cat.latitude > 35.7) & (cat.latitude < 36)
              & (cat.longitude > -97.6) & (cat.longitude < -97.2)]
    # Filter events during July 2014
    begin_aug = UTCDateTime(2014, 7, 1, 00, 00, 00, 500000)
    end_december = UTCDateTime(2014, 7, 31, 23, 59, 59, 500000)
    cat = cat[(cat.utc_timestamp > begin_aug) & (cat.utc_timestamp < end_december)]
    lat = cat['latitude'].as_matrix()
    lon = cat['longitude'].as_matrix()
    return lat, lon


def get_distance_matrix(X):
    """ Get distance matrix between events
    X_dist[i,j] = distance between event i and j
    """
    # implement the sparse matrix of distance
    X_dist = np.zeros((len(X), len(X)), dtype=np.float64)
    # select on index
    for i1 in range(len(X)):
        # loop over all the other indices
        for i2 in range(len(X)):
            #  now find the distance
            if i1 < i2:
                # if i1 = i2 , distance = 0
                X_dist[i1, i2] = distance(
                    X[i1, 0], X[i1, 1], 0, X[i2, 0], X[i2, 1], 0)
                # fill the symetric part of the matrix
                # since distance(x1, x2) = distance(x2, x1)
                X_dist[i2, i1] = X_dist[i1, i2]
    return X_dist


def distance_to_station(lat, long, depth):
    """
    station GPS coordinates
    """
    lat0 = 35.796570
    long0 = -97.454860
    depth0 = -0.333
    # return distance of the event to the station
    return distance(long, lat, depth, long0, lat0, depth0)


def main(argv):
    try:
        argv = FLAGS(argv)  # parse flags
    except gflags.FlagsError as e:
        print('%s\nUsage: %s ARGS\n%s' % (e, sys.argv[0], FLAGS))
        sys.exit(1)

    if not os.path.exists(FLAGS.dst):
        os.makedirs(FLAGS.dst)

    cat = data_io.load_catalog(FLAGS.src)
    # Filter catalog
    cat = filter_catalog(cat)
    lat = cat['latitude'].as_matrix()
    lon = cat['longitude'].as_matrix()
    depth = cat['depth'].as_matrix()
    utc_time = cat["utc_timestamp"].as_matrix()

    print("Number of events to cluster: ", len(lat))

    feats = np.hstack((lon[:, None], lat[:, None]))
    print(" + Calculating the distance matrix")

    print(" + Running {}".format(FLAGS.model))
    if FLAGS.model == 'DBSCAN':
        distance_matrix = get_distance_matrix(feats)
        clust = DBSCAN(
            eps=2.3, min_samples=10, metric='precomputed').fit(distance_matrix)
        labels = list(set(clust.labels_))
        clust_labels = clust.labels_
    elif FLAGS.model == 'KMeans':

        if FLAGS.n_components == 6:
            initialization = np.array([[-97.6, 36],
                                       [-97.4, 35.85],
                                       [-97.2, 35.85],
                                       [-97.3, 35.75],
                                       [-97.4, 35.95],
                                       [-97.6, 35.75]])
        elif FLAGS.n_components == 50:
            init_50 = os.path.join(FLAGS.dst, 'centroids_50.npy')
            initialization = np.load(init_50)
        else:
            # random initialization
            initialization = 'k-means++'

        clust = KMeans(FLAGS.n_components,
                       n_init=10, init=initialization).fit(feats)
        clust_labels = clust.labels_
        labels = list(set(clust_labels))

        # Display predicted scores by the model as contour plot
        x = np.linspace(min(lon), max(lon), 1000)
        y = np.linspace(min(lat), max(lat), 1000)
        X, Y = np.meshgrid(x, y)
        XX = np.array([X.ravel(), Y.ravel()]).T
        Z = clust.predict(XX)
        Z = Z.reshape(X.shape)
        plt.contour(X, Y, Z, colors='k', levels=range(FLAGS.n_components))
        plt.xlabel("longitude")
        plt.ylabel("latitude")

    elif FLAGS.model == 'GMM':
        clust = GaussianMixture(FLAGS.n_components, covariance_type="full").fit(feats)
        clust_labels = clust.predict(feats)
        labels = list(set(clust_labels))

        # Display predicted scores by the model as contour plot
        x = np.linspace(min(lon), max(lon), 10)
        y = np.linspace(min(lat), max(lat), 10)
        X, Y = np.meshgrid(x, y)
        XX = np.array([X.ravel(), Y.ravel()]).T
        print('XX', XX)
        Z = clust.score_samples(XX)
        Z = Z.reshape(X.shape)
        plt.contourf(X, Y, Z, alpha=0.3)
        plt.colorbar()

    else:
        raise ValueError("set clustering model to DBSCAN or KMeans")

    # Add station on plot
    plt.plot(OK029_LONG, OK029_LAT, '*', linewidth=10)
    plt.plot(OK027_LONG, OK027_LAT, '*', linewidth=10)

    # Save labels metadata
    metadata_path = os.path.join(FLAGS.dst, "clusters_metadata.json")
    counter = collections.Counter(clust_labels)
    pd.DataFrame.from_dict(counter, orient='index')[0].to_json(metadata_path)

    # Save output catalog
    output_path = os.path.join(FLAGS.dst, "catalog_with_cluster_ids.csv")
    write_catalog_with_clusters(utc_time, clust_labels, lat, lon, depth, output_path)

    # plot the labels
    for label in labels:
        colors = sns.color_palette('hls', len(labels))[label]
        plt.scatter(lon[clust_labels == label], lat[clust_labels == label],
                    c=colors,
                    linewidth=0, label=label)

    if FLAGS.n_components < 8:
        plt.legend(bbox_to_anchor=(1.01, 1), loc=2, borderaxespad=0.)

    # plot test data
    lat_test, lon_test = get_test_coordinates(cat)
    plt.scatter(lon_test, lat_test, marker='+', linewidth=4, c='k')
    print('number of events', lat_test.shape[0])
    plt.grid(False)

    fig_name = os.path.join(FLAGS.dst,
                            "cluster_ids_{}_comp.eps".format(FLAGS.n_components))
    # plt.show()
    plt.savefig(fig_name)

    # Couple of files useful to keep
    np_name = "cluster_ids_{}_comp.npy".format(FLAGS.n_components)
    np.save(np_name, Z)

    np_name = "cluster_ids_{}_comp_lon.npy".format(FLAGS.n_components)
    np.save(np_name, x)
    np_name = "cluster_ids_{}_comp_lat.npy".format(FLAGS.n_components)
    np.save(np_name, y)


if __name__ == "__main__":
    main(sys.argv)
