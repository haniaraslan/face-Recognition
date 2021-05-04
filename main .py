import os
import cv2
from PIL import Image
import numpy as np
from os import listdir
from os.path import isfile, join
from matplotlib import pyplot, pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA


def read_file(dir_name):
    matrix_input = np.zeros(shape=(1, 10304))
    imgs_counter = 0
    files_counter = 0
    for subdir, dirs, files in sorted(os.walk(dir_name)):
        files_counter += 1
        for filename in sorted(files, key=len):
            filepath = subdir + os.sep + filename
            if filepath.endswith(".pgm"):
                imgs_counter += 1
                new_row = convert_img(filepath)
                matrix_input = np.append(matrix_input, np.matrix(new_row), axis=0)
            if filepath.endswith(".jpg"):
                imgs_counter += 1
                img = Image.open(filepath)
                resized_img = img.resize((92, 112))
                img = resized_img.convert('L')
                ready_image = np.array(img).reshape((1, 10304))
                matrix_input = np.append(matrix_input, ready_image, axis=0)
    files_counter -= 1
    matrix_input = np.delete(matrix_input, 0, 0)
    return matrix_input, imgs_counter, files_counter


def convert_img(img_path):
    img = cv2.imread(img_path, -1).flatten()
    return img


def data_split(imgs, imgs_counter, files_counter, faces_flag):
    training = np.zeros(shape=(1, 10304))
    testing = np.zeros(shape=(1, 10304))
    split_labels_vector = np.arange(imgs_counter / 2)
    k = 0
    print(imgs_counter / 2)
    print(imgs_counter / 4)
    for i in range(int(imgs_counter / 2)):
        if (faces_flag):  # faces dataset
            if (i % int((imgs_counter / files_counter) / 2) != 0):
                split_labels_vector[i] = k
            else:
                k = k + 1
                split_labels_vector[i] = k
        else:  # faces and non faces dataset
            if (i < int(imgs_counter / 4)):
                split_labels_vector[i] = 0  # face
            else:
                split_labels_vector[i] = 1  # noface
    print(imgs_counter)
    for i in range(int(imgs_counter)):
        if (i % 2 == 0):
            testing = np.append(testing, np.matrix(imgs[i]), axis=0)
        else:
            training = np.append(training, np.matrix(imgs[i]), axis=0)
    testing = np.delete(testing, 0, 0)
    training = np.delete(training, 0, 0)
    print("labels : ", split_labels_vector)
    return testing, training, split_labels_vector


def PCA(training, testing, alpha, split_labels_vectors):
    print("------------------PCA------------------")
    mean = np.mean(training, axis=0)
    centralized_matrix = training - mean
    cov = np.cov(centralized_matrix, bias=True, rowvar=False)
    eigenvalues, eigenvectors = np.linalg.eigh(cov)
    evalues_mat = np.diag(eigenvalues)
    index = np.argsort(eigenvalues)[::-1]
    sorted_evalues = eigenvalues[index]
    sorted_evectors = eigenvectors[:, index]
    r = 0
    while (np.sum(sorted_evalues[0:r]) / np.sum(sorted_evalues) - alpha <= 1e-6):
        r += 1
    projected_matrix_training = np.dot(training, sorted_evectors[:, :r])
    projected_matrix_testing = np.dot(testing, sorted_evectors[:, :r])
    knn = [1, 3, 5, 7]
    score_calc(knn, projected_matrix_training, projected_matrix_testing, split_labels_vectors, "PCA")


def lda_no_face(training_no_face, split_labels_vectors, testing_no_face):
    mean_training = np.mean(training_no_face, axis=0)
    mean = np.zeros(shape=(1, 10304))
    S = np.zeros(shape=(1, 10304))
    Sb = np.zeros(shape=(1, 10304))
    for i in range(200, len(training_no_face) + 200, 200):
        mean = np.append(mean, np.matrix(np.mean(training_no_face[i - 200:i], axis=0)), axis=0)
    mean = np.delete(mean, 0, 0)
    print(mean)
    print(mean.shape)
    for i in range(2):
        Sb = Sb + np.dot(200 * ((mean[i] - mean_training).T), mean[i] - mean_training)
    k = 0
    Z = np.zeros(shape=(1, 10304))
    for i in range(200, len(training_no_face) + 200, 200):
        Z = np.append(Z, np.matrix(training_no_face[i - 200:i]) - mean[k], axis=0)
        k = k + 1
    Z = np.delete(Z, 0, 0)
    for i in range(10):
        S = S + (np.dot(Z[i].T, Z[i]))
    eigenvalues, eigenvectors = np.linalg.eigh(np.dot(np.linalg.inv(S), Sb))
    index = np.argsort(eigenvalues)[::-1]
    sorted_evectors = eigenvectors[:, index]
    dims = sorted_evectors[:, 0:39]
    projected_matrix_training = np.dot(training_no_face, dims)
    projected_matrix_testing = np.dot(testing_no_face, dims)
    score_calc_lda2(projected_matrix_training, projected_matrix_testing, split_labels_vectors, "LDA NON-FACES")


def lda(training, split_labels_vectors, testing):
    print("------------------------------LDA------------------------------")
    mean_training = np.mean(training, axis=0)
    mean = np.zeros(shape=(1, 10304))
    S = np.zeros(shape=(1, 10304))
    Sb = np.zeros(shape=(1, 10304))
    for i in range(5, len(training) + 5, 5):
        mean = np.append(mean, np.matrix(np.mean(training[i - 5:i], axis=0)), axis=0)
    mean = np.delete(mean, 0, 0)
    for i in range(40):
        Sb = Sb + np.dot(5 * ((mean[i] - mean_training).T), mean[i] - mean_training)
    k = 0
    Z = np.zeros(shape=(1, 10304))
    for i in range(5, len(training) + 5, 5):
        Z = np.append(Z, np.matrix(training[i - 5:i]) - mean[k], axis=0)
        k = k + 1
    Z = np.delete(Z, 0, 0)
    for i in range(200):
        S = S + (np.dot(Z[i].T, Z[i]))
    eigenvalues, eigenvectors = np.linalg.eigh(np.dot(np.linalg.inv(S), Sb))
    knn = [1, 3, 5, 7]
    index = np.argsort(eigenvalues)[::-1]
    sorted_evectors = eigenvectors[:, index]
    dims = sorted_evectors[:, 0:39]
    projected_matrix_training = np.dot(training, dims)
    projected_matrix_testing = np.dot(testing, dims)
    score_calc(knn, projected_matrix_training, projected_matrix_testing, split_labels_vectors, "LDA")


def score_calc_lda2(projected_matrix_training, projected_matrix_testing, split_labels_vectors, title):
    print("------------------------score LDA non faces--------------------------")
    scores = [0, 0, 0, 0]
    k = 0
    for i in range(250, 401, 50):
        neigh = KNeighborsClassifier(n_neighbors=1, weights='distance')
        neigh.fit(projected_matrix_training[0:i, :], split_labels_vectors[:i])
        scores[k] = neigh.score(projected_matrix_testing, split_labels_vectors)
        k += 1
    non_faces_num = [250, 300, 350, 400]
    print("non face scores: ", scores)
    plt.scatter(non_faces_num, scores)
    plt.title(title)
    plt.show()


def score_calc(knn, projected_matrix_training, projected_matrix_testing, split_labels_vectors, title):
    print("------------------------------score ", title, " ------------------------------")
    scores = [0, 0, 0, 0]
    for i in range(len(knn)):
        neigh = KNeighborsClassifier(n_neighbors=knn[i], weights='distance')
        neigh.fit(projected_matrix_training, split_labels_vectors)
        scores[i] = neigh.score(projected_matrix_testing, split_labels_vectors)
    print(title, " scores: ", scores)
    plt.scatter(knn, scores)
    plt.title(title)
    plt.show()


if __name__ == "__main__":
    imgs, imgs_counter, files_counter = read_file("C:/Users/Hania/Desktop/faces")
    imgs_no_face, imgs_no_face_counter, files_no_face_counter = read_file("C:/Users/Hania/Desktop/non faces")
    testing, training, split_labels_vectors = data_split(imgs, imgs_counter, files_counter, True)
    testing_no_face, training_no_face, split_labels_vector_no_face = data_split(imgs_no_face, imgs_no_face_counter,
                                                                                files_no_face_counter, False)
    ALPHA = [0.8, 0.85, 0.9, 0.95]
    lda_no_face(training_no_face, split_labels_vector_no_face, testing_no_face)
    PCA(training, testing, 0.9, split_labels_vectors)
    lda(training, split_labels_vectors,testing)
