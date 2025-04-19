
# Εισάγω τις αντίστοιχες βιβλιοθήκες για τα μοντέλα
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.cluster import MiniBatchKMeans, DBSCAN, AgglomerativeClustering
from sklearn.metrics import calinski_harabasz_score, davies_bouldin_score, silhouette_score, adjusted_rand_score
from tensorflow.keras.datasets import fashion_mnist
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Input, Dense
!pip install umap-learn #κατεβάζω το πακέτο για το μοντέλο UMap
import umap
from sklearn.cluster import SpectralClustering
from sklearn.mixture import GaussianMixture
import pandas as pd
import time

# Φορτώνω τα δεδομένα απο το api
(x_train_full, y_train_full), (x_test, y_test) = fashion_mnist.load_data()



#Κανονικοποίηση των τιμών των pixel στο εύρος 0-1
x_train_full = x_train_full / 255.0
x_test = x_test / 255.0

#Μετατροπή των εικόνων σε διανυσμα 1D (784 χαρακτηριστικά ανά εικόνα)
x_train_full = x_train_full.reshape(-1, 28 * 28)
x_test = x_test.reshape(-1, 28 * 28)


#Διαχωρισμός των training data σε 20 validation και 80 training
x_train, x_val, y_train, y_val = train_test_split(x_train_full, y_train_full, test_size=0.2, random_state=42)

#Δημιουργώ το dataframe για την αποθηκευση των δεδομένων μου
results_df = pd.DataFrame(columns=[
    "Dimensionality Reduction Technique",
    "Clustering Algorithm",
    "Dimensionality Reduction Time (s)",
    "Clustering Time (s)",
    "Number of Suggested Clusters",
    "Calinski-Harabasz Index",
    "Davies-Bouldin Index",
    "Silhouette Score",
    "Adjusted Rand Index"
])

#Dimensionality reduction με AutoEncoder
def create_autoencoder(input_dim):
    input_layer = Input(shape=(input_dim,))
    encoded = Dense(256, activation='relu')(input_layer)
    encoded = Dense(128, activation='relu')(encoded)
    encoded = Dense(64, activation='relu')(encoded)

    decoded = Dense(64, activation='relu')(encoded)
    decoded = Dense(128, activation='relu')(decoded)
    decoded = Dense(256, activation='relu')(decoded)
    decoded = Dense(input_dim, activation='sigmoid')(decoded)

    autoencoder = Model(input_layer, decoded)
    encoder = Model(input_layer, encoded)
    return autoencoder, encoder

#Εκπαιδέυω τον SAE
#Μετράω τον χρόνο εκπαίδευσης
start_time = time.time()
autoencoder, encoder = create_autoencoder(x_train.shape[1])
autoencoder.compile(optimizer='adam', loss='binary_crossentropy')
training_start_time = time.time()
autoencoder.fit(x_train, x_train, validation_data=(x_val, x_val), epochs=35, batch_size=512, verbose=2)
training_time = time.time() - training_start_time
print(f"Training Time for SAE: {training_time:.4f} seconds")

#Εκτελω dimensionality reduction με τον SAE
#Μετράω τον χρόνο που θέλει
prediction_start_time = time.time()
x_train_encoded_sae = encoder.predict(x_train)
x_test_encoded_sae = encoder.predict(x_test)
prediction_time = time.time() - prediction_start_time
print(f"Prediction Time for SAE: {prediction_time:.4f} seconds")

# Συνολικός Χρόνος
total_time = time.time() - start_time
print(f"Total Time for SAE (Training + Prediction): {total_time:.4f} seconds")

# Commented out IPython magic to ensure Python compatibility.
#Εμφανίζω μία φωτογραφία απο κάθε κλάση μόνο η τεχνική SAE το επιτρέπει
#Πάνω είναι οι αρχικές φωτογραφίες και κάτω οι φωτογραφίες μετά την τεχνική dr
n_classes = 10
# %matplotlib inline
plt.figure(figsize=(15, 4))
for i in range(n_classes):
    idx = np.where(y_test == i)[0][0]
    plt.subplot(2, n_classes, i + 1)
    plt.imshow(x_test[idx].reshape(28, 28), cmap='gray')
    plt.axis('off')
    reconstructed = autoencoder.predict(x_test[idx].reshape(1, -1))
    plt.subplot(2, n_classes, i + 1 + n_classes)
    plt.imshow(reconstructed.reshape(28, 28), cmap='gray')
    plt.axis('off')
plt.suptitle("Αρχικές (πάνω) and Ανακατασκευαμένες (κάτω) Φωτογραφίες")
plt.show()

#Dimensionality reduction με pca
#Μετράω τον χρόνο
start_time = time.time()
pca = PCA(n_components=64)
x_train_encoded_pca = pca.fit_transform(x_train)
x_test_encoded_pca = pca.transform(x_test)
pca_time = time.time() - start_time
print(f"PCA Dimensionality Reduction Time: {pca_time:.4f} seconds")

#Dimensionality reduction με UMAP + μετράω τον χρόνο
start_time = time.time()
umap_reducer = umap.UMAP(n_components=64, random_state=42)
x_train_encoded_umap = umap_reducer.fit_transform(x_train)
x_test_encoded_umap = umap_reducer.transform(x_test)
umap_time = time.time() - start_time
print(f"UMAP Dimensionality Reduction Time: {umap_time:.4f} seconds")

#Δημιουργία 2D γραφικών παραστάσεων

#Οπτικοποίηση των δεδομένων μετα την τεχνική Autoencoder
encoder_2d = Model(inputs=autoencoder.input, outputs=autoencoder.get_layer(index=2).output)
x_train_encoded_sae_2d = encoder_2d.predict(x_train)
plt.figure(figsize=(10, 7))
plt.scatter(x_train_encoded_sae_2d[:, 0], x_train_encoded_sae_2d[:, 1], c=y_train, cmap='Spectral', s=5)
plt.colorbar()
plt.title('SAE - 2D Projection of Fashion-MNIST')
plt.show()

#Οπτικοποίηση των δεδομένων μετα την τεχνική PCA
pca_2d = PCA(n_components=64)
x_train_encoded_pca_2d = pca_2d.fit_transform(x_train)
plt.figure(figsize=(10, 7))
plt.scatter(x_train_encoded_pca_2d[:, 0], x_train_encoded_pca_2d[:, 1], c=y_train, cmap='Spectral', s=5)
plt.colorbar()
plt.title('PCA - 2D Projection of Fashion-MNIST')
plt.show()

#Οπτικοποίηση των δεδομένων μετα την τεχνική UMAP
umap_2d = umap.UMAP(n_components=2, random_state=42)
x_train_encoded_umap_2d = umap_2d.fit_transform(x_train)
plt.figure(figsize=(10, 7))
plt.scatter(x_train_encoded_umap_2d[:, 0], x_train_encoded_umap_2d[:, 1], c=y_train, cmap='Spectral', s=5)
plt.colorbar()
plt.title('UMAP - 2D Projection of Fashion-MNIST')
plt.show()

# Ορισμός των μοντέλων clustering και των παραμέτρων τους
clustering_methods = {
    "MiniBatchKMeans": MiniBatchKMeans(n_clusters=n_classes, batch_size=64,random_state=42),
    "DBSCAN" : DBSCAN(eps=5,min_samples=5),
    "Agglomerative": AgglomerativeClustering(n_clusters=n_classes,linkage='average'),
}

results = {}

# Εφαρμογή των μεθόδων clustering + Μετράω τον χρόνο + Υπολογίζω τις μετρικές + Προσθέτω τα δεδομένα στο dataframe
for reduction_name, reduced_data_train, reduced_data_test in zip(
    ["Raw", "PCA", "SAE", "UMAP"],
    [x_train, x_train_encoded_pca, x_train_encoded_sae, x_train_encoded_umap],
    [x_test, x_test_encoded_pca, x_test_encoded_sae, x_test_encoded_umap]
):
    if reduction_name == "Raw":
        reduction_time = 0
    else:
        start_time = time.time()
        reduction_time = time.time() - start_time

    for name, clustering_method in clustering_methods.items():
        start_time = time.time()
        clusters = clustering_method.fit_predict(reduced_data_test) # Εφαρμογή του clustering
        clustering_time = time.time() - start_time

        # Υπολογισμός μετρικών
        ch_score = calinski_harabasz_score(reduced_data_test, clusters)
        db_score = davies_bouldin_score(reduced_data_test, clusters)
        silhouette = silhouette_score(reduced_data_test, clusters)
        ari = adjusted_rand_score(y_test, clusters)

        # Αποθήκευση στο results
        if reduction_name not in results:
            results[reduction_name] = {}
        results[reduction_name][name] = {
            "Dimensionality Reduction Time (s)": reduction_time,
            "Clustering Time (s)": clustering_time,
            "Number of Suggested Clusters": len(set(clusters)) - (1 if -1 in clusters else 0),
            "Calinski-Harabasz Index": ch_score,
            "Davies-Bouldin Index": db_score,
            "Silhouette Score": silhouette,
            "Adjusted Rand Index": ari,
        }

        # Προσθήκη στο DataFrame
        new_row = {
            "Dimensionality Reduction Technique": reduction_name,
            "Clustering Algorithm": name,
            "Dimensionality Reduction Time (s)": reduction_time,
            "Clustering Time (s)": clustering_time,
            "Number of Suggested Clusters": len(set(clusters)) - (1 if -1 in clusters else 0),
            "Calinski-Harabasz Index": ch_score,
            "Davies-Bouldin Index": db_score,
            "Silhouette Score": silhouette,
            "Adjusted Rand Index": ari
        }
        results_df = pd.concat([results_df, pd.DataFrame([new_row])], ignore_index=True)

# Αποθήκευση αποτελεσμάτων σε CSV
results_df.to_csv("clustering_results.csv", index=False)

# Εμφάνιση αποτελεσμάτων από το results
for reduction_name, algorithms in results.items():
    print(f"\nResults for {reduction_name}:\n")
    for algorithm_name, metrics in algorithms.items():
        print(f"  Algorithm: {algorithm_name}")
        for metric, value in metrics.items():
            print(f"    {metric}: {value:.4f}")

import matplotlib.pyplot as plt
import numpy as np

selected_classes = [1, 2, 7, 8]

clusters_dict = {}
for reduction_name, reduced_data_test in zip(
    ["Raw Data", "SAE", "PCA", "UMAP"],
    [x_test, x_test_encoded_sae, x_test_encoded_pca, x_test_encoded_umap]
):
    clusters_dict[reduction_name] = clustering_methods["Agglomerative"].fit_predict(reduced_data_test)

for class_label in selected_classes:
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))

    # Επιλογή 8 διαφορετικών δειγμάτων από την κλάση
    idx = np.where(y_test == class_label)[0][:8]

    for i in range(2):  # 2 σειρές (1 raw + 3 DR)
        for j, reduction_name in enumerate(["Raw Data", "SAE", "PCA", "UMAP"]):
            img_idx = idx[i*4 + j]  # Διαφορετικές εικόνες για κάθε subplot

            cluster_label = clusters_dict[reduction_name][img_idx]

            subplot_index = i * 4 + j
            ax = axes.flat[subplot_index]

            ax.imshow(x_test[img_idx].reshape(28, 28), cmap='gray')
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_title(reduction_name, fontsize=10)
            ax.set_xlabel(f"Cluster: {cluster_label}", fontsize=10)

    plt.suptitle(f"Clustering Results for Class {class_label}", fontsize=16)
    plt.tight_layout()
    plt.show()
