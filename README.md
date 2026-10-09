# Grokking Deep Learning (Python Implementation)

Repositori ini berisi implementasi praktis konsep *deep learning* dari dasar (*from scratch*) menggunakan Python murni (*pure Python*) dan NumPy. Materi dan kode pada repositori ini mengacu pada buku **"Grokking Deep Learning"** oleh Andrew W. Trask.

Setiap bab disusun dalam format Jupyter Notebook interaktif yang menggabungkan penjelasan konsep teoritis, penurunan matematis praktis, serta implementasi kode langsung tanpa bergantung pada *black-box framework* (seperti TensorFlow atau PyTorch). Tujuannya adalah memahami secara utuh mekanisme sains internal di balik arsitektur jaringan saraf tiruan (*artificial neural network*).

---

## Daftar Isi dan Rincian Tiap Bab

### [Bab 1: Introducing Deep Learning: Why You Should Learn It](Bab1_Introducing_Deep_Learning.ipynb)
Fokus pada pengantar konseptual mengenai urgensi dan pendekatan pembelajaran deep learning:
- **Mengapa Deep Learning Penting**: Peran otomasi kecerdasan buatan, perbandingan inovasi mental vs. mekanis, serta dampak teknologi terhadap otomatisasi pekerjaan berbasis keahlian.
- **Filosofi Pembelajaran Buku**: Pendekatan dengan hambatan masuk rendah (*low barrier to entry*) yang hanya mensyaratkan matematika tingkat SMA, fokus pada pemahaman cara kerja internal di balik framework, dan penyampaian rumus matematika melalui analogi intuitif.
- **Prasyarat dan Fondasi**: Lingkungan kerja dasar (Python, Jupyter Notebook, NumPy) dan pentingnya memecahkan masalah nyata yang relevan secara mandiri.

### [Bab 2: Fundamental Concepts: How Do Machines Learn?](Bab2_Fundamental_Concepts.ipynb)
Membahas taksonomi, terminologi, dan fondasi konseptual algoritma machine learning:
- **Hierarki Kecerdasan Buatan**: Hubungan berjenjang antara Artificial Intelligence (AI), Machine Learning (ML), dan Deep Learning (DL), serta penerapannya untuk masalah praktis industri vs. Artificial General Intelligence (AGI).
- **Supervised vs. Unsupervised Learning**: Perbedaan antara pembelajaran dengan bimbingan label target (pemetaan input ke output) dan penemuan struktur laten tanpa label apriori.
- **Parametric vs. Nonparametric Learning**: Karakteristik model dengan jumlah parameter tetap (*fixed capacity*) dibandingkan model dengan parameter dinamis yang bertumbuh mengikuti volume data.
- **Klasifikasi Kuadran Algoritma ML**: Pemetaan algoritma ke dalam empat kategori: Supervised Parametric, Supervised Nonparametric, Unsupervised Parametric, dan Unsupervised Nonparametric.

### [Bab 3: Introduction to Neural Prediction: Forward Propagation](Bab3_Introduction_to_Neural_Prediction.ipynb)
Membangun fondasi prediksi jaringan saraf (perambatan maju / *forward propagation*) dari struktur paling sederhana hingga jaringan bertingkat:
- **Jaringan Saraf Satu Input dan Satu Bobot**: Mekanisme prediksi sebagai perkalian skalar antara input dan bobot (*scaling factor*).
- **Prediksi dengan Banyak Input (Multiple Inputs)**: Konsep *dot product* (penjumlahan berbobot / *weighted sum*), representasi informasi, dan analogi operasi logika (AND, partial AND, serta bobot negatif sebagai operator NOT).
- **Prediksi dengan Banyak Output (Multiple Outputs)**: Distribusi informasi dari satu input ke banyak node prediksi independen.
- **Prediksi Banyak Input dan Banyak Output**: Pemodelan relasi multivariat menggunakan perkalian matriks-vektor untuk menghasilkan luaran multi-dimensi secara simultan.
- **Jaringan Saraf Bertingkat (Stacking / Hidden Layers)**: Penyusunan representasi perantara (*intermediate representation*) dengan menumpuk layer jaringan saraf.
- **Primer Vektorisasi NumPy**: Operasi array, kaidah keselarasan dimensi *dot product*, dan efisiensi komputasi matriks.

### [Bab 4: Introduction to Neural Learning: Gradient Descent](Bab4_Introduction_to_Neural_Learning.ipynb)
Membahas mekanisme evaluasi dan pembaruan bobot agar jaringan saraf dapat belajar dari kesalahan:
- **Pengukuran Error (Compare)**: Perhitungan Mean Squared Error (MSE), alasan matematis pengkuadratan deviasi (selalu positif dan memberi penalti lebih besar pada selisih ekstrem), serta sifat turunan yang halus.
- **Hot and Cold Learning**: Metode penyesuaian bobot naif melalui perturbasi acak serta keterbatasannya (inefisiensi komputasi dan risiko osilasi).
- **Kalkulasi Arah dan Besaran Koreksi**: Konsep *pure error* (`pred - goal`) dan *weight delta* (`delta * input`) sebagai turunan parsial dari fungsi loss terhadap bobot.
- **Satu Iterasi Gradient Descent**: Pembaruan bobot secara analitis menuju titik minimum error.
- **Divergensi dan Peran Alpha (Learning Rate)**: Analisis fenomena *overshooting*, kondisi instabilitas numerik ketika langkah pembaruan terlalu besar, dan modulasi konvergensi menggunakan parameter alpha.

### [Bab 5: Learning Multiple Weights at a Time: Generalizing Gradient Descent](Bab5_Learning_Multiple_Weights_at_a_Time.ipynb)
Generalisasi metode penurunan gradien untuk arsitektur multi-variabel:
- **Gradient Descent dengan Banyak Input**: Distribusi satu sinyal error (*delta*) menjadi banyak *weight delta* proporsional terhadap kontribusi masing-masing input.
- **Dinamika Pembelajaran dan Freezing Weight**: Eksperimen pembekuan salah satu parameter untuk mengamati mekanisme kompensasi bobot lain dalam mencapai titik konvergensi.
- **Gradient Descent dengan Banyak Output**: Pembaruan bobot untuk beberapa target output independen secara bersamaan.
- **Gradient Descent Banyak Input dan Banyak Output**: Pemanfaatan operasi *outer product* antara vektor input dan node delta untuk memperbarui matriks bobot secara terpadu.
- **Studi Kasus Citra MNIST**: Pelatihan model klasifikasi digit angka tulisan tangan 0-9 dan interpretasi visual bobot sebagai filter pola gambar spasial.

### [Bab 6: Building Your First Deep Neural Network: Introduction to Backpropagation](Bab6_Building_Your_First_Deep_Neural_Network.ipynb)
Membangun arsitektur jaringan saraf dalam (*deep neural network*) lengkap dengan propagasi balik:
- **Studi Kasus Lampu Lalu Lintas (Streetlight Problem)**: Formulasi masalah korelasi pola lampu jalanan biner (3 input boolean) ke keputusan jalan atau berhenti.
- **Varian Gradient Descent**: Perbedaan implementasi dan karakteristik konvergensi antara Stochastic Gradient Descent (SGD), Batch Gradient Descent, dan Full Gradient Descent.
- **Dinamika Tekanan Korelasi dan Overfitting**: Mekanisme jaringan saraf mempelajari korelasi fitur, tantangan tekanan koreksi yang saling bertentangan, serta penanganan overfitting.
- **Keterbatasan Jaringan Linear**: Ketidakmampuan representasi linear murni memodelkan korelasi kondisional (seperti relasi logika XOR atau korelasi tidak langsung).
- **Fungsi Aktivasi Non-linear (ReLU)**: Penggunaan Rectified Linear Unit (`relu(x) = max(0, x)`) dan turunannya (`relu2deriv`) untuk memutus linearitas antar-layer.
- **Algoritma Backpropagation**: Propagasi sinyal error dari layer output kembali ke hidden layer (`layer_1_delta = layer_2_delta.dot(weights_1_2.T) * relu2deriv(layer_1)`) untuk atribusi kesalahan jarak jauh.
- **Deep Network Pipeline Lengkap**: Integrasi menyeluruh perambatan maju, kalkulasi error, propagasi balik, dan pembaruan bobot dalam satu loop pelatihan terintegrasi.

---

## Dataset dan Studi Kasus yang Digunakan

Implementasi kode pada notebook menggunakan data sintetis maupun data riil untuk menguji pemahaman konsep:

- **Toy Sports Data (Baseball/Team Stats)**: Fitur rata-rata jumlah sentuhan/jari kaki (*toes*), rekor menang-kalah (*win/loss record*), dan jumlah penggemar (*number of fans*) untuk memprediksi probabilitas kemenangan atau cedera pemain (Bab 3, 4, 5).
- **MNIST Handwritten Digits**: Dataset gambar grayscale berukuran 28x28 piksel berisi digit tulisan tangan (0-9) untuk klasifikasi citra multikelas dan visualisasi bobot filter (Bab 5).
- **Streetlight Problem**: Dataset logika 3 lampu jalanan biner untuk mensimulasikan korelasi langsung, korelasi saling bertentangan, serta korelasi non-linear/kondisional yang memerlukan hidden layer dan fungsi aktivasi (Bab 6).

---

## Teknologi dan Library Python

Proyek ini dibangun secara mandiri (*from scratch*) tanpa framework deep learning level tinggi (seperti PyTorch, TensorFlow, atau Keras) agar mekanisme kalkulasi matematis dapat dipelajari secara transparan:

- **Python 3**: Bahasa pemrograman utama.
- **NumPy**: Komputasi numerik, aljabar linear, representasi array n-dimensi, dan operasi dot/outer product.
- **Jupyter Notebook**: Lingkungan interaktif untuk dokumentasi, penjelasan formula matematika, dan eksekusi kode.

---

## Cara Menjalankan Proyek

1. **Clone repositori ini**:
   ```bash
   git clone https://github.com/sandyxd18/grokking-Deep-Learning.git
   cd grokking-Deep-Learning
   ```

2. **Siapkan virtual environment (opsional tetapi disarankan)**:
   ```bash
   python3 -m venv venv
   source venv/bin/activate  # Untuk Linux/macOS
   # venv\Scripts\activate   # Untuk Windows
   ```

3. **Install dependensi yang diperlukan**:
   ```bash
   pip install numpy jupyter
   ```

4. **Jalankan Jupyter Notebook**:
   ```bash
   jupyter notebook
   ```
   Buka berkas notebook bab yang ingin dipelajari (misalnya `Bab1_Introducing_Deep_Learning.ipynb` hingga `Bab6_Building_Your_First_Deep_Neural_Network.ipynb`).
