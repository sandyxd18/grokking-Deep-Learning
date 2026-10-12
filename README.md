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

### [Bab 7: How to Picture Neural Networks — In Your Head and on Paper](Bab7_How_to_Picture_Neural_Networks.ipynb)
Membahas teknik visualisasi dan abstraksi mental untuk memahami cara kerja jaringan saraf tanpa kewalahan oleh detail grafis:
- **Penyederhanaan Representasi Jaringan**: Mengapa visualisasi diagram node dan bobot individu menjadi tidak praktis saat ukuran model membesar, serta transisi ke representasi tingkat layer.
- **Correlation Summarization**: Konsep bahwa hidden layer berfungsi merangkum korelasi fitur input menjadi representasi perantara tingkat tinggi yang lebih mudah dipisahkan oleh output layer.
- **Visualisasi Matriks dan Vektor**: Alur komputasi forward propagation sebagai perkalian vektor-matriks beruntun (`layer_0.dot(weights_0_1)` dilanjutkan ReLU, lalu `.dot(weights_1_2)`).
- **Notasi Formal dan Huruf**: Penggunaan notasi aljabar linear ringkas ($x$, $W$, $h$, $y$) sebagai alat berpikir efisien untuk memetakan arsitektur kompleks.

### [Bab 8: Learning Signal and Ignoring Noise — Introduction to Regularization and Batching](Bab8_Learning_Signal_and_Ignoring_Noise.ipynb)
Membahas teknik regularisasi dan optimasi batch untuk mengatasi overfitting dan mempercepat konvergensi:
- **Jaringan Tiga Layer pada Dataset MNIST**: Pelatihan deep neural network penuh pada citra digit tulisan tangan MNIST menggunakan representasi piksel mentah.
- **Memorization vs. Generalization**: Membedakan antara menghafal *noise* pada data latih (overfitting) dan mempelajari pola sinyal sejati yang mampu digeneralisasikan ke data uji baru.
- **Teknik Regularisasi**:
  - *Early Stopping*: Pemantauan akurasi validasi/uji dan penghentian training sebelum model mulai mengalami degradasi generalisasi.
  - *Dropout*: Regularisasi standar industri dengan menonaktifkan sebagian neuron secara acak selama training (`dropout_mask`), yang secara efektif melatih ensembel dari jutaan sub-network mini.
- **Mini-Batch Gradient Descent**: Pengelompokan data ke dalam batch kecil (misal 100 sampel) untuk memaksimalkan efisiensi komputasi vektorisasi NumPy dan menghaluskan fluktuasi pembaruan gradien.

### [Bab 9: Modeling Probabilities and Nonlinearities — Activation Functions](Bab9_Modeling_Probabilities_and_Nonlinearities.ipynb)
Membahas fungsi aktivasi modern, pemodelan probabilitas, dan penanganan non-linearitas:
- **Anatomi dan Kriteria Fungsi Aktivasi**: Peran penting fungsi aktivasi dalam memutus linearitas, syarat kontinuitas diferensiabel, dan sifat monotonik.
- **Fungsi Aktivasi Hidden Layer**: Karakteristik dan komparasi antara Sigmoid, Tanh (rentang -1 hingga 1, zero-centered), dan ReLU (efisiensi komputasi dan ketahanan terhadap gradien jenuh).
- **Fungsi Aktivasi Output Layer**: Pemilihan Sigmoid untuk probabilitas kejadian independen/biner vs. Softmax untuk distribusi probabilitas kelas yang saling eksklusif (*mutually exclusive*).
- **Implementasi Komputasi Softmax**: Normalisasi eksponensial numerik stabil (`np.exp(x) / np.sum(np.exp(x))`) untuk mengubah skor mentah (*logits*) menjadi distribusi probabilitas yang berjumlah tepat 1.
- **Backpropagation dengan Turunan Aktivasi**: Formulasi analitis turunan fungsi aktivasi yang dihitung langsung dari output layer (`tanh2deriv`, `sigmoid2deriv`, `relu2deriv`) untuk modulasi sinyal delta error.
- **Peningkatan Model MNIST**: Implementasi jaringan dengan aktivasi Tanh pada hidden layer dan Softmax pada output layer.

### [Bab 10: Neural Learning About Edges and Corners — Intro to Convolutional Neural Networks](Bab10_Neural_Learning_About_Edges_and_Corners.ipynb)
Membangun Convolutional Neural Network (CNN) dari dasar menggunakan NumPy untuk pengenalan pola spasial:
- **Motivasi CNN dan Weight Sharing**: Mengatasi inefisiensi parameter fully connected layer pada citra berdimensi tinggi melalui penggunaan ulang bobot (*weight sharing*).
- **Convolutional Layer**:
  - Prinsip operasi konvolusi: Menggeser filter/kernel kecil (misal 3x3) ke seluruh gambar untuk mendeteksi fitur lokal invarian (tepi horizontal, vertikal, dan diagonal).
  - Ekstraksi patch gambar berukuran $k \times k$ dan kalkulasi dot product dengan kernel bobot.
- **Pooling Layer**: Konsep subsampling spasial (Max Pooling) untuk mereduksi ukuran representasi fitur dan memberikan invariansi terhadap pergeseran kecil (*translational invariance*).
- **Implementasi CNN Lengkap dari Nol**: Alur forward pass konvolusi, aktivasi, dan perambatan balik gradien (*backpropagation*) ke bobot kernel konvolusi menggunakan operasi array NumPy murni.
- **Efisiensi Parameter**: Perbandingan drastis efisiensi jumlah bobot antara arsitektur Dense/Linear konvensional dan Convolutional Layer.

---

## Dataset dan Studi Kasus yang Digunakan

Implementasi kode pada notebook menggunakan data sintetis maupun data riil untuk menguji pemahaman konsep:

- **Toy Sports Data (Baseball/Team Stats)**: Fitur rata-rata jumlah sentuhan/jari kaki (*toes*), rekor menang-kalah (*win/loss record*), dan jumlah penggemar (*number of fans*) untuk memprediksi probabilitas kemenangan atau cedera pemain (Bab 3, 4, 5).
- **MNIST Handwritten Digits**: Dataset gambar grayscale berukuran 28x28 piksel berisi digit tulisan tangan (0-9) untuk klasifikasi citra multikelas, evaluasi regularisasi dropout/batching, aktivasi tanh/softmax, dan arsitektur CNN (Bab 5, 8, 9, 10).
- **Streetlight Problem**: Dataset logika 3 lampu jalanan biner untuk mensimulasikan korelasi langsung, korelasi saling bertentangan, serta korelasi non-linear/kondisional yang memerlukan hidden layer dan fungsi aktivasi (Bab 6).
- **Synthetic 8x8 Pattern & Edge Kernels**: Citra sintetis sederhana dan filter konvolusi 3x3 untuk mendemonstrasikan deteksi tepi horizontal, vertikal, dan diagonal secara visual (Bab 10).

---

## Teknologi dan Library Python

Proyek ini dibangun secara mandiri (*from scratch*) tanpa framework deep learning level tinggi (seperti PyTorch, TensorFlow, atau Keras) agar mekanisme kalkulasi matematis dapat dipelajari secara transparan:

- **Python 3**: Bahasa pemrograman utama.
- **NumPy**: Komputasi numerik, aljabar linear, representasi array n-dimensi, dan operasi dot/outer product.
- **matplotlib**: Visualisasi kurva loss/akurasi pelatihan, bobot filter, dan representasi spasial (Bab 8, 9, 10).
- **scikit-learn** (atau **keras**): Pengunduhan dataset MNIST (`fetch_openml` / `keras.datasets`) sebagai data input evaluasi model (Bab 8, 9, 10).
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
   pip install numpy matplotlib scikit-learn jupyter
   ```

4. **Jalankan Jupyter Notebook**:
   ```bash
   jupyter notebook
   ```
   Buka berkas notebook bab yang ingin dipelajari (misalnya `Bab1_Introducing_Deep_Learning.ipynb` hingga `Bab10_Neural_Learning_About_Edges_and_Corners.ipynb`).
