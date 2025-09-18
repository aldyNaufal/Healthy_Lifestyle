

# 🩺 Analisis Gaya Hidup Sehat: Dari Clustering ke Klasifikasi Prediktif

**Segmentasi Pola Hidup dan Pembangunan Model Prediksi Menggunakan Machine Learning**

---



## 📌 1. Domain Proyek: Kesehatan Masyarakat & Ilmu Data

Gaya hidup merupakan faktor fundamental yang menentukan kesehatan dan kesejahteraan seorang individu. Kebiasaan sehari-hari—mulai dari pola makan, aktivitas fisik, manajemen stres, hingga interaksi sosial—secara kumulatif berkontribusi pada risiko penyakit kronis dan kualitas hidup secara keseluruhan. Dalam bidang kesehatan masyarakat dan kedokteran preventif, kemampuan untuk mengidentifikasi **pola atau profil gaya hidup** yang berbeda dalam populasi sangatlah berharga. Segmentasi ini memungkinkan intervensi kesehatan yang lebih personal dan tepat sasaran, serta membantu dalam perumusan kebijakan publik yang efektif.

Namun, menganalisis data gaya hidup sangatlah kompleks karena melibatkan banyak variabel yang saling berinteraksi. Di sinilah ilmu data dan *machine learning* memainkan peran kunci. Dengan memanfaatkan data multidimensional yang mencakup demografi, kebiasaan, dan kondisi kesehatan, kita dapat mengungkap pola-pola tersembunyi yang tidak terlihat melalui analisis statistik konvensional.

Proyek ini mengadopsi pendekatan dua tahap untuk menganalisis **Healthy Lifestyle Dataset**:
1.  **Tahap Unsupervised (Clustering)**: Pertama, kita akan menggunakan algoritma *clustering* untuk mengidentifikasi segmen-segmen atau "profil gaya hidup" yang berbeda secara alami dalam data, tanpa asumsi awal.
2.  **Tahap Supervised (Klasifikasi)**: Setelah profil-profil ini didefinisikan, kita akan membangun model klasifikasi prediktif yang sangat akurat untuk secara otomatis menetapkan individu baru ke dalam profil gaya hidup yang paling sesuai.

Kombinasi ini menciptakan sebuah kerangka kerja yang kuat, dari penemuan wawasan hingga aplikasi prediktif praktis di bidang kesehatan.

---

## 🎯 2. Business Understanding

### 🔍 Problem Statements

1.  Apakah terdapat kelompok-kelompok (cluster) yang berbeda dan dapat diidentifikasi secara statistik dalam populasi berdasarkan kombinasi fitur gaya hidup, kesehatan, dan demografi?
2.  Setelah cluster-cluster ini diidentifikasi, dapatkah kita membangun sebuah model klasifikasi yang sangat akurat untuk memprediksi keanggotaan cluster seorang individu berdasarkan fitur-fitur mereka?
3.  Model *machine learning* manakah (baik untuk clustering maupun klasifikasi) yang memberikan performa paling optimal, dan apa karakteristik utama yang membedakan setiap cluster gaya hidup yang ditemukan?

### 🎯 Objectives

1.  **Tahap 1 (Unsupervised Learning)**:
    * Menerapkan dan membandingkan tiga algoritma clustering: **K-Means**, **MiniBatchKMeans**, dan **Gaussian Mixture Model (GMM)** untuk mensegmentasi data.
    * Mengevaluasi jumlah cluster yang optimal menggunakan **Elbow Method** dan **Silhouette Score**.
    * Menganalisis karakteristik setiap cluster untuk memberikan interpretasi yang bermakna (misalnya, "Sehat Aktif", "Risiko Tinggi Sedentari").

2.  **Tahap 2 (Supervised Learning)**:
    * Menggunakan label cluster yang dihasilkan sebagai variabel target untuk melatih model klasifikasi.
    * Membangun dan membandingkan tiga model klasifikasi: **Support Vector Machine (SVC)**, **Gaussian Naïve Bayes**, dan **Random Forest Classifier**.
    * Melakukan *hyperparameter tuning* menggunakan `GridSearchCV` untuk mengoptimalkan performa model terbaik.
    * Mengevaluasi model menggunakan metrik akurasi, *classification report*, dan *learning curve* untuk mendeteksi *overfitting*.

### 💡 Solusi yang Diusulkan

Sebuah *pipeline* machine learning dua tahap:
1.  Pertama, **K-Means Clustering** akan digunakan untuk segmentasi data, karena terbukti memberikan hasil dengan *Silhouette Score* tertinggi.
2.  Kedua, label dari model K-Means akan digunakan untuk melatih model **Support Vector Machine (SVC)**, yang kemudian dioptimalkan untuk menciptakan alat prediksi akhir.

---

## 📁 3. Dataset Overview

* **Sumber**: Kaggle - [Healthy Lifestyle Dataset](https://www.kaggle.com/datasets/aditibabu/healthy-lifestyle?select=Test_Data.csv)
* **Jumlah Entri**: 6.480 baris
* **Jumlah Fitur**: 17 kolom
* **Penulis Dataset**: Aditi Babu

---

## 📋 4. Fitur Dataset

| Fitur | Deskripsi |
| :--- | :--- |
| `Age` | Usia individu dalam tahun. |
| `BMI` | Body Mass Index, rasio berat terhadap tinggi badan. |
| `Smoker?` | Status merokok (Ya/Tidak/Tidak bisa dikatakan). |
| `Living in?` | Lingkungan tempat tinggal (Perkotaan/Pedesaan). |
| `Follow Diet` | Kepatuhan terhadap diet tertentu (Ya/Tidak). |
| `Physical activity` | Tingkat aktivitas fisik. |
| `Regular sleeping hours`| Durasi tidur teratur per malam. |
| `Alcohol consumption` | Tingkat konsumsi alkohol. |
| `Social interaction` | Tingkat interaksi sosial. |
| `Taking supplements` | Kebiasaan mengonsumsi suplemen. |
| `Illness count last year`| Jumlah penyakit yang diderita dalam setahun terakhir. |
| `Food preference`, `Specific ailments`| Preferensi makanan dan kondisi kesehatan spesifik. |
| `Any hereditary condition?`| Adanya kondisi kesehatan turunan. |
| `ID1`, `ID2` | ID unik responden (tidak digunakan dalam pemodelan). |

---

## 🔍 5. Data Understanding & EDA

### 📊 **Distribusi dan Tipe Data**
Dataset terdiri dari campuran tipe data: **numerik** (Age, BMI, Illness count), **kategorikal biner** (Living in?), dan **kategorikal multi-kelas** (Smoker?, Food preference).

### 💧 **Missing Values**
Beberapa kolom, terutama yang terkait kebiasaan (`Follow Diet`, `Physical activity`, dll.), memiliki **262 nilai yang hilang**. Kolom `Food preference` memiliki 3 nilai yang hilang. Ini menandakan perlunya strategi **imputasi** pada tahap persiapan data.

### ↔️ **Analisis Korelasi**
*Heatmap* korelasi menunjukkan hubungan yang umumnya lemah antar fitur numerik. Namun, beberapa korelasi positif yang cukup terlihat antara lain:
* **`Social interaction`** dan **`Illness count last year`** (r = 0.55)
* **`Regular sleeping hours`** dan **`Illness count last year`** (r = 0.45)
* **`Physical activity`** dan **`Illness count last year`** (r = 0.41)
Ini memberikan petunjuk awal bahwa faktor-faktor gaya hidup ini mungkin saling terkait dengan kondisi kesehatan.



### 💎 **Analisis Nilai Unik**
* Kolom `Any hereditary condition?` hanya berisi satu nilai unik ('Stable'), sehingga tidak memberikan variasi informasi dan dapat dihapus.
* Kolom `Smoker?` memiliki tiga kategori ('YES', 'NO', 'Cannot say'), yang perlu di-encode.

---

## 🧹 6. Data Preparation

Data mentah diproses melalui *pipeline* yang ketat untuk memastikan kualitas dan kesiapannya untuk pemodelan.

| Langkah | Penjelasan |
| :--- | :--- |
| **Imputasi Missing Values** | Nilai hilang pada kolom **numerik** diisi dengan **median**. Nilai hilang pada kolom **kategorikal** diisi dengan **modus (nilai tersering)**. |
| **Encoding Kategorikal** | **Label Encoding** digunakan untuk `Smoker?`. **One-Hot Encoding** digunakan untuk `Living in?` untuk menghindari asumsi urutan. |
| **Penanganan Outlier** | Metode **IQR (Interquartile Range)** digunakan untuk "memangkas" (*capping*) nilai-nilai ekstrem pada fitur-fitur kunci seperti `BMI` dan `Age`, sehingga model tidak terdistorsi. |
| **Seleksi Fitur** | Fitur yang tidak relevan (seperti `ID1`, `ID2`, `Any hereditary condition?`) dihapus. |
| **Normalisasi (Scaling)** | **MinMaxScaler** diterapkan pada semua fitur numerik untuk mengubah skala nilainya ke rentang [0, 1]. Ini krusial untuk algoritma berbasis jarak seperti K-Means dan SVC. |

---

## ⚙️ 7. Tahap 1: Pemodelan Clustering (Unsupervised)

Tujuan tahap ini adalah menemukan segmen atau profil gaya hidup yang tersembunyi dalam data.

### 🧪 **Model yang Diuji**
* **K-Means**: Algoritma berbasis centroid, efisien dan populer.
* **MiniBatchKMeans**: Varian K-Means yang lebih cepat untuk data besar.
* **Gaussian Mixture Model (GMM)**: Model probabilistik yang lebih fleksibel untuk cluster non-bulat.

### 📈 **Evaluasi dan Hasil**
Jumlah cluster optimal dievaluasi menggunakan **Elbow Method** (untuk K-Means) dan **Silhouette Score** (untuk semua model) pada rentang k dari 2 hingga 10.

* **Hasil Silhouette Score**:
    * **K-Means**: Mencapai skor tertinggi **0.8331** pada **k=10**.
    * **GMM**: Juga mencapai skor **0.8331** pada **k=10**.
    * **MiniBatchKMeans**: Skor sedikit lebih rendah (0.7966) pada k=10.
* **Dampak Feature Selection**: Eksperimen tambahan menunjukkan bahwa melakukan seleksi fitur (mengurangi dari 11 menjadi 6 fitur) justru **menurunkan kualitas cluster** secara drastis (Silhouette Score turun menjadi ~0.58).

### 🏆 **Keputusan Model Clustering**
**K-Means dengan k=10 pada set fitur asli (tanpa seleksi fitur)** dipilih sebagai model clustering terbaik karena memberikan skor silhouette tertinggi, yang menandakan cluster yang padat dan terpisah dengan baik.

---

## 🔬 8. Analisis & Interpretasi Cluster

Setelah mendapatkan 10 cluster, kami menganalisis karakteristik rata-rata dari setiap cluster, terutama berdasarkan **BMI** dan **`Illness count last year`**.

| Cluster | Rata-rata BMI | Rata-rata Jumlah Penyakit | Interpretasi Profil | Kategori Risiko |
| :--- | :---: | :---: | :--- | :--- |
| **Cluster 1, 2, 9**| ~26.98 | 5.27 - 5.65 | Individu dengan **berat badan berlebih (overweight)**. | **Risiko Tinggi** |
| **Cluster 0, 6, 7**| ~20.02 | 5.23 - 5.38 | Individu dengan **berat badan cenderung rendah (underweight)**. | **Risiko Menengah** |
| **Cluster 3, 4, 5**| ~23.36 | 5.36 - 5.45 | Individu dengan **BMI normal** namun jumlah penyakit masih cukup tinggi. | **Risiko Menengah** |
| **Cluster 8** | 23.83 | **5.30** | Individu dengan **BMI normal dan jumlah penyakit relatif terendah**. | **Kesehatan Optimal**|

Analisis ini berhasil memberikan makna pada setiap segmen, mengubah angka menjadi profil gaya hidup yang dapat ditindaklanjuti.

---

## 🤖 9. Tahap 2: Pemodelan Klasifikasi (Supervised)

Tujuan tahap ini adalah membangun model yang dapat secara akurat memprediksi profil gaya hidup (cluster 0-9) untuk data individu baru.

### 🎯 **Fitur dan Target**
* **Fitur (X)**: 11 fitur gaya hidup yang telah diproses.
* **Target (y)**: Label cluster (0-9) dari hasil model K-Means terbaik.

### 🧪 **Model yang Diuji**
1.  **Support Vector Machine (SVC)**
2.  **Gaussian Naïve Bayes (GNB)**
3.  **Random Forest Classifier (RF)**

### 🔧 **Hyperparameter Tuning**
`GridSearchCV` digunakan untuk mencari kombinasi parameter terbaik untuk model SVC dan Random Forest.

---

## 📊 10. Hasil Evaluasi Klasifikasi

### 📈 **Perbandingan Akurasi**

| Model | Akurasi (Sebelum Tuning) | Akurasi (Setelah Tuning) | Perubahan |
| :--- | :---: | :---: | :---: |
| **Support Vector Machine (SVC)** | 98.53% | **99.07%** | ✅ Meningkat |
| **Random Forest Classifier** | 98.07% | 97.68% | 🔻 Menurun |
| **Gaussian Naïve Bayes** | 92.43% | (Tidak dituning) | - |

**SVM dengan kernel 'linear' dan C=1** muncul sebagai model dengan performa tertinggi setelah tuning.

### 📉 **Analisis Overfitting (Learning Curves)**
* **SVM & GNB**: Kurva *training* dan *cross-validation* saling mendekat, menunjukkan **generalisasi yang baik** dan tidak ada *overfitting* yang signifikan.
* **Random Forest**: Terdapat celah yang cukup besar antara kurva *training* (akurasi hampir 100%) dan *cross-validation*, yang merupakan **indikasi jelas adanya overfitting**.



---

## ✅ 11. Kesimpulan

Proyek ini berhasil menunjukkan kekuatan pendekatan dua tahap dalam analisis data kesehatan:
1.  **Penemuan Segmen**: Dengan *unsupervised learning*, kami berhasil mengidentifikasi **10 profil gaya hidup** yang berbeda dan bermakna dari data, dengan **K-Means** sebagai algoritma terbaik (*Silhouette Score = 0.83*).
2.  **Pembangunan Model Prediktif**: Dengan *supervised learning*, kami berhasil membangun model klasifikasi yang sangat akurat. **Support Vector Machine (SVC) yang telah dioptimalkan** mampu memprediksi profil gaya hidup individu dengan **akurasi 99.07%**, menunjukkan generalisasi yang sangat baik dan tanpa *overfitting*.

Kerangka kerja ini menyediakan alat yang kuat bagi praktisi kesehatan, peneliti, dan pembuat kebijakan untuk memahami dinamika gaya hidup dalam populasi dan mengembangkan intervensi yang lebih personal dan efektif.

---

## 🚀 12. Rekomendasi untuk Penelitian Selanjutnya

* **Mengatasi Overfitting RF**: Lakukan regularisasi lebih lanjut pada model Random Forest, misalnya dengan mengurangi `max_depth` atau menambah `min_samples_leaf`.
* **Eksplorasi Model Lain**: Uji algoritma *boosting* seperti **XGBoost** atau **LightGBM** yang dikenal memiliki performa tinggi.
* **Interpretabilitas Model**: Gunakan teknik seperti **SHAP** atau **LIME** pada model klasifikasi terbaik (SVC) untuk memahami fitur mana yang paling berpengaruh dalam menentukan profil gaya hidup seseorang.
* **Pengayaan Data**: Tambahkan lebih banyak fitur, seperti data diet yang lebih detail, tingkat stres, atau faktor sosio-ekonomi untuk mendapatkan wawasan yang lebih dalam.
