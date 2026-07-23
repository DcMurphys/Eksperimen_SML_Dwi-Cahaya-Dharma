# -*- coding: utf-8 -*-
"""Eksperimen MSML_Telco Customer Churn

# **1. Perkenalan Dataset**

Tahap pertama, Anda harus mencari dan menggunakan dataset dengan ketentuan sebagai berikut:

1. **Sumber Dataset**:  
   Dataset dapat diperoleh dari berbagai sumber, seperti public repositories (*Kaggle*, *UCI ML Repository*, *Open Data*) atau data primer yang Anda kumpulkan sendiri.

Untuk proyek **Customer Churn Analytics** ini, dataset yang digunakan adalah **Telco customer churn (11.1.3+)** dari Al Fath Terry, yang filenya bisa di-download secara langsung dari Kaggle melalui link URL berikut: [Telco customer churn (11.1.3+) on Kaggle](https://www.kaggle.com/datasets/alfathterry/telco-customer-churn-11-1-3).

Dataset ini terdiri dari 50 kolom/atribut, berisikan informasi mengenai sebuah perusahaan telco fiktif di California yang menyediakan layanan internet dan telepon rumah kepada 7.043 pelanggan di daerah tersebut.

Dataset ini dapat digunakan untuk berbagai macam hal terkait proses pembangunan machine learning prediktif untuk prediksi churn pelanggan, termasuk analisis data eksploratif (EDA) secara mendalam, prapemrosesan, hingga pemodelan prediktif dengan sistem machine learning.

Berikut adalah deskripsi singkat dari masing-masing fitur (kolom) dalam dataset ini.
* `Customer ID`: ID pengenal unik dari setiap pelanggan
* `Gender`: Jenis kelamin masing-masing pelanggan: Male, Female
* `Age`: Umur pelanggan (dalam tahun)
* `Senior Citizen`: Apakah pelanggan berumur 65 tahun atau lebih tua?: Yes, No
* `Married`: Apakah pelanggan sudah menikah?: Yes, No
* `Dependents`: Mengindikasikan apakah pelanggan bergantung kepada orang lain (Yes, No); misalnya anak, orang tua, kakek/nenek, dsb.
* `Number of Dependents`: Jumlah orang lain yang tinggal bersama pelanggan yang bersangkutan
* `Country`: Negara domisili pelanggan
* `City`: Kota domisili pelanggan
* `Zip Code`: Kode pos domisili pelanggan
* `Latitude`: Posisi tempat tinggal domisili pelanggan mengacu pada garis lintang
* `Longitude`: Posisi tempat tinggal domisili pelanggan mengacu pada garis bujur
* `Population`: Estimasi jumlah penduduk untuk seluruh area kode pos
* `Quarter`: Kuartal fiskal di mana data terakhir diambil
* `Referred a Friend`: Apakah pelanggan merekomendasikan layanan perusahaan kepada teman atau anggota keluarga lain: Yes, No
* `Number of Referrals`: Jumlah referral yang dihasilkan dari pelanggan yang bersangkutan
* `Tenure in Months`: Lama waktu pelanggan berlangganan dengan perusahaan dalam bulan
* `Offer`: Penawaran terakhir yang diterima dan diambil oleh pelanggan. Ini melibatkan None, Offer A, Offer B, Offer C, Offer D, dan Offer E
* `Phone Service`: Apakah pelanggan menggunakan layanan telepon rumah yang ditawarkan?: Yes, No
*  `Avg Monthly Long Distance Charges`: Biaya rata-rata pelanggan untuk pemakaian jarak jauh
* `Multiple Lines`: Apakah pelanggan menggunakan lebih dari satu jaringan telepon dari perusahaan yang sama?: Yes, No
* `Internet Service`: Jenis layanan internet yang digunakan oleh pelanggan: No, DSL, Fiber Optic, Cable
* `Avg Monthly GB Download`: Rata-rata volume download pelanggan dalam GB
* `Online Security`: Apakah pelanggan menggunakan layanan online security dari perusahaan?: Yes, No
* `Online Backup`: Apakah pelanggan menggunakan layanan backup data langsung dari perusahaan?: Yes, No
* `Device Protection Plan`: Apakah pelanggan menggunakan layanan tambahan berupa device protection plan untuk peralatan internet yang disediakan oleh perusahaan?: Yes, No
* `Online Security`: Apakah pelanggan menggunakan layanan tambahan berupa technical support plan dari perusahaan, dengan waktu tunggu yang lebih pendek?: Yes, No
* `Streaming TV`: Apakah pelanggan menggunakan layanan internet untuk mengakses siaran TV (IPTV)?: Yes, No
* `Streaming Movies`: Apakah pelanggan menggunakan layanan internet untuk mengakses konten film?: Yes, No
* `Streaming Music`: Apakah pelanggan menggunakan layanan internet untuk mengakses konten musik?: Yes, No
* `Unlimited Data`: Apakah pelanggan membayar biaya tambahan untuk download/upload tanpa batas?: Yes, No
* `Contract`: Jenis kontrak yang diambil pelanggan: Month-to-Month, One Year, Two Year
* `Paperless Billing`: Apakah pelanggan memilih e-billing (paperless)?: Yes, No
* `Payment Method`: Jenis pembayaran yang digunakan pelanggan dalam membayar tagihannya: Bank Withdrawal, Credit Card, Mailed Check
* `Monthly Charge`: Jumlah tagihan bulanan pelanggan untuk seluruh layanan yang digunakan
* `Total Charges`: Jumlah tagihan pelanggan secara keseluruhan yang dihitung pada kuartal tertentu
* `Total Refunds`: Jumlah pengembalian dana dari perusahaan ke pelanggan yang dihitung pada kuartal tertentu
* `Total Extra Data Charges`: Jumlah tagihan yang dibebankan ke pelanggan atas pemakaian data melebihi apa yang dicantumkan dalam paket (plan), pada akhir kuartal tertentu
* `Total Long Distance Charges`: Biaya pelanggan secara total untuk pemakaian jarak jauh
* `Satisfaction Score`: Rating kepuasan secara keseluruhan dari pelanggan, dari 1 (Very Unsatisfied) ke 5 (Very Satisfied)
* `Satisfaction Score Label`: Versi teks (String) dari kolom `Satisfaction Score`
* `Customer Status`: Status pelanggan pada akhir kuartal: Churned, Stayed, atau Joined
* `Churn Label`: Yes = Pelanggan meninggalkan perusahaan pada kuartal tertentu, No = Pelanggan masih tetap berlangganan dengan perusahaan hingga saat ini
* `Churn Value`: 1 = Pelanggan meninggalkan perusahaan pada kuartal tertentu, 0 = Pelanggan masih tetap berlangganan dengan perusahaan hingga saat ini. Secara langsung berhubungan dengan kolom `Churn Label`
* `Churn Score`: Nilai skor dalam skala 1 - 100 yang dihasilkan menggunakan IBM SPSS Modeler, ditentukan oleh banyak faktor yang diyakini menyebabkan churn. Semakin tinggi Churn Score, semakin besar kemungkinan pelanggan melakukan churn
* `Churn Score Category`: Kategori berdasarkan nilai `Churn Score`: 0-10, 11-20, 21-30, 31-40, 41-50, 51-60, 61-70, 71-80, 81-90, dan 91-100
* `CLTV`: Customer Lifetime Value, dihasilkan menggunakan rumus yang ditentukan perusahaan dan data yang sudah ada. Semakin tinggi nilai CLTV, semakin bernilai pelanggan tersebut
* `CLTV Category`: Kategori berdasarkan nilai `CLTV`: 2000-2500, 2501-3000, 3001-3500, 3501-4000, 4001-4500, 4501-5000, 5001-5500, 5501-6000, 6001-6500, dan 6501-7000
* `Churn Category`: Kategori tingkat tinggi mengenai alasan pelanggan melakukan churn: Attitude, Competitor, Dissatisfaction, Other, Price
* `Churn Reason`: Alasan pelanggan melakukan churn secara lebih detail, di luar poin jawaban yang tersedia pada `Churn Category`

# **2. Import Library**

Pada tahap ini, Anda perlu mengimpor beberapa pustaka (library) Python yang dibutuhkan untuk analisis data dan pembangunan model machine learning atau deep learning.
"""

# Import library yang diperlukan
# Library pengolahan & visualisasi data
import gdown
import zipfile
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Library transformasi data
import numpy as np
import joblib
from scipy import stats
from scipy.stats import chi2_contingency, randint
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# Meniadakan pesan "warning"
import warnings
warnings.filterwarnings("ignore")

"""# **3. Memuat Dataset**

Pada tahap ini, Anda perlu memuat dataset ke dalam notebook. Jika dataset dalam format CSV, Anda bisa menggunakan pustaka pandas untuk membacanya. Pastikan untuk mengecek beberapa baris awal dataset untuk memahami strukturnya dan memastikan data telah dimuat dengan benar.

Jika dataset berada di Google Drive, pastikan Anda menghubungkan Google Drive ke Colab terlebih dahulu. Setelah dataset berhasil dimuat, langkah berikutnya adalah memeriksa kesesuaian data dan siap untuk dianalisis lebih lanjut.

Jika dataset berupa unstructured data, silakan sesuaikan dengan format seperti kelas Machine Learning Pengembangan atau Machine Learning Terapan

"""

# Mengambil file dataset 'telco' untuk disimpan sebagai dataframe 'df_churn'
file_path = "dataset_raw/telco.csv"

df_churn = pd.read_csv(file_path)

# ===== Menampilkan lima baris data pertama dengan fungsi .head() =====
df_churn.head()

"""# **4. Exploratory Data Analysis (EDA)**

Pada tahap ini, Anda akan melakukan **Exploratory Data Analysis (EDA)** untuk memahami karakteristik dataset.

Tujuan dari EDA adalah untuk memperoleh wawasan awal yang mendalam mengenai data dan menentukan langkah selanjutnya dalam analisis atau pemodelan.
"""

# ===== Mengecek tipe data dari setiap kolom pada dataset =====
print(df_churn.dtypes)

# ===== Mengecek isi nilai dataframe =====
print(f"Terdapat sebanyak: \n{df_churn.shape[1]} kolom dan {df_churn.shape[0]} baris dalam dataframe ini.")
print(f"{df_churn['Customer ID'].duplicated().sum().sum()} baris data yang duplikat.")

# ===== Memeriksa adanya missing value pada setiap kolom =====
missing_counts = df_churn.isna().sum()
missing_counts[missing_counts > 0].sort_values(ascending=False)
print(("=" * 50) + "\nKolom dengan missing value:")
print(f"{missing_counts}")

# ===== Melihat gambaran statistik dari kolom kategorikal =====
cat_cols = df_churn.select_dtypes(include=object).columns.tolist()

df_summary = pd.DataFrame()

for col in cat_cols:
    freqs = df_churn[col].value_counts().reset_index()
    freqs.columns = ["Unique Value", "Frequency"]
    freqs["Variable"] = col
    df_summary = pd.concat([df_summary, freqs], ignore_index=True)

display(df_churn.select_dtypes(include=object).describe())
print()
display(df_summary)

# ====== Membuat fungsi helper untuk visualisasi =====
# 1) Fungsi helper bar chart
def plot_bar(data, y_col, x_col=None, hue=None, title="Bar Chart", xlabel="X-axis", ylabel="Y-axis", ax=None):
    if ax is None:
        ax = plt.gca()
    if x_col:
        sns.barplot(data=data, y=y_col, x=x_col, hue=hue, orient="h", ax=ax)
    else:
        sns.countplot(data=data, y=y_col, hue=hue, ax=ax)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    ax.tick_params(axis='y', labelrotation=0)

    for container in ax.containers:
        ax.bar_label(container, fmt='%.2f', padding=3)

    return ax


# 2) Fungsi pie chart
def plot_pie(data, column_name, title="Pie Chart", ax=None):
    if isinstance(data, pd.DataFrame):
        pie_data = data[column_name].value_counts()
    else:
        pie_data = data.value_counts()
    if ax is None:
        ax = plt.gca()

    ax.pie(pie_data, labels=pie_data.index, autopct='%1.1f%%', startangle=90)
    ax.set_title(title)
    ax.axis('equal')


# 3). Fungsi box plot chart
def plot_box(data, x_col, y_col=None, hue=None, title="Box Plot", xlabel="X-axis", ylabel="Y-axis", ax=None):
    if ax is None:
        ax = plt.gca()

    sns.boxplot(data=data, x=x_col, y=y_col, hue=hue, ax=ax)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

# ===== Eksplorasi Churn Rate =====
# Filter data Churn berdasarkan status Churn
churn_rate = (
    df_churn['Churn Label']
    .value_counts(normalize=True)
    .mul(100)
    .reset_index()
)

churn_rate.columns = ['Churn Label', 'Percentage']

# Visualisasi Churn Rate
fig, ax = plt.subplots(1, 2, figsize=(12, 5))

# Plot visualisasi Churn Rate menggunakan bar chart
plot_bar(
    data=churn_rate,
    x_col='Percentage',
    y_col='Churn Label',
    title='Distribution of Churn Rate (%)',
    xlabel='Churn Status',
    ylabel='Percentage (%)',
    ax=ax[0]
)

# Plot visualisasi Churn Rate by Contract menggunakan pie chart
plot_pie(
    data=df_churn[df_churn["Churn Label"] == 'Yes'],
    column_name="Contract",
    title="Distribution of Churn Rate by Contract",
    ax=ax[1])

plt.tight_layout()
plt.show()

# ===== Eksplorasi Churn Rate berdasarkan faktor demografi selain usia dan geografi =====
# Membuat fitur untuk flag hanya pada Churn = Yes
df_churn["Churn_flag"] = (
    df_churn["Churn Label"]
    .apply(lambda x: 1 if x == "Yes" else 0)
)

# Menentukan fitur terkait demografi selain usia dan geografi
demography_features = [
    "Gender", "Senior Citizen",
    "Under 30", "Married"
]

# Visualisasi dengan horizontal bar plot
fig, axes = plt.subplots(2, 2, figsize=(12, 5))
axes = axes.flatten()

for i, col in enumerate(demography_features):
    temp = df_churn.groupby(col)["Churn_flag"].mean().mul(100).reset_index()
    plot_bar(
        data=temp,
        y_col=col, x_col="Churn_flag",
        title=f"Churn Rate \nby {col} (%)",
        ylabel="", xlabel="",
        ax=axes[i]
    )

plt.tight_layout()
plt.show()

# ===== Eksplorasi Churn Count berdasarkan Kota (hanya 10 tertinggi dan terendah ditampilkan) =====
# Mendapatkan Churn Count untuk setiap Kota
df_churn_count = (
    df_churn
    .groupby("City")["Churn Label"]
    .apply(lambda x: (x == "Yes").sum())
    .reset_index(name="Churn Count")
)

# Sortir data berdasarkan 10 kota dengan jumlah churn tertinggi
df_churn_top10_cities = (
    df_churn_count[df_churn_count["Churn Count"] > 0]
    .sort_values("Churn Count", ascending=False)
    .head(10)
)

# Sortir data berdasarkan 10 kota dengan jumlah churn terendah
df_churn_btm10_cities = (
    df_churn_count[df_churn_count["Churn Count"] > 0]
    .sort_values("Churn Count", ascending=True)
    .head(10)
)

# Visualisasi dengan bar plot
fig, ax = plt.subplots(1, 2, figsize=(12, 5))

plot_bar(
    data=df_churn_top10_cities,
    x_col='Churn Count',
    y_col='City',
    title='Top 10 Cities by Count of Customer Chunt',
    xlabel='Count of Chunts',
    ylabel='',
    ax=ax[0]
)

plot_bar(
    data=df_churn_btm10_cities,
    x_col='Churn Count',
    y_col='City',
    title='Bottom 10 Cities by Count of Customer Chunt \n(Count >= 1)',
    xlabel='Count of Chunts',
    ylabel='',
    ax=ax[1]
)

plt.tight_layout()
plt.show()

# ===== Eksplorasi Churn Rate berdasarkan jenis produk yang ditawarkan =====
# Menentukan fitur terkait jenis produk langganan
product_features = [
    "Phone Service", "Multiple Lines", "Internet Service",
    "Internet Type", "Online Security", "Online Backup",
    "Device Protection Plan", "Premium Tech Support", "Streaming TV",
    "Streaming Movies", "Streaming Music", "Unlimited Data"
]

# Visualisasi dengan horizontal bar plot
fig, axes = plt.subplots(4, 3, figsize=(18, 9))
axes = axes.flatten()

for i, col in enumerate(product_features):
    temp = df_churn.groupby(col)["Churn_flag"].mean().mul(100).reset_index()
    plot_bar(
        data=temp,
        y_col=col, x_col="Churn_flag",
        title=f"Churn Rate \nby {col} (%)",
        ylabel="", xlabel="",
        ax=axes[i]
    )

plt.tight_layout()
plt.show()

# ===== Eksplorasi Churn Rate berdasarkan metode langganan dan bayar =====
# Menentukan fitur terkait metode langganan dan bayar
subscription_payment_features = [
    "Offer", "Contract",
    "Paperless Billing", "Payment Method"
]

# Visualisasi dengan horizontal bar plot
fig, axes = plt.subplots(2, 2, figsize=(12, 5))
axes = axes.flatten()

for i, col in enumerate(subscription_payment_features):
    temp = df_churn.groupby(col)["Churn_flag"].mean().mul(100).reset_index()
    plot_bar(
        data=temp,
        y_col=col, x_col="Churn_flag",
        title=f"Churn Rate \nby {col} (%)",
        ylabel="", xlabel="",
        ax=axes[i]
    )

plt.tight_layout()
plt.show()

# ===== Eksplorasi Distribusi Churn berdasarkan Fitur Numerik =====
# Sortir hanya pada pelanggan yang melakukan churn
churn_yes_data = df_churn[df_churn['Churn Label'] == 'Yes']

# Menentukan fitur numerik
numerical_cols = [
    "Age", "Tenure in Months", "Avg Monthly Long Distance Charges",
    "Avg Monthly GB Download", "Monthly Charge", "Total Charges",
    "Total Refunds", "Total Extra Data Charges", "Total Long Distance Charges",
    "Total Revenue", "Satisfaction Score", "CLTV"
]

# Visualisasi dengan Kernel Density (KDE) plot
fig, axes = plt.subplots(4, 3, figsize=(16, 10))
axes = axes.flatten()

for ax, col in zip(axes, numerical_cols):
    sns.kdeplot(
        data=churn_yes_data, x=col,
        fill=True, alpha=0.6,
        ax=ax
    )

    mode_value = churn_yes_data[col].mode()[0]
    ax.axvline(mode_value, color='red', linestyle='dashed', label=f"Mode: {mode_value}")

    ax.set_title(f"Kernel Density Plot of \n{col} (Churn: Yes)")
    ax.set_xlabel("")
    ax.set_ylabel("Density")
    ax.legend()

plt.tight_layout()
plt.show()

# ===== Eksplorasi Distribusi Outlier pada Fitur Numerik =====
# Visualisasi dengan box plot (untuk mendeteksi outlier)
fig, axes = plt.subplots(4, 3, figsize=(16, 8))
axes = axes.flatten()

for i, col in enumerate(numerical_cols):
    plot_box(
        data=df_churn,
        x_col=col,
        y_col="Churn Label",
        title=f"{col} Distribution",
        xlabel="",
        ylabel="Churn Status",
        ax=axes[i]
    )

plt.tight_layout()
plt.show()

# ===== Eksplorasi Korelasi antar Fitur Numerik =====
# Menentukan fitur numerik dalam dataset 'df_churn'
df_churn_num_only = df_churn[numerical_cols]

# Melakukan visualisasi terhadap korelasi fitur antar fitur dalam dataset 'df_churn' dengan correlation plot
fig, ax = plt.subplots(figsize=(10, 5))
corr = df_churn_num_only.corr()
sns.heatmap(corr, cmap='rocket', annot=True)
plt.show()

# ===== Menentukan fitur dengan multikolinearitas tinggi =====
# Sortir fitur dengan multikolinearitas tinggi (> 0.6)
corr_unstacked = corr.unstack()

strong_pos_corr = corr_unstacked[
    (corr_unstacked >= 0.6) & (corr_unstacked < 1)
].sort_values(ascending=False)

printed_pairs = set()

# Menampilkan fitur dengan multikolinearitas tinggi
print("Fitur dengan multikolineritas tinggi:")
for(col1, col2), corr_value in strong_pos_corr.items():
    pair = tuple(sorted((col1, col2)))
    if pair not in printed_pairs:
        print(f"'{col1}' dan '{col2}': {corr_value:.4f}")
        printed_pairs.add(pair)

"""# **5. Data Preprocessing**

Pada tahap ini, data preprocessing adalah langkah penting untuk memastikan kualitas data sebelum digunakan dalam model machine learning.

Jika Anda menggunakan data teks, data mentah sering kali mengandung nilai kosong, duplikasi, atau rentang nilai yang tidak konsisten, yang dapat memengaruhi kinerja model. Oleh karena itu, proses ini bertujuan untuk membersihkan dan mempersiapkan data agar analisis berjalan optimal.

Berikut adalah tahapan-tahapan yang bisa dilakukan, tetapi **tidak terbatas** pada:
1. Menghapus atau Menangani Data Kosong (Missing Values)
2. Menghapus Data Duplikat
3. Normalisasi atau Standarisasi Fitur
4. Deteksi dan Penanganan Outlier
5. Encoding Data Kategorikal
6. Binning (Pengelompokan Data)

Cukup sesuaikan dengan karakteristik data yang kamu gunakan yah. Khususnya ketika kami menggunakan data tidak terstruktur.
"""

# ===== Chi-square Test =====
categorical_cols = df_churn.select_dtypes(include="object").columns.tolist()
chi_statistic = []
p_value = []
vars_rm = []

for i in df_churn[categorical_cols]:
    observed = pd.crosstab(index=df_churn["Churn Label"], columns=df_churn[i])
    stat, p, dof, expected = chi2_contingency(observed)
    chi_statistic.append(stat)
    p_value.append(p)
    if p >= 0.05:
        print("'Churn Label' dan '{}' sama-sama independen (p-value = {:.2f}).".format(i, p))
        vars_rm.append(i)

# Menampilkan variabel independen yang perlu dihilangkan
if vars_rm:
    print("\nVariabel independen yang perlu dihilangkan (p < 0.05):")
    print(vars_rm)

# ===== Feature Selection =====
# Menghilangkan fitur yang tidak relevan dengan proyek
irrelevant_cols = [
    "Zip Code", "Latitude", "Longitude",
    "Population", "Quarter", "Referred a Friend",
    "Number of Referrals", "City", "Churn_flag",
    "Customer Status", "Churn Score"
]

df_churn_curated = df_churn.drop(irrelevant_cols, axis=1, errors='ignore')

# Menghilangkan fitur dengan tingkat signifikansi (p-value) < 0.05
df_churn_curated = df_churn_curated.drop(vars_rm, axis=1, errors='ignore')

# ===== Handling Data Missing Value =====
# Mendapatkan fitur yang berisikan missing values
missing_value_cols = ["Offer", "Internet Type"]

# Mengisi data missing values dengan nilai "No"
for col in missing_value_cols:
    df_churn_curated[col] = df_churn_curated[col].fillna(value="No")

# ===== Penanganan Outlier dengan Metode IQR dengan Winsorization =====
numerical_cols = df_churn_curated.select_dtypes(include='number').columns.tolist()

# Membuat fungsi helper untuk proses winsorization
for col in numerical_cols:
    Q1 = df_churn_curated[col].quantile(0.25) # 25% data
    Q3 = df_churn_curated[col].quantile(0.75) # 75% data
    IQR = Q3 - Q1
    lower_bound = Q1 - (1.5 * IQR)
    upper_bound = Q3 + (1.5 * IQR)

    # Proses capping (winsorization)
    df_churn_curated[col] = (
        df_churn_curated[col]
        .clip(lower=lower_bound, upper=upper_bound)
    )

# ===== Eksplorasi Distribusi Outlier pada Fitur Numerik pasca Outlier Handling =====
# Visualisasi dengan box plot (untuk mendeteksi outlier)
fig, axes = plt.subplots(5, 3, figsize=(16, 10))
axes = axes.flatten()

for i, col in enumerate(numerical_cols):
    plot_box(
        data=df_churn_curated,
        x_col=col,
        y_col="Churn Label",
        title=f"{col} Distribution",
        xlabel="",
        ylabel="Churn Status",
        ax=axes[i]
    )

for i in range(-2, 0):
    axes[i].axis("off")
plt.tight_layout()
plt.show()

# ===== Log Transformation =====
# Memeriksa variabel/fitur dengan skewness > 0.66
MAX_SKEW_THRESHOLD = 0.66
skew_cols = df_churn_curated.skew(numeric_only=True)
df_skew_cols = pd.DataFrame(
    skew_cols[(abs(skew_cols) > MAX_SKEW_THRESHOLD) & (skew_cols.index != 'Churn Label')],
    columns=['Skewness']
).sort_values(
    by=['Skewness'], ascending=False
)

# Menerapkan log-transformation pada variabel/fitur dengan skewness > 0.66
print(f"Variabel/fitur dengan skewness > 0.66: \n\n{df_skew_cols}")
for col in df_skew_cols.index:
    df_churn_curated[col] = np.log1p(df_churn_curated[col])
    df_churn_curated[col] = df_churn_curated[col].replace([np.inf, -np.inf], np.nan)
    df_churn_curated[col] = df_churn_curated[col].fillna(0)

# ===== Data splitting menjadi 80% Data Latih + 20% Data Uji =====
# Mengambil rasio data uji
TEST_SIZE_RATIO = 0.2

# Proses data splitting
df_train, df_test = train_test_split(
    df_churn_curated,
    test_size=TEST_SIZE_RATIO,
    random_state=126,
    shuffle=True
)

df_train.reset_index(drop=True, inplace=True)
df_test.reset_index(drop=True, inplace=True)

# Menampilkan jumlah sampel pada data latih dan data uji
print(f"Jumlah sampel data latih: {len(df_train)} baris\nJumlah sampel data uji: {len(df_test)}")

# ===== Undersampling pada Kelas Mayoritas =====
# Menentukan fitur target
target_col = "Churn Label"

# Menghitung jumlah sampel terhadap setiap kelas pada fitur target
class_counts = df_train[target_col].value_counts()
major_class = class_counts.idxmax()
minor_class = class_counts.idxmin()

# Memisahkan dataset berdasarkan kelas
df_major = df_train[df_train[target_col] == major_class]
df_minor = df_train[df_train[target_col] == minor_class]

# Mendapatkan jumlah sampel pada setiap kelas
n_major = len(df_major)
n_minor = len(df_minor)

# Melakukan operasi undersampling pada dataset mayoritas
df_major_shuffled = df_major.sample(frac=1, random_state=126)

df_major_undersampled = df_major_shuffled.sample(
    n=n_minor,
    random_state=126
).reset_index(drop=True)

# Menggabungkan dataset mayoritas dengan dataset hasil undersampling
df_train_undersampled = pd.concat(
    [df_minor, df_major_undersampled],
    ignore_index=True
)

# Shuffling pada data latih hasil undersampling
df_train_undersampled = df_train_undersampled.sample(
    frac=1, random_state=126
).reset_index(drop=True)

# Menampilkan distribusi data latih setelah undersampling
print("Distribusi Data Latih seteleh Undersampling")
print(df_train_undersampled["Churn Label"].value_counts())

# ===== Encoding Fitur Prediktor dengan One-Hot Encoder =====
# Fungsi helper untuk One-Hot Encoding
def encode_cat_col(columns, df, df_test=None):
    if df_test is not None:
        df = df.copy()
        df_test = df_test.copy()

        for column in columns:
            ohe = pd.get_dummies(df[column], prefix=column).astype(int)
            df = pd.concat([df.drop(columns=[column]), ohe], axis=1)

            ohe_test = pd.get_dummies(df_test[column], prefix=column).astype(int)
            df_test = pd.concat([df_test.drop(columns=[column]), ohe_test], axis=1)

            # Menyamakan struktur kolom train-test
            df, df_test = df.align(df_test, join="left", axis=1, fill_value=0)
            feature_columns = df.columns.tolist()
            joblib.dump(feature_columns, "preprocessing/encoder/predictor_encoder.joblib")

        return df, df_test

    else:
        df = df.copy()
        for column in columns:
            ohe = pd.get_dummies(df[column], prefix=column).astype(int)
            df = pd.concat([df.drop(columns=[column]), ohe], axis=1)
            feature_columns = df.columns.tolist()
            joblib.dump(feature_columns, "preprocessing/encoder/predictor_encoder.joblib")

        return df


# Menjalankan proses encoding fitur prediktor pada data latih dan data uji
categorical_cols = [
    'Under 30', 'Senior Citizen', 'Married', 'Dependents',
    'Offer', 'Multiple Lines', 'Internet Service', 'Internet Type',
    'Online Security', 'Online Backup', 'Device Protection Plan', 'Premium Tech Support',
    'Streaming TV', 'Streaming Movies', 'Streaming Music', 'Unlimited Data',
    'Contract', 'Paperless Billing', 'Payment Method'
]

df_train_new, df_test_new = encode_cat_col(
    columns=categorical_cols,
    df=df_train_undersampled,
    df_test=df_test
)

# ===== Encoding Fitur Target dengan Label Encoder =====
# Fungsi helper untuk Label Encoding
def encode_target_col(target_column, df, df_test=None):
    if df_test is not None:
        df = df.copy()
        df_test = df_test.copy()

        le = LabelEncoder()
        df[target_column] = le.fit_transform(df[target_column])
        joblib.dump(le, "preprocessing/encoder/target_encoder.joblib")

        df_test[target_column] = le.transform(df_test[target_column])

        return df, df_test

    else:
        df = df.copy()
        le = LabelEncoder()
        df[target_column] = le.fit_transform(df[target_column])
        joblib.dump(le, "preprocessing/encoder/target_encoder.joblib")

        return df


# Meng-encode fitur target `Churn Label` pada data latih dan data uji
df_train_new, df_test_new = encode_target_col(
    target_column="Churn Label",
    df=df_train_new,
    df_test=df_test_new
)

# ===== Standarisasi Nilai dengan StandardScaler =====
# Fungsi helper untuk standarisasi fitur numerik
def scale_num_col(columns, df, df_test=None):
    if df_test is not None:
        df = df.copy()
        df_test = df_test.copy()
        for column in columns:
            scaler = StandardScaler()
            X = np.asanyarray(df[column])
            X = X.reshape(-1, 1)
            scaler.fit(X)
            df[column] = scaler.transform(X)
            joblib.dump(scaler, f"preprocessing/scaler/{column}_scaler.joblib")

            X_test = np.asanyarray(df_test[column])
            X_test = X_test.reshape(-1, 1)
            df_test[column] = scaler.transform(X_test)
        return df, df_test

    else:
        df = df.copy()
        for column in columns:
            scaler = StandardScaler()
            X = np.asanyarray(df[column])
            X = X.reshape(-1, 1)
            scaler.fit(X)
            df[column] = scaler.transform(X)
            joblib.dump(scaler, f"preprocessing/scaler/{column}_scaler.joblib")
        return df


# Menjalankan proses standarisasi pada data latih dan data uji
df_train_new, df_test_new = scale_num_col(
    columns=numerical_cols,
    df=df_train_new,
    df_test=df_test_new
)

# ===== Reduksi Dimensi dengan PCA =====
# Membuat dataframe baru bernama `df_train_pca` dan `df_test_pca`
df_train_pca = df_train_new.copy().reset_index(drop=True)
df_test_pca = df_test_new.copy().reset_index(drop=True)

# Menentukan fitur dengan tingkat multikolinearitas > 0.6
pca_cols = [
    'Total Revenue', 'Total Charges', 'Tenure in Months',
    'Total Long Distance Charges', 'Monthly Charge'
]

# Melakukan eksperimen PCA pada fitur dengan multikolinearitas > 0.6
pca_experimental = PCA(n_components=len(pca_cols), random_state=126)
pca_experimental.fit(df_train_new[pca_cols])
princ_comp = pca_experimental.transform(df_train_new[pca_cols])

# Visualisasi hasil eksperimen PCA pada fitur dengan multikolinearitas > 0.6
var_exp = pca_experimental.explained_variance_ratio_.round(3)
cum_var_exp = np.cumsum(var_exp)

plt.bar(range(len(pca_cols)), var_exp, alpha=0.5, align='center', label='Individual explained variance')
plt.step(range(len(pca_cols)), cum_var_exp, where='mid', label='Cumulative explained variance')
plt.ylabel('Explained variance ratio')
plt.xlabel('Principal component index')
plt.legend(loc='best')
plt.show()

# Menerapkan PCA pada data latih dan data uji
pca = PCA(n_components=2, random_state=126)
pca.fit(df_train_pca[pca_cols])
joblib.dump(pca, "preprocessing/pca/pca.joblib")

train_princ_comp = pca.transform(df_train_pca[pca_cols])
df_train_pca[["Pc_1", "Pc_2"]] = pd.DataFrame(train_princ_comp, columns=["Pc_1", "Pc_2"])
df_train_pca.drop(columns=pca_cols, axis=1, inplace=True)

test_princ_comp = pca.transform(df_test_pca[pca_cols])
df_test_pca[["Pc_1", "Pc_2"]] = pd.DataFrame(test_princ_comp, columns=["Pc_1", "Pc_2"])
df_test_pca.drop(columns=pca_cols, axis=1, inplace=True)

# Menampilkan data latih hasil PCA
df_train_pca.head()

# Menampilkan data uji hasil PCA
df_test_pca.head()

# ===== Meng-export dataset hasil transformasi =====
# Membuat fungsi helper untuk menyimpan dataframe ke dalam file .csv
def save_to_csv(data, file_path):
    try:
        data.to_csv(file_path, index=False)
    except Exception as error:
        print(f"Terjadi error saat menyimpan data: {error}")
    else:
        print(f"Data berhasil disimpan sebagai: {file_path}")


# Menyiapkan direktori file untuk menyimpan dataframe ke dalam file .csv
train_pca_file = "preprocessing/telco_preprocessing/train_pca.csv"
test_pca_file = "preprocessing/telco_preprocessing/test_pca.csv"

# Menyimpan dataframe `df_train_pca` dan `df_test_pca` sebagai file .csv
save_to_csv(df_train_pca, train_pca_file)
save_to_csv(df_test_pca, test_pca_file)