import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import geobr
import pkg_resources
import streamlit as st
from babel.numbers import format_currency
import requests
import pickle
sns.set(style='dark')

def dataset(url):
    response = requests.get(url)
    response.raise_for_status()  
    df = pickle.loads(response.content)
    return df

def demografi_dan_typepayment_dan_statusorder(df, kolom):
    df = df[f'{kolom}'].value_counts().head(5)
    return df

def top_kategori_produk(df, kolom):
    df = df[f'{kolom}'].value_counts().head(5)
    return df

def bottom_kategori_produk(df, kolom):
    df = df[f'{kolom}'].value_counts().tail(5).sort_values(ascending=True)
    return df

def order_dan_revenue(df, time):
    monthly_orders_df = df.resample(rule='ME', on=f'{time}').agg({
        "order_id": "nunique",
        "price": "sum"
    })
    monthly_orders_df.index = monthly_orders_df.index.strftime('%B') #mengubah format order date menjadi Tahun-Bulan
    monthly_orders_df = monthly_orders_df.reset_index()
    monthly_orders_df.rename(columns={
        "order_id": "order_count",
        "price": "revenue"
    }, inplace=True)
    return monthly_orders_df

def rmf(df, df2, mak, order):
    df[f'{mak}'] = df[f'{mak}'].dt.date
    recent_date = df2[f'{order}'].dt.date.max()
    df["recency"] = df[f'{mak}'].apply(lambda x: (recent_date - x).days)
    df = df.drop(f'{mak}', axis=1)
    return df

def plot_rmf(df):
    fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(30, 8))
    colors_ = ["#72BCD4"] + ["#D3D3D3"] * (len(df) - 1)
    sns.barplot(y="recency", x="customer_id", data=df.sort_values(by="recency", ascending=False).head(5), palette=colors_, ax=ax[0])
    ax[0].set_ylabel(None)
    ax[0].set_xlabel("customer_id", fontsize=30)
    ax[0].set_title("By Recency (days)", loc="center", fontsize=45)
    ax[0].tick_params(axis ='x', labelsize=15, rotation=90)

    sns.barplot(y="frequency", x="customer_id", data=df.sort_values(by="frequency", ascending=False).head(5), palette=colors_, ax=ax[1])
    ax[1].set_ylabel(None)
    ax[1].set_xlabel("customer_id", fontsize=30)
    ax[1].set_title("By Frequency", loc="center", fontsize=45)
    ax[1].tick_params(axis ='x', labelsize=15, rotation=90)

    sns.barplot(y="monetary", x="customer_id", data=df.sort_values(by="monetary", ascending=False).head(5), palette=colors_, ax=ax[2])
    ax[2].set_ylabel(None)
    ax[2].set_xlabel("customer_id", fontsize=30)
    ax[2].set_title("By Monetary", loc="center", fontsize=45)
    ax[2].tick_params(axis ='x', labelsize=15, rotation=90)

    return fig

def cluster_rmf(df, col1,col2,col3):
    df['r_rank'] = df[f'{col1}'].rank(ascending=False)
    df['f_rank'] = df[f'{col2}'].rank(ascending=True)
    df['m_rank'] = df[f'{col3}'].rank(ascending=True)

    df['r_rank_norm'] = (df['r_rank']/df['r_rank'].max())*100
    df['f_rank_norm'] = (df['f_rank']/df['f_rank'].max())*100
    df['m_rank_norm'] = (df['m_rank']/df['m_rank'].max())*100
    
    df.drop(columns=['r_rank', 'f_rank', 'm_rank'], inplace=True)

    df['RFM_score'] = 0.15*df['r_rank_norm']+0.28 * \
    df['f_rank_norm']+0.57*df['m_rank_norm']
    df['RFM_score'] *= 0.05
    df = df.round(2)
   
    df["customer_segment"] = np.where(
      df['RFM_score'] > 4.5, "Top customers", (np.where(
        df['RFM_score'] > 4, "High value customer",(np.where(
            df['RFM_score'] > 3, "Medium value customer", np.where(
                df['RFM_score'] > 1.6, 'Low value customers', 'lost customers'))))))

    df = df['customer_segment'].value_counts()
    return df

def geo_state(df, col):
    df_states= df[f'{col}'].value_counts().reset_index()
    df_states.columns = ['state', 'customer_count']

    states = geobr.read_state(year=2019)

    states["abbrev_state"] = states["abbrev_state"].str.lower()
    df_states["state"] = df_states["state"].str.lower()
    brasil = states.merge(df_states, how="left", left_on="abbrev_state", right_on="state")
    return brasil
    #plt.rcParams.update({"font.size": 5})

def plot_geo(df):
    plt.rcParams.update({"font.size": 2})
    fig, ax = plt.subplots(figsize=(4, 4), dpi=300)
    df.plot(
        column="customer_count",
        cmap="inferno",
        legend=True,
        legend_kwds={
            "label": "Parameter",
            "orientation": "horizontal",
            "shrink": 0.6
        },
        ax=ax,)
    ax.axis("off")
    return fig


def geo_city(df, city):
    df_cities = df[f'{city}'].value_counts().reset_index()
    df_cities.columns = ['customer_city', 'customer_count']

    all_city = geobr.read_municipality(year=2019)

    all_city["name_muni"] = all_city["name_muni"].str.replace("'", '')
    all_city["name_muni"] = all_city["name_muni"].str.lower()
    df_cities["customer_city"] = df_cities["customer_city"].str.lower()
    all_city = all_city.merge(df_cities, how="left", left_on="name_muni", right_on="customer_city")
    return all_city

def sao(df, col):
    cities = df[f'{col}'].value_counts().reset_index()
    cities.columns = ['customer_city', 'customer_count']

    sao = geobr.read_municipality(code_muni="SP", year=2019)

    sao["name_muni"] = sao["name_muni"].str.replace("'", '')
    sao["name_muni"] = sao["name_muni"].str.lower()
    cities["customer_city"] = cities["customer_city"].str.lower()

    sao = sao.merge(cities, how="left", left_on="name_muni", right_on="customer_city")
    return sao

def cluster(df, cust):
    jumlah_transaksi_per_pelanggan = df[f'{cust}'].value_counts().reset_index()
    jumlah_transaksi_per_pelanggan.columns = ['customer_id', 'jumlah_transaksi']

    # Tetapkan batasan atau kriteria untuk klaster
    bins = [0, 5, 15, 20, 25, float('inf')]
    labels = ['Low', 'Medium', 'High', 'Very High', 'Extremely High']

    # Buat klaster pelanggan berdasarkan kriteria
    jumlah_transaksi_per_pelanggan['klaster'] = pd.cut(jumlah_transaksi_per_pelanggan['jumlah_transaksi'], bins=bins, labels=labels)
    df = jumlah_transaksi_per_pelanggan['klaster'].value_counts()
    return df

# Fungsi untuk plot
def bar_chart(df):
    fig, ax = plt.subplots(figsize=(25, 15))
    colors_ = ["#72BCD4"] + ["#D3D3D3"] * (len(df) - 1)
    sns.barplot(x=df.values, y=df.index, palette=colors_, dodge=False)
    ax.set_xlabel(None)
    ax.set_ylabel(None)
    ax.tick_params(axis='x', labelsize=30)
    ax.tick_params(axis='y', labelsize=30)
    return fig

def line_chart(df, time, count):
    fig, ax = plt.subplots(figsize=(25, 15))
    ax.plot(
        df[f'{time}'],
        df[f'{count}'],
        marker='o', 
        linewidth=5,
        color='#72BCD4'
    )
    ax.tick_params(axis='y', labelsize=30)
    ax.tick_params(axis='x', labelsize=30, rotation=45)
    return fig

df_all = dataset('https://github.com/RiyZ411/Proyek-Analisis-Data-LaskarAi/raw/refs/heads/main/dashboard/Data%20Clean/df_all.pkl')
rmf_df = dataset('https://github.com/RiyZ411/Proyek-Analisis-Data-LaskarAi/raw/refs/heads/main/dashboard/Data%20Clean/rfm_df.pkl')


min_date = df_all["order_purchase_timestamp"].min()
max_date = df_all["order_purchase_timestamp"].max()
 
with st.sidebar:
    # Menambahkan logo perusahaan
    st.image("https://github.com/dicodingacademy/assets/raw/main/logo.png")
    
    # Mengambil start_date & end_date dari date_input
    start_date, end_date = st.date_input(
        label='Rentang Waktu',min_value=min_date,
        max_value=max_date,
        value=[min_date, max_date]
    )

df_all = df_all[(df_all["order_purchase_timestamp"] >= str(start_date)) & 
                (df_all["order_purchase_timestamp"] <= str(end_date))]

# Pertanyaan 1
state = demografi_dan_typepayment_dan_statusorder(df_all, 'customer_state')
city = demografi_dan_typepayment_dan_statusorder(df_all, 'customer_city')

# Pertanyaan 2
top_kategori = top_kategori_produk(df_all, 'product_category_name')
bottom_kategori = bottom_kategori_produk(df_all, 'product_category_name')

# Pertanyaan 3
top_produk = top_kategori_produk(df_all, 'product_id')
bottom_produk = bottom_kategori_produk (df_all, 'product_id')

# Pertanyaan 4
order_revenue = order_dan_revenue(df_all, 'order_purchase_timestamp')

# Pertanyaan 5
payment = demografi_dan_typepayment_dan_statusorder(df_all,'payment_type')

# Pertanyaan 6
status = demografi_dan_typepayment_dan_statusorder(df_all,'order_status')

# Analisis lanjutan (RMF, Geospatial, Clustering)
# RMF
rmf = rmf(rmf_df, df_all, 'max_order_timestamp', 'order_purchase_timestamp')
klaster_rmf = cluster_rmf(rmf, 'recency', 'frequency', 'monetary')

# Geospatial
states = geo_state(df_all, 'customer_state')
cities = geo_city(df_all, 'customer_city')
saop = sao(df_all, 'customer_city')

#cluster
klaster = cluster(df_all, 'customer_id')

# Dasboard streamlit
st.header('Dicoding Collection Dashboard :sparkles:')
st.write("")
st.subheader("Demografi Customer")
st.write("Berdasarkan Negara Bagian")
state = bar_chart(state)
st.pyplot(state)

st.write("")
st.write("Berdasarkan Kota")
city = bar_chart(city)
st.pyplot(city)

st.write("")
st.subheader("Kategori Produk Paling Banyak dan Paling Sedikit Terjual")
st.write("Kategori Produk Yang Banyak Terjual")
top_kategori = bar_chart(top_kategori)
st.pyplot(top_kategori)

st.write("")
st.write("Kategori Produk Yang Sedikit Terjual")
bottom_kategori = bar_chart(bottom_kategori)
st.pyplot(bottom_kategori)

st.write("")
st.subheader("Produk Paling Banyak dan Paling Sedikit Terjual")
st.write("Produk Yang Banyak Terjual")
top_produk = bar_chart(top_produk)
st.pyplot(top_produk)

st.write("")
st.write("Produk Yang Sedikit Terjual")
bottom_produk = bar_chart(bottom_produk)
st.pyplot(bottom_produk)

st.write("")
st.subheader('Monthly Orders')
col1, col2 = st.columns(2)
with col1:
    total_orders = order_revenue.order_count.sum()
    st.metric("Total orders", value=total_orders)
 
with col2:
    total_revenue = format_currency(order_revenue.revenue.sum(), "AUD", locale='es_CO') 
    st.metric("Total Revenue", value=total_revenue)
order_revenue = line_chart(order_revenue, 'order_purchase_timestamp', 'order_count')
st.pyplot(order_revenue)

st.write("")
st.subheader("Type Pembayaran Yang Paling Banyak Digunakan Customer")
payment = bar_chart(payment)
st.pyplot(payment)

st.write("")
st.subheader("Status Order")
status = bar_chart(status)
st.pyplot(status)

st.write("")
st.subheader("RMF Parameter")
st.write("Customer Terbaik Berdasarkan RFM Parameters")
col1, col2, col3 = st.columns(3)
with col1:
    avg_recency = round(rmf.recency.mean(), 1)
    st.metric("Average Recency (days)", value=avg_recency)
 
with col2:
    avg_frequency = round(rmf.frequency.mean(), 2)
    st.metric("Average Frequency", value=avg_frequency)
 
with col3:
    avg_frequency = format_currency(rmf.monetary.mean(), "AUD", locale='es_CO') 
    st.metric("Average Monetary", value=avg_frequency)
rmf = plot_rmf(rmf)
st.pyplot(rmf)

st.write("Cluster Customer Berdasarkan RFM Parameters")
klaster_rmf = bar_chart(klaster_rmf)
st.pyplot(klaster_rmf)

st.write("")
st.subheader("Geospasial Negara Bagian di Brasil")
states = plot_geo(states)
st.pyplot(states)

st.write("")
st.subheader("Geospasial Kota di Brasil")
cities = plot_geo(cities)
st.pyplot(cities)

st.write("")
st.subheader("Geospasial Negara Bagian Sao Paulo")
sao = plot_geo(saop)
st.pyplot(sao)

st.write("")
st.subheader("Clustering Customer")
klaster = bar_chart(klaster)
st.pyplot(klaster)