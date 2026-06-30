import os
import re
import json
from collections import Counter
import pandas as pd
import numpy as np
import tensorflow as tf
from flask import Flask, jsonify, render_template, request

app = Flask(__name__)

# Tentukan path model dan data
MODEL_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_POS_PATH = os.path.join(MODEL_DIR, 'model_cnn_lstm_positif.h5')
MODEL_NEG_PATH = os.path.join(MODEL_DIR, 'model_cnn_lstm_negatif.h5')
DEFAULT_DATASET_PATH = os.path.join(MODEL_DIR, 'dataset baru.csv')
CUSTOM_DATASET_PATH = os.path.join(MODEL_DIR, 'labeled_data_custom.csv')
CUSTOM_METADATA_PATH = os.path.join(MODEL_DIR, 'custom_metadata.json')

def get_active_dataset_path():
    if os.path.exists(CUSTOM_DATASET_PATH):
        return CUSTOM_DATASET_PATH
    return DEFAULT_DATASET_PATH

def get_dataset_info():
    if os.path.exists(CUSTOM_DATASET_PATH):
        original_name = 'labeled_data_custom.csv'
        if os.path.exists(CUSTOM_METADATA_PATH):
            try:
                with open(CUSTOM_METADATA_PATH, 'r') as f:
                    meta = json.load(f)
                    original_name = meta.get('original_name', 'labeled_data_custom.csv')
            except Exception:
                pass
        return {'type': 'custom', 'filename': original_name}
    return {'type': 'default', 'filename': 'dataset baru.csv'}

# Variabel global untuk model
model_pos = None
model_neg = None

def load_models():
    global model_pos, model_neg
    try:
        print("--> Memuat model positif dari:", MODEL_POS_PATH)
        model_pos = tf.keras.models.load_model(MODEL_POS_PATH)
        print("--> Memuat model negatif dari:", MODEL_NEG_PATH)
        model_neg = tf.keras.models.load_model(MODEL_NEG_PATH)
        print("--> Berhasil memuat semua model.")
    except Exception as e:
        print("--> GAGAL memuat model:", str(e))

# Muat model saat startup
load_models()

def get_processed_data():
    """
    Membaca dataset aktif (default atau custom), melakukan agregasi harian,
    dan menghitung persentase sentimen.
    """
    path = get_active_dataset_path()
    if not os.path.exists(path):
        raise FileNotFoundError(f"Dataset tidak ditemukan di: {path}")
        
    df = pd.read_csv(path)
    
    # Pemetaan kolom kustom (skema baru: text, createTimeISO, sentiment)
    if 'createTimeISO' in df.columns:
        df['date'] = df['createTimeISO']
    
    if 'platform' not in df.columns:
        df['platform'] = 'Media Sosial'
        
    if 'final_text' not in df.columns and 'text' in df.columns:
        df['final_text'] = df['text']
        
    # Pastikan kolom date ada
    if 'date' not in df.columns:
        raise ValueError("Kolom tanggal ('date' atau 'createTimeISO') wajib ada dalam CSV.")
        
    # Pastikan kolom text ada
    if 'text' not in df.columns:
        raise ValueError("Kolom 'text' wajib ada dalam CSV.")
        
    # Gunakan sentimen dari CSV jika tersedia, jika tidak gunakan analisis leksikon
    if 'sentiment' in df.columns:
        label_mapping = {
            'Negative': 'Negatif',
            'Positive': 'Positif',
            'Neutral': 'Netral',
            'Negatif': 'Negatif',
            'Positif': 'Positif',
            'Netral': 'Netral'
        }
        df['sentiment'] = df['sentiment'].map(label_mapping).fillna('Netral')
    else:
        df['sentiment'] = df['text'].apply(analyze_sentiment)

    # Paksa sentimen negatif jika teks mengandung kata umpatan "tai"/"tae" (dan variasinya)
    def check_swear(row):
        txt = str(row['text']).lower() if isinstance(row['text'], str) else ""
        if re.search(r'\b(t+a*i+|t+a*e+h*)\b', txt):
            return 'Negatif'
        return row['sentiment']
    df['sentiment'] = df.apply(check_swear, axis=1)
    
    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    df = df.dropna(subset=['date'])
    df['date_only'] = df['date'].dt.date
    
    # Agregasi harian berdasarkan tipe sentimen
    daily = df.groupby(['date_only', 'sentiment']).size().unstack(fill_value=0).reset_index()
    
    # Pastikan ketiga kolom sentimen ada
    for col in ['Negatif', 'Netral', 'Positif']:
        if col not in daily.columns:
            daily[col] = 0
            
    daily = daily.sort_values('date_only').reset_index(drop=True)
    daily['Total'] = daily['Positif'] + daily['Negatif'] + daily['Netral']
    
    # Cegah pembagian dengan nol
    daily['Total'] = daily['Total'].replace(0, 1)
    
    # Hitung persentase sentimen (skala 0 - 1)
    daily['Pos_Pct'] = daily['Positif'] / daily['Total']
    daily['Neg_Pct'] = daily['Negatif'] / daily['Total']
    daily['Neu_Pct'] = daily['Netral'] / daily['Total']
    
    # Hitung moving average 7 hari
    daily['Pos_Pct_Smooth'] = daily['Pos_Pct'].rolling(window=7, min_periods=1).mean()
    daily['Neg_Pct_Smooth'] = daily['Neg_Pct'].rolling(window=7, min_periods=1).mean()
    daily['Neu_Pct_Smooth'] = daily['Neu_Pct'].rolling(window=7, min_periods=1).mean()
    
    return df, daily

def get_top_topics(df, limit=5):
    all_texts = df['final_text'].dropna().astype(str).values
    extra_stopwords = {
        'yang', 'dan', 'di', 'ke', 'dari', 'untuk', 'dengan', 'ini', 'itu', 'ada', 
        'bisa', 'saya', 'kami', 'mereka', 'dia', 'adalah', 'akan', 'telah', 'sudah', 
        'oleh', 'pada', 'juga', 'atau', 'tidak', 'ya', 'saja', 'bahwa', 'sebagai', 
        'lah', 'kah', 'tapi', 'namun', 'yg', 'dgn', 'dr', 'aja', 'kalo', 'kalau', 
        'ga', 'gak', 'bukan', 'sangat', 'secara', 'karena', 'bila', 'jika', 'sehingga', 
        'sebelum', 'setelah', 'saat', 'ketika', 'dalam', 'tentang', 'dan', 'with', 
        'program', 'makan', 'gratis', 'bergizi', 'mbg', 'anak', 'sekolah', 'bagi'
    }
    
    word_counter = Counter()
    for text in all_texts:
        words = re.findall(r'\b[a-zA-Z]{4,15}\b', text.lower())
        filtered_words = [w for w in words if w not in extra_stopwords]
        word_counter.update(filtered_words)
        
    top_words = word_counter.most_common(limit)
    topics = []
    topic_mapping = {
        'gizi': 'Kualitas Gizi',
        'distribusi': 'Distribusi Makanan',
        'anggaran': 'Anggaran APBN',
        'susu': 'Pembagian Susu',
        'daerah': 'Pemerataan Daerah',
        'menu': 'Variasi Menu',
        'uji': 'Tahap Uji Coba',
        'pemerintah': 'Dukungan Pemerintah',
        'sehat': 'Kesehatan Anak',
        'masyarakat': 'Respon Masyarakat'
    }
    
    max_vol = top_words[0][1] if top_words else 1
    for word, count in top_words:
        mapped_name = topic_mapping.get(word, word.capitalize())
        is_neg = word in ['anggaran', 'daerah', 'mahal', 'biaya', 'korupsi']
        color = '#E24B4A' if is_neg else '#639922'
        pct = (count / max_vol) * 100
        topics.append({
            'name': mapped_name,
            'vol': count,
            'color': color,
            'pct': pct
        })
    return topics

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/dashboard')
def api_dashboard():
    try:
        df, daily = get_processed_data()
        if daily is None or df is None:
            return jsonify({'error': 'Gagal memproses data'}), 500
            
        latest_day = daily.iloc[-1]
        
        # Hitung rata-rata mingguan sentimen harian historis untuk blended seasonality
        daily_temp = daily.copy()
        daily_temp['date_dt'] = pd.to_datetime(daily_temp['date_only'])
        daily_temp['day_of_week'] = daily_temp['date_dt'].dt.weekday
        mean_pos = daily_temp.groupby('day_of_week')['Pos_Pct'].mean().to_dict()
        mean_neg = daily_temp.groupby('day_of_week')['Neg_Pct'].mean().to_dict()

        # Siapkan input history 14 hari terakhir
        pos_history = list(daily['Pos_Pct'].tail(14).values)
        neg_history = list(daily['Neg_Pct'].tail(14).values)

        forecast_steps = 30  # Tepat 30 Hari Ke Depan
        forecast_pos = []
        forecast_neg = []
        
        cur_pos_seq = pos_history.copy()
        cur_neg_seq = neg_history.copy()
        
        alpha = 0.4
        last_date = daily['date_only'].iloc[-1]
        last_date_dt = pd.to_datetime(last_date)
        
        forecast_dates_display = []
        
        # Lakukan loop ramalan dengan model CNN-LSTM
        for i in range(1, forecast_steps + 1):
            in_pos = np.array(cur_pos_seq[-14:]).reshape(1, 14, 1)
            in_neg = np.array(cur_neg_seq[-14:]).reshape(1, 14, 1)
            
            p_pred = 0.0
            n_pred = 0.0
            
            if model_pos is not None:
                p_pred = float(model_pos.predict(in_pos, verbose=0)[0, 0])
            else:
                p_pred = cur_pos_seq[-1] * 1.01
                
            if model_neg is not None:
                n_pred = float(model_neg.predict(in_neg, verbose=0)[0, 0])
            else:
                n_pred = cur_neg_seq[-1] * 0.99
                
            forecast_date = last_date_dt + pd.Timedelta(days=i)
            day_of_week = forecast_date.weekday()
            
            # Blended Seasonality (Model + Rata-rata Mingguan rill)
            p_pred_final = alpha * p_pred + (1 - alpha) * mean_pos.get(day_of_week, 0.18)
            n_pred_final = alpha * n_pred + (1 - alpha) * mean_neg.get(day_of_week, 0.60)
            
            p_pred_final = max(0.0, min(1.0, p_pred_final))
            n_pred_final = max(0.0, min(1.0, n_pred_final))
            
            forecast_pos.append(p_pred_final)
            forecast_neg.append(n_pred_final)
            
            cur_pos_seq.append(p_pred_final)
            cur_neg_seq.append(n_pred_final)
            
            forecast_dates_display.append(forecast_date.strftime('%d %b'))

        # Hitung tren perubahan dibanding hari sebelumnya
        if len(daily) > 1:
            pos_change_up = latest_day['Pos_Pct'] > daily['Pos_Pct'].iloc[-2]
            pos_change_diff = abs(latest_day['Pos_Pct'] - daily['Pos_Pct'].iloc[-2]) * 100
            pos_change_str = f"{'+' if pos_change_up else '-'}{pos_change_diff:.1f}%"
            
            neg_change_up = latest_day['Neg_Pct'] > daily['Neg_Pct'].iloc[-2]
            neg_change_diff = abs(latest_day['Neg_Pct'] - daily['Neg_Pct'].iloc[-2]) * 100
            neg_change_str = f"{'+' if neg_change_up else '-'}{neg_change_diff:.1f}%"
        else:
            pos_change_up = True
            pos_change_str = "0.0%"
            neg_change_up = False
            neg_change_str = "0.0%"

        # Format historical
        hist_dates = [d.strftime('%d %b') for d in daily['date_only']]
        
        # Ambil feed komentar terbaru (4 komentar)
        recent_comments_df = df.sort_values('date', ascending=False).head(4)
        recent_comments = []
        for _, r in recent_comments_df.iterrows():
            sentiment_class = 'pill-pos' if r['sentiment'] == 'Positif' else ('pill-neg' if r['sentiment'] == 'Negatif' else 'pill-neu')
            text_short = str(r['text'])[:110] + '...' if len(str(r['text'])) > 110 else str(r['text'])
            date_str = pd.to_datetime(r['date']).strftime('%d %b %H:%M')
            recent_comments.append({
                'platform': str(r['platform']).capitalize(),
                'date': date_str,
                'sentiment': r['sentiment'],
                'sentiment_class': sentiment_class,
                'text': text_short
            })

        # Topik teratas
        topics = get_top_topics(df, limit=5)
        
        return jsonify({
            'dataset_info': get_dataset_info(),
            'history': {
                'dates_display': hist_dates,
                'pos_raw': [float(x * 100) for x in daily['Pos_Pct']],
                'neg_raw': [float(x * 100) for x in daily['Neg_Pct']],
                'pos_smooth': [float(x * 100) for x in daily['Pos_Pct_Smooth']],
                'neg_smooth': [float(x * 100) for x in daily['Neg_Pct_Smooth']]
            },
            'forecast': {
                'dates_display': forecast_dates_display,
                'pos': [float(x * 100) for x in forecast_pos],
                'neg': [float(x * 100) for x in forecast_neg]
            },
            'metrics': {
                'total_mention': int(len(df)),
                'latest_pos': f"{latest_day['Pos_Pct']*100:.1f}%",
                'latest_neg': f"{latest_day['Neg_Pct']*100:.1f}%",
                'latest_neu': f"{latest_day['Neu_Pct']*100:.1f}%",
                'model_accuracy': "96.6%",
                'f1_score': "0.95",
                'pos_change': pos_change_str,
                'pos_change_up': bool(pos_change_up),
                'neg_change': neg_change_str,
                'neg_change_up': bool(neg_change_up)
            },
            'donut': {
                'labels': ['Positif', 'Negatif', 'Netral'],
                'values': [
                    float(latest_day['Pos_Pct'] * 100),
                    float(latest_day['Neg_Pct'] * 100),
                    float(latest_day['Neu_Pct'] * 100)
                ]
            },
            'topics': topics,
            'recent_comments': recent_comments
        })
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

@app.route('/api/custom-forecast', methods=['POST'])
def api_custom_forecast():
    try:
        data = request.get_json()
        pos_arr = np.array(data.get('pos_history', []), dtype=np.float32)
        neg_arr = np.array(data.get('neg_history', []), dtype=np.float32)
        
        if len(pos_arr) != 14 or len(neg_arr) != 14:
            return jsonify({'error': 'Panjang history harus tepat 14 hari.'}), 400
            
        pos_scaled = pos_arr / 100.0 if np.max(pos_arr) > 1.0 else pos_arr
        neg_scaled = neg_arr / 100.0 if np.max(neg_arr) > 1.0 else neg_arr
        
        in_pos = pos_scaled.reshape(1, 14, 1)
        in_neg = neg_scaled.reshape(1, 14, 1)
        
        pred_pos = 0.0
        pred_neg = 0.0
        
        if model_pos is not None:
            pred_pos = float(model_pos.predict(in_pos, verbose=0)[0, 0])
        else:
            pred_pos = float(pos_scaled[-1] * 1.01)
            
        if model_neg is not None:
            pred_neg = float(model_neg.predict(in_neg, verbose=0)[0, 0])
        else:
            pred_neg = float(neg_scaled[-1] * 0.99)
            
        pred_pos_pct = max(0.0, min(100.0, pred_pos * 100.0))
        pred_neg_pct = max(0.0, min(100.0, pred_neg * 100.0))
        
        return jsonify({
            'next_pos': pred_pos_pct,
            'next_neg': pred_neg_pct,
            'message': 'Simulasi ramalan berhasil dihitung.'
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# Lexicon-based Indonesian sentiment analysis for MBG comments
INDONESIAN_POSITIVE_WORDS = {
    'enak', 'mantap', 'mntp', 'bagus', 'bgs', 'setuju', 'senang', 'suka', 'bergizi', 'sehat', 
    'bantu', 'membantu', 'terima kasih', 'makasih', 'mksih', 'alhamdulillah', 'mantul', 
    'lezat', 'apresiasi', 'hebat', 'dukung', 'mendukung', 'top', 'keren', 'manfaat', 
    'bermanfaat', 'lancar', 'bersyukur', 'bersukur', 'untung', 'hebat', 'setuju', 
    'sip', 'oke', 'ok', 'good', 'nice', 'lajut', 'lanjut', 'lanjutkan', 'terimakasih'
}

INDONESIAN_NEGATIVE_WORDS = {
    'gagal', 'jelek', 'buruk', 'kecewa', 'rugi', 'merugikan', 'basi', 'monoton', 'lambat', 
    'terhambat', 'mahal', 'biaya', 'anggaran', 'korupsi', 'sepi', 'tolak', 'menolak', 
    'tidak setuju', 'gak setuju', 'susah', 'sulit', 'protes', 'sedikit', 'kurang', 
    'amburadul', 'keterlaluan', 'kecewa', 'bohong', 'hoax', 'rugi', 'sayang', 'sayangnya', 
    'potong', 'dipotong', 'korup', 'hambat', 'lambat', 'henti', 'hentikan', 'stop',
    'hancur', 'hancoor', 'ancur', 'ancoor', 'ancooor', 'morat', 'marit', 'miskin', 'melarat', 'habis', 'utang', 'hutang'
}

def analyze_sentiment(text):
    if not isinstance(text, str):
        return 'Netral'
    text_lower = text.lower()
    
    # Paksa sentimen negatif untuk kata umpatan seperti "tai"/"tae" (dan variasinya)
    if re.search(r'\b(t+a*i+|t+a*e+h*)\b', text_lower):
        return 'Negatif'
    
    # Tokenize sederhana menggunakan regex
    words = re.findall(r'\b\w+\b', text_lower)
    
    pos_score = sum(1 for w in words if w in INDONESIAN_POSITIVE_WORDS)
    neg_score = sum(1 for w in words if w in INDONESIAN_NEGATIVE_WORDS)
    
    # Tambahan pengecekan frasa khusus
    if 'tidak setuju' in text_lower or 'ga setuju' in text_lower or 'gak setuju' in text_lower:
        neg_score += 1.5
    if 'terima kasih' in text_lower or 'terimakasih' in text_lower:
        pos_score += 1.5
    if 'gagal total' in text_lower:
        neg_score += 1.5
        
    if pos_score > neg_score:
        return 'Positif'
    elif neg_score > pos_score:
        return 'Negatif'
    else:
        return 'Netral'


@app.route('/api/upload-dataset', methods=['POST'])
def api_upload_dataset():
    try:
        # Periksa apakah ada file dalam request
        if 'file' not in request.files:
            # "ketika input tidak diisi maka tetap dataset yang lama"
            return jsonify({
                'success': True,
                'message': 'Tidak ada file baru yang diunggah. Tetap menggunakan dataset sebelumnya.',
                'dataset_info': get_dataset_info()
            })
            
        file = request.files['file']
        
        # Jika nama file kosong (misal submit form kosong)
        if file.filename == '':
            return jsonify({
                'success': True,
                'message': 'Tidak ada file baru yang dipilih. Tetap menggunakan dataset sebelumnya.',
                'dataset_info': get_dataset_info()
            })
            
        if not file.filename.endswith('.csv'):
            return jsonify({'error': 'File harus berformat CSV (.csv)'}), 400
            
        # Simpan sementara untuk validasi
        temp_path = CUSTOM_DATASET_PATH + '.tmp'
        file.save(temp_path)
        
        try:
            df = pd.read_csv(temp_path)
            uploaded_cols = set(df.columns)
            
            # Periksa apakah ada kolom teks (text) dan tanggal (createTimeISO atau date)
            has_text = 'text' in uploaded_cols
            has_date = 'createTimeISO' in uploaded_cols or 'date' in uploaded_cols
            
            if not (has_text and has_date):
                os.remove(temp_path)
                return jsonify({
                    'error': 'Skema CSV tidak valid. CSV wajib memiliki kolom teks [text] dan kolom tanggal [createTimeISO atau date].'
                }), 400
                
            # Jika kolom sentiment tidak ada, lakukan analisis sentimen otomatis
            if 'sentiment' not in uploaded_cols:
                print("--> Mendeteksi kolom 'sentiment' tidak ada. Melakukan klasifikasi sentimen otomatis...")
                df['sentiment'] = df['text'].apply(analyze_sentiment)
                # Simpan ulang ke temp_path agar CSV yang tersimpan memiliki kolom sentiment
                df.to_csv(temp_path, index=False)
                
            # Validasi tipe data atau minimal baris jika perlu
            if len(df) == 0:
                os.remove(temp_path)
                return jsonify({'error': 'File CSV tidak boleh kosong.'}), 400
                
            # Ganti file kustom lama dengan file baru
            if os.path.exists(CUSTOM_DATASET_PATH):
                os.remove(CUSTOM_DATASET_PATH)
            os.rename(temp_path, CUSTOM_DATASET_PATH)
            
            # Simpan metadata file asli
            with open(CUSTOM_METADATA_PATH, 'w') as f:
                json.dump({'original_name': file.filename}, f)
                
            return jsonify({
                'success': True,
                'message': f'Dataset kustom "{file.filename}" berhasil diunggah dan diterapkan.',
                'dataset_info': get_dataset_info()
            })
            
        except Exception as e:
            if os.path.exists(temp_path):
                os.remove(temp_path)
            return jsonify({'error': f'Gagal membaca/memvalidasi file CSV: {str(e)}'}), 400
            
    except Exception as e:
        return jsonify({'error': f'Terjadi kesalahan server: {str(e)}'}), 500


@app.route('/api/reset-dataset', methods=['POST'])
def api_reset_dataset():
    try:
        if os.path.exists(CUSTOM_DATASET_PATH):
            os.remove(CUSTOM_DATASET_PATH)
        if os.path.exists(CUSTOM_METADATA_PATH):
            os.remove(CUSTOM_METADATA_PATH)
        return jsonify({
            'success': True,
            'message': 'Dataset telah dikembalikan ke bawaan (dataset baru.csv).',
            'dataset_info': get_dataset_info()
        })
    except Exception as e:
        return jsonify({'error': f'Gagal mereset dataset: {str(e)}'}), 500

if __name__ == '__main__':
    # Jika dipanggil dengan flag --test-models, uji pemuatan dan keluar
    import sys
    if '--test-models' in sys.argv:
        print("Testing model loading...")
        load_models()
        if model_pos is not None and model_neg is not None:
            print("Model load test passed!")
            sys.exit(0)
        else:
            print("Model load test failed!")
            sys.exit(1)
            
    print("Memulai server Flask di http://localhost:5000")
    app.run(host='0.0.0.0', port=5000, debug=True)
