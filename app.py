import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import LSTM, Dense, Bidirectional, GRU, Conv1D, Attention, Input, GlobalAveragePooling1D

# 1. ΦΟΡΤΩΣΗ ΚΑΙ ΠΡΟΕΤΟΙΜΑΣΙΑ
file_path = 'archive-2/LSTM-Multivariate_pollution.csv' 
df = pd.read_csv(file_path)

encoder = LabelEncoder()
df['wnd_dir'] = encoder.fit_transform(df['wnd_dir'])
data_columns = ['pollution', 'dew', 'temp', 'press', 'wnd_dir', 'wnd_spd', 'snow', 'rain']
values = df[data_columns].values.astype('float32')

scaler = MinMaxScaler(feature_range=(0, 1))
scaled = scaler.fit_transform(values)

def create_dataset(data, steps=24):
    X, y = [], []
    for i in range(len(data) - steps):
        X.append(data[i:i+steps, :])
        y.append(data[i+steps, 0])
    return np.array(X), np.array(y)

X, y = create_dataset(scaled)
split = int(len(X) * 0.8)
train_X, test_X = X[:split], X[split:]
train_y, test_y = y[:split], y[split:]

input_shape = (train_X.shape[1], train_X.shape[2])

# 2. ΟΡΙΣΜΟΣ ΜΟΝΤΕΛΩΝ
def build_all_models():
    return {
        'LSTM Classic': Sequential([Input(shape=input_shape), LSTM(50), Dense(1)]),
        'BiLSTM': Sequential([Input(shape=input_shape), Bidirectional(LSTM(50)), Dense(1)]),
        'GRU': Sequential([Input(shape=input_shape), GRU(50), Dense(1)]),
        'ConvLSTM': Sequential([Input(shape=input_shape), Conv1D(64, 3, activation='relu'), LSTM(50), Dense(1)]),
        'Attention LSTM': (lambda: {
            'i': (inp := Input(shape=input_shape)),
            'l': (lstm := LSTM(50, return_sequences=True)(inp)),
            'm': Model(inputs=inp, outputs=Dense(1)(GlobalAveragePooling1D()(Attention()([lstm, lstm]))))
        })()['m']
    }

models = build_all_models()
results_list = []

# 3. ΕΚΠΑΙΔΕΥΣΗ ΚΑΙ ΑΝΑΛΥΤΙΚΕΣ ΜΕΤΡΙΚΕΣ
for name, model in models.items():
    print(f"\n>>> Έναρξη εκπαίδευσης: {name} (50 Epochs)")
    start_time = time.time()
    
    model.compile(optimizer='adam', loss='mse')
    model.fit(train_X, train_y, epochs=50, batch_size=72, validation_split=0.1, verbose=0) # verbose=0 για καθαρό τερματικό
    
    end_time = time.time()
    training_duration = end_time - start_time
    
    # Προβλέψεις και Μετρικές
    y_pred = model.predict(test_X)
    mse = mean_squared_error(test_y, y_pred)
    mae = mean_absolute_error(test_y, y_pred)
    r2 = r2_score(test_y, y_pred)
    
    results_list.append({
        'Μοντέλο': name,
        'MSE': f"{mse:.6f}",
        'MAE': f"{mae:.6f}",
        'R2 Score': f"{r2:.4f}",
        'Χρόνος (sec)': f"{training_duration:.2f}"
    })
    print(f"Ολοκληρώθηκε: {name} σε {training_duration:.2f} δευτερόλεπτα.")

# 4. ΕΚΤΥΠΩΣΗ ΤΕΛΙΚΟΥ ΠΙΝΑΚΑ
results_df = pd.DataFrame(results_list)
print("\n" + "="*60)
print("             ΤΕΛΙΚΟΣ ΠΙΝΑΚΑΣ ΑΞΙΟΛΟΓΗΣΗΣ")
print("="*60)
print(results_df.to_string(index=False))
print("="*60)