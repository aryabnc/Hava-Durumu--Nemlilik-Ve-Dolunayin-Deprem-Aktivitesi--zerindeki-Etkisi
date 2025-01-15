import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from skyfield.api import load
from datetime import datetime, timedelta
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score

file_path_weather = r'C:/Users/aryab/Desktop/Yüksek Lisan Bitirme Projesi/hava durumu verileri/2010_2024_weather.csv'
file_path_other = r'C:/Users/aryab/Desktop/Yüksek Lisan Bitirme Projesi/Veriler/20090101_20241031_4.0_9.0_19_42.csv'
file_path_humidity = r'C:/Users/aryab/Desktop/Yüksek Lisan Bitirme Projesi/nemlilik verileri/daily_humidity_data.csv'

try:
    weather_df = pd.read_csv(file_path_weather, encoding='ISO-8859-1')
    print("Hava Durumu Verisi:")
    print(weather_df.head()) 

    other_df = pd.read_csv(file_path_other, encoding='ISO-8859-1')
    print("\nÜçüncü Dosya Verisi:")
    print(other_df.head())

    humidity_df = pd.read_csv(file_path_humidity, encoding='ISO-8859-1')
    print("\nNemlilik Verisi:")
    print(humidity_df.head())

    # Nemlilik verisini interpolasyon ile tamamla
    print("\nEksik Nemlilik Verisi Tamamlanıyor...")
    humidity_df['date'] = pd.to_datetime(humidity_df['date'], format='%Y-%m-%d')
    humidity_df = humidity_df.set_index('date')
    humidity_df = humidity_df.sort_index()
    humidity_df['relative_humidity_2m'] = humidity_df['relative_humidity_2m'].interpolate(method='time')
    print("\nTamamlanmış Nemlilik Verisi:")
    print(humidity_df.head())

    # Dolunay tarihleri
    print("\nDolunay tarihleri hesaplanıyor...")
    eph = load('de430t.bsp') 
    earth, moon, sun = eph['earth'], eph['moon'], eph['sun']

    start_date = datetime(2010, 1, 1)
    end_date = datetime(2024, 10, 31)
    delta = timedelta(days=1)

    dolunay_tarihleri = []
    date = start_date
    while date <= end_date:
        ts = load.timescale()
        t = ts.utc(date.year, date.month, date.day)

        earth_position = earth.at(t)
        moon_position = moon.at(t)
        sun_position = sun.at(t)

        elongation = moon_position.separation_from(sun_position).degrees

        if abs(elongation - 180) < 1: 
            dolunay_tarihleri.append(date)

        date += delta

    moon_df = pd.DataFrame({'date': pd.to_datetime(dolunay_tarihleri)})
    print("\nHesaplanan Dolunay Verisi:")
    print(moon_df.head())

    moon_df['date'] = moon_df['date'].dt.strftime('%Y.%m.%d')

    # Verileri birleştirme
    weather_df['date'] = pd.to_datetime(weather_df['Date'], format='%Y-%m-%d')  
    other_df['date'] = pd.to_datetime(other_df['Olus tarihi'], format='%Y-%m-%d') 
    moon_df['date'] = pd.to_datetime(moon_df['date'], format='%Y.%m.%d') 

    humidity_df = humidity_df.reset_index()

    merged_df = pd.merge(weather_df, moon_df, on='date', how='left', indicator='is_fullmoon')
    merged_df = pd.merge(merged_df, other_df, on='date', how='left')
    merged_df = pd.merge(merged_df, humidity_df, on='date', how='left')

    merged_df['is_fullmoon'] = merged_df['is_fullmoon'].apply(lambda x: 1 if x == 'both' else 0)

    print("\nBirleştirilmiş Veri:")
    print(merged_df.head())

except FileNotFoundError:
    print("Bir veya daha fazla dosya bulunamadı.")
except Exception as e:
    print(f"Bir hata oluştu: {e}")

# Deprem var/yok (1/0) 
merged_df['is_earthquake'] = merged_df['Mw'].apply(lambda x: 1 if x >= 3.0 else 0)

def categorize_temperature(temp):
    if temp < 20:
        return 'below_20'
    elif temp >= 20 and temp <= 30:
        return 'between_20_and_30'
    else:
        return 'above_30'

merged_df['temperature_category'] = merged_df['Average Temperature (°C)'].apply(categorize_temperature)

features = ['relative_humidity_2m', 'is_fullmoon', 'temperature_category']  
X = pd.get_dummies(merged_df[features], drop_first=True)  

y = merged_df['is_earthquake']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestClassifier(random_state=42, class_weight='balanced')

param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [10, 20, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=5, n_jobs=-1, scoring='accuracy')
grid_search.fit(X_train, y_train)

best_model = grid_search.best_estimator_

y_pred = best_model.predict(X_test)

# Performans değerlendirmesi
print("Model Performansı (Class Weight Balanced ile):")
print(classification_report(y_test, y_pred))
print("Accuracy:", accuracy_score(y_test, y_pred))

feature_importances = pd.DataFrame(best_model.feature_importances_, index=X.columns, columns=['importance']).sort_values('importance', ascending=False)
print("\nÖzellik Önemleri:")
print(feature_importances)

# Nemlilik kategorilerine göre deprem oranlarını incele
def categorize_humidity(humidity):
    if humidity <= 30:
        return 'low'
    elif 31 <= humidity <= 60:
        return 'medium'
    else:
        return 'high'

merged_df['humidity_category'] = merged_df['relative_humidity_2m'].apply(categorize_humidity)

# Nemlilik kategorilerine göre deprem oranları
earthquake_by_humidity = merged_df.groupby('humidity_category')['is_earthquake'].mean()
print("\nNemi Kategorilerine Göre Deprem Oranları:")
print(earthquake_by_humidity)

sns.barplot(x=earthquake_by_humidity.index, y=earthquake_by_humidity.values)
plt.xlabel('Nemi Kategorisi')
plt.ylabel('Deprem Olma Oranı')
plt.title('Nem Kategorilerine Göre Deprem Olma Oranı')
plt.show()

# Sıcaklık seviyelerine göre deprem oranları
earthquake_by_temperature = merged_df.groupby('temperature_category')['is_earthquake'].mean()

print("\nSıcaklık Kategorilerine Göre Deprem Oranları:")
print(earthquake_by_temperature)

sns.barplot(x=earthquake_by_temperature.index, y=earthquake_by_temperature.values, palette="viridis")
plt.xlabel('Sıcaklık Kategorisi')
plt.ylabel('Deprem Olma Oranı')
plt.title('Sıcaklık Kategorilerine Göre Deprem Olma Oranı')
plt.show()

# Sıcaklık Kategorilerine Göre Deprem Büyüklükleri
magnitude_by_temperature = merged_df.groupby('temperature_category')['Mw'].mean()
print("\nSıcaklık Kategorilerine Göre Ortalama Deprem Büyüklükleri:")
print(magnitude_by_temperature)

sns.barplot(x=magnitude_by_temperature.index, y=magnitude_by_temperature.values, palette="coolwarm")
plt.xlabel('Sıcaklık Kategorisi')
plt.ylabel('Ortalama Deprem Büyüklüğü')
plt.title('Sıcaklık Kategorilerine Göre Ortalama Deprem Büyüklükleri')
plt.show()

# Nem ve Sıcaklık Etkileşimleri ile Deprem Oranları
interaction = merged_df.groupby(['humidity_category', 'temperature_category'])['is_earthquake'].mean().unstack()
print("\nNem ve Sıcaklık Etkileşimleri ile Deprem Oranları:")
print(interaction)

sns.heatmap(interaction, annot=True, fmt=".2f", cmap="YlGnBu")
plt.xlabel('Sıcaklık Kategorisi')
plt.ylabel('Nem Kategorisi')
plt.title('Nem ve Sıcaklık Etkileşimleri ile Deprem Oranları')
plt.show()

def calculate_fullmoon(date):
    ts = load.timescale()
    t = ts.utc(date.year, date.month, date.day)
    earth_position = earth.at(t)
    moon_position = moon.at(t)
    sun_position = sun.at(t)
    elongation = moon_position.separation_from(sun_position).degrees
    return 1 if abs(elongation - 180) < 1 else 0

print("\nLütfen tahmin için aşağıdaki bilgileri giriniz:")
date_input = input("Tarih (yyyy-mm-dd): ")
temperature_input = float(input("Sıcaklık (°C): "))
humidity_input = float(input("Nem (%): "))

date_obj = datetime.strptime(date_input, '%Y-%m-%d')
is_fullmoon = calculate_fullmoon(date_obj)

temperature_category = categorize_temperature(temperature_input)

user_data = pd.DataFrame({
    'relative_humidity_2m': [humidity_input],
    'is_fullmoon': [is_fullmoon],
    'temperature_category': [temperature_category]
})
user_data = pd.get_dummies(user_data, drop_first=True)

for col in X.columns:
    if col not in user_data.columns:
        user_data[col] = 0

# Tahmin yapma
user_prediction = best_model.predict(user_data)[0]
print("\nTahmin edilen deprem durumu:", "Deprem Bekleniyor" if user_prediction == 1 else "Deprem Beklenmiyor")
