import requests
import csv
import numpy as np

# İzmir'in enlem ve boylam bilgileri
latitude = 36.4018  # enlem
longitude = 36.3498 #boylam

# Tarih aralığı
start_date = '2010-01-01'
end_date = '2024-10-31'

# Open-Meteo API URL'si (Saatlik sıcaklık verisi)
url = f"https://archive-api.open-meteo.com/v1/era5?latitude={latitude}&longitude={longitude}&start_date={start_date}&end_date={end_date}&hourly=temperature_2m"

# API isteği gönder
response = requests.get(url)

if response.status_code == 200:
    # JSON formatındaki veriyi al
    data = response.json()

    # Saatlik sıcaklık verisini al
    temperature_hourly = data['hourly']['temperature_2m']
    time = data['hourly']['time']

    # Saatlik verileri günlük ortalamaya dönüştürme
    daily_data = []
    for i in range(0, len(temperature_hourly), 24):  # Her 24 saati bir gün olarak kabul et
        daily_avg = np.mean(temperature_hourly[i:i+24])  # 24 saatin ortalamasını al
        date = time[i][:10]  # Tarih kısmını al (yyyy-mm-dd)
        daily_data.append([date, daily_avg])

    # CSV dosyasına yazma
    file_name = f"hatay_2010_2024_weather.csv"  # Dosya adı İzmir için belirli

    with open(file_name, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Date", "Average Temperature (°C)"])  # Başlık satırı
        writer.writerows(daily_data)  # Günlük sıcaklık ortalamaları

    print(f"Hava durumu verisi başarıyla '{file_name}' dosyasına kaydedildi.")
else:
    # Hata durumunda detaylı yanıtı yazdır
    print(f"Veri alınırken hata oluştu: {response.status_code}")
    print("Yanıt:", response.text)  # Yanıt metnini yazdır
