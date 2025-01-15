import requests
import pandas as pd

# WeatherStack API bilgileri
api_key = "69b952404281f3ec073f602f7dd72a7a"  # Dashboard'dan aldığınız API Access Key
base_url = "http://api.weatherstack.com/historical"

# API parametreleri (İstanbul için)
params = {
    "access_key": api_key,
    "query": "Istanbul",  # Şehir adı
    "historical_date_start": "2023-01-01",  # Başlangıç tarihi
    "historical_date_end": "2023-01-31",  # Daha kısa tarih aralığı
    "hourly": "1",  # Saatlik veriler
    "units": "m"  # Metric birimler
}

# API isteği
response = requests.get(base_url, params=params)

# Yanıt kontrolü
if response.status_code == 200:
    data = response.json()
    print(data)  # Yanıtı kontrol edin

    if "historical" in data:
        historical_data = data["historical"]
        daily_data = []

        # Her gün için verileri işleme
        for date, details in historical_data.items():
            if "temperature" in details and "humidity" in details:
                avg_temperature = details["temperature"]
                avg_humidity = details["humidity"]

                daily_data.append({
                    "date": date,
                    "avg_temperature": avg_temperature,
                    "avg_humidity": avg_humidity
                })

        # DataFrame oluşturma
        df = pd.DataFrame(daily_data)
        print(df)

        # CSV olarak kaydetme
        df.to_csv("istanbul_daily_weather.csv", index=False)
        print("Günlük ortalama veriler CSV olarak kaydedildi.")
    else:
        print("Geçmiş veri bulunamadı:", data)
else:
    print("API isteği başarısız:", response.status_code)
