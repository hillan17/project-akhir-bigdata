import time, json, requests
from kafka import KafkaProducer

producer = KafkaProducer(
    bootstrap_servers="localhost:9092",
    value_serializer=lambda v: json.dumps(v).encode("utf-8")
)

while True:
    res = requests.get("https://api.binance.com/api/v3/ticker/24hr?symbol=BTCUSDT").json()
    payload = {
        "symbol": res["symbol"],
        "price": float(res["lastPrice"]),
        "priceChangePercent": float(res["priceChangePercent"]),
        "volume": float(res["volume"]),
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
    }
    producer.send("binance-klines-stream", payload)
    print("Streaming to Kafka:", payload)
    time.sleep(5)