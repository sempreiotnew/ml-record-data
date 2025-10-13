import os
import serial
import threading
from collections import deque, defaultdict
from datetime import datetime
import csv
import numpy as np
import pandas as pd

# ---------------- CONFIG ----------------
SERIAL_PORT = "/dev/cu.usbserial-0289714A"
BAUDRATE = 115200
MAX_BUFFER_LINES = 500

serial_lock = threading.Lock()

# Buffers per sensor ID
serial_buffer = defaultdict(lambda: deque(maxlen=MAX_BUFFER_LINES))
start_time = datetime.now()

# ---------------- CSV LOGGING ----------------
timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
csv_file = f"{timestamp_str}.csv"
annotation_file = f"{timestamp_str}.txt"

csv_columns = ["id","index","millis","gas_index","mes_index","temperature","pressure","humidity","gas_resistance","status","date_time"]

# Initialize CSV
with open(csv_file, mode="w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=csv_columns)
    writer.writeheader()

def try_serial():
    try:
        ser = serial.Serial(SERIAL_PORT, BAUDRATE, timeout=0.5)
        return ser
    except Exception as e:
        print(f"[serial_reader] Error opening {SERIAL_PORT}: {e}")  

def serial_reader_function():
    ser = try_serial()
    while ser == None:
        ser = try_serial()

    while True:
        try:
            line_bytes = ser.readline()
            if not line_bytes:
                continue
            line = line_bytes.decode("utf-8", errors="ignore").strip()
            if not line:
                continue

            parts = line.split(",")
            if len(parts) < 10:
                continue
            sensor_id = parts[0]
            gas_resistance = float(parts[8])
            

            # Append to serial buffer for plotting
            with serial_lock:
                # serial_buffer[sensor_id].append({"millis": millis, "gas_resistance": gas_resistance})
                serial_buffer[sensor_id].append({
                    "timestamp": datetime.now(),
                    "gas_resistance": gas_resistance,
                })

            
            save_csv_file(parts)
            
            #write_excel_row(row, csv_columns_full, f"{timestamp_str}.xlsx")
            # ---------------- PRINT OUTPUT ----------------
            #print(row)

        except Exception as e:
            print("[serial_reader] Error:", e)
            ser = try_serial()
            while ser == None:
                ser = try_serial()

def save_csv_file(splitted_data):
    sensor_id = splitted_data[0]
    gas_resistance = float(splitted_data[8])
    temperature = float(splitted_data[5])
    pressure = float(splitted_data[6])
    humidity = float(splitted_data[7])
    status = splitted_data[9]
    gas_index = int(splitted_data[3])
    mes_index = int(splitted_data[4])
    index = int(splitted_data[1])
    millis = int(splitted_data[2])

    row = {
        "id": sensor_id,
        "index": index,
        "millis": millis,
        "gas_index": gas_index,
        "mes_index": mes_index,
        "temperature": temperature,
        "pressure": pressure,
        "humidity": humidity,
        "gas_resistance": gas_resistance,
        "status": status,
        "date_time" : datetime.now().strftime("%d-%m-%Y-%H:%M:%S%f")[:-2]
    }
    csv_columns_full = list(row.keys())
    with open(csv_file, mode="a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=csv_columns_full)
        writer.writerow(row)

def save_annotations(value):
    """Writes a value to a text file; appends if it exists, creates if not."""
    mode = "a" if os.path.exists(annotation_file) else "w"
    with open(annotation_file, mode, encoding="utf-8") as file:
        file.write(f"{value}\n")