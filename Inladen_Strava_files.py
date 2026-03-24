from bikeride import BikeRide
import pandas as pd
import numpy as np
from haversine import haversine, Unit

ride = BikeRide('data/StravaRobin/Passo_Duran_Duran_.fit')

df_records = pd.DataFrame(ride.records)
desired_rec_cols = ['timestamp', 'lat', 'lon', 'enhanced_altitude', 'power', 'cadence', 'temperature']
present_rec_cols = [c for c in desired_rec_cols if c in df_records.columns]
if not present_rec_cols:
    print('Warning: none of the expected record columns found. Available:', list(df_records.columns))
records = df_records[present_rec_cols].copy()
for c in desired_rec_cols:
    if c not in records.columns:
        records[c] = np.nan
records = records[desired_rec_cols]

# vraag 1: Afstand en snelheid berekenen tussen ieder opvolgend datapunt van activiteit

# Maak kolommen met de lat/lon van het VORIGE datapunt
records['lat_prev'] = records['lat'].shift(1)
records['lon_prev'] = records['lon'].shift(1)

# Bereken haversine afstand per rij (in meters)
def bereken_afstand(row):
    if pd.isna(row['lat_prev']) or pd.isna(row['lat']):
        return np.nan
    return haversine(
        (row['lat_prev'], row['lon_prev']),
        (row['lat'],      row['lon']),
        unit=Unit.METERS
    )

records['afstand_m'] = records.apply(bereken_afstand, axis=1)



# Tijdsverschil in seconden tussen opeenvolgende datapunten
records['tijd_s'] = records['timestamp'].diff().dt.total_seconds()

# Snelheid in m/s en km/h
records['snelheid_ms']  = records['afstand_m'] / records['tijd_s']
records['snelheid_kmh'] = records['snelheid_ms'] * 3.6

print(records[['timestamp', 'lat', 'lon', 'afstand_m', 'tijd_s', 'snelheid_ms', 'snelheid_kmh']].head())