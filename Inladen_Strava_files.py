from bikeride import BikeRide
import pandas as pd
import numpy as np

ride = BikeRide('data/StravaRobin/Passo_Duran_Duran_.fit')
print(ride.summary)

# Build records DataFrame but only select columns that actually exist in the data.
df_records = pd.DataFrame(ride.records)
desired_rec_cols = ['timestamp', 'lat', 'lon', 'enhanced_altitude', 'power', 'cadence', 'temperature']
present_rec_cols = [c for c in desired_rec_cols if c in df_records.columns]
if not present_rec_cols:
	print('Warning: none of the expected record columns found. Available:', list(df_records.columns))
records = df_records[present_rec_cols].copy()

# Ensure expected schema: add missing columns with NaN and reorder to desired list
for c in desired_rec_cols:
	if c not in records.columns:
		records[c] = np.nan
records = records[desired_rec_cols]

# Build segments DataFrame similarly
df_segs = pd.DataFrame(ride.segments)
desired_seg_cols = ['timestamp_start', 'timestamp_end', 'lat_start', 'lon_start', 'lat_end', 'lon_end', 'temp_recorded_start']
present_seg_cols = [c for c in desired_seg_cols if c in df_segs.columns]
if not present_seg_cols:
	print('Warning: none of the expected segment columns found. Available:', list(df_segs.columns))
sgms = df_segs[present_seg_cols].copy()
for c in desired_seg_cols:
	if c not in sgms.columns:
		sgms[c] = np.nan
sgms = sgms[desired_seg_cols]


