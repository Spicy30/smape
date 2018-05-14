# SMAPE

## Usage
```
python2 compuate_smape.py --ld_aq_path {ld_aq_path} --bj_aq_path {bj_aq_path} --submit_path {submit_path} --submit_date {submit_date}
```

* submit_date : ex 2018-04-29, 2018-05-01
* submit_path : submission path 
* ld_aq_path : london air quality path
* bj_aq_path : beijing air quality path

### ld_aq

| station_id | MeasurementDateGMT | PM2.5 | PM10 | NO2 | 
| ---------- | ------------------ | ----- | ---- | --- |
|BL0 | 2017-01-01 14:00:00 | 5.3 | 12.7 | 31.0 |

### bj_aq

| stationId | utc_time | PM2.5 | PM10 | O3 | NO2 | SO2 | CO |
| --------- | -------- | ----- | ---- | -- | --- | --- | -- |
| aotizhongxin_aq | 2017-01-01 14:00:00 | 453.0 | 467.0 | 3.0 | 156.0 | 9.0 | 7.2 |
