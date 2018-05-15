import pandas as pd
import numpy as np
import os
import logging
import argparse
import json

bj_station_list = [
        'aotizhongxin_aq', 'badaling_aq', 'beibuxinqu_aq', 'daxing_aq',
       'dingling_aq', 'donggaocun_aq', 'dongsi_aq', 'dongsihuan_aq',
       'fangshan_aq', 'fengtaihuayuan_aq', 'guanyuan_aq', 'gucheng_aq',
       'huairou_aq', 'liulihe_aq', 'mentougou_aq', 'miyun_aq',
       'miyunshuiku_aq', 'nansanhuan_aq', 'nongzhanguan_aq',
       'pingchang_aq', 'pinggu_aq', 'qianmen_aq', 'shunyi_aq',
       'tiantan_aq', 'tongzhou_aq', 'wanliu_aq', 'wanshouxigong_aq',
       'xizhimenbei_aq', 'yanqin_aq', 'yizhuang_aq', 'yongdingmennei_aq',
       'yongledian_aq', 'yufa_aq', 'yungang_aq', 'zhiwuyuan_aq',]

ld_station_list = ['CD1', 'BL0', 'GR4', 'MY7', 'HV1', 'GN3', 'GR9', 'LW2',
                'GN0', 'KF1', 'CD9', 'ST5', 'TH4']

def symmetric_mean_absolute_percentage_error(actual, predicted):

    actual = actual.ravel()
    predicted = predicted.ravel()

    a = np.abs(np.array(actual) - np.array(predicted))
    b = np.array(actual) + np.array(predicted)

    return 2 * np.mean(np.divide(a, b, out=np.zeros_like(a), where=b!=0, casting='unsafe'))

def get_logger():
    logging.basicConfig(format='%(asctime)s [%(levelname)s] [%(filename)s:%(lineno)s - %(funcName)s() ] %(message)s')
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    return logger

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--submit_date', type=str)
    parser.add_argument('--submit_path', required=True, type=str)
    parser.add_argument('--ld_aq_path', type=str)
    parser.add_argument('--bj_aq_path', type=str)
    args = parser.parse_args()
    print(json.dumps(vars(args), indent=2, sort_keys=True))
    return args

args = parse_args()
logger = get_logger()

##### load bj and ld aq file
ld_aq_path = args.ld_aq_path
ld_aq_df = pd.read_csv(ld_aq_path, parse_dates=['MeasurementDateGMT'])
ld_aq_df = ld_aq_df.rename(columns={'MeasurementDateGMT': 'time'})
ld_aq_df = ld_aq_df[ld_aq_df['station_id'].isin(ld_station_list)]

bj_aq_path = args.bj_aq_path
bj_aq_df = pd.read_csv(bj_aq_path, parse_dates=['utc_time'])
bj_aq_df['stationId'] = bj_aq_df['stationId'].apply(lambda x: x.replace('_aq','')[:10] + '_aq' if 'aq' in x else x)
bj_aq_df = bj_aq_df.rename(columns={'utc_time': 'time', 'stationId': 'station_id'})

##### load pred file
pred_df = pd.read_csv(args.submit_path)
if 'submit_date' not in pred_df:
    pred_df['submit_date'] = args.submit_date
    logger.info("Need submit date sinice no submite_date column")

##### compute smape
task = {'bj': ['PM2.5', 'PM10', 'O3'], 'ld': ['PM2.5', 'PM10']  }
submit_dates_str = pred_df['submit_date'].unique()
date_frames = []
for submit_date_str in submit_dates_str:

    submit_date = pd.to_datetime(submit_date_str)
    start_date = submit_date + pd.offsets.Day(1)#'2018-03-19 00:00:00'
    end_date = submit_date + pd.offsets.Day(2) + pd.offsets.Hour(23) #'2018-03-20 23:00:00'

    city_frames = []
    for city in list(task.keys()):

        pollutants = task[city]

        aq_df = bj_aq_df if city == 'bj' else ld_aq_df

        ## ans dataframe
        date_filter = (aq_df['time'] >= start_date) & (aq_df['time'] <= end_date)
        ans_df_city = aq_df[ date_filter ]

        ## formating
        ans_df_city['hour'] = ((ans_df_city['time']- start_date)/pd.Timedelta('1 hour')).astype(int)
        ans_df_city['test_id'] = ans_df_city['station_id'] + '#' + ans_df_city['hour'].apply(str)
        ans_df_city = ans_df_city.drop(['time','station_id','hour'], axis = 1)

        city_frames.append(ans_df_city[pollutants+['test_id']])

    ans_df_date = pd.concat(city_frames)
    ans_df_date['submit_date'] = submit_date_str
    date_frames.append(ans_df_date)

ans_df = pd.concat(date_frames)

#### reindex for ans_df and pred_df
ans_df = ans_df.set_index(['submit_date', 'test_id'])
pred_df = pred_df.set_index(['submit_date', 'test_id'])
pred_df = pred_df.reindex(ans_df.index)
ans_df = ans_df.reset_index(['submit_date', 'test_id'])
pred_df = pred_df.reset_index(['submit_date', 'test_id'])


#### SMAPE code from TA
pd_merged_all = pd.merge( ans_df, pred_df, how='left', on=['submit_date','test_id'] )

# Data preprocessing: Split test_id
pd_merged_all['stat_id'], pd_merged_all['hr'] = pd_merged_all['test_id'].str.split('#', 1).str

# Data preprocessing: Replace negative values with np.nan ( for ans )
# Data preprocessing: Replace negative values with 1.0 ( for pred )
pd_merged_all[['PM2.5_x', 'PM10_x', 'O3_x']] = pd_merged_all[['PM2.5_x', 'PM10_x', 'O3_x']].mask(pd_merged_all[['PM2.5_x', 'PM10_x', 'O3_x']] < 0.0, np.nan)
pd_merged_all[['PM2.5_y', 'PM10_y', 'O3_y']] = pd_merged_all[['PM2.5_y', 'PM10_y', 'O3_y']].mask(pd_merged_all[['PM2.5_y', 'PM10_y', 'O3_y']] < 0.0, 1.0)


# Data preprocessing: Remove missing value in gt
pd_merged_lon = pd_merged_all.loc[ pd_merged_all['stat_id'].isin(ld_station_list) ]
pd_merged_bj = pd_merged_all.loc[ ~pd_merged_all['stat_id'].isin(ld_station_list) ]

pd_merged_lon = pd_merged_lon.replace('', np.nan, regex=True).dropna( subset=['PM2.5_x', 'PM10_x'], how='any' )
pd_merged_bj = pd_merged_bj.replace('', np.nan, regex=True).dropna( subset=['PM2.5_x', 'PM10_x', 'O3_x'], how='any' )

pd_merged_all = pd.concat( [pd_merged_lon, pd_merged_bj] )

for submit_date_str in pd_merged_all['submit_date'].unique():


    pd_merged_bj_temp = pd_merged_bj[pd_merged_bj['submit_date'] == submit_date_str]
    pd_merged_lon_temp = pd_merged_lon[pd_merged_lon['submit_date'] == submit_date_str]

    bj_SMAPE = symmetric_mean_absolute_percentage_error(pd_merged_bj_temp[['PM2.5_x','PM10_x','O3_x']].values,
                                                        pd_merged_bj_temp[['PM2.5_y','PM10_y','O3_y']].values)
    bj_PM25_SMAPE = symmetric_mean_absolute_percentage_error(pd_merged_bj_temp[['PM2.5_x']].values,
                                                        pd_merged_bj_temp[['PM2.5_y']].values)
    bj_PM10_SMAPE = symmetric_mean_absolute_percentage_error(pd_merged_bj_temp[['PM10_x']].values,
                                                        pd_merged_bj_temp[['PM10_y']].values)
    bj_O3_SMAPE = symmetric_mean_absolute_percentage_error(pd_merged_bj_temp[['O3_x']].values,
                                                        pd_merged_bj_temp[['O3_y']].values)
    lon_SMAPE = symmetric_mean_absolute_percentage_error(pd_merged_lon_temp[['PM2.5_x','PM10_x']].values,
                                                        pd_merged_lon_temp[['PM2.5_y','PM10_y']].values)
    lon_PM25_SMAPE = symmetric_mean_absolute_percentage_error(pd_merged_lon_temp[['PM2.5_x']].values,
                                                        pd_merged_lon_temp[['PM2.5_y']].values)
    lon_PM10_SMAPE = symmetric_mean_absolute_percentage_error(pd_merged_lon_temp[['PM10_x']].values,
                                                        pd_merged_lon_temp[['PM10_y']].values)

    official_SMAPE = (bj_SMAPE+lon_SMAPE)/2

    pd_merged = pd_merged_all[pd_merged_all['submit_date'] == submit_date_str]
    new_SMAPE = symmetric_mean_absolute_percentage_error(pd_merged[['PM2.5_x','PM10_x','O3_x']].values,
                                                  pd_merged[['PM2.5_y','PM10_y','O3_y']].values)

    ans_date_filter = ans_df['submit_date'] == submit_date_str
    pred_date_filter = pred_df['submit_date'] == submit_date_str
    old_SMAPE = symmetric_mean_absolute_percentage_error(
        ans_df.loc[ans_date_filter, ['PM2.5','PM10','O3']].values,
        pred_df.loc[pred_date_filter, ['PM2.5','PM10','O3']].values)

    ###### feature index
    tuples = [( submit_date_str, os.path.basename(args.submit_path)) ]
    task_index = pd.MultiIndex.from_tuples(tuples, names=[
         'submit_date', 'filename'])

    SMAPE_df = pd.DataFrame(data={'old SMAPE':[old_SMAPE],
                                  'new SMAPE':[new_SMAPE],
                                  'bj SMAPE':[bj_SMAPE],
                                  'lon SMAPE':[lon_SMAPE],
                                  'official SMAPE': [official_SMAPE],
                                  'bj_PM25':[bj_PM25_SMAPE],
                                  'bj_PM10':[bj_PM10_SMAPE],
                                  'bj_O3':[bj_O3_SMAPE],
                                  'lon_PM25':[lon_PM25_SMAPE],
                                  'lon_PM10':[lon_PM10_SMAPE]}, index=task_index)

    logger.info("submit_date : {}".format(submit_date.strftime("%Y-%m-%d")))
    logger.info("official : {}".format(official_SMAPE))
    logger.info("bj : {}".format(bj_SMAPE))
    logger.info("ld : {}".format(lon_SMAPE))
    logger.info("bj_PM25 : {}".format(bj_PM25_SMAPE))
    logger.info("bj_PM10 : {}".format(bj_PM10_SMAPE))
    logger.info("bj_O3 : {}".format(bj_O3_SMAPE))
    logger.info("lon_PM25 : {}".format(lon_PM25_SMAPE))
    logger.info("lon_PM10 : {}".format(lon_PM10_SMAPE))

