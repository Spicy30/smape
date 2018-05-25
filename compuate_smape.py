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

bj_station_dict = {'aotizhongx': 'aotizhongxin',
                    'fengtaihua': 'fengtaihuayuan',
                    'miyunshuik': 'miyunshuiku',
                    'nongzhangu': 'nongzhanguan',
                    'wanshouxig': 'wanshouxigong',
                    'xizhimenbe': 'xizhimenbei',
                    'yongdingme': 'yongdingmennei'}

def station_transfrom(test_id):
    if '_aq' in test_id:  ## station_id in bj_station_list
        old_station_id, hour = test_id.split('#')
        old_station_name, _ = old_station_id.split('_')
        if old_station_name in bj_station_dict: ## station name to modify
            return bj_station_dict[old_station_name] + '_aq#' + hour
    return test_id

def smape(actual, predicted):

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

def read_ans(submit_dates_str):

    ##### load bj and ld aq file
    # ld_aq_file = 'ld_aq_{}.csv'.format(args.today)
    # ld_aq_path = os.path.join(args.aq_dir, ld_aq_file)
    ld_aq_path = args.ld_aq_path
    ld_aq_df = pd.read_csv(ld_aq_path, parse_dates=['MeasurementDateGMT'])
    ld_aq_df = ld_aq_df.rename(columns={'MeasurementDateGMT': 'time'})
    ld_aq_df = ld_aq_df[ld_aq_df['station_id'].isin(ld_station_list)]

    # bj_aq_file = 'bj_aq_{}.csv'.format(args.today)
    # bj_aq_path = os.path.join(args.aq_dir, bj_aq_file)
    bj_aq_path = args.bj_aq_path
    bj_aq_df = pd.read_csv(bj_aq_path, parse_dates=['utc_time'])
    # bj_aq_df['stationId'] = bj_aq_df['stationId'].apply(lambda x: x.replace('_aq','')[:10] + '_aq' if 'aq' in x else x)
    bj_aq_df = bj_aq_df.rename(columns={'utc_time': 'time', 'stationId': 'station_id'})

    ##### build bj and ld aq ans to submission format
    task = {'bj': ['PM2.5', 'PM10', 'O3'], 'ld': ['PM2.5', 'PM10']  }
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

    return ans_df

def _compute_smape(smape_name, ans_values, pred_values):

    if len(ans_values) == 0:
        SMAPE = 0.0
    else:
        SMAPE = smape(ans_values, pred_values)
    SMAPE_dict = { smape_name : [SMAPE] }

    return SMAPE_dict

def compute_smape(ans_df, pred_df):

    #### SMAPE code from TA
    pd_merged_all = pd.merge( ans_df, pred_df, how='left', on=['submit_date','test_id'] )

    # Data preprocessing: Split test_id
    pd_merged_all['station_id'], pd_merged_all['hour'] = pd_merged_all['test_id'].str.split('#', 1).str
    pd_merged_all['hour'] = pd_merged_all['hour'].astype(int)

    # Data preprocessing: Replace negative values with np.nan ( for ans )
    # Data preprocessing: Replace negative values with 1.0 ( for pred )
    ans_columns = ['PM2.5_x', 'PM10_x', 'O3_x']
    pred_columns = ['PM2.5_y', 'PM10_y', 'O3_y']
    pd_merged_all[ans_columns] = pd_merged_all[ans_columns].mask(pd_merged_all[ans_columns] < 0.0, np.nan)
    pd_merged_all[pred_columns] = pd_merged_all[pred_columns].mask(pd_merged_all[pred_columns] < 0.0, 1.0)

    # Data preprocessing: Remove missing value in gt
    pd_merged_lon = pd_merged_all.loc[ pd_merged_all['station_id'].isin(ld_station_list) ]
    pd_merged_bj = pd_merged_all.loc[ ~pd_merged_all['station_id'].isin(ld_station_list) ]

    pd_merged_lon = pd_merged_lon.replace('', np.nan, regex=True).dropna( subset=['PM2.5_x', 'PM10_x'], how='any' )
    pd_merged_bj = pd_merged_bj.replace('', np.nan, regex=True).dropna( subset=['PM2.5_x', 'PM10_x', 'O3_x'], how='any' )

    pd_merged_all = pd.concat( [pd_merged_lon, pd_merged_bj] )

    bj_SMAPE = smape(pd_merged_bj[['PM2.5_x','PM10_x','O3_x']].values,
                        pd_merged_bj[['PM2.5_y','PM10_y','O3_y']].values)
    bj_PM25_SMAPE = smape(pd_merged_bj[['PM2.5_x']].values,
                            pd_merged_bj[['PM2.5_y']].values)
    bj_PM10_SMAPE = smape(pd_merged_bj[['PM10_x']].values,
                            pd_merged_bj[['PM10_y']].values)
    bj_O3_SMAPE = smape(pd_merged_bj[['O3_x']].values,
                        pd_merged_bj[['O3_y']].values)
    lon_SMAPE = smape(pd_merged_lon[['PM2.5_x','PM10_x']].values,
                        pd_merged_lon[['PM2.5_y','PM10_y']].values)
    lon_PM25_SMAPE = smape(pd_merged_lon[['PM2.5_x']].values,
                            pd_merged_lon[['PM2.5_y']].values)
    lon_PM10_SMAPE = smape(pd_merged_lon[['PM10_x']].values,
                            pd_merged_lon[['PM10_y']].values)

    official_SMAPE = (bj_SMAPE+lon_SMAPE)/2

    ### hour split for 0-23 24-47
    bj_hour_filter = (pd_merged_bj['hour'] <= 23) & (pd_merged_bj['hour'] >= 0 )
    pd_merged_bj_0_23 = pd_merged_bj[bj_hour_filter]
    pd_merged_bj_24_47 = pd_merged_bj[~bj_hour_filter]

    lon_hour_filter = (pd_merged_lon['hour'] <= 23) & (pd_merged_lon['hour'] >= 0 )
    pd_merged_lon_0_23 = pd_merged_lon[lon_hour_filter]
    pd_merged_lon_24_47 = pd_merged_lon[~lon_hour_filter]

    bj_0_23_SMAPE = smape(pd_merged_bj_0_23[['PM2.5_x','PM10_x','O3_x']].values,
                            pd_merged_bj_0_23[['PM2.5_y','PM10_y','O3_y']].values)
    bj_PM25_0_23_SMAPE = smape(pd_merged_bj_0_23[['PM2.5_x']].values,
                                pd_merged_bj_0_23[['PM2.5_y']].values)
    bj_PM10_0_23_SMAPE = smape(pd_merged_bj_0_23[['PM10_x']].values,
                                pd_merged_bj_0_23[['PM10_y']].values)
    bj_O3_0_23_SMAPE = smape(pd_merged_bj_0_23[['O3_x']].values,
                                pd_merged_bj_0_23[['O3_y']].values)
    lon_0_23_SMAPE = smape(pd_merged_lon_0_23[['PM2.5_x','PM10_x']].values,
                            pd_merged_lon_0_23[['PM2.5_y','PM10_y']].values)
    lon_PM25_0_23_SMAPE = smape(pd_merged_lon_0_23[['PM2.5_x']].values,
                                pd_merged_lon_0_23[['PM2.5_y']].values)
    lon_PM10_0_23_SMAPE = smape(pd_merged_lon_0_23[['PM10_x']].values,
                                pd_merged_lon_0_23[['PM10_y']].values)

    bj_24_47_SMAPE = smape(pd_merged_bj_24_47[['PM2.5_x','PM10_x','O3_x']].values,
                            pd_merged_bj_24_47[['PM2.5_y','PM10_y','O3_y']].values)
    bj_PM25_24_47_SMAPE = smape(pd_merged_bj_24_47[['PM2.5_x']].values,
                                pd_merged_bj_24_47[['PM2.5_y']].values)
    bj_PM10_24_47_SMAPE = smape(pd_merged_bj_24_47[['PM10_x']].values,
                                pd_merged_bj_24_47[['PM10_y']].values)
    bj_O3_24_47_SMAPE = smape(pd_merged_bj_24_47[['O3_x']].values,
                                pd_merged_bj_24_47[['O3_y']].values)
    lon_24_47_SMAPE = smape(pd_merged_lon_24_47[['PM2.5_x','PM10_x']].values,
                            pd_merged_lon_24_47[['PM2.5_y','PM10_y']].values)
    lon_PM25_24_47_SMAPE = smape(pd_merged_lon_24_47[['PM2.5_x']].values,
                                pd_merged_lon_24_47[['PM2.5_y']].values)
    lon_PM10_24_47_SMAPE = smape(pd_merged_lon_24_47[['PM10_x']].values,
                                pd_merged_lon_24_47[['PM10_y']].values)

    SMAPE_dicts = []
    task = {'bj':['PM2.5', 'PM10', 'O3'], 'ld':['PM2.5', 'PM10']}
    for city in list(task.keys()):
        pollutants = task[city]
        for pollutant in pollutants:
            station_list = bj_station_list if city == 'bj' else ld_station_list
            for station_id in station_list:
                station_filter = pd_merged_all['station_id'] == station_id
                hour_ranges = [(0, 47), (0, 23), (24, 47)]
                for hour_range in hour_ranges:
                    start_hour, end_hour = hour_range[0], hour_range[1]
                    hour_filter = ( (pd_merged_all['hour'] <= end_hour)
                            & (pd_merged_all['hour'] >= start_hour ) )
                    pd_merged_temp = pd_merged_all[station_filter & hour_filter]
                    if start_hour == 0 and end_hour == 47:
                        smape_name='{}_{}_{}'.format(city, pollutant, station_id)
                    else:
                        smape_name='{}_{}_{}_{}-{}'.format(city, pollutant,
                                    station_id, start_hour, end_hour)
                    SMAPE_dict_temp = _compute_smape(
                        smape_name=smape_name,
                        ans_values=pd_merged_temp[['{}_x'.format(pollutant)]].values,
                        pred_values=pd_merged_temp[['{}_y'.format(pollutant)]].values)
                    SMAPE_dicts.append(SMAPE_dict_temp)
                    if pd.isnull(SMAPE_dict_temp[smape_name][0]) :
                        import ipdb; ipdb.set_trace()  # XXX BREAKPOINT
                    logger.info("{} : {}".format(smape_name, SMAPE_dict_temp[smape_name]))

    SMAPE_dict = {}
    for SMAPE_dict_temp in SMAPE_dicts:
        SMAPE_dict.update(SMAPE_dict_temp)

    submit_date = ans_df['submit_date'].unique()[0]
    SMAPE_dict1 = {'official': [official_SMAPE],
                    'bj':[bj_SMAPE],
                    'ld':[lon_SMAPE],
                    'bj_PM2.5':[bj_PM25_SMAPE],
                    'bj_PM10':[bj_PM10_SMAPE],
                    'bj_O3':[bj_O3_SMAPE],
                    'ld_PM2.5':[lon_PM25_SMAPE],
                    'ld_PM10':[lon_PM10_SMAPE],
                    'bj_0_23':[bj_0_23_SMAPE],
                    'ld_0_23':[lon_0_23_SMAPE],
                    'bj_PM2.5_0_23':[bj_PM25_0_23_SMAPE],
                    'bj_PM10_0_23':[bj_PM10_0_23_SMAPE],
                    'bj_O3_0_23':[bj_O3_0_23_SMAPE],
                    'ld_PM2.5_0_23':[lon_PM25_0_23_SMAPE],
                    'ld_PM10_0_23':[lon_PM10_0_23_SMAPE],
                    'bj_24_47':[bj_24_47_SMAPE],
                    'ld_24_47':[lon_24_47_SMAPE],
                    'bj_PM2.5_24_47':[bj_PM25_24_47_SMAPE],
                    'bj_PM10_24_47':[bj_PM10_24_47_SMAPE],
                    'bj_O3_24_47':[bj_O3_24_47_SMAPE],
                    'ld_PM2.5_24_47':[lon_PM25_24_47_SMAPE],
                    'ld_PM10_24_47':[lon_PM10_24_47_SMAPE],
                        }

    SMAPE_dict.update(SMAPE_dict1)

    SMAPE_df = pd.DataFrame(data=SMAPE_dict,
                            index=[submit_date])

    SMAPE_df.fillna(0, inplace=True)

    logger.info("submit_date : {}".format(submit_date))
    logger.info("official : {}".format(official_SMAPE))
    logger.info("bj : {}".format(bj_SMAPE))
    logger.info("ld : {}".format(lon_SMAPE))
    logger.info("bj_PM25 : {}".format(bj_PM25_SMAPE))
    logger.info("bj_PM25_0_23 : {}".format(bj_PM25_0_23_SMAPE))
    logger.info("bj_PM25_24_47 : {}".format(bj_PM25_24_47_SMAPE))
    logger.info("bj_PM10 : {}".format(bj_PM10_SMAPE))
    logger.info("bj_PM10_0_23 : {}".format(bj_PM10_0_23_SMAPE))
    logger.info("bj_PM10_24_47 : {}".format(bj_PM10_24_47_SMAPE))
    logger.info("bj_O3 : {}".format(bj_O3_SMAPE))
    logger.info("bj_O3_0_23 : {}".format(bj_O3_0_23_SMAPE))
    logger.info("bj_O3_24_47 : {}".format(bj_O3_24_47_SMAPE))
    logger.info("lon_PM25 : {}".format(lon_PM25_SMAPE))
    logger.info("lon_PM25_0_23 : {}".format(lon_PM25_0_23_SMAPE))
    logger.info("lon_PM25_24_47 : {}".format(lon_PM25_24_47_SMAPE))
    logger.info("lon_PM10 : {}".format(lon_PM10_SMAPE))
    logger.info("lon_PM10_0_23 : {}".format(lon_PM10_0_23_SMAPE))
    logger.info("lon_PM10_24_47 : {}".format(lon_PM10_24_47_SMAPE))

    return SMAPE_df

def main():

    ##### load pred file
    pred_df = pd.read_csv(args.submit_path)
    pred_df['test_id'] = pred_df['test_id'].apply(station_transfrom)
    if 'submit_date' not in pred_df:
        pred_df['submit_date'] = args.submit_date
        logger.info("Need submit date sinice no submite_date column")

    ##### read ans file
    task = {'bj': ['PM2.5', 'PM10', 'O3'], 'ld': ['PM2.5', 'PM10']  }
    submit_dates_str = pred_df['submit_date'].unique()
    ans_df = read_ans(submit_dates_str)

    for submit_date_str in submit_dates_str:

        ##### compute smape
        ans_date_filter = ans_df['submit_date'] == submit_date_str
        ans_df_temp = ans_df[ans_date_filter]

        pred_date_filter = pred_df['submit_date'] == submit_date_str
        pred_df_temp = pred_df[pred_date_filter]

        SMAPE_df = compute_smape(ans_df_temp, pred_df_temp)

if __name__ == '__main__':

    args = parse_args()
    logger = get_logger()

    main()

