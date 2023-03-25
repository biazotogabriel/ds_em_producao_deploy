import pickle
import inflection
import pandas as pd
import numpy as np
import math
import datetime
import locale
from datetime import timedelta



class Rossmann(object):
    def __init__(self):
        self.home_path = ''
        self.competition_distance_rescaler = pickle.load(open(self.home_path+'parameter/competition_distance_rescaler.pkl', 'rb'))
        self.competition_time_month_rescaler = pickle.load(open(self.home_path+'parameter/competition_time_month_rescaler.pkl', 'rb'))
        self.promo_time_week_rescaler = pickle.load(open(self.home_path+'parameter/promo_time_week_rescaler.pkl', 'rb'))
        self.year_rescaler = pickle.load(open(self.home_path+'parameter/year_rescaler.pkl', 'rb'))
        self.store_type_label_encoder = pickle.load(open(self.home_path+'parameter/store_type_label_encoder.pkl', 'rb'))
        self.assortment_ordinal_encoder = pickle.load(open(self.home_path+'parameter/assortment_ordinal_encoder.pkl', 'rb'))

    def data_cleaning(self, df1):
        #rename
        old_cols = df1.columns
        new_cols = list(map(inflection.underscore, old_cols))
        df1.columns = new_cols
        df1['date'] = pd.to_datetime(df1['date'], )
        #fill NAs
        max_value = 10 * df1['competition_distance'].max()
        df1.loc[df1['competition_distance'].isna(), 'competition_distance'] = max_value
        df1['competition_open_since_month'] = df1.apply(lambda row: row['date'].month if math.isnan(row['competition_open_since_month']) else row['competition_open_since_month'], axis=1)
        df1['competition_open_since_year'] = df1.apply(lambda row: row['date'].year if math.isnan(row['competition_open_since_year']) else row['competition_open_since_year'], axis=1)
        df1['promo2_since_week'] = df1.apply(lambda row: row['date'].week if math.isnan(row['promo2_since_week']) else row['promo2_since_week'], axis=1)
        df1['promo2_since_year'] = df1.apply(lambda row: row['date'].year if math.isnan(row['promo2_since_year']) else row['promo2_since_year'], axis=1)
        df1.loc[df1['promo_interval'].isna(), 'promo_interval'] = ''
        #convert dtype
        df1['competition_open_since_month'] = df1['competition_open_since_month'].astype(int)
        df1['competition_open_since_year'] = df1['competition_open_since_year'].astype(int)
        df1['promo2_since_week'] = df1['promo2_since_week'].astype(int)
        df1['promo2_since_year'] = df1['promo2_since_year'].astype(int)
        #feature engenering
        # definir se a loja estava em promoção no dia da venda
        locale.setlocale(locale.LC_TIME, ('en_US', 'UTF-8'))
        df1['in_promo'] = df1[['date', 'promo_interval']].apply(lambda x: 1 if x['date'].strftime('%b') in x['promo_interval'].split(',') else 0, axis=1)
        
        return df1
    
    def feature_engineering(self, df2):
        df2['year'] = df2['date'].dt.year
        df2['month'] = df2['date'].dt.month
        df2['day'] = df2['date'].dt.day
        df2['week_of_year'] = df2['date'].dt.isocalendar().week.astype('int64')
        df2['year_week'] = df2['date'].dt.strftime('%Y-%W')
        df2['competition_since'] = pd.to_datetime({'year': df2['competition_open_since_year'],
                                                'month': df2['competition_open_since_month'],
                                                'day': 1})
        df2['competition_time_month'] = ((df2['date'] - df2['competition_since'])/30).apply(lambda x: x.days).astype(int)
        df2['promo_since'] = pd.to_datetime(df2['promo2_since_year'] * 1000 + df2['promo2_since_week'] * 10, format='%Y%W%w')
        df2['promo_since'] = df2['promo_since'] - timedelta(days=7)
        df2['promo_time_week'] = ((df2['date'] - df2['promo_since'])/7).apply(lambda x: x.days).astype(int)
        df2['assortment'] = df2['assortment'].map(lambda x: {'a': 'basic',
                                                            'b': 'extra',
                                                            'c': 'extended'}.get(x, x))
        df2['state_holiday'] = df2['state_holiday'].map(lambda x: {'a': 'public',
                                                                'b': 'easter',
                                                                'c': 'christmas',
                                                                '0': 'regular'}.get(x, x))
        df2 = df2[df2['open'] != 0]
        cols_drop = ['open', 'promo_interval']
        df2 = df2.drop(cols_drop, axis=1)

        return df2

    def data_preparation(self, df5):
        #fit_transform ??
        df5['competition_distance'] = self.competition_distance_rescaler.transform(df5[['competition_distance']].values)
        df5['competition_time_month'] = self.competition_time_month_rescaler.transform(df5[['competition_time_month']].values)
        df5['promo_time_week'] = self.promo_time_week_rescaler.transform(df5[['promo_time_week']].values)
        df5['year'] = self.year_rescaler.transform(df5[['year']].values)

        df5 = pd.get_dummies(df5, prefix=['state_holiday'], columns=['state_holiday'])
        df5['store_type'] = self.store_type_label_encoder.transform(df5['store_type'])
        df5['assortment'] = self.assortment_ordinal_encoder.transform(df5[['assortment']].values)
        
        #df5['sales'] = np.log1p(df5['sales'])
        
        df5['day_of_week_sin'] = df5['day_of_week'].apply(lambda x: np.sin(x * (2. *  np.pi / 7)))
        df5['day_of_week_cos'] = df5['day_of_week'].apply(lambda x: np.cos(x * (2. *  np.pi / 7)))

        df5['month_sin'] = df5['month'].apply(lambda x: np.sin(x * (2. *  np.pi / 12)))
        df5['month_cos'] = df5['month'].apply(lambda x: np.cos(x * (2. *  np.pi / 12)))

        df5['day_sin'] = df5['day'].apply(lambda x: np.sin(x * (2. *  np.pi / 30)))
        df5['day_cos'] = df5['day'].apply(lambda x: np.cos(x * (2. *  np.pi / 30)))

        df5['week_of_year_sin'] = df5['week_of_year'].apply(lambda x: np.sin(x * (2. *  np.pi / 52)))
        df5['week_of_year_cos'] = df5['week_of_year'].apply(lambda x: np.cos(x * (2. *  np.pi / 52)))

        #Final cols selected according with boruta
        selected = [
            'store', 
            'promo', 
            'store_type', 
            'assortment', 
            'competition_distance',
            'competition_open_since_month', 
            'competition_open_since_year', 
            'promo2',
            'promo2_since_week', 
            'promo2_since_year', 
            'competition_time_month',
            'promo_time_week', 
            'day_of_week_sin', 
            'day_of_week_cos', 
            'month_sin',
            'month_cos',
            'day_sin', 
            'day_cos',
            'week_of_year_sin', 
            'week_of_year_cos'] 

        feature_add = ['date']#, 'sales'
        #selected.extend(feature_add)

        return df5[selected]

    def get_prediction(self, model, original_data, test_data):
        pred = model.predict(test_data)

        original_data['prediction'] = np.expm1(pred)

        return original_data.to_json(orient='records', date_format='iso')