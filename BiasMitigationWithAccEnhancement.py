import copy
import warnings
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.svm import SVC
from sklearn.manifold import MDS
from itertools import combinations
from collections import defaultdict
from sklearn.linear_model import LassoCV
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.inspection import permutation_importance
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.feature_selection import mutual_info_classif

warnings.filterwarnings("ignore")

# default parameters
default_params = {
    'seed': 0,
    'alpha_O': 0.8,
    'clf_name': 'LR',
    'h_order': 'default',
    'output_step': False,
    'epsilon_threshold': 0.001,
    'is_bias_mitigation': True,
    'is_acc_enhancement': False,
    'rate_acc_enhancement': 0.02,
    'transform_method': 't1',
    'rebin_method': 'r1',
    'select_method': 'a2',
    'cal_dist_method': 'd1A',
    'train_rate': 0.2,
    'step_method': 'd3A',
    'exponent_stream': 'd4A',
}

dataset_info = {
    'uci_adult': {
        'load_path': './adult.data.csv',
        'attrs': [
            'age', 'workclass', 'fnlwgt', 'education', 'education num','marital status',
            'occupation', 'relationship', 'race', 'sex','capital gain', 'capital loss',
            'hours per week', 'native country', 'income'
        ],
        'num_attrs': [
            'age', 'fnlwgt', 'education num', 'capital gain', 'capital loss', 'hours per week'
        ],
        'cate_attrs': [
            'workclass', 'education', 'marital status', 'occupation', 'relationship', 'race',
            'sex', 'native country', 'income'
        ]
    },

    'esg_footprint_indi': {
        'load_path': './ESG_footprint_UK_indi.csv',
        'attrs': [
            'sex', 'geography', 'ethnic_group', 'nationality', 'occupation', 'Diesel car',
            'Diesel consumption', 'Electricity consumption', 'Family type', 'Gas Availability',
            'Gas consumption', 'House members', 'Individual type', 'Motor cycle', 'Number of bedrooms',
            'Petrol car', 'Petrol consumption'
        ],
        'cate_attrs': [
            'sex', 'geography', 'ethnic_group', 'nationality', 'occupation', 'Diesel car',
            'Diesel consumption', 'Electricity consumption', 'Family type', 'Gas Availability',
            'Gas consumption', 'House members', 'Individual type', 'Motor cycle', 'Number of bedrooms',
            'Petrol car', 'Petrol consumption'
        ],
        'num_attrs': []
    },

    'esg_footprint_busi': {
        'load_path': './ESG_footprint_UK_busi.csv',
        'attrs': [
            'coal_consumption', 'electricity_consumption', 'gas_consumption',
            'petrol_consumption', 'NG_consumption', 'Structure_Type', 'primary_sector',
            'entity_status', 'annual_turnover', 'Geographic'
        ],
        'cate_attrs': [
            'coal_consumption', 'electricity_consumption', 'gas_consumption',
            'petrol_consumption', 'NG_consumption', 'Structure_Type',
            'primary_sector', 'entity_status', 'annual_turnover', 'Geographic'
        ],
       'num_attrs': []
    },

    'corporate_env_impact': {
        'load_path': './Corporate_Environmental_Impact.csv',
        'attrs': [
            'year', 'industry', 'establishment_year', 'annualIncrease', 'continent',
            'eco_level', 'revenue', 'operatingIncome', 'environmentalCost',
            'WorkingCapacity', 'fishProductionCapacity', 'CropProductionCapacity',
            'MeatProductionCapacity', 'Biodiversity', 'AbioticResources',
            'Waterproductioncapacity', 'WoodProductionCapacity', 'imputed'
        ],
        'cate_attrs': [
            'year', 'industry', 'establishment_year', 'annualIncrease', 'continent',
            'eco_level', 'environmentalCost'
        ],
        'num_attrs': [
            'revenue', 'operatingIncome', 'WorkingCapacity', 'fishProductionCapacity',
            'CropProductionCapacity', 'MeatProductionCapacity', 'Biodiversity',
            'AbioticResources', 'Waterproductioncapacity', 'WoodProductionCapacity', 'imputed'
        ]
    },

    'unicon_campus_energy_consumption': {
        'load_path': 'UNICON_Campus_Energy_Consumption.csv',
        'attrs': [
            'campus_id', 'building_id', 'energy_consumption', 'gas_consumption',
            'built_year', 'category', 'gross_floor_area', 'room_area', 'capacity',
            'air_temperature', 'apparent_temperature', 'dew_point_temperature',
            'relative_humidity', 'wind_direction', 'wind_speed', 'energy_daily_increase',
            'gas_daily_increase'
        ],
        'num_attrs': [
            'gross_floor_area', 'room_area', 'capacity', 'air_temperature',
            'apparent_temperature', 'dew_point_temperature', 'relative_humidity',
            'wind_direction', 'wind_speed'
        ],
        'cate_attrs': [
            'campus_id', 'building_id', 'energy_consumption', 'gas_consumption',
            'built_year', 'category', 'energy_daily_increase', 'gas_daily_increase'
        ]
    }
}


# Framework of bias mitigation with accuracy accuracy
class BMwithAE:
    def __init__(self, label_O='race', label_Y='income', dataset_name='uci_adult') -> None:
        self.result_list = []
        self.label_O = label_O
        self.label_Y = label_Y

        default_params['dataset_name'] = dataset_name
        self.attrs = dataset_info[dataset_name]['attrs']
        self.num_attrs = dataset_info[dataset_name]['num_attrs']
        self.load_path = dataset_info[dataset_name]['load_path']
        self.cate_attrs = dataset_info[dataset_name]['cate_attrs']


    def load_data(self) -> None:
        self.df_data = pd.read_csv(self.load_path, header=None)
        self.df_data.columns = self.attrs
        
        if type(self.label_O) == list:
            def combine_col(row):
                return ' '.join(row.values.astype(str))
            
            col_name = ' '.join(self.label_O)
            self.df_data[col_name] = self.df_data[self.label_O].apply(combine_col, axis=1)
            self.df_data.drop(columns=self.label_O, inplace=True)
            self.cate_attrs = list(set(self.cate_attrs) - set(self.label_O))
            self.cate_attrs.append(col_name)
            self.label_O = col_name
        
        df = self.df_data[self.cate_attrs].copy()
        self.cate_dict = defaultdict(LabelEncoder)
        encoded_series = df.apply(lambda x: self.cate_dict[x.name].fit_transform(x))
        df_data_processed = pd.concat([self.df_data.drop(self.cate_attrs,axis=1), encoded_series], axis=1)
        self.df_data_bk = (pd.concat([df_data_processed[self.num_attrs].astype(np.float64), df_data_processed[self.cate_attrs]], axis=1)).copy()


    def get_sub(self, indexes, h_order) -> list:
        if h_order < -1:
            h_order = -1
        
        sub_index = []
        
        for i in range(len(indexes), h_order, -1):
            temp = [list(x) for x in combinations(indexes, i)]
            
            if len(temp) > 0:
                sub_index.extend(temp)
        
        return sub_index
    

    def cal_epsilon(self, df_data_temp) -> pd.Series:
        comb_label_arr = combinations(df_data_temp[self.label_O].unique(), 2)
        df_S_full = pd.DataFrame()
        for comb_label in comb_label_arr:
            df_data_processed_tmp = df_data_temp[df_data_temp[self.label_O].isin(comb_label)].copy()
            df_data_processed_tmp[self.num_attrs] = df_data_processed_tmp[self.num_attrs].apply(lambda x:x if x.min()==x.max() else (x-x.min())/(x.max()-x.min()))
            p, n = comb_label
            temp_diff_dict = {}
            for attribute in [x for x in self.cate_attrs if x not in [self.label_O, self.label_Y]]:
                df_1 = df_data_processed_tmp[[attribute, self.label_O]][df_data_processed_tmp[self.label_O] == p].groupby(attribute).count()
                df_2 = df_data_processed_tmp[[attribute, self.label_O]][df_data_processed_tmp[self.label_O] == n].groupby(attribute).count()
                df_0 = pd.concat([df_1, df_2], axis=1).fillna(0)
                df_0.columns = ['p', 'n']
                df_c = (df_0['p']/(df_0['p'].sum()) - df_0['n']/(df_0['n'].sum())).abs()
                temp_diff_dict[attribute] = df_c.mean()
            df_1 = df_data_processed_tmp[self.num_attrs][df_data_processed_tmp[self.label_O] == p].mean()
            df_2 = df_data_processed_tmp[self.num_attrs][df_data_processed_tmp[self.label_O] == n].mean()
            df_c = (df_1 - df_2).abs()
            df_S_full[str(comb_label)] = pd.concat([df_c, pd.Series(temp_diff_dict)])
        
        index_arr = list(df_S_full.index) + ['origin']
        distance_matrix = pd.DataFrame(index=index_arr, columns=index_arr, data=np.nan)

        for x_a in range(len(index_arr)):
            distance_matrix.iloc[x_a, x_a] = 0

        df_S_sq  = df_S_full ** 2
        temp_dict = {}
        for sub in self.get_sub(df_S_full.index, self.test_params['h_order']-1):
            if self.test_params['cal_dist_method'] == 'd1A':
                temp_dict[tuple(sorted(sub))] = df_S_sq.loc[sub].mean() ** (1/2)
            elif self.test_params['cal_dist_method'] == 'd1B':
                temp_dict[tuple(sorted(sub))] = (df_S_sq.loc[sub].sum()) ** (1/2)
        
        
        def cal_dist(attr_1, attr_2):
            temp_index_list = df_S_full.index.tolist()
            if attr_1 != 'origin':
                temp_index_list.remove(attr_1)
            if attr_2 != 'origin':
                temp_index_list.remove(attr_2)
            result_list = []
            temp_index_list = self.get_sub(temp_index_list, self.test_params['h_order']-(len(df_S_full.index.tolist()) - len(temp_index_list)))
            for s in temp_index_list:
                s_1 = (lambda x:sorted(s + [x]) if x != 'origin' else sorted(s))(attr_1)
                s_2 = (lambda x:sorted(s + [x]) if x != 'origin' else sorted(s))(attr_2)
                if (attr_1 == 'origin') and (len(s) == 0):
                    result_list.append(temp_dict[tuple(s_2)].max())
                elif (attr_2 == 'origin') and (len(s) == 0):
                    result_list.append(temp_dict[tuple(s_1)].max())
                else:
                    result_list.append(temp_dict[tuple(s_1)].max() - temp_dict[tuple(s_2)].max())
            distance_matrix_val = np.mean(np.abs(result_list))
            return distance_matrix_val
        

        for attr_1,attr_2 in combinations(index_arr, 2):
            dist = cal_dist(attr_1, attr_2)
            distance_matrix.loc[attr_1, attr_2] = dist
            distance_matrix.loc[attr_2, attr_1] = dist
        mds = MDS(n_components=2, dissimilarity='precomputed', random_state=self.test_params['seed'], n_init=4, max_iter=10000, eps=1e-10, normalized_stress='auto')
        pts = mds.fit_transform(distance_matrix)
        pts = pts - pts[-1]
        epsilon_concentration_matrix = pd.DataFrame(data=pts, index=distance_matrix.index).T
        df_epsilon = (epsilon_concentration_matrix**2).sum().apply(np.sqrt)
        return df_epsilon


    def process_data(self, df_data_processed, changed_dict) -> pd.DataFrame:
        df_data_temp = df_data_processed.copy()
        for attribute in changed_dict.keys():
            df_temp_attr = df_data_temp[attribute].copy()
            if attribute in self.num_attrs:
                if self.test_params['transform_method'] == 't1':
                    change = changed_dict[attribute]
                    if change['beta_O']==0 or change['beta_Y']==0:
                        df_temp_attr = self.test_params['alpha_O'] * ((df_temp_attr.abs()) ** (change['beta_O'])) * (df_temp_attr.apply(np.sign)) \
                                        + (1-self.test_params['alpha_O']) * ((df_temp_attr.abs()) ** (change['beta_Y'])) * (df_temp_attr.apply(np.sign))
                    else:
                        up_index = self.test_params['exponent_stream']
                        df_temp_attr = self.test_params['alpha_O'] * ((df_temp_attr.abs()) ** (change['beta_O']**up_index)) * (df_temp_attr.apply(np.sign)) \
                                            + (1-self.test_params['alpha_O']) * ((df_temp_attr.abs()) ** (change['beta_Y']**up_index)) * (df_temp_attr.apply(np.sign))
                elif self.test_params['transform_method'] == 't2':
                    change = changed_dict[attribute]
                    if change['beta_O']==0 or change['beta_Y']==0:
                        df_temp_attr = self.test_params['alpha_O'] * ((df_temp_attr.abs()) ** (change['beta_O'])) * (df_temp_attr.apply(np.sign)) \
                                        + (1-self.test_params['alpha_O']) * ((df_temp_attr.abs()) ** (change['beta_Y'])) * (df_temp_attr.apply(np.sign))
                    else:
                        up_index = self.test_params['exponent_stream']
                        beta_O_list = np.arange(1, change['beta_O']+1, 2)
                        beta_Y_list = np.arange(1, change['beta_Y']+1, 2)
                        df_temp_attr_O = df_temp_attr.copy()
                        df_temp_attr_Y = df_temp_attr.copy()
                        beta_O = 1.0
                        beta_Y = 1.0
                        for temp_beta_O in beta_O_list:
                            beta_O = beta_O*temp_beta_O
                        for temp_beta_Y in beta_Y_list:
                            beta_Y = beta_Y*temp_beta_Y
                        df_temp_attr = self.test_params['alpha_O'] * ((df_temp_attr.abs()) ** (beta_O**up_index)) * (df_temp_attr.apply(np.sign)) \
                                            + (1-self.test_params['alpha_O']) * ((df_temp_attr.abs()) ** (beta_Y**up_index)) * (df_temp_attr.apply(np.sign))
                elif self.test_params['transform_method'] == 't3':
                    change = changed_dict[attribute]
                    if change['beta_O']==0 or change['beta_Y']==0:
                        df_temp_attr = self.test_params['alpha_O'] * ((df_temp_attr.abs()) ** (change['beta_O'])) * (df_temp_attr.apply(np.sign)) \
                                        + (1-self.test_params['alpha_O']) * ((df_temp_attr.abs()) ** (change['beta_Y'])) * (df_temp_attr.apply(np.sign))
                    else:
                        up_index = self.test_params['exponent_stream']
                        beta_O_list = np.arange(1, change['beta_O']+1, 2)
                        beta_Y_list = np.arange(1, change['beta_Y']+1, 2)
                        df_temp_attr_O = df_temp_attr.copy().abs() ** (float(beta_O_list[0])**up_index)
                        df_temp_attr_Y = df_temp_attr.copy().abs() ** (float(beta_O_list[0])**up_index)
                        for beta_O in beta_O_list[1:]:
                            df_temp_attr_O += (df_temp_attr_O ** (float(beta_O)**up_index))
                        for beta_Y in beta_Y_list[1:]:
                            df_temp_attr_Y += (df_temp_attr_Y ** (float(beta_Y)**up_index))
                        df_temp_attr = self.test_params['alpha_O'] * df_temp_attr_O * (df_temp_attr.apply(np.sign)) \
                                            + (1-self.test_params['alpha_O']) * df_temp_attr_Y * (df_temp_attr.apply(np.sign))
            elif attribute in self.cate_attrs:
                re_bin_dict = changed_dict[attribute]
                for cate in list(re_bin_dict.keys()):
                    df_temp_attr[df_temp_attr == cate] = re_bin_dict[cate]
            df_data_temp[attribute] = df_temp_attr
        return df_data_temp
    

    def sum_of_differences_using_combinations(self, nums):
        return sum(abs(a - b) for a, b in combinations(nums, 2)) / len(list(combinations(nums, 2)))


    def fit_model(self, df_data_train, df_data_test, train_rate, is_start=False):
        if is_start:
            if self.test_params['clf_name'] == 'LR':
                self.model = LogisticRegression(random_state=self.test_params['seed'])
            elif self.test_params['clf_name'] == 'DT':
                self.model = DecisionTreeClassifier(random_state=self.test_params['seed'])
            elif self.test_params['clf_name'] == 'KNN':
                self.model = KNeighborsClassifier()
            elif self.test_params['clf_name'] == 'GBDT':
                self.model = GradientBoostingClassifier(random_state=self.test_params['seed'])
            elif self.test_params['clf_name'] == 'ADABoost':
                self.model = AdaBoostClassifier(random_state=self.test_params['seed'])
            elif self.test_params['clf_name'] == 'NB':
                self.model = GaussianNB()
            elif self.test_params['clf_name'] == 'SVM':
                self.model = SVC(random_state=self.test_params['seed'], max_iter=100)
            elif self.test_params['clf_name'] == 'MLP':
                self.model = MLPClassifier(random_state=self.test_params['seed'])
            elif self.test_params['clf_name'] == 'XGBoost':
                self.model = xgb.XGBClassifier(random_state=self.test_params['seed'])
            self.model.fit(df_data_train.drop(columns=[self.label_O, self.label_Y]), df_data_train[self.label_Y])
        elif train_rate == 1:
            self.model.fit(df_data_train.drop(columns=[self.label_O, self.label_Y]), df_data_train[self.label_Y])
        elif 0 < train_rate < 1:
            train_X = df_data_train.drop(columns=[self.label_O, self.label_Y]).sample(frac=self.test_params['train_rate'], random_state=self.test_params['seed'])
            train_Y = self.df_data_bk.loc[train_X.index][self.label_Y]
            self.model.fit(train_X, train_Y)
        
        acc = self.model.score(df_data_test.drop(columns=[self.label_O, self.label_Y]), df_data_test[self.label_Y])
        pred_y = pd.Series(self.model.predict(df_data_test.drop(columns=[self.label_O, self.label_Y])), name='pred_y')
        df_data_test.index = pred_y.index
        cross_df = pd.concat([df_data_test[self.label_O], df_data_test[self.label_Y], pred_y], axis=1)
        cross_df.columns = ['label_O', 'label_Y', 'pred_Y']
        temp_list = []
        for i in cross_df['label_O'].unique():
            temp_list.append(len(cross_df[(cross_df['label_O'] == i) & (cross_df['pred_Y'] == 1)]) / len(cross_df[(cross_df['label_O'] == i)]))
        delta_sp = self.sum_of_differences_using_combinations(temp_list)

        delta_eo = 0
        for y in cross_df['label_Y'].unique():
            temp = cross_df[cross_df['label_Y'] == y]
            cross_tab_temp = pd.crosstab(temp['label_O'], temp['pred_Y'], normalize=0)

            try:
                if 1 in cross_tab_temp.columns:
                    delta_eo += self.sum_of_differences_using_combinations(cross_tab_temp[1].values)
                elif 0 in cross_tab_temp.columns:
                    delta_eo += self.sum_of_differences_using_combinations(cross_tab_temp[0].values)
            except:
                pass
        
        if acc < 0.5:
            acc = 1 - acc
        delta_bias = 0.5 * delta_sp + 0.25 * delta_eo

        bias_dict = {'delta_bias': delta_bias,
                     'delta_sp': delta_sp,
                     'delta_eo': delta_eo}
        return acc, bias_dict


    def BMwithAE_loop(self, **kwgs):
        self.test_params = copy.deepcopy(default_params)
        self.test_params.update(kwgs)
        
        if self.test_params['h_order'] == 'default':
            if type(self.label_O) != list:
                self.test_params['h_order'] = len(self.attrs) - 2 - 1
            else:
                self.test_params['h_order'] = len(self.attrs) - 2 - len(self.label_O)
        
        if self.test_params['exponent_stream'] == 'd4A':
            self.test_params['exponent_stream'] = 1
        elif self.test_params['exponent_stream'] == 'd4B':
            self.test_params['exponent_stream'] = -1
        
        df_data_train, df_data_test = train_test_split(self.df_data_bk, train_size=0.8, test_size=0.2, random_state=self.test_params['seed'])
        acc_base, delta_bias_base = self.fit_model(df_data_train, df_data_test, self.test_params['train_rate'], is_start=True)
        df_epsilon = self.cal_epsilon(df_data_train)
        result = {
            'label_O': self.label_O,
            'label_Y': self.label_Y,
            'acc_base': acc_base,
            'delta_bias_base': delta_bias_base,
            'epsilon_base': df_epsilon.sort_values(ascending=False)[0],
            'epsilon_dict': df_epsilon.to_dict(),
        }
        result.update(self.test_params)
        
        step_num = 0
        epsilon_step = df_epsilon.sort_values(ascending=False)[0]
        acc_step = acc_base
        info_step = {}
        changed_dict = {}
        attr_tag_ndrop = {}
        attr_tag_nskip = {}
        
        for attr in self.attrs:
            if (attr is not self.label_O) and (attr is not self.label_Y):
                attr_tag_ndrop[attr] = True
                attr_tag_nskip[attr] = True

                if attr in self.num_attrs:
                    changed_dict[attr] = {'beta_O':1, 'beta_Y':1}
                elif attr in self.cate_attrs:
                    changed_dict[attr] = {}

        if self.test_params['step_method'] == 'd3A':
            while step_num<20:
                step_num += 1
                info_step[step_num] = {}

                # bias mitigation
                if self.test_params['is_bias_mitigation']:
                    df_data_temp = self.process_data(df_data_train, changed_dict)
                    df_epsilon = self.cal_epsilon(df_data_temp).sort_values(ascending=False)
                
                    if df_epsilon.index[0] == 'origin':
                        print('Error Origin!')
                        break

                    try:
                        for attr_bias in df_epsilon.index:
                            if attr_tag_ndrop[attr_bias]:
                                break
                    except:
                        break

                    epsilon_step = df_epsilon[attr_bias]

                    if self.test_params['output_step']:
                        print(step_num, 'start mitigation', attr_bias, changed_dict[attr_bias])

                    temp_dict = copy.deepcopy(changed_dict)
                    step_num_0 = 0
                    while True:
                        step_num_0 += 1
                        df_data_bm_temp = self.process_data(df_data_train, temp_dict)
                        
                        if attr_bias in self.num_attrs:
                            temp_dict[attr_bias]['beta_O'] = temp_dict[attr_bias]['beta_O'] + 2
                        elif attr_bias in self.cate_attrs:
                            diff_change = []
                            for i,j in combinations(df_data_bm_temp[self.label_O].unique(), 2):
                                df_0 = df_data_bm_temp[[attr_bias, self.label_O]][df_data_bm_temp[self.label_O]==i].groupby(attr_bias).count()
                                df_1 = df_data_bm_temp[[attr_bias, self.label_O]][df_data_bm_temp[self.label_O]==j].groupby(attr_bias).count()
                                df_c = (df_1/(df_1.sum()) - df_0/(df_0.sum())).fillna(0).sort_values(by=self.label_O)
                                diff = (df_c.max() + df_c.min()).values[0]
                                change_temp = {df_c.index[-1]:df_c.index[0]}
                                diff_change.append([change_temp, diff])
                            diff_change = pd.DataFrame(diff_change, columns=['change', 'diff'])
                            change = diff_change[diff_change['diff'].abs() == diff_change['diff'].abs().max()]['change'].values[0]
                            temp_dict[attr_bias].update(change)
                        
                        df_temp_iter = self.process_data(df_data_train, temp_dict)
                        
                        if attr_bias in self.num_attrs:
                            if (df_temp_iter[attr_bias].abs().max() > 1e+5) or (temp_dict[attr_bias]['beta_O'] > 10):
                                attr_tag_ndrop[attr_bias] = False
                                changed_dict[attr_bias] = {'beta_O':0, 'beta_Y':0}
                                break
                        
                        if attr_bias in self.cate_attrs:
                            if (len(df_temp_iter[attr_bias].unique()) == 1):
                                attr_tag_ndrop[attr_bias] = False
                                changed_dict.update(temp_dict)
                                break
                        
                        df_epsilon = self.cal_epsilon(df_temp_iter)
                        temp_epsilon = df_epsilon[attr_bias]
                        
                        if temp_epsilon<epsilon_step:
                            epsilon_step = temp_epsilon
                            changed_dict.update(temp_dict)
                            break
                
                    df_temp_train_after_BM = self.process_data(df_data_train, changed_dict)
                    df_temp_test_after_BM = self.process_data(df_data_test, changed_dict)
                    df_epsilon = self.cal_epsilon(df_temp_train_after_BM)
                    acc_step, delta_bias_step = self.fit_model(df_temp_train_after_BM, df_temp_test_after_BM, train_rate=0)
                    info_step[step_num].update({
                        'attr_bias': attr_bias,
                        'epsilon_step_after_BM': df_epsilon[attr_bias],
                        'acc_step_after_BM': acc_step,
                        'epsilon_dict_after_BM': df_epsilon.to_dict(),
                        'delta_bias_step_after_BM': delta_bias_step
                    })
                
                # acc enhancement
                if self.test_params['is_acc_enhancement']:
                    df_data_temp = self.process_data(df_data_train, changed_dict)    

                    if self.test_params['output_step']:
                        print(step_num, 'start enhancement')
                    
                    if self.test_params['select_method'] == 'LassoCV_max':
                        clf = LassoCV(random_state=self.test_params['seed']).fit(df_data_temp.drop(columns=[self.label_O, self.label_Y]), df_data_temp[self.label_Y])
                        importance = np.abs(clf.coef_)
                        df_importance = pd.DataFrame(importance)
                        df_importance.index = df_data_temp.drop(columns=[self.label_O, self.label_Y]).columns
                        df_importance.columns = ['importance']
                        df_importance = df_importance.sort_values(by='importance', ascending=False)
                    elif self.test_params['select_method'] == 'a2':
                        clf = LassoCV(random_state=self.test_params['seed']).fit(df_data_temp.drop(columns=[self.label_O, self.label_Y]), df_data_temp[self.label_Y])
                        importance = np.abs(clf.coef_)
                        df_importance = pd.DataFrame(importance)
                        df_importance.index = df_data_temp.drop(columns=[self.label_O, self.label_Y]).columns
                        df_importance.columns = ['importance']
                        df_importance = df_importance.sort_values(by='importance', ascending=True)
                    elif self.test_params['select_method'] == 'a1':
                        importance = mutual_info_classif(df_data_temp.drop(columns=[self.label_O, self.label_Y]), df_data_temp[self.label_Y], random_state=self.test_params['seed'])
                        df_importance = pd.DataFrame(importance)
                        df_importance.index = df_data_temp.drop(columns=[self.label_O, self.label_Y]).columns
                        df_importance.columns = ['importance']
                        df_importance = df_importance.sort_values(by='importance', ascending=True)
                    elif self.test_params['select_method'] == 'MIC_max':
                        importance = mutual_info_classif(df_data_temp.drop(columns=[self.label_O, self.label_Y]), df_data_temp[self.label_Y], random_state=self.test_params['seed'])
                        df_importance = pd.DataFrame(importance)
                        df_importance.index = df_data_temp.drop(columns=[self.label_O, self.label_Y]).columns
                        df_importance.columns = ['importance']
                        df_importance = df_importance.sort_values(by='importance', ascending=False)
                    elif self.test_params['select_method'] == 'a3':
                        importance = permutation_importance(self.model, df_data_temp.drop(columns=[self.label_O, self.label_Y]), df_data_temp[self.label_Y], random_state=self.test_params['seed']).importances_mean
                        df_importance = pd.DataFrame(importance)
                        df_importance.index = df_data_temp.drop(columns=[self.label_O, self.label_Y]).columns
                        df_importance.columns = ['importance']
                        df_importance = df_importance.sort_values(by='importance', ascending=True)

                    for attr_acc in df_importance.index:
                        if attr_tag_ndrop[attr_acc]:
                            break
                    
                    if self.test_params['output_step']:
                        print(step_num, 'acc enhancement', attr_acc)

                    temp_dict = copy.deepcopy(changed_dict)
                    while attr_tag_ndrop[attr_acc]:
                        df_data_ae_temp = self.process_data(df_data_train, temp_dict)

                        if attr_acc in self.num_attrs:
                            temp_dict[attr_acc]['beta_Y'] = temp_dict[attr_acc]['beta_Y'] + 2
                        elif attr_acc in self.cate_attrs:
                            if self.test_params['rebin_method'] == 'r1':
                                diff_change = []
                                for i,j in combinations(df_data_ae_temp[self.label_Y].unique(), 2):
                                    df_0 = df_data_ae_temp[[attr_acc, self.label_Y]][df_data_ae_temp[self.label_Y]==i].groupby(attr_acc).count()
                                    df_1 = df_data_ae_temp[[attr_acc, self.label_Y]][df_data_ae_temp[self.label_Y]==j].groupby(attr_acc).count()
                                    df_c = (df_1 - df_0).fillna(0).abs().sort_values(by=self.label_Y)
                                    diff = (df_c.max() + df_c.min()).values[0]
                                    change_temp = {df_c.index[-1]:df_c.index[0]}
                                    diff_change.append([change_temp, diff])
                                diff_change = pd.DataFrame(diff_change, columns=['change', 'diff'])
                                change = diff_change[diff_change['diff'].abs() == diff_change['diff'].abs().max()]['change'].values[0]
                                temp_dict[attr_acc].update(change)
                            
                            elif self.test_params['rebin_method'] == 'r2':
                                df_c = df_data_ae_temp[attr_acc].value_counts().sort_values()
                                # print(df_c)
                                change = {df_c.index[1]:df_c.index[0]}
                                temp_dict[attr_acc].update(change)
                            
                            elif self.test_params['rebin_method'] == 'r3':
                                change = {}
                                best_acc_temp = 0
                                for i,j in combinations(df_data_ae_temp[attr_acc].unique(), 2):
                                    change_temp = {i:j}
                                    temp_dict_0 = temp_dict.copy()
                                    temp_dict_0[attr_acc].update(change_temp)
                                    df_temp_iter_train_0 = self.process_data(df_data_train, temp_dict_0)
                                    df_temp_iter_test_0 = self.process_data(df_data_test, temp_dict_0)
                                    acc_temp, _ = self.fit_model(df_temp_iter_train_0, df_temp_iter_test_0, train_rate=0)
                                    if acc_temp > best_acc_temp:
                                        change = change_temp
                                        best_acc_temp = acc_temp

                        df_temp_iter_train = self.process_data(df_data_train, temp_dict)
                        df_temp_iter_test = self.process_data(df_data_test, temp_dict)

                        try:
                            acc_temp, _ = self.fit_model(df_temp_iter_train, df_temp_iter_test, train_rate=self.test_params['train_rate'])
                        except:
                            attr_tag_nskip[attr_acc] = False
                            break

                        if attr_acc in self.num_attrs:
                            if (df_temp_iter_train[attr_acc].abs().max() > 1e+5) or (temp_dict[attr_acc]['beta_Y'] > 10):
                                attr_tag_nskip[attr_acc] = False
                                attr_tag_ndrop[attr_acc] = False
                                break
                        
                        if attr_acc in self.cate_attrs:
                            if (len(df_temp_iter_train[attr_acc].unique()) == 1):
                                attr_tag_nskip[attr_acc] = False
                                attr_tag_ndrop[attr_acc] = False
                                break
  
                        if acc_temp > acc_step:
                            changed_dict.update(temp_dict)
                            break

                    df_temp_train_after_AE = self.process_data(df_data_train, changed_dict)
                    df_temp_test_after_AE = self.process_data(df_data_test, changed_dict)
                    df_epsilon = self.cal_epsilon(df_temp_train_after_AE)
                    try:
                        epsilon_step = df_epsilon[attr_bias]
                    except:
                        epsilon_step = df_epsilon.sort_values(ascending=False)[0]
                    acc_step, delta_bias_step = self.fit_model(df_temp_train_after_AE, df_temp_test_after_AE, train_rate=self.test_params['train_rate'])
                    info_step[step_num].update({
                        'attr_acc': attr_acc,
                        'epsilon_step_after_AE': epsilon_step,
                        'acc_step_after_AE': acc_step,
                        'epsilon_dict_after_AE': df_epsilon.to_dict(),
                        'delta_bias_step_after_AE': delta_bias_step
                    })
                
                if self.test_params['is_bias_mitigation']:
                    if epsilon_step<self.test_params['epsilon_threshold']:
                        break
                elif self.test_params['is_acc_enhancement']:
                    if (acc_step >= ((1 + self.test_params['rate_acc_enhancement']) * acc_base)):
                        break
                else:
                    break
        
        elif self.test_params['step_method'] == 'd3B':

            while step_num<11:
                step_num += 1
                info_step[step_num] = {}

                # bias mitigation
                if self.test_params['is_bias_mitigation']:
                    epsilon_num = 0
                    info_iter_bias = {}
                    
                    while epsilon_step>self.test_params['epsilon_threshold']:
                        epsilon_num += 1
                        if epsilon_num>5:
                            break
                        info_iter_bias[epsilon_num] = {}

                        df_data_temp = self.process_data(df_data_train, changed_dict)
                        df_epsilon = self.cal_epsilon(df_data_temp).sort_values(ascending=False)
                
                        if df_epsilon.index[0] == 'origin':
                            print('Error Origin!')
                            break

                        try:
                            for attr_bias in df_epsilon.index:
                                if attr_tag_ndrop[attr_bias]:
                                    break
                        except:
                            break

                        epsilon_step = df_epsilon[attr_bias]

                        if self.test_params['output_step']:
                            print(step_num, 'start mitigation', attr_bias)

                        temp_dict = copy.deepcopy(changed_dict)
                        while True:
                            if self.test_params['output_step']:
                                print(changed_dict[attr_bias])
                            df_data_bm_temp = self.process_data(df_data_train, temp_dict)
                            
                            if attr_bias in self.num_attrs:
                                temp_dict[attr_bias]['beta_O'] = temp_dict[attr_bias]['beta_O'] + 2
                            elif attr_bias in self.cate_attrs:
                                diff_change = []
                                for i,j in combinations(df_data_bm_temp[self.label_O].unique(), 2):
                                    df_0 = df_data_bm_temp[[attr_bias, self.label_O]][df_data_bm_temp[self.label_O]==i].groupby(attr_bias).count()
                                    df_1 = df_data_bm_temp[[attr_bias, self.label_O]][df_data_bm_temp[self.label_O]==j].groupby(attr_bias).count()
                                    df_c = (df_1/(df_1.sum()) - df_0/(df_0.sum())).fillna(0).sort_values(by=self.label_O)
                                    diff = (df_c.max() + df_c.min()).values[0]
                                    change_temp = {df_c.index[-1]:df_c.index[0]}
                                    diff_change.append([change_temp, diff])
                                diff_change = pd.DataFrame(diff_change, columns=['change', 'diff'])
                                change = diff_change[diff_change['diff'].abs() == diff_change['diff'].abs().max()]['change'].values[0]
                                temp_dict[attr_bias].update(change)
                            
                            df_temp_iter = self.process_data(df_data_train, temp_dict)
                            
                            if attr_bias in self.num_attrs:
                                if (df_temp_iter[attr_bias].abs().max() > 1e+5) or (temp_dict[attr_bias]['beta_O'] > 10):
                                    attr_tag_ndrop[attr_bias] = False
                                    changed_dict[attr_bias] = {'beta_O':0, 'beta_Y':0}
                                    break
                            
                            if attr_bias in self.cate_attrs:
                                if (len(df_temp_iter[attr_bias].unique()) == 1):
                                    attr_tag_ndrop[attr_bias] = False
                                    changed_dict.update(temp_dict)
                                    break
                            
                            df_epsilon = self.cal_epsilon(df_temp_iter)
                            temp_epsilon = df_epsilon[attr_bias]
                            
                            if temp_epsilon<epsilon_step:
                                epsilon_step = temp_epsilon
                                changed_dict.update(temp_dict)
                                break
                        
                        df_temp_train_after_BM = self.process_data(df_data_train, changed_dict)
                        df_temp_test_after_BM = self.process_data(df_data_test, changed_dict)
                        df_epsilon = self.cal_epsilon(df_temp_train_after_BM)
                        acc_step, delta_bias_step = self.fit_model(df_temp_train_after_BM, df_temp_test_after_BM, train_rate=0)
                        info_iter_bias[epsilon_num].update({
                            'attr_bias': attr_bias,
                            'epsilon_step_after_BM': df_epsilon[attr_bias],
                            'acc_step_after_BM': acc_step,
                            'epsilon_dict_after_BM': df_epsilon.to_dict(),
                            'delta_bias_step_after_BM': delta_bias_step
                        })
                    
                    df_temp_train_after_BM = self.process_data(df_data_train, changed_dict)
                    df_temp_test_after_BM = self.process_data(df_data_test, changed_dict)
                    df_epsilon = self.cal_epsilon(df_temp_train_after_BM)
                    acc_step, delta_bias_step = self.fit_model(df_temp_train_after_BM, df_temp_test_after_BM, train_rate=0)
                    info_step[step_num].update({
                        'epsilon_step_after_BM': epsilon_step,
                        'acc_step_after_BM': acc_step,
                        'epsilon_dict_after_BM': df_epsilon.to_dict(),
                        'delta_bias_step_after_BM': delta_bias_step,
                        'info_iter_bias':info_iter_bias
                    })
                
                # acc enhancement
                if self.test_params['is_acc_enhancement']:
                    acc_num = 0
                    info_iter_acc = {}

                    while acc_step < ((1 + self.test_params['rate_acc_enhancement']) * acc_base):
                        acc_num += 1
                        if acc_num>5:
                            break
                        info_iter_acc[acc_num] = {}
                        
                        df_data_temp = self.process_data(df_data_train, changed_dict)
                        
                        if self.test_params['output_step']:
                            print(step_num, 'start enhancement')
                        
                        if self.test_params['select_method'] == 'LassoCV_max':
                            clf = LassoCV(random_state=self.test_params['seed']).fit(df_data_temp.drop(columns=[self.label_O, self.label_Y]), df_data_temp[self.label_Y])
                            importance = np.abs(clf.coef_)
                            df_importance = pd.DataFrame(importance)
                            df_importance.index = df_data_temp.drop(columns=[self.label_O, self.label_Y]).columns
                            df_importance.columns = ['importance']
                            df_importance = df_importance.sort_values(by='importance', ascending=False)
                        elif self.test_params['select_method'] == 'LassoCV_min':
                            clf = LassoCV(random_state=self.test_params['seed']).fit(df_data_temp.drop(columns=[self.label_O, self.label_Y]), df_data_temp[self.label_Y])
                            importance = np.abs(clf.coef_)
                            df_importance = pd.DataFrame(importance)
                            df_importance.index = df_data_temp.drop(columns=[self.label_O, self.label_Y]).columns
                            df_importance.columns = ['importance']
                            df_importance = df_importance.sort_values(by='importance', ascending=True)
                        elif self.test_params['select_method'] == 'MIC_min':
                            importance = mutual_info_classif(df_data_temp.drop(columns=[self.label_O, self.label_Y]), df_data_temp[self.label_Y], random_state=self.test_params['seed'])
                            df_importance = pd.DataFrame(importance)
                            df_importance.index = df_data_temp.drop(columns=[self.label_O, self.label_Y]).columns
                            df_importance.columns = ['importance']
                            df_importance = df_importance.sort_values(by='importance', ascending=True)
                        elif self.test_params['select_method'] == 'MIC_max':
                            importance = mutual_info_classif(df_data_temp.drop(columns=[self.label_O, self.label_Y]), df_data_temp[self.label_Y], random_state=self.test_params['seed'])
                            df_importance = pd.DataFrame(importance)
                            df_importance.index = df_data_temp.drop(columns=[self.label_O, self.label_Y]).columns
                            df_importance.columns = ['importance']
                            df_importance = df_importance.sort_values(by='importance', ascending=False)
                        elif self.test_params['select_method'] == 'PI_min':
                            importance = permutation_importance(self.model, df_data_temp.drop(columns=[self.label_O, self.label_Y]), df_data_temp[self.label_Y], random_state=self.test_params['seed']).importances_mean
                            df_importance = pd.DataFrame(importance)
                            df_importance.index = df_data_temp.drop(columns=[self.label_O, self.label_Y]).columns
                            df_importance.columns = ['importance']
                            df_importance = df_importance.sort_values(by='importance', ascending=True)

                        for attr_acc in df_importance.index:
                            if attr_tag_ndrop[attr_acc]:
                                break
                        
                        if self.test_params['output_step']:
                            print(step_num, 'acc enhancement', attr_acc)

                        temp_dict = copy.deepcopy(changed_dict)
                        while attr_tag_ndrop[attr_acc]:
                            if self.test_params['output_step']:
                                print(temp_dict[attr_acc])
                            df_data_ae_temp = self.process_data(df_data_train, temp_dict)

                            if attr_acc in self.num_attrs:
                                temp_dict[attr_acc]['beta_Y'] = temp_dict[attr_acc]['beta_Y'] + 2
                            elif attr_acc in self.cate_attrs:
                                if self.test_params['rebin_method'] == 'r1':
                                    diff_change = []
                                    for i,j in combinations(df_data_ae_temp[self.label_Y].unique(), 2):
                                        df_0 = df_data_ae_temp[[attr_acc, self.label_Y]][df_data_ae_temp[self.label_Y]==i].groupby(attr_acc).count()
                                        df_1 = df_data_ae_temp[[attr_acc, self.label_Y]][df_data_ae_temp[self.label_Y]==j].groupby(attr_acc).count()
                                        df_c = (df_1 - df_0).fillna(0).abs().sort_values(by=self.label_Y)
                                        diff = (df_c.max() + df_c.min()).values[0]
                                        change_temp = {df_c.index[-1]:df_c.index[0]}
                                        diff_change.append([change_temp, diff])
                                    diff_change = pd.DataFrame(diff_change, columns=['change', 'diff'])
                                    change = diff_change[diff_change['diff'].abs() == diff_change['diff'].abs().max()]['change'].values[0]
                                    temp_dict[attr_acc].update(change)
                                
                                elif self.test_params['rebin_method'] == 'r2':
                                    df_c = df_data_ae_temp[attr_acc].value_counts().sort_values()
                                    # print(df_c)
                                    change = {df_c.index[1]:df_c.index[0]}
                                    temp_dict[attr_acc].update(change)
                                
                                elif self.test_params['rebin_method'] == 'r3':
                                    change = {}
                                    best_acc_temp = 0
                                    for i,j in combinations(df_data_ae_temp[attr_acc].unique(), 2):
                                        change_temp = {i:j}
                                        temp_dict_0 = temp_dict.copy()
                                        temp_dict_0[attr_acc].update(change_temp)
                                        df_temp_iter_train_0 = self.process_data(df_data_train, temp_dict_0)
                                        df_temp_iter_test_0 = self.process_data(df_data_test, temp_dict_0)
                                        acc_temp, _ = self.fit_model(df_temp_iter_train_0, df_temp_iter_test_0, train_rate=0)
                                        if acc_temp > best_acc_temp:
                                            change = change_temp
                                            best_acc_temp = acc_temp

                            df_temp_iter_train = self.process_data(df_data_train, temp_dict)
                            df_temp_iter_test = self.process_data(df_data_test, temp_dict)

                            try:
                                acc_temp, _ = self.fit_model(df_temp_iter_train, df_temp_iter_test, train_rate=0)
                            except:
                                attr_tag_nskip[attr_acc] = False
                                break

                            if attr_acc in self.num_attrs:
                                if (df_temp_iter_train[attr_acc].abs().max() > 1e+5) or (temp_dict[attr_acc]['beta_Y'] > 10):
                                    attr_tag_nskip[attr_acc] = False
                                    attr_tag_ndrop[attr_acc] = False
                                    break
                            
                            if attr_acc in self.cate_attrs:
                                if (len(df_temp_iter_train[attr_acc].unique()) == 1):
                                    attr_tag_nskip[attr_acc] = False
                                    attr_tag_ndrop[attr_acc] = False
                                    break

                            if acc_temp > acc_step:
                                changed_dict.update(temp_dict)
                                break
                        
                        df_temp_train_after_AE = self.process_data(df_data_train, changed_dict)
                        df_temp_test_after_AE = self.process_data(df_data_test, changed_dict)
                        df_epsilon = self.cal_epsilon(df_temp_train_after_AE)
                        try:
                            epsilon_step = df_epsilon[attr_bias]
                        except:
                            epsilon_step = df_epsilon.sort_values(ascending=False)[0]
                        acc_step, delta_bias_step = self.fit_model(df_temp_train_after_AE, df_temp_test_after_AE, train_rate=self.test_params['train_rate'])
                        info_iter_acc[acc_num].update({
                            'attr_acc': attr_acc,
                            'epsilon_step_after_AE': epsilon_step,
                            'acc_step_after_AE': acc_step,
                            'epsilon_dict_after_AE': df_epsilon.to_dict(),
                            'delta_bias_step_after_AE': delta_bias_step
                        })
                    
                    df_temp_train_after_AE = self.process_data(df_data_train, changed_dict)
                    df_temp_test_after_AE = self.process_data(df_data_test, changed_dict)
                    df_epsilon = self.cal_epsilon(df_temp_train_after_AE)
                    try:
                        epsilon_step = df_epsilon[attr_bias]
                    except:
                        epsilon_step = df_epsilon.sort_values(ascending=False)[0]
                    acc_step, delta_bias_step = self.fit_model(df_temp_train_after_AE, df_temp_test_after_AE, train_rate=0)
                    info_step[step_num].update({
                        'epsilon_step_after_AE': epsilon_step,
                        'acc_step_after_AE': acc_step,
                        'epsilon_dict_after_AE': df_epsilon.to_dict(),
                        'delta_bias_step_after_AE': delta_bias_step,
                        'info_iter_acc': info_iter_acc
                    })
                
                if self.test_params['is_bias_mitigation']:
                    if epsilon_step<self.test_params['epsilon_threshold']:
                        break
                elif self.test_params['is_acc_enhancement']:
                    if (acc_step >= ((1 + self.test_params['rate_acc_enhancement']) * acc_base)):
                        break
                else:
                    break
        
        result['epsilon_end'] = epsilon_step
        result['acc_end'] = acc_step
        
        try:
            result['delta_bias_end'] = delta_bias_step
        except:
            result['delta_bias_end'] = delta_bias_base
        
        result['epsilon_dict_end'] = df_epsilon
        result['info_step'] = info_step
        result['changed_dict'] = changed_dict
        self.result_list.append(result)

