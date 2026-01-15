import pandas as pd
import numpy as np
import xgboost as xgb
import gc
import datetime
import optuna
import os
from sklearn.metrics import roc_auc_score

def reduce_mem_usage(df, verbose=True):
    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    start_mem = df.memory_usage().sum() / 1024**2
    for col in df.columns:
        col_type = df[col].dtypes
        if col_type in numerics:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)
            else:
                if c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)
    end_mem = df.memory_usage().sum() / 1024**2
    if verbose: print('Mem. usage decreased to {:5.2f} Mb ({:.1f}% reduction)'.format(end_mem, 100 * (start_mem - end_mem) / start_mem))
    return df

# HYPERPARAMETER TUNING
def objective(trial, X_tr, y_tr, X_va, y_va):
    param = {
        'n_estimators': 5000,
        'max_depth': trial.suggest_int('max_depth', 9, 15),
        'learning_rate': trial.suggest_float('learning_rate', 0.005, 0.05, log=True),
        'subsample': trial.suggest_float('subsample', 0.7, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.3, 0.7),
        'gamma': trial.suggest_float('gamma', 0, 10),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 20),
        'missing': -1,
        'eval_metric': 'auc',
        'tree_method': 'hist',
        'grow_policy': 'lossguide',
        'n_jobs': -1,
        'early_stopping_rounds': 200,
        'random_state': 42
    }
    
    clf = xgb.XGBClassifier(**param)
    clf.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)
    
    return roc_auc_score(y_va, clf.predict_proba(X_va)[:, 1])

# ENCODING FUNCTIONS

# FREQUENCY ENCODE TOGETHER
def encode_FE(df1, df2, cols):
    new_df1_cols = {}
    new_df2_cols = {}
    for col in cols:
        df = pd.concat([df1[col], df2[col]])
        vc = df.value_counts(dropna=True, normalize=True).to_dict()
        vc[-1] = -1
        nm = col+'_FE'
        new_df1_cols[nm] = df1[col].map(vc).astype('float32')
        new_df2_cols[nm] = df2[col].map(vc).astype('float32')
        print(nm, ', ', end='')
    if new_df1_cols:
        df1 = pd.concat([df1, pd.DataFrame(new_df1_cols, index=df1.index)], axis=1)
    if new_df2_cols:
        df2 = pd.concat([df2, pd.DataFrame(new_df2_cols, index=df2.index)], axis=1)
    return df1, df2

# LABEL ENCODE
def encode_LE(col, train, test, verbose=True):
    df_comb = pd.concat([train[col], test[col]], axis=0)
    df_comb, _ = df_comb.factorize(sort=True)
    if df_comb.max() > 32000:
        train[col] = df_comb[:len(train)].astype('int32')
        test[col] = df_comb[len(train):].astype('int32')
    else:
        train[col] = df_comb[:len(train)].astype('int16')
        test[col] = df_comb[len(train):].astype('int16')
    if verbose: print(col, ', ', end='')

# COMBINE TWO COLUMNS
def encode_CB(col1, col2, train, test):
    nm = col1 + '_' + col2
    train[nm] = train[col1].astype(str) + '_' + train[col2].astype(str)
    test[nm] = test[col1].astype(str) + '_' + test[col2].astype(str)
    encode_LE(nm, train, test)

# GROUP AGGREGATION MEAN AND STD
def encode_AG(main_columns, uids, aggregations=['mean'], train_df=None, test_df=None, fillna=True, usena=False):
    new_train_cols = {}
    new_test_cols = {}
    for main_column in main_columns:
        for col in uids:
            for agg_type in aggregations:
                new_col_name = main_column + '_' + col + '_' + agg_type
                # LEAKAGE FIX: Only use train_df to calculate statistics
                temp_df = train_df[[col, main_column]].copy()
                if usena: temp_df.loc[temp_df[main_column] == -1, main_column] = np.nan
                
                # Calculate aggregation on train only
                temp_stat = temp_df.groupby([col])[main_column].agg([agg_type]).reset_index().rename(
                    columns={agg_type: new_col_name})
                
                temp_stat.index = list(temp_stat[col])
                temp_stat_dict = temp_stat[new_col_name].to_dict()

                new_train_cols[new_col_name] = train_df[col].map(temp_stat_dict).astype('float32')
                new_test_cols[new_col_name] = test_df[col].map(temp_stat_dict).astype('float32')

                if fillna:
                    new_train_cols[new_col_name] = new_train_cols[new_col_name].fillna(-1)
                    new_test_cols[new_col_name] = new_test_cols[new_col_name].fillna(-1)

                print(new_col_name, ', ', end='')
    
    if new_train_cols:
        train_df = pd.concat([train_df, pd.DataFrame(new_train_cols, index=train_df.index)], axis=1)
    if new_test_cols:
        test_df = pd.concat([test_df, pd.DataFrame(new_test_cols, index=test_df.index)], axis=1)
    return train_df, test_df

# GROUP AGGREGATION NUNIQUE
def encode_AG2(main_columns, uids, train_df=None, test_df=None):
    new_train_cols = {}
    new_test_cols = {}
    for main_column in main_columns:
        for col in uids:
            new_col_name = main_column + '_' + col + '_ct'
            # LEAKAGE FIX: Only use train_df to calculate statistics
            temp_df = train_df[[col, main_column]].copy()
            temp_stat = temp_df.groupby([col])[main_column].agg(['nunique']).reset_index().rename(
                columns={'nunique': new_col_name})
            
            temp_stat.index = list(temp_stat[col])
            temp_stat_dict = temp_stat[new_col_name].to_dict()
            
            # Use float32 to avoid IntCastingNaNError if NaNs are present before fillna
            temp_train = train_df[col].map(temp_stat_dict).astype('float32')
            temp_test = test_df[col].map(temp_stat_dict).astype('float32')
            
            # Fillna for unknown UIDs in test
            temp_train = temp_train.fillna(-1)
            temp_test = temp_test.fillna(-1)
            
            # Convert to int32 after filling NaNs
            new_train_cols[new_col_name] = temp_train.astype('int32')
            new_test_cols[new_col_name] = temp_test.astype('int32')
            
            print(new_col_name, ', ', end='')
            
    if new_train_cols:
        train_df = pd.concat([train_df, pd.DataFrame(new_train_cols, index=train_df.index)], axis=1)
    if new_test_cols:
        test_df = pd.concat([test_df, pd.DataFrame(new_test_cols, index=test_df.index)], axis=1)
    return train_df, test_df

def main():
    # LOAD DATA
    print('Loading data...')
    # Define columns to load to save memory and match the magic notebook
    v_cols = ['V'+str(x) for x in range(1,340)]
    cols = ['TransactionID', 'TransactionDT', 'TransactionAmt', 'ProductCD', 'card1', 'card2', 'card3', 'card4', 'card5', 'card6', 'addr1', 'addr2', 'dist1', 'dist2', 'P_emaildomain', 'R_emaildomain', 'C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8', 'C9', 'C10', 'C11', 'C12', 'C13', 'C14', 'D1', 'D2', 'D3', 'D4', 'D5', 'D6', 'D7', 'D8', 'D9', 'D10', 'D11', 'D12', 'D13', 'D14', 'D15', 'M1', 'M2', 'M3', 'M4', 'M5', 'M6', 'M7', 'M8', 'M9'] + v_cols
    
    train_transaction = pd.read_csv('train_transaction.csv', usecols=cols + ['isFraud'])
    train_identity = pd.read_csv('train_identity.csv')
    test_transaction = pd.read_csv('test_transaction.csv', usecols=cols)
    test_identity = pd.read_csv('test_identity.csv')

    # Rename identity columns for consistency
    train_identity.columns = [col.replace('-', '_') if col.startswith('id-') else col for col in train_identity.columns]
    test_identity.columns = [col.replace('-', '_') if col.startswith('id-') else col for col in test_identity.columns]

    X_train = train_transaction.merge(train_identity, on='TransactionID', how='left')
    X_test = test_transaction.merge(test_identity, on='TransactionID', how='left')
    
    X_train = reduce_mem_usage(X_train)
    X_test = reduce_mem_usage(X_test)

    y_train = X_train['isFraud'].copy()

    del train_transaction, train_identity, test_transaction, test_identity
    gc.collect()

    print('Train shape', X_train.shape, 'test shape', X_test.shape)

    # NORMALIZE D COLUMNS
    print('Normalizing D columns...')
    # The D Columns are "time deltas" from some point in the past. 
    # We transform them into absolute points in the past.
    for i in range(1, 16):
        if i in [1, 2, 3, 5, 9]: continue 
        X_train['D'+str(i)] = X_train['D'+str(i)] - X_train.TransactionDT/np.float32(24*60*60)
        X_test['D'+str(i)] = X_test['D'+str(i)] - X_test.TransactionDT/np.float32(24*60*60)

    # FEATURE ENGINEERING
    print('Feature engineering...')

    # CLEAN IDENTITY FEATURES
    print('Cleaning Identity features...')
    
    # OS cleaning
    def group_os(df):
        df['id_30'] = df['id_30'].fillna('unknown')
        df.loc[df['id_30'].str.contains('Windows', na=False), 'id_30'] = 'windows'
        df.loc[df['id_30'].str.contains('iOS', na=False), 'id_30'] = 'ios'
        df.loc[df['id_30'].str.contains('Mac OS', na=False), 'id_30'] = 'mac'
        df.loc[df['id_30'].str.contains('Android', na=False), 'id_30'] = 'android'
        df.loc[df['id_30'].str.contains('Linux', na=False), 'id_30'] = 'linux'
        return df
    
    X_train = group_os(X_train)
    X_test = group_os(X_test)

    # Browser cleaning
    X_train['id_31'] = X_train['id_31'].str.lower().str.replace(r'[^a-z]', '', regex=True)
    X_test['id_31'] = X_test['id_31'].str.lower().str.replace(r'[^a-z]', '', regex=True)
    
    def group_browser(df):
        df.loc[df['id_31'].str.contains('chrome', na=False), 'id_31'] = 'chrome'
        df.loc[df['id_31'].str.contains('firefox', na=False), 'id_31'] = 'firefox'
        df.loc[df['id_31'].str.contains('safari', na=False), 'id_31'] = 'safari'
        df.loc[df['id_31'].str.contains('edge', na=False), 'id_31'] = 'edge'
        df.loc[df['id_31'].str.contains('ie', na=False), 'id_31'] = 'ie'
        df.loc[df['id_31'].str.contains('opera', na=False), 'id_31'] = 'opera'
        return df
    
    X_train = group_browser(X_train)
    X_test = group_browser(X_test)

    # Resolution features
    print('Processing resolution features...')
    X_train['screen_width'] = X_train['id_33'].str.split('x', expand=True)[0].fillna(-1)
    X_train['screen_height'] = X_train['id_33'].str.split('x', expand=True)[1].fillna(-1)
    X_test['screen_width'] = X_test['id_33'].str.split('x', expand=True)[0].fillna(-1)
    X_test['screen_height'] = X_test['id_33'].str.split('x', expand=True)[1].fillna(-1)
    
    # Convert to numeric safely
    X_train['screen_width'] = pd.to_numeric(X_train['screen_width'], errors='coerce').fillna(-1).astype(np.int32)
    X_train['screen_height'] = pd.to_numeric(X_train['screen_height'], errors='coerce').fillna(-1).astype(np.int32)
    X_test['screen_width'] = pd.to_numeric(X_test['screen_width'], errors='coerce').fillna(-1).astype(np.int32)
    X_test['screen_height'] = pd.to_numeric(X_test['screen_height'], errors='coerce').fillna(-1).astype(np.int32)
    
    # Pixel count feature
    X_train['pixel_count'] = X_train['screen_width'] * X_train['screen_height']
    X_test['pixel_count'] = X_test['screen_width'] * X_test['screen_height']

    # EMAIL DOMAIN CLEANING
    print('Cleaning email domains...')
    emails = {'gmail': 'google', 'att.net': 'att', 'twc.com': 'spectrum', 
              'scantech.com': 'etc', 'netzero.net': 'netzero', 
              'prodigy.net.mx': 'at&t', 'charter.net': 'spectrum', 
              'live.com.mx': 'microsoft', 'connection.com.mx': 'etc', 
              'icloud.com': 'apple', 'ymail.com': 'yahoo', 
              'frontier.com': 'yahoo', 'rocketmail.com': 'yahoo', 
              'nmgb.com.mx': 'etc', 'netzero.com': 'netzero', 
              'bellsouth.net': 'at&t', 'hotmail.es': 'microsoft', 
              'hotmail.com': 'microsoft', 'live.com': 'microsoft', 
              'me.com': 'apple', 'msn.com': 'microsoft', 
              'yahoo.com.mx': 'yahoo', 'yahoo.com': 'yahoo', 
              'earthlink.net': 'earthlink', 'roadrunner.com': 'spectrum', 
              'verizon.net': 'verizon', 'outlook.com': 'microsoft', 
              'cox.net': 'cox', 'att.net': 'att', 'sbcglobal.net': 'at&t', 
              'aim.com': 'aol', 'foxmail.com': 'etc', 'twc.com': 'spectrum', 
              'frontiernet.net': 'yahoo', 'gmail.com': 'google', 
              'juno.com': 'etc', 'optimum.net': 'etc', 
              'cableone.net': 'etc', 'windstream.net': 'etc', 
              'suddenlink.net': 'etc', 'web.de': 'etc', 
              'outlook.es': 'microsoft', 'gmx.de': 'etc', 
              'yahoo.fr': 'yahoo', 'yahoo.es': 'yahoo', 
              'yahoo.de': 'yahoo', 'yahoo.co.uk': 'yahoo', 
              'yahoo.co.jp': 'yahoo', 'live.fr': 'microsoft', 
              'live.de': 'microsoft', 'hotmail.fr': 'microsoft', 
              'hotmail.co.uk': 'microsoft', 'hotmail.de': 'microsoft'}

    for col in ['P_emaildomain', 'R_emaildomain']:
        X_train[col+'_bin'] = X_train[col].map(emails)
        X_test[col+'_bin'] = X_test[col].map(emails)
    
    # TRANSACTION AMT FEATURES
    print('Processing TransactionAmt...')
    X_train['TransactionAmt_Log'] = np.log(X_train['TransactionAmt'])
    X_test['TransactionAmt_Log'] = np.log(X_test['TransactionAmt'])
    X_train['TransactionAmt_decimal'] = ((X_train['TransactionAmt'] - X_train['TransactionAmt'].astype(int)) * 1000).astype(int)
    X_test['TransactionAmt_decimal'] = ((X_test['TransactionAmt'] - X_test['TransactionAmt'].astype(int)) * 1000).astype(int)

    # LABEL ENCODE CATEGORICAL
    cat_cols = ['ProductCD', 'card4', 'card6', 'P_emaildomain', 'R_emaildomain', 'P_emaildomain_bin', 'R_emaildomain_bin', 'M1', 'M2', 'M3', 'M4', 'M5', 'M6', 'M7', 'M8', 'M9', 'id_12', 'id_15', 'id_16', 'id_23', 'id_27', 'id_28', 'id_29', 'id_30', 'id_31', 'id_33', 'id_34', 'id_35', 'id_36', 'id_37', 'id_38', 'DeviceType', 'DeviceInfo']
    for col in cat_cols:
        if col in X_train.columns:
            encode_LE(col, X_train, X_test)

    # COMBINE FEATURES
    encode_CB('card1', 'addr1', X_train, X_test)
    encode_CB('card1_addr1', 'P_emaildomain', X_train, X_test)

    # CREATE UID 
    X_train['day'] = X_train.TransactionDT / (24*60*60)
    X_test['day'] = X_test.TransactionDT / (24*60*60)
    
    X_train['uid'] = X_train.card1.astype(str)+'_'+X_train.card2.astype(str)+'_'+X_train.card3.astype(str)+'_'+X_train.card4.astype(str)+'_'+X_train.addr1.astype(str)+'_'+X_train.addr2.astype(str)
    X_test['uid'] = X_test.card1.astype(str)+'_'+X_test.card2.astype(str)+'_'+X_test.card3.astype(str)+'_'+X_test.card4.astype(str)+'_'+X_test.addr1.astype(str)+'_'+X_test.addr2.astype(str)

    # FREQUENCY ENCODING
    X_train, X_test = encode_FE(X_train, X_test, ['card1', 'card2', 'card3', 'card5', 'card1_addr1', 'card1_addr1_P_emaildomain', 'id_30', 'id_31'])

    # GROUP AGGREGATIONS
    X_train, X_test = encode_AG(['TransactionAmt', 'D4', 'D9', 'D10', 'D15'], ['card1', 'card1_addr1', 'uid'], ['mean', 'std'], X_train, X_test)
    X_train, X_test = encode_AG2(['C1', 'C2', 'C4', 'C5', 'C6', 'C7', 'C8', 'C10', 'C11', 'C12', 'C13', 'C14'], ['card1', 'card1_addr1', 'uid'], X_train, X_test)
    X_train, X_test = encode_AG2(['M1','M2','M3','M4','M5','M6','M7','M8','M9'], ['card1','card1_addr1','uid'], X_train, X_test)
    X_train, X_test = encode_AG(['C13', 'V310'], ['card1', 'card1_addr1', 'uid'], ['mean'], X_train, X_test)
    
    # NEW: Aggregations by email
    X_train, X_test = encode_AG(['TransactionAmt'], ['P_emaildomain_bin'], ['mean', 'std'], X_train, X_test)

    X_train['Transaction_hour'] = np.floor(X_train['TransactionDT'] / 3600) % 24
    X_test['Transaction_hour'] = np.floor(X_test['TransactionDT'] / 3600) % 24
    X_train['Transaction_day'] = np.floor(X_train['TransactionDT'] / (3600*24)) % 7
    X_test['Transaction_day'] = np.floor(X_test['TransactionDT'] / (3600*24)) % 7
    X_train, X_test = encode_FE(X_train, X_test, ['Transaction_hour', 'Transaction_day'])
    
    # NULL COUNT
    print('Calculating null counts...')
    X_train['null_count'] = X_train.isnull().sum(axis=1)
    X_test['null_count'] = X_test.isnull().sum(axis=1)

    # Prune Redundant V Columns
    # The 1st place solution drops 219 V columns.
    # We will drop columns with high correlation (>0.99)
    v_cols_to_drop = [
        'V300', 'V309', 'V111', 'V124', 'V106', 'V125', 'V315', 'V134', 'V102', 'V123', 'V316', 'V113',
        'V136', 'V305', 'V322', 'V313', 'V296', 'V110', 'V107', 'V331', 'V303', 'V290', 'V116', 'V118',
        'V119', 'V120', 'V121', 'V122', 'V273', 'V274', 'V275', 'V276', 'V277', 'V278', 'V279', 'V280',
        'V1', 'V14', 'V41', 'V65', 'V88', 'V94', 'V107', 'V120', 'V121', 'V122', 'V144', 'V145', 'V150',
        'V151', 'V153', 'V154', 'V155', 'V157', 'V158', 'V159', 'V161', 'V162', 'V163', 'V164', 'V166',
        'V178', 'V181', 'V183', 'V190', 'V191', 'V193', 'V196', 'V199', 'V200', 'V201', 'V205', 'V207',
        'V212', 'V213', 'V214', 'V218', 'V219', 'V223', 'V224', 'V225', 'V230', 'V232', 'V233', 'V235',
        'V238', 'V239', 'V240', 'V241', 'V242', 'V244', 'V246', 'V247', 'V248', 'V249', 'V250', 'V252',
        'V254', 'V260', 'V263', 'V264', 'V265', 'V269', 'V272', 'V297', 'V298', 'V299', 'V302', 'V304',
        'V306', 'V307', 'V308', 'V310', 'V311', 'V312', 'V317', 'V318', 'V319', 'V320', 'V321', 'V325',
        'V327', 'V330', 'V333', 'V334', 'V335', 'V336', 'V337', 'V338', 'V339'
    ]
    X_train.drop(v_cols_to_drop, axis=1, inplace=True, errors='ignore')
    X_test.drop(v_cols_to_drop, axis=1, inplace=True, errors='ignore')

    # TRAIN XGBOOST
    print('\nTraining XGBoost...')
    
    print('Sorting by TransactionDT...')
    order = X_train['TransactionDT'].values.argsort()
    X_train = X_train.iloc[order].reset_index(drop=True)
    y_train = y_train.iloc[order].reset_index(drop=True)

    cols_to_drop = ['uid', 'day', 'isFraud', 'TransactionID', 'TransactionDT']
    X_train.drop([c for c in cols_to_drop if c in X_train.columns], axis=1, inplace=True)
    X_test.drop([c for c in cols_to_drop if c in X_test.columns], axis=1, inplace=True)
    
    split_idx = int(len(X_train)*0.8)
    X_tr = X_train.iloc[:split_idx]
    y_tr = y_train.iloc[:split_idx]
    X_va = X_train.iloc[split_idx:]
    y_va = y_train.iloc[split_idx:]
    
    non_numeric = X_train.select_dtypes(exclude=[np.number]).columns
    if len(non_numeric) > 0:
        print(f'Warning: Non-numeric columns found: {list(non_numeric)}. Label encoding them...')
        for col in non_numeric:
            encode_LE(col, X_train, X_test, verbose=False)
            
    X_train = X_train.astype('float32')
    X_test = X_test.astype('float32')
    
    X_train = reduce_mem_usage(X_train)
    X_test = reduce_mem_usage(X_test)
    gc.collect()

    X_tr = X_train.iloc[:split_idx]
    X_va = X_train.iloc[split_idx:]

    # OPTIONAL HYPERPARAMETER TUNING
    run_tuning = False # Set to True to run Optuna tuning
    if run_tuning:
        print('Starting hyperparameter tuning...')
        study = optuna.create_study(direction='maximize')
        study.optimize(lambda trial: objective(trial, X_tr, y_tr, X_va, y_va), n_trials=20)
        print('Best trial:', study.best_trial.params)
        best_params = study.best_trial.params
        best_params.update({
            'n_estimators': 5000,
            'missing': -1,
            'eval_metric': 'auc',
            'tree_method': 'hist',
            'grow_policy': 'lossguide',
            'n_jobs': -1,
            'early_stopping_rounds': 200,
            'random_state': 42
        })
    else:
        best_params = {
            'n_estimators': 5000,
            'max_depth': 12,
            'learning_rate': 0.0107,
            'subsample': 0.7632,
            'colsample_bytree': 0.58,
            'gamma': 2.75,
            'min_child_weight': 4,
            'missing': -1,
            'eval_metric': 'auc',
            'tree_method': 'hist',
            'grow_policy': 'lossguide',
            'n_jobs': -1,
            'early_stopping_rounds': 200,
            'random_state': 42
        }

    clf = xgb.XGBClassifier(**best_params)

    clf.fit(X_tr, y_tr, 
        eval_set=[(X_va, y_va)],
        verbose=50)

    # PREDICT AND SUBMIT
    print('Generating submission...')
    preds = clf.predict_proba(X_test)[:, 1]
    
    submission = pd.read_csv('test_transaction.csv', usecols=['TransactionID'])
    submission['isFraud'] = preds
    submission.to_csv('submission_xgb_magic.csv', index=False)
    print('Done!')

if __name__ == "__main__":
    main()
