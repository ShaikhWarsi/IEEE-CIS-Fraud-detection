import pandas as pd
import numpy as np
import xgboost as xgb
import gc
import datetime
from sklearn.metrics import roc_auc_score

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
    
    # LABEL ENCODE CATEGORICAL
    cat_cols = ['ProductCD', 'card4', 'card6', 'P_emaildomain', 'R_emaildomain', 'M1', 'M2', 'M3', 'M4', 'M5', 'M6', 'M7', 'M8', 'M9', 'id_12', 'id_15', 'id_16', 'id_23', 'id_27', 'id_28', 'id_29', 'id_30', 'id_31', 'id_33', 'id_34', 'id_35', 'id_36', 'id_37', 'id_38', 'DeviceType', 'DeviceInfo']
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
    X_train, X_test = encode_FE(X_train, X_test, ['card1', 'card2', 'card3', 'card5', 'card1_addr1', 'card1_addr1_P_emaildomain'])

    # GROUP AGGREGATIONS
    X_train, X_test = encode_AG(['TransactionAmt', 'D4', 'D9', 'D10', 'D15'], ['card1', 'card1_addr1', 'uid'], ['mean', 'std'], X_train, X_test)
    X_train, X_test = encode_AG2(['C1', 'C2', 'C4', 'C5', 'C6', 'C7', 'C8', 'C10', 'C11', 'C12', 'C13', 'C14'], ['card1', 'card1_addr1', 'uid'], X_train, X_test)
    X_train, X_test = encode_AG2(['M1','M2','M3','M4','M5','M6','M7','M8','M9'], ['card1','card1_addr1','uid'], X_train, X_test)

    
    X_train['Transaction_hour'] = np.floor(X_train['TransactionDT'] / 3600) % 24
    X_test['Transaction_hour'] = np.floor(X_test['TransactionDT'] / 3600) % 24
    X_train, X_test = encode_FE(X_train, X_test, ['Transaction_hour'])
    
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
            
    # Final check and fillna for XGB
    X_train = X_train.astype('float32')
    X_test = X_test.astype('float32')
    X_tr = X_train.iloc[:split_idx]
    X_va = X_train.iloc[split_idx:]

    clf = xgb.XGBClassifier( 
        n_estimators=2000,
        max_depth=12, 
        learning_rate=0.02, 
        subsample=0.8,
        colsample_bytree=0.4, 
        missing=-1, 
        eval_metric='auc',
        tree_method='hist', 
        early_stopping_rounds=100
    )

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
