import pandas as pd
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tools.tools import add_constant

def remove_features_vif(df, threshold, removed_columns=[]):
    """
    Recursively removes features with variance inflation factor (VIF) above the specified threshold.
    """
    # Add a constant for VIF calculation (required by statsmodels).
    df_with_constant = add_constant(df)

    # Create a second DataFrame to record the features and their corresponding VIF.
    vif_data = pd.DataFrame()
    vif_data['feature'] = df_with_constant.columns
    vif_data['VIF'] = [variance_inflation_factor(df_with_constant.values, i) for i in range(df_with_constant.shape[1])]

    # Exclude the 'const' column from VIF analysis.
    vif_data = vif_data[vif_data['feature'] != 'const']

    max_vif_row = vif_data.loc[vif_data['VIF'].idxmax()]
    max_vif_feature = max_vif_row['feature']
    max_vif_value = max_vif_row['VIF']

    if max_vif_value > threshold:
        print(f"Removing '{max_vif_feature}' with VIF: {max_vif_value:.2f}")
        removed_columns.append(max_vif_feature)
        df_reduced = df.drop(columns=[max_vif_feature])
        return remove_features_vif(df_reduced, threshold, removed_columns)
    else:
        print(f'All remaining features have VIFs below the threshold of {threshold}.')
        return df, removed_columns

if __name__ == "__main__":

    # Load h1 data.
    h1_train = pd.read_csv('../data/processed/h1_train.csv')
    h1_val = pd.read_csv('../data/processed/h1_val.csv')
    h1_test = pd.read_csv('../data/processed/h1_test.csv')

    # Load h7 data.
    h7_train = pd.read_csv('../data/processed/h7_train.csv')
    h7_val = pd.read_csv('../data/processed/h7_val.csv')
    h7_test = pd.read_csv('../data/processed/h7_test.csv')

    print(f"H1 - Train: {h1_train.shape}, Val: {h1_val.shape}, Test: {h1_test.shape}")
    print(f"H7 - Train: {h7_train.shape}, Val: {h7_val.shape}, Test: {h7_test.shape}")

    # Check for colinearity and remove features that have a high variance inflation factor (VIF).
    h1_X_train = h1_train[[c for c in h1_train.columns if c not in ['date', 'y_btc_close_t+1']]]
    h1_X_train, removed_columns = remove_features_vif(h1_X_train, threshold=40)
    h1_train.drop(columns=removed_columns).to_csv(f'../data/processed/h1_vif_train.csv', index=False)

    # Remove these same columns for the other data sets and save each of them to a new csv file.
    file_names = ['h1_vif_val.csv', 'h1_vif_test.csv', 'h7_vif_train.csv', 'h7_vif_val.csv', 'h7_vif_test.csv']
    for index, X in enumerate([h1_val, h1_test, h7_train, h7_val, h7_test]):
        X = X.drop(columns=removed_columns)
        X.to_csv(f'../data/processed/{file_names[index]}', index=False)

    print("\nAfter the VIF analysis and feature removal...")
    h1_train = pd.read_csv('../data/processed/h1_vif_train.csv')
    h1_val = pd.read_csv('../data/processed/h1_vif_val.csv')
    h1_test = pd.read_csv('../data/processed/h1_vif_test.csv')
    h7_train = pd.read_csv('../data/processed/h7_vif_train.csv')
    h7_val = pd.read_csv('../data/processed/h7_vif_val.csv')
    h7_test = pd.read_csv('../data/processed/h7_vif_test.csv')
    print(f"H1 - Train: {h1_train.shape}, Val: {h1_val.shape}, Test: {h1_test.shape}")
    print(f"H7 - Train: {h7_train.shape}, Val: {h7_val.shape}, Test: {h7_test.shape}")