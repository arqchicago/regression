import pandas as pd
from sklearn.linear_model import LinearRegression
import scipy.stats as sstats




if __name__ == '__main__':

    #----  features and target variable
    features = ['age', 'bidders'] 
    price = 'auction_price'
    cols = features + [price]

    # uploading data
    clock_df = pd.read_csv('data\\antique_clocks.csv', usecols=cols)
    clock_df.fillna(0, inplace=True)
    rows, cols = clock_df.shape
    print(f'> rows = {rows},  cols = {cols}')
    
    
    # multiple regression
    #     - modeling price using age and bidders
    #     - y = B0 + B1(x1) + B2(x2) + e
    #     - deterministic portion: B0 + B1(x1) + B2(x2),  random error = e
    
    X = clock_df[['age', 'bidders']]
    y = clock_df['auction_price']

    reg = LinearRegression().fit(X, y)
    print('> sklearn based regression')
    print(f'  coefficients:   b1={reg.coef_[0]},   b2={reg.coef_[1]},   b0={reg.intercept_}\n')

    clock_df['pred_price'] = reg.intercept_ + reg.coef_[0]*clock_df['age'] + reg.coef_[1]*clock_df['bidders']
    residual_error = (clock_df["auction_price"] - clock_df["pred_price"]).sum()
    sum_of_squared_error = ((clock_df["auction_price"] - clock_df["pred_price"])**2).sum()
    print(f'> residual error = {round(residual_error, 4)}')
    print(f'> sum of squared error (SSE) = {round(sum_of_squared_error, 4)}\n')

