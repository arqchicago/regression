import pandas as pd
from sklearn.linear_model import LinearRegression
import scipy.stats as sstats

def analysis_of_variance(df, y, y_hat, k):
    n = df.shape[0]
    
    # DF
    DF_model = k
    DF_error = n-(k+1)
    DF_total = k+(n-(k+1))
    
    # sum of squares
    mean = df[y].mean()
    ss_model = ((df[y_hat] - mean)**2).sum()
    ss_error = ((df[y] - df[y_hat])**2).sum()
    ss_total = ((df[y] - mean)**2).sum()
    
    # mean square
    ms_model = ss_model / DF_model
    ms_error = ss_error / DF_error
    
    # F stats
    F_value = ms_model/ms_error
    p_value = 1-sstats.f.cdf(F_value, DF_model, DF_error)
    
    anova_packet = {'df': {'model': DF_model, 'error': DF_error, 'total': DF_total}, 
                    'ss': {'model': ss_model, 'error': ss_error, 'total': ss_total},
                    'ms': {'model': ms_model, 'error': ms_error},
                    'f':  {'f_val': F_value, 'p_val': p_value}}
    
    return anova_packet



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


    anova = analysis_of_variance(clock_df, "auction_price", "pred_price", 2)
    
    print("> Analysis of Variance")
    print ("{:<10} {:>6} {:>20} {:>15} {:>10} {:>10}".format('  Source','DF','Sum of Squares','Mean Square', 'F Value', 'Pr > F'))
    print ("-"*90)
    print ("{:<10} {:>6} {:>20} {:>15} {:>10} {:>10}".format('  Model', str(anova['df']['model']), str(round(anova['ss']['model'],0)), str(round(anova['ms']['model'],0)), str(round(anova['f']['f_val'],2)), str(round(anova['f']['p_val'], 4))))
    print ("{:<10} {:>6} {:>20} {:>15} {:>10} {:>10}".format('  Error', str(anova['df']['error']), str(round(anova['ss']['error'],0)), str(round(anova['ms']['error'],0)), ' ', ' '))
    print ("{:<10} {:>6} {:>20} {:>15} {:>10} {:>10}".format('  Total', str(anova['df']['total']), str(round(anova['ss']['total'],0)), ' ', ' ', ' '))
    print('')
    
    print(f'> Estimator of sigma square (variance of the error term) is s^2 --> MSE:  {round(anova["ms"]["error"],0)}')
    print(f'> s is --> Root MSE:  {round((anova["ms"]["error"])**0.50,2)}')

    