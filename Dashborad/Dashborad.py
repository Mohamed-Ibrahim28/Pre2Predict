import pandas as pd
import joblib
from datetime import timedelta

import dash
import dash_bootstrap_components as dbc
from dash import dcc, html, Input, Output
import plotly.express as px

# --- 1. Load Data & Pre-trained Model --------------------------------

# Paths
DATA_PATH = r"F:\Courses\DEPI\Data Scientist\finalproject\Pre2Predict\Datasets\ready_for_training.csv"
MODEL_PATH = r"F:\Courses\DEPI\Data Scientist\finalproject\Pre2Predict\Model\trained_model.pkl"

# Load dataset
df_all = pd.read_csv(DATA_PATH)
# Recreate Date from year/month/day
df_all['Date'] = pd.to_datetime(
    df_all[['Date_year','Date_month','Date_day']]
      .rename(columns={'Date_year':'year','Date_month':'month','Date_day':'day'})
)

# Load your trained model (e.g., XGBoost regressor)
sales_model = joblib.load(MODEL_PATH)

# --- 2. Prediction Helper --------------------------------------------

def predict_sales(store_id, last_date, periods, df_ref):
    # Build future dates
    future_dates = pd.date_range(start=last_date + timedelta(days=1), periods=periods, freq='D')
    fut = pd.DataFrame({'Date': future_dates})

    # Static store info
    store_info = df_ref[df_ref['Store']==store_id].iloc[-1]
    fut['Store'] = store_id
    fut['StoreType'] = store_info['StoreType']
    fut['StoreType'] = store_info['StoreType']
    fut['CompetitionDistance'] = store_info['CompetitionDistance']
    fut['Promo2'] = store_info['Promo2']
    fut['Promo2SinceWeek'] = store_info['Promo2SinceWeek']
    fut['Promo2SinceYear'] = store_info['Promo2SinceYear']

    # Default dynamic features
    fut['Customers'] = df_ref[df_ref['Store']==store_id]['Customers'].median()
    fut['Open'] = 1
    fut['Promo'] = 0
    fut['SchoolHoliday'] = 0

    # One-hot intervals
    for col in ['PromoInterval_0','PromoInterval_1','PromoInterval_2','PromoInterval_3']:
        fut[col] = store_info.get(col, False)
    # One-hot holiday & assortment
    for col in ['StateHoliday_0','Assortment_0','Assortment_1','Assortment_2']:
        fut[col] = store_info.get(col, False)
    # Date features
    fut['DayOfWeek'] = fut['Date'].dt.dayofweek + 1
    fut['Date_year'] = fut['Date'].dt.year
    fut['Date_month'] = fut['Date'].dt.month
    fut['Date_day'] = fut['Date'].dt.day
    # One-hot day of week
    for i in range(1,8): fut[f'DayOfWeek_{i}'] = (fut['DayOfWeek']==i)

    # Select features in training order
    feature_cols = [
        'Store','StoreType','CompetitionDistance','Promo2','Promo2SinceWeek','Promo2SinceYear',
        'Customers','Open','Promo','SchoolHoliday',
        'Date_year','Date_month','Date_day',
        'PromoInterval_0','PromoInterval_1','PromoInterval_2','PromoInterval_3',
        'StateHoliday_0','Assortment_0','Assortment_1','Assortment_2',
        'DayOfWeek_1','DayOfWeek_2','DayOfWeek_3','DayOfWeek_4','DayOfWeek_5','DayOfWeek_6','DayOfWeek_7'
    ]
    X = fut[feature_cols]

    # Predict
    yhat = sales_model.predict(X)
    return pd.DataFrame({'ds': fut['Date'], 'yhat': yhat})

# --- 3. Dash App Initialization --------------------------------------

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.FLATLY])
app.title = "Sales Forecasting & Optimization"

app.layout = dbc.Container([
    html.H1("📈 Sales Forecasting & Optimization", className="text-center my-4"),
    dbc.Row([
        dbc.Col([
            html.Label("Store:"),
            dcc.Dropdown(
                id='store-dropdown', clearable=False,
                options=[{'label': f"Store {i}", 'value': i} for i in sorted(df_all['Store'].unique())],
                value=sorted(df_all['Store'].unique())[0]
            ),
            html.Br(),
            html.Label("Date Range:"),
            dcc.DatePickerRange(
                id='date-range',
                min_date_allowed=df_all['Date'].min(),
                max_date_allowed=df_all['Date'].max(),
                start_date=df_all['Date'].min(),
                end_date=df_all['Date'].max()
            ),
            html.Br(), html.Br(),
            html.Label("Forecast Horizon (days):"),
            dcc.Slider(
                id='horizon-slider', min=7, max=90, step=1, value=30,
                marks={7:'7',30:'30',60:'60',90:'90'}
            ),
        ], width=3, className="shadow-sm p-3 mb-4 bg-white rounded"),
        dbc.Col(
            dbc.Tabs(id='tabs', active_tab='tab-explore', children=[
                dbc.Tab(label='Exploration', tab_id='tab-explore'),
                dbc.Tab(label='Forecast',    tab_id='tab-forecast'),
                dbc.Tab(label='Optimize',    tab_id='tab-optimize'),
            ]), width=9
        )
    ]),
    html.Div(id='tab-content')
], fluid=True)

# --- 4. Callback -------------------------------------------------------

@app.callback(
    Output('tab-content', 'children'),
    Input('tabs', 'active_tab'),
    Input('store-dropdown', 'value'),
    Input('date-range', 'start_date'),
    Input('date-range', 'end_date'),
    Input('horizon-slider', 'value')
)
def render_tab(tab, store_id, start_date, end_date, horizon):
    # Filter historical data
    d = df_all[(df_all['Store']==store_id) &
               (df_all['Date']>=start_date) &
               (df_all['Date']<=end_date)].copy()
    # Recreate numeric DayOfWeek
    d['DayOfWeek'] = d['Date'].dt.dayofweek + 1

    if tab == 'tab-explore':
        figs = []
        figs.append(dcc.Graph(figure=px.line(d, x='Date', y='Sales', title='Sales Over Time')))
        dow = d.groupby('DayOfWeek', as_index=False)['Sales'].mean()
        figs.append(dcc.Graph(figure=px.bar(dow, x='DayOfWeek', y='Sales', title='Avg Sales by Day of Week', labels={'Sales':'Avg Sales','DayOfWeek':'Day'})))
        figs.append(dcc.Graph(figure=px.scatter(d, x='CompetitionDistance', y='Sales', trendline='ols', title='Sales vs Competition Distance', labels={'CompetitionDistance':'Comp Dist (m)','Sales':'Sales'})))
        mth = d.groupby('Date_month', as_index=False)['Sales'].mean()
        figs.append(dcc.Graph(figure=px.bar(mth, x='Date_month', y='Sales', title='Avg Sales by Month', labels={'Date_month':'Month','Sales':'Avg Sales'}, category_orders={'Date_month':list(range(1,13))})))
        figs.append(dcc.Graph(figure=px.box(d, x='StoreType', y='Sales', title='Sales Distribution by Store Type', labels={'StoreType':'Type','Sales':'Sales'})))
        figs.append(dcc.Graph(figure=px.box(d, x='Promo', y='Sales', title='Promo On vs Off', labels={'Promo':'Promo Active','Sales':'Sales'}, category_orders={'Promo':[0,1]})))
        figs.append(dcc.Graph(figure=px.box(d, x='StateHoliday_0', y='Sales', title='Holiday vs Regular Sales', labels={'StateHoliday_0':'Holiday Flag','Sales':'Sales'})))
        ts = d.set_index('Date').sort_index()
        ts['Sales_7d'] = ts['Sales'].rolling(7, min_periods=1).mean()
        figs.append(dcc.Graph(figure=px.line(ts.reset_index(), x='Date', y='Sales_7d', title='7-Day Rolling Avg', labels={'Sales_7d':'7-Day Avg'})))
        return dbc.Row([dbc.Col(f, width=6, className="mb-4") for f in figs])

    elif tab == 'tab-forecast':
        last_date = d['Date'].max()
        fc_df = predict_sales(store_id, last_date, horizon, df_all)
        hist = d.rename(columns={'Date':'ds','Sales':'y'}).assign(type='history')
        fut  = fc_df.rename(columns={'yhat':'y'}).assign(type='forecast')
        combined = pd.concat([hist[['ds','y','type']], fut])
        fig_fc = px.line(combined, x='ds', y='y', color='type', labels={'ds':'Date','y':'Sales','type':'Series'}, title=f'Actual vs Forecasted Sales ({horizon}d)')
        return dcc.Graph(figure=fig_fc)

    elif tab == 'tab-optimize':
        last_date = d['Date'].max()
        fc_df = predict_sales(store_id, last_date, horizon, df_all)
        fc_df['safety_stock'] = fc_df['yhat'] * 0.2
        fc_df['reorder_level'] = fc_df['yhat'] + fc_df['safety_stock']
        fig_opt = px.line(fc_df, x='ds', y='reorder_level', labels={'ds':'Date','reorder_level':'Reorder Level'}, title='Reorder Level Recommendation')
        return dcc.Graph(figure=fig_opt)

# --- 5. Run -----------------------------------------------------------

if __name__ == '__main__':
    app.run(debug=True)