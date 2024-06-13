import pandas as pd


def debug_on_success_(data: pd.DataFrame, dummy_value: int) -> None:
    
    # Print columns
    if True:
        print(data.columns)

    print("pipeline succeed !", dummy_value)

    return


def sales_columns_naming_(data: pd.DataFrame) -> pd.DataFrame:
    
    # Rename columns
    # data.rename({'ï»¿DATE':'Full_Date', 'Mapped_GCK':'GCK', 'Sales_EUR':'Sales €'}, axis=1, inplace=True)
    # data.rename({'п»їDATE':'Full_Date', 'Mapped_GCK':'GCK', 'Sales_EUR':'Sales €'}, axis=1, inplace=True)
    data.rename({'DATE':'Full_Date', 'Mapped_GCK':'GCK', 'Sales_EUR':'Sales €'}, axis=1, inplace=True)

    return data


def full_date_col_(data: pd.DataFrame) -> pd.DataFrame:
    
    # Format data
    data['Full_Date'] = pd.to_datetime(data['Full_Date'], format='%d.%m.%Y').dt.strftime('%d-%m-%Y')
    
    return data


def sales_col_(data):
    
    # Format sales
    data["Sales €"] = data["Sales €"].str.replace(",", ".").astype(float)
    # preprocessed_sales = sales_raw_data.fillna(0)
    # preprocessed_sales = sales_raw_data
    
    return data


def market_columns_naming_(market_data: pd.DataFrame) -> pd.DataFrame:
    
    # Combine the headers from the first row with the names in the second row
    new_headers = []
    for header, name in zip(market_data.iloc[0], market_data.iloc[1]):
        if isinstance(header, str) and isinstance(name, str):
            new_headers.append(f'{header}_{name}')
        else:
            new_headers.append(name if isinstance(name, str) else '')

    # Assign the new headers to the DataFrame
    market_data.columns = new_headers

    # Drop the first two rows as they are now redundant
    market_data = market_data.drop([0, 1])

    # Now df has the updated headers
    market_data.rename(
        {
            'Index 2010=100 (if not otherwise noted)_date':'Month Year',
            'China_Production Index Machinery & Electricals':'China Production Index M&E',
            'China_Shipments Index Machinery & Electricals':'China Shipments Index M&E',
            'France_Production Index Machinery & Electricals':'France Production Index M&E',
            'France_Shipments Index Machinery & Electricals':'France Shipments Index M&E',
            'Germany_Production Index Machinery & Electricals':'Germany Production Index M&E',
            'Germany_Shipments Index Machinery & Electricals':'Germany Shipments Index M&E',
            'Italy_Production Index Machinery & Electricals':'Italy Production Index M&E',
            'Italy_Shipments Index Machinery & Electricals':'Italy Shipments Index M&E',
            'Japan_Production Index Machinery & Electricals':'Japan Production Index M&E',
            'Japan_Shipments Index Machinery & Electricals':'Japan Shipments Index M&E',
            'Switzerland_Production Index Machinery & Electricals':'Switzerland Production Index M&E',
            'Switzerland_Shipments Index Machinery & Electricals':'Switzerland Shipments Index M&E',
            'United Kingdom_Production Index Machinery & Electricals':'UK Production Index M&E',
            'United Kingdom_Shipments Index Machinery & Electricals':'UK Shipments Index M&E',
            'United States_Production Index Machinery & Electricals':'US Production Index M&E',
            'United States_Shipments Index Machinery & Electricals':'US Shipments Index M&E',
            'Europe_Production Index Machinery & Electricals':'Europe Production Index M&E',
            'Europe_Shipments Index Machinery & Electricals':'Europe Shipments Index M&E',
            'World: Price of Base Metals':'Price of Base Metals',
            'World: Price of Energy':'Price of Energy',
            'World: Price of Metals  & Minerals':'Price of Metals & Minerals',
            'World: Price of Natural gas index':'Price of Natural gas index',
            'World: Price of Crude oil, average':'Price of Crude oil (AVG)',
            'World: Price of Copper':'Price of Copper',
            'Producer Prices_United States: Electrical equipment':'US Producer Prices Electrical eq.',
            'Producer Prices_United Kingdom: Electrical equipment':'UK Producer Prices Electrical eq.',
            'Producer Prices_Italy: Electrical equipment':'Italy Producer Prices Electrical eq.',
            'Producer Prices_France: Electrical equipment':'France Producer Prices Electrical eq.',
            'Producer Prices_Germany: Electrical equipment':'Germany Producer Prices Electrical eq.',
            'Producer Prices_China: Electrical equipment':'China Producer Prices Electrical eq.',
            'production index_United States: Machinery and equipment n.e.c.':'US Production Index Machinery eq.',
            'production index_World: Machinery and equipment n.e.c.':'Global Production Index Machinery eq.',
            'production index_Switzerland: Machinery and equipment n.e.c.':'Switzerland Production Index Machinery eq.',
            'production index_United Kingdom: Machinery and equipment n.e.c.':'UK Production Index Machinery eq.',
            'production index_Italy: Machinery and equipment n.e.c.':'Italy Production Index Machinery eq.',
            'production index_Japan: Machinery and equipment n.e.c.':'Japan Production Index Machinery eq.',
            'production index_France: Machinery and equipment n.e.c.':'France Production Index Machinery eq.',
            'production index_Germany: Machinery and equipment n.e.c.':'Germany Production Index Machinery eq.',
            'production index_United States: Electrical equipment':'US Production Index Electrical eq.',
            'production index_World: Electrical equipment':'Global Production Index Electrical eq.',
            'production index_Switzerland: Electrical equipment':'Switzerland Production Index Electrical eq.',
            'production index_United Kingdom: Electrical equipment':'UK Production Index Electrical eq.',
            'production index_Italy: Electrical equipment':'Italy Production Index Electrical eq.',
            'production index_Japan: Electrical equipment':'Japan Production Index Electrical eq.',
            'production index_France: Electrical equipment':'France Production Index Electrical eq.',
            'production index_Germany: Electrical equipment':'Germany Production Index Electrical eq.',
        },
        axis=1,
        inplace=True
    )

    return market_data


if False:
# def age_(data):
    
#     data['bin_age'] = 0  
#     data.loc[(data['age'] <= 35) & (data['age'] >= 18),'bin_age'] = 1
#     data.loc[(data['age'] <= 60) & (data['age'] >= 36),'bin_age'] = 2
#     data.loc[data['age'] >=61,'bin_age'] = 3
    
#     return data

# def campaign_(data):
    
    
#     data.loc[data['campaign'] == 1,'campaign'] = 1
#     data.loc[(data['campaign'] >= 2) & (data['campaign'] <= 3),'campaign'] = 2
#     data.loc[data['campaign'] >= 4,'campaign'] = 3
    
#     return data

# def duration_(data):
    
#     data['t_min'] = 0
#     data['t_e_min'] = 0
#     data['e_min']=0
#     data.loc[data['duration'] <= 5,'t_min'] = 1
#     data.loc[(data['duration'] > 5) & (data['duration'] <= 10),'t_e_min'] = 1
#     data.loc[data['duration'] > 10,'e_min'] = 1
    
#     return data

# def pdays_(data):
#     data['pdays_not_contacted'] = 0
#     data['months_passed'] = 0
#     data.loc[data['pdays'] == -1 ,'pdays_not_contacted'] = 1
#     data['months_passed'] = data['pdays']/30
#     data.loc[(data['months_passed'] >= 0) & (data['months_passed'] <=2) ,'months_passed'] = 1
#     data.loc[(data['months_passed'] > 2) & (data['months_passed'] <=6),'months_passed'] = 2
#     data.loc[data['months_passed'] > 6 ,'months_passed'] = 3
    
#     return data


# def balance_(data):
#     data['Neg_Balance'] = 0
#     data['No_Balance'] = 0
#     data['Pos_Balance'] = 0
#     data.loc[~data['balance']<0,'Neg_Balance'] = 1
#     data.loc[data['balance'] < 1,'bin_Balance'] = 0
#     data.loc[(data['balance'] >= 1) & (data['balance'] < 100),'bin_Balance'] = 1
#     data.loc[(data['balance'] >= 100) & (data['balance'] < 500),'bin_Balance'] = 2
#     data.loc[(data['balance'] >= 500) & (data['balance'] < 2000),'bin_Balance'] = 3
#     data.loc[(data['balance'] >= 2000) & (data['balance'] < 5000),'bin_Balance'] = 4
#     data.loc[data['balance'] >= 5000,'bin_Balance'] = 5
    
#     return data
    pass