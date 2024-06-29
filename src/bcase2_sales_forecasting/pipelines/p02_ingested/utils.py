import pandas as pd
import re
from typing import Any, Dict


def debug_on_success_(data: pd.DataFrame, dummy_value: int, pipeline_name: str = "", f_verbose: bool = False) -> None:
    
    # Print columns
    if f_verbose:
        print(data.dtypes)

    # dummy_value is for checking pipelines sequence
    dummy_value.append(dummy_value[-1] + 1) 
    print(f"pipeline {pipeline_name} succeed !; f_verbose={f_verbose};", dummy_value)

    return


def print_to_debug_(data: pd.DataFrame) -> None:
    
    data_copy = data.copy()
    headers = data_copy.columns.to_list()
    valid_headers = market_columns_list_()
    print(f'{30*"#"} {"first 5 columns of data".upper()} {30*"#"}')
    print(headers[:5])
    print(f'{30*"#"} {"first 5 columns of validation_columns_list_".upper()} {30*"#"}')
    print(valid_headers[:5])
    print(f'{30*"#"} {"last 5 columns of data".upper()} {30*"#"}')
    print(headers[-5:])
    print(f'{30*"#"} {"last 5 columns of validation_columns_list_".upper()} {30*"#"}')
    print(valid_headers[-5:])
    # print(f'{30*"#"} {"length of market_columns_list_()".upper()} {30*"#"}')
    print(90*"#")

    data_copy = columns_sanitation_(data_copy)
    headers = data_copy.columns.to_list()
    print(f'{30*"#"} {"columns after sanitation".upper()} {30*"#"}')
    print("col count:", len(headers), "valid count:", len(valid_headers))
    print("first 5:")
    print(headers[:5])
    print(valid_headers[:5])
    print("last 5:")
    print(headers[-5:])
    print(valid_headers[-5:])
    print(data_copy.dtypes)
    print(90*"#")

    # if False:
    #     # print(market_data.dtypes)
    #     print(market_data['month_year'].dtypes)
    #     print(f'{30*"#"} {"market_categorical_features".upper()} {30*"#"}')
    #     print(market_categorical_features)
    #     print(f'{30*"#"} {"market_numerical_features".upper()} {30*"#"}')
    #     print(market_numerical_features)
    #     return market_data

    pass


def sales_columns_naming_(data: pd.DataFrame) -> pd.DataFrame:
    
    # Rename columns
    data.rename({'DATE':'Full_Date', 'Mapped_GCK':'GCK', 'Sales_EUR':'Sales €'}, axis=1, inplace=True)

    return data


def full_date_col_(data: pd.DataFrame) -> pd.DataFrame:
    
    # Format data
    data['Full_Date'] = pd.to_datetime(data['Full_Date'], format='%d.%m.%Y', dayfirst=True)#.dt.strftime('%d-%m-%Y')
    
    return data


def sales_col_(data):
    
    # Format sales
    data["Sales €"] = data["Sales €"].str.replace(",", ".").astype(float)
    
    return data


def market_columns_list_() -> list:

    columns_list = [
        'index',
        'month_year',
        'china_production_index_m_e',
        'china_shipments_index_m_e',
        'france_production_index_m_e',
        'france_shipments_index_m_e',
        'germany_production_index_m_e',
        'germany_shipments_index_m_e',
        'italy_production_index_m_e',
        'italy_shipments_index_m_e',
        'japan_production_index_m_e',
        'japan_shipments_index_m_e',
        'switzerland_production_index_m_e',
        'switzerland_shipments_index_m_e',
        'uk_production_index_m_e',
        'uk_shipments_index_m_e',
        'us_production_index_m_e',
        'us_shipments_index_m_e',
        'europe_production_index_m_e',
        'europe_shipments_index_m_e',
        'price_of_base_metals',
        'price_of_energy',
        'price_of_metals___minerals',
        'price_of_natural_gas_index',
        'price_of_crude_oil_avg',
        'price_of_copper',
        'united_states__eur_in_lcu',
        'us_producer_prices_electrical_eq',
        'uk_producer_prices_electrical_eq',
        'italy_producer_prices_electrical_eq',
        'france_producer_prices_electrical_eq',
        'germany_producer_prices_electrical_eq',
        'china_producer_prices_electrical_eq',
        'us_production_index_machinery_eq',
        'global_production_index_machinery_eq',
        'switzerland_production_index_machinery_eq',
        'uk_production_index_machinery_eq',
        'italy_production_index_machinery_eq',
        'japan_production_index_machinery_eq',
        'france_production_index_machinery_eq',
        'germany_production_index_machinery_eq',
        'us_production_index_electrical_eq',
        'global_production_index_electrical_eq',
        'switzerland_production_index_electrical_eq',
        'uk_production_index_electrical_eq',
        'italy_production_index_electrical_eq',
        'japan_production_index_electrical_eq',
        'france_production_index_electrical_eq',
        'germany_production_index_electrical_eq',
    ]

    return columns_list


def market_columns_naming_(market_data: pd.DataFrame) -> pd.DataFrame:

    # Combine the headers from the first row with the names in the second row
    new_headers = []
    for header, name in zip(market_data.iloc[0], market_data.iloc[1]):
        if isinstance(header, str) and isinstance(name, str):
            new_headers.append(f'{header}_{name}')
        else:
            new_headers.append(name if isinstance(name, str) else '')

    # print(len(new_headers), new_headers)
    # print("##############################")

    # Assign the new headers to the DataFrame
    market_data.columns = new_headers

    # Drop the first two rows as they are now redundant
    market_data = market_data.drop([0, 1])

    # Now df has the updated headers
    market_data.rename(
        {
            'Index 2010=100 (if not otherwise noted)':'Month Year',
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


def columns_sanitation_(market_data: pd.DataFrame) -> pd.DataFrame:
    """
        error code: 270040, error msg: Illegal feature name, user msg: , the provided feature name month year is
            invalid.
        Legal usage:
        Feature names can only contain lower case characters, numbers and underscores, have to start with a
            letter and cannot be longer than 63 characters or empty
    """

    new_headers = market_data.columns
    # For feature store seamless usasge
    symbols_to_replace_pattern = r'[ &:]'
    replacement_char = "_"
    new_headers = [re.sub(symbols_to_replace_pattern, replacement_char, col) for col in new_headers]
    symbols_to_replace_pattern = r'[().]'
    replacement_char = ""
    new_headers = [re.sub(symbols_to_replace_pattern, replacement_char, col) for col in new_headers]
    symbols_to_replace_pattern = r'[€]'
    replacement_char = "eur"
    new_headers = [re.sub(symbols_to_replace_pattern, replacement_char, col) for col in new_headers]
    
    new_headers = [col.lower() for col in new_headers]

    market_data.columns = new_headers

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