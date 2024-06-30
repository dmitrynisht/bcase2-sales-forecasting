import pandas as pd
import re
from typing import Any, Dict
import logging


logger = logging.getLogger(__name__)

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
