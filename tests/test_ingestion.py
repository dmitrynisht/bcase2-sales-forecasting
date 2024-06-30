import pandas as pd
from src.bcase2_sales_forecasting.pipelines.p02_ingested.nodes import ingest_sales, ingest_markets, ingest_gdp, ingest_test_data

def test_ingest_sales():
    sample_data = pd.DataFrame({
        'DATE': ['01.01.2022', '02.01.2022', '03.01.2022'],
        'Sales_EUR': ["100", "200", "300,5"],
        'Mapped_GCK': ["#1", "#2", "#3"]
    })
    result = ingest_sales(sample_data, parameters={"to_feature_store": False, "debug_output": {}})

    assert isinstance(result, pd.DataFrame)

    expected_columns = ['full_date', 'gck', 'sales_eur']
    assert set(result.columns.to_list()) == set(expected_columns)

    assert result['full_date'].dtype == 'datetime64[ns]'
    assert result['sales_eur'].dtype == 'float64'

def test_ingest_markets():
    # Here we're using the original values instead of sample data for simplicity
    # We know that relying on the file isn't the best practice though
    sample_data = pd.read_excel("data/01_raw/Case2_Market_data.xlsx", header=None, engine='openpyxl')
    
    result = ingest_markets(sample_data, {"to_feature_store": False, "debug_output": {}})

    expected_columns = [
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

    assert isinstance(result, pd.DataFrame)
    assert set(result.columns.to_list()) == set(expected_columns)

def test_ingest_gdp():
    sample_data = pd.DataFrame({
        "Date": ["2004-04-01", "2004-07-01", "2004-10-01"],
        "GDP": ["560160.0", "563267.0", "566374"],
    })

    result = ingest_gdp(sample_data, {"to_feature_store": False, "debug_output": {}})

    expected_columns = ['index', 'month_year', 'gdp']

    assert isinstance(result, pd.DataFrame)
    assert result['month_year'].dtype == 'datetime64[ns]'
    assert set(result.columns.to_list()) == set(expected_columns)


def test_ingest_test_data():
    sample_data = pd.DataFrame({
        'Month Year': ['Mai 22', 'Okt 22', 'Dez 22'],
        'Mapped_GCK': ["#1", "#1", "#3"],
        'Sales_EUR': ["105.5", "200", "300.5"]
    })

    result = ingest_test_data(sample_data, {"target_product": "#1"})

    assert isinstance(result, pd.DataFrame)

    expected_columns = ['index', 'full_date', 'sales_eur']
    assert set(result.columns.to_list()) == set(expected_columns)
    print(result)
    # assert that full_date column has the format %Y-%m-%d
    assert result['full_date'].to_list() == ['2022-05-01', '2022-10-01']
    assert result['sales_eur'].to_list() == [105.5, 200.0]