import pandas as pd
from .utils import *


# Examples from classes
# can be removed later
if False:
    
# def _is_true(x: pd.Series) -> pd.Series:
#     return x == "t"


# def _parse_percentage(x: pd.Series) -> pd.Series:
#     x = x.str.replace("%", "")
#     x = x.astype(float) / 100
#     return x


# def _parse_money(x: pd.Series) -> pd.Series:
#     x = x.str.replace("$", "").str.replace(",", "")
#     x = x.astype(float)
#     return x


# def preprocess_companies(companies: pd.DataFrame) -> pd.DataFrame:
#     """Preprocesses the data for companies.

#     Args:
#         companies: Raw data.
#     Returns:
#         Preprocessed data, with `company_rating` converted to a float and
#         `iata_approved` converted to boolean.
#     """
#     companies["iata_approved"] = _is_true(companies["iata_approved"])
#     companies["company_rating"] = _parse_percentage(companies["company_rating"])
#     return companies


# def preprocess_shuttles(shuttles: pd.DataFrame) -> pd.DataFrame:
#     """Preprocesses the data for shuttles.

#     Args:
#         shuttles: Raw data.
#     Returns:
#         Preprocessed data, with `price` converted to a float and `d_check_complete`,
#         `moon_clearance_complete` converted to boolean.
#     """
#     shuttles["d_check_complete"] = _is_true(shuttles["d_check_complete"])
#     shuttles["moon_clearance_complete"] = _is_true(shuttles["moon_clearance_complete"])
#     shuttles["price"] = _parse_money(shuttles["price"])
#     return shuttles


# def create_model_input_table(
#     shuttles: pd.DataFrame, companies: pd.DataFrame, reviews: pd.DataFrame
# ) -> pd.DataFrame:
#     """Combines all data to create a model input table.

#     Args:
#         shuttles: Preprocessed data for shuttles.
#         companies: Preprocessed data for companies.
#         reviews: Raw data for reviews.
#     Returns:
#         Model input table.

#     """
#     rated_shuttles = shuttles.merge(reviews, left_on="id", right_on="shuttle_id")
#     rated_shuttles = rated_shuttles.drop("id", axis=1)
#     model_input_table = rated_shuttles.merge(
#         companies, left_on="company_id", right_on="id"
#     )
#     model_input_table = model_input_table.dropna()
#     return model_input_table

# def preprocess_ibm(ibm: pd.DataFrame) -> pd.DataFrame:
#     """Preprocesses the data for companies.

#     Args:
#         companies: Raw data.
#     Returns:
#         Preprocessed data
#     """

#     ibm = ibm.fillna(0)

#     return ibm
    pass


def preprocess_sales(data: pd.DataFrame) -> pd.DataFrame:
    """Preprocesses the sales data.

    Args:
        data: Raw sales.
    Returns:
        Preprocessed sales
    """

    sales_data = data.copy()
    
    sales_data = sales_columns_naming_(sales_data)

    # Format data
    sales_data = full_date_col_(sales_data)

    # Format sales
    sales_data = sales_col_(sales_data)

    # Printing something from dataframe (usually columns)
    debug_on_success_(sales_data)

    return sales_data


def preprocess_markets(data: pd.DataFrame) -> pd.DataFrame:
    """Preprocesses the market data.

    Args:
        data: Raw markets.
    Returns:
        Preprocessed market data
    """

    market_data = data.copy()

    market_data = market_columns_naming_(market_data)

    market_data = market_data.drop(market_data.index[0]).reset_index(drop=True)

    # We believed it would be easier to understand the columns with clearer names
    market_data['Month Year'] = market_data['Month Year'].str.strip()

    # Splitting the column into year and month
    market_data[['year', 'month']] = market_data['Month Year'].str.split('m', expand=True)

    # Converting year and month to datetime
    market_data['Month Year'] = pd.to_datetime(market_data['year'] + '-' + market_data['month'], format='%Y-%m')

    # Formatting datetime to Month Year
    market_data['Month Year'] = market_data['Month Year'].dt.strftime('%b %Y')

    # Dropping intermediate columns if needed
    market_data.drop(columns=['year', 'month'], inplace=True)

    # Printing something from dataframe (usually columns)
    debug_on_success_(market_data)

    return market_data

