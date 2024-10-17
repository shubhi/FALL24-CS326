import numpy as np
import pandas as pd
import scipy
from scipy.stats import pearsonr, spearmanr, ttest_ind, shapiro, levene


def get_float64_column_names(df: pd.DataFrame) -> list[str]:
    """Get the float64 column names from a DataFrame.

    NOTE: This method should return a list of column names (str). NOT
    an Index. Pandas will often use non-standard python data
    structures for indices and arrays, and we want to be explicit
    about the return type. To cast a pandas Index to a list of strings,
    you can use the .tolist() method.

    Args:
        df: DataFrame to extract float64 columns from.

    Returns:
        List of strings with the float64 column names.
    """

    return df.select_dtypes(include=['float64']).columns.tolist()
    
    raise NotImplementedError("You need to implement this function.")

def get_missing_value_indices(df: pd.DataFrame, column: str) -> list[int]:
    """Get the row indices of missing values within a column.

    NOTE: This method should return a list of ints. NOT an Index.
    Pandas will often use non-standard python data structures for indices
    and arrays, and we want to be explicit about the return type.
    To cast a pandas Index to a list of integers, you can use the .tolist()
    method.

    Args:
        df: DataFrame with missing values.
        column: Column with missing values.

    Returns:
        Indices of missing values, as a list of ints.
    """

    return df[df[column].isna()].index.tolist()

    raise NotImplementedError("You need to implement this function.")

def drop_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """Drop rows with missing values.

    If **any** rows have missing values, drop those
    rows and return the resulting DataFrame.

    Args:
        df: DataFrame with missing values.

    Returns:
        DataFrame with missing values dropped.
    """
    return df.dropna()

    raise NotImplementedError("You need to implement this function.")

def fill_float64_cols_with_random_sample(df: pd.DataFrame, random_state=2024) -> pd.DataFrame:
    """Fill float64 columns with a random sample from the column.
    
    Args:
        df: DataFrame with missing values and float64 type columns.
        random_state: Random seed for reproducibility, to be used with df.sample().

    Returns:
        DataFrame with missing values filled with random sample from the column.
    """
    float_columns = get_float64_column_names(df)
    for column in float_columns:
        missing_indices = get_missing_value_indices(df, column)
        random_sample = df[column].dropna().sample(len(missing_indices),
                                                   replace=True,
                                                   random_state=random_state)
        df.loc[missing_indices, column] = random_sample.values

    return df

def fill_float64_cols_with_mean(df: pd.DataFrame) -> pd.DataFrame:
    """Fill float64 columns with the mean of the column.
    
    Args:
        df: DataFrame with missing values and float64 type columns.

    Returns:
        DataFrame with missing values filled with the mean of the column.
    """
    float_columns = get_float64_column_names(df)
    for column in float_columns:        
        missing_indices = get_missing_value_indices(df, column)
        mean = df[column].mean(skipna=True)
        mean_values = [mean] * len(missing_indices)
        df.loc[missing_indices, column] = mean_values

    return df

def calculate_covariance_numpy(x: np.array, y: np.array) -> float:
    """Use only numpy to calculate the covariance.

    Do NOT use np.cov().
    Use only basic numpy operations (like np.sum, np.mean, etc.)
    
    Args:   
        x: First array.
        y: Second array.
    
    Returns:
        Covariance between x and y.
    """
    mean_x = np.mean(x)
    mean_y = np.mean(y)

    deviation_x = x - mean_x
    deviation_y = y - mean_y

    covariance = np.sum(deviation_x * deviation_y) / (len(x) - 1)

    return covariance

    raise NotImplementedError("You need to implement this function.")

def calculate_pearson_correlation_numpy(x: np.array, y: np.array) -> float:
    """Use only numpy to calculate pearson's correlation coefficient.

    Do NOT use np.corrcoef().
    Use only basic numpy operations (like sum, mean, etc.)
    
    Args:
        x: First array.
        y: Second array.
    
    Returns:
        Pearson's correlation coefficient between x and y.
    """
    mean_x = np.mean(x)
    mean_y = np.mean(y)

    deviation_x = x - mean_x
    deviation_y = y - mean_y

    covariance = np.sum(deviation_x * deviation_y) / (len(x) - 1)

    std_x = np.sqrt(np.sum(deviation_x ** 2) / (len(x) - 1))
    std_y = np.sqrt(np.sum(deviation_y ** 2) / (len(y) - 1))

    pearson_corr = covariance / (std_x * std_y)

    return pearson_corr

    raise NotImplementedError("You need to implement this function.")

def calculate_pearson_correlation_scipy(x: np.array, y: np.array) -> float:
    """Use scipy to calculate pearson's correlation coefficient.

    You can use scipy.stats.pearsonr() to calculate the correlation
    coefficient. Please return the "statistic" field from the function.
    
    Args:
        x: First array.
        y: Second array.
    
    Returns:
        Pearson's correlation coefficient between x and y.
    """
    correlation_coefficient, _ = pearsonr(x, y)

    return correlation_coefficient

    raise NotImplementedError("You need to implement this function.")

def calculate_spearman_correlation_scipy(x: np.array, y: np.array) -> float:
    """Use scipy to calculate spearman's correlation coefficient.

    You can use scipy.stats.spearmanr() to calculate the correlation
    coefficient. Please return the statistic from the function.
    
    Args:
        x: First array.
        y: Second array.
    
    Returns:
        Spearman's correlation coefficient between x and y.
    
    """
    spearman_coefficient, _ = spearmanr(x, y)

    return spearman_coefficient
    
    raise NotImplementedError("You need to implement this function.")

def perform_independent_t_test(x: np.array, y: np.array) -> tuple[float, float]:
    """Use scipy to calculate independent t-test.

    You can use scipy.stats.ttest_ind() to calculate the t-test.
    Please return the test statistic and p-value.
    
    Args:
        x: First array.
        y: Second array.
    
    Returns:
        Test statistic and p-value (float, float) from the independent t-test.
    """
    t_statistic, p_value = ttest_ind(x, y)

    return t_statistic, p_value
    
    raise NotImplementedError("You need to implement this function.")

def check_normality(x: np.array) -> tuple[float, float]:
    """Check if a sample is normally distributed using Shapiro-Wilk test.

    You can use scipy.stats.shapiro() to check for normality.
    Please return the test statistic and p-value.
    
    Args:
        x: Array to check for normality.
    
    Returns:
        Test statistic and p-value (float, float)from the Shapiro-Wilk test.
    """
    test_statistic, p_value = shapiro(x)

    return test_statistic, p_value

    raise NotImplementedError("You need to implement this function.")

def check_variance_homogeneity(x: np.array, y: np.array) -> tuple[float, float]:
    """Check if two samples have equal variance using Levene's test.

    You can use scipy.stats.levene() to check for equal variance.
    Please return the test statistic and p-value.
    
    Args:
        x: First array.
        y: Second array.
    
    Returns:
        Test statistic and p-value (float, float) from the Levene's test.
    """
    test_statistic, p_value = levene(x, y)

    return test_statistic, p_value

    raise NotImplementedError("You need to implement this function.")