from src.modules.column_datatypes_definition import \
    double_type_names,\
    split_dataframe_weight, \
    target_column,\
    target_values, \
    default_drop_col
from src.modules.env.config import TRAIN_CSV_PATH, DEF_CSV_PATH
from pyspark.sql.functions import countDistinct, when, udf
from pyspark.sql.types import DoubleType, IntegerType, StringType

import os
import logging

logging.getLogger().setLevel(logging.INFO)


class ETL(object):
    """Class to perform basic ETL process from the original dataset to our training
    datasets.

    """

    def __init__(self, spark, delimiter, header, inferschema):
        """

        :param spark:
        :param delimiter:
        :param header:
        """
        self.spark = spark
        self.delimiter = delimiter
        self.header = header
        self.inferschema = inferschema

    def extract(self, csv_path):
        """Function to read csv to pyspark dataframe
        :return: pyspark Dataframe
        """

        return self.spark.read.option("delimiter", self.delimiter) \
            .option('header', self.header) \
            .option('inferSchema', self.inferschema) \
            .csv(csv_path)

    def run(self):
        """Method to run the data processing process

        :return: pyspark dataframe
        """

        # I do this because, I tried to split them in memory, but I got
        # problems to count the rows, I think there might be some problems
        # with some column values, so to basically speed up the process I save
        # sub parts of the original dataset into several files to try to apply
        # KNN on one or several datasets, however I will aim to do it on 1 to begin with.
        # The idea is to perform KNN on each and every file to then stack then combine
        # them to get a solid model. If I had more time I would implement more of a SGD
        # approach on the whole dataset.
        #
        # This also gives me a fairly good overview of the data and I am able to analyze
        # subparts of the data to get a better understanding of it
        if not os.path.isfile(TRAIN_CSV_PATH):
            df = self.extract(DEF_CSV_PATH)
            df = self.transform(df)
            dfs = df.randomSplit(split_dataframe_weight)
            write_split_dataset(dfs)
        else:
            df = self.extract(TRAIN_CSV_PATH)
            df = self.transform(df)

        return df

    def transform(self, df):
        df = clean_columns_to_double(df)
        targetparser = udf(typecast_target_value, StringType())
        df = df.withColumn(target_column, targetparser(df[target_column]))
        df = df.drop(default_drop_col)
        df = clean_target_column_to_integer(df)

        return df

    def load(self, df):
        df_ = df.groupBy(target_column).count()
        df_.show()
        logging.info(df.printSchema())
        logging.info('total rows: ' + f'{df.count()}')


def cast_to_datatype(df, colnames, datatype):
    """Function casts column values to given datatype, should be udf, fix later
    Args:
        df: Spark Dataframe
        colname: Name of column
        datatype: pyspark datatype for given column
    Returns cleaned initial transaction dataframe
    """
    for name in colnames:
        coldatatype = df.schema[name].dataType
        if isinstance(coldatatype, datatype):
            logging.info('Already this datatype')
        else:
            df = df.withColumn(name, df[name].cast(datatype()))

    return df


def split_dataframe(df):
    """Function to split dataframe into subparts

    :param df: pyspark dataframe
    :return: a list containing pyspark dataframes with a weight
    of 0.1 (~1*10^6 -> ~83333-100000 rows per dataset (tot 12))
    """
    dfs = [df.randomSplit(split_dataframe_weight)]
    return dfs


def write_split_dataset(dfs):
    """Function to write down subparts of the original dataset,
    in a real life situation this could be a write process to S3 or minio which
    can be used to replicate S3 but don't have time for that now

    :param dfs:
    :return:
    """
    count = 0
    for df in dfs:
        write_to_csv(df, 'dfsplit_' + f'{count}')
        count = count + 1


def write_to_csv(_df, _name):
    """Function to save the datafranes to CSV, with coalesce the whole file get written in on go,
    not nice in a real environment but ok for this setup
    Args:
        _df: spark dataframe
        _name: name of the folder containing the dataframe as CSV
    """
    _df.coalesce(1).write.format('com.databricks.spark.csv') \
        .option('header', 'true') \
        .mode('overwrite') \
        .save(os.getcwd() + '/data/split_data/' + _name)


def clean_target_column_to_integer(df):
    df = cast_to_datatype(df, [target_column], IntegerType)
    return df


def clean_columns_to_double(df):
    """Function to make sure correct columns are in type double,
    and clean dataframe from nulls to 0.0

    :param df: pyspark dataframe
    :return: cleaned columns in a pyspark dataframe
    """
    df = cast_to_datatype(df, double_type_names, DoubleType)
    df = df.na.fill(0.0)
    return df


def typecast_target_value(col_val):
    """Function to change target value to a numerical one instead for the model (Y)
    """
    if col_val in str(target_values):
        return target_values.index(col_val)
