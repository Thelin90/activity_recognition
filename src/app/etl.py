from src.modules.column_datatypes_definition import \
    double_type_names,\
    split_dataframe_weight, \
    target_column,\
    target_values, \
    default_drop_col, \
    knnr_train_test_weight
from src.modules.env.config import TRAIN_CSV_PATH, DEF_CSV_PATH, END_CSV_PATH, END_FOLDER_PATH
from pyspark.sql.functions import udf, monotonically_increasing_id
from pyspark.sql.types import DoubleType, IntegerType, StringType

import pandas as pd
import numpy as np
import os
import sys
import shutil
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
        # them to get a solid model. I would like to implement a SGD
        # approach on the whole dataset.
        #
        # This also gives me a fairly good overview of the data and I am able to analyze
        # subparts of the data to get a better understanding of it
        if not os.path.isfile(TRAIN_CSV_PATH):
            df = self.extract(DEF_CSV_PATH)
            df = self.transform(df)
            dfs = split_dataframe(df, True)
            write_split_dataset(dfs)

        df = self.extract(TRAIN_CSV_PATH)

        return self.transform(df)

    def transform(self, df):
        df = clean_columns_to_double(df)
        targetparser = udf(typecast_target_to_index_value, StringType())
        df = df.withColumn(target_column, targetparser(df[target_column]))
        df = df.drop(default_drop_col)
        df = clean_target_column_to_integer(df)

        train, test = split_dataframe(df, False)

        # extract train and some test data to validate against
        train_y = convert_col_to_list(train, target_column)
        train_x = convert_col_to_list(train, double_type_names)
        test_y = convert_col_to_list(test, target_column)
        test_x = convert_col_to_list(test, double_type_names)

        return train_x, train_y, test_x, test_y

    def load(self, results):
        logging.info('results')

        df = create_final_df(results, self.spark)

        write_to_csv(df, 'end_result')

        logging.info('mse: ' f'{results[2]}')


def create_final_df(results, spark):
    """

    :param results:
    :param spark:
    :return:
    """

    if os.path.isfile(END_CSV_PATH):
        try:
            shutil.rmtree(END_FOLDER_PATH)
        except OSError as e:
            logging.warning(f'{e.filename}' '-'  f'{e.strerror}')

    df1 = spark.createDataFrame(pd.DataFrame(results[0], columns=['predicted_target']))
    df2 = spark.createDataFrame(pd.DataFrame(results[1], columns=['actual_target']))

    df1 = df1.withColumn('id1', monotonically_increasing_id())
    df2 = df2.withColumn('id2', monotonically_increasing_id())

    target_parser = udf(typecast_target_to_str_value, StringType())
    df1_parsed_targets = df1.withColumn('predicted_target', target_parser(df1['predicted_target']))
    df2_parsed_targets = df2.withColumn('actual_target', target_parser(df2['actual_target']))

    df3 = df1_parsed_targets.join(df2_parsed_targets, df1_parsed_targets.id1 == df2_parsed_targets.id2)
    df3 = df3.select("predicted_target", "actual_target", "id1")

    df3 = df3.withColumnRenamed('predicted_target', 'predicted_target_str')
    df3 = df3.withColumnRenamed('actual_target', 'actual_target_str')
    df3 = df3.withColumnRenamed('id1', 'id')

    df3 = df3.join(df1, df1.id1 == df3.id)
    df3 = df3.select("predicted_target", "predicted_target_str", "actual_target_str", "id")

    return df3


def convert_col_to_list(df, col_name):
    return np.array(df.select(col_name).collect())


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


def split_dataframe(df, flag):
    """Function to split dataframe into subparts

    :param df: pyspark dataframe
    :param flag: determine if the split is for the initial split of the dataset or for the KNNR model
    :return: a list containing pyspark dataframes with a weight
    of 0.1 (~1*10^6 -> ~83333-100000 rows per dataset (tot 12))
    """
    if flag:
        return [df.randomSplit(split_dataframe_weight)]
    else:
        return df.randomSplit(knnr_train_test_weight)


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


def typecast_target_to_index_value(col_val):
    """Function to change target value to a numerical one instead for the model (Y)
    """
    if col_val in str(target_values):
        return target_values.index(col_val)


def typecast_target_to_str_value(col_val):
    # ex: 2.5 -> 3, 2.4 -> 2
    item = round(col_val)

    if 9 > item >= 0:
        return target_values[item]
    else:
        return 'out of scope'
