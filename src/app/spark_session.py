from pyspark.sql import SparkSession


class InitSpark(object):

    def __init__(self, app_name, warehouse_location):
        self.app_name = app_name
        self.warehouse_location = warehouse_location

    def spark_init(self):

        return SparkSession \
            .builder \
            .appName(self.app_name) \
            .config("spark.sql.warehouse.dir", self.warehouse_location) \
            .getOrCreate()
