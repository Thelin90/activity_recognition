

class ETL(object):
    """

    """

    def __init__(self, spark, delimiter, header, inferschema, csv_path):
        """

        :param spark:
        :param delimiter:
        :param header:
        :param csv_path:
        """
        self.spark = spark
        self.delimiter = delimiter
        self.header = header
        self.inferschema = inferschema
        self.csv_path = csv_path

    def extract(self):
        """Function to read csv to pyspark dataframe
        :return: pyspark Dataframe
        """
        return self.spark.read.option("delimiter", self.delimiter) \
            .option('header', self.header) \
            .option('inderSchema', self.inferschema) \
            .csv(self.csv_path)

    def transform(self):
        print("jhas")

    def load(self, df):
        print(df.printSchema())
        print(df.show(10))
        print(type(df))
        print(df.count())
