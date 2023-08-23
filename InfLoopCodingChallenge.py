from pyspark.ml import Pipeline
from pyspark.mllib.evaluation import BinaryClassificationMetrics
from pyspark.sql import SparkSession

spark = SparkSession.builder.appName("read csv file").getOrCreate()

ip_df = spark.read.csv('IpAddress_to_Country.csv', header=True, inferSchema=True)
transaction_df = spark.read.csv('Fraud_Data_subset.csv', header=True, inferSchema=True)

cond = [transaction_df.ip_address >= ip_df.lower_bound_ip_address,
        transaction_df.ip_address <= ip_df.upper_bound_ip_address]
transaction_df = transaction_df.join(ip_df, cond, 'left')


from collections import defaultdict
from pyspark.sql.functions import countDistinct
data_types = defaultdict(list)
for entry in transaction_df.schema.fields:
    data_types[str(entry.dataType)].append(entry.name)

counts_summary = transaction_df.agg(*[countDistinct(c).alias(c) for c in data_types["StringType"]])
counts_summary = counts_summary.toPandas()
import pandas as pd
counts = pd.Series(counts_summary.values.ravel())
counts.index = counts_summary.columns
sorted_vars = counts.sort_values(ascending = False)
print(sorted_vars)

import math
from pyspark.sql.functions import col, expr, when, substring, unix_timestamp
transaction_df = transaction_df.withColumn('purchase_tod', substring(col('purchase_time'), 12, 2))
transaction_df = transaction_df.withColumn('purchase_size', when(col('purchase_value').cast('int') < 40, 'small').otherwise(when(col('purchase_value').cast('int') < 60, 'medium').otherwise(when(col('purchase_value').cast('int') < 90, 'large').otherwise('max'))))
transaction_df = transaction_df.withColumn('age_bin', when(col('age').cast('int') < 30, 'youth').otherwise(when(col('age').cast('int') < 65, 'adult').otherwise('elder')))
transaction_df = transaction_df.withColumn('waited_seconds', (unix_timestamp(col('purchase_time')) - unix_timestamp(col('signup_time'))).cast('long'))
transaction_df = transaction_df.withColumn('waited', when(col('waited_seconds').cast('int') < 60, 'seconds').otherwise(when(col('waited_seconds').cast('int') < 'hours', 1).otherwise(when(col('waited_seconds').cast('int') < 3600*24*30, 'days').otherwise('months'))))

model_df = transaction_df.select('source', 'browser', 'sex', 'country', 'purchase_tod', 'purchase_size', 'age_bin', 'waited', 'class')
model_df = model_df.where("source is not null and browser is not null and sex is not null and country is not null and purchase_tod is not null and waited is not null and class is not null")

model_df.printSchema()

model_df.show(10)

from pyspark.ml import Pipeline
from pyspark.ml.feature import StringIndexer
from pyspark.ml.feature import OneHotEncoder, OneHotEncoderModel

# just in case.  should all be categorical
categoricalColumns = [item[0] for item in model_df.dtypes if item[1].startswith('string')]
# iterate through them
stage_string = [StringIndexer(inputCol= c, outputCol = c + 'Index') for c in categoricalColumns] + [StringIndexer(inputCol='class', outputCol = 'objective')]
stage_encoder = [OneHotEncoder(inputCol= c+'Index', outputCol= c+'Vector') for c in categoricalColumns] + [OneHotEncoder(inputCol='objective', outputCol = 'objectiveVector')]
transform_pipeline = Pipeline(stages = stage_string + stage_encoder)
pm = transform_pipeline.fit(model_df)
new_model_df = pm.transform(model_df)

new_model_df.show(10)

# Replace the old model_df with one that can be operated on by VectorAssembler
model_df = new_model_df.select('sourceVector', 'browserVector', 'sexVector', 'countryVector', 'purchase_todVector', 'purchase_sizeVector', 'age_binVector', 'waitedVector', 'objective')

model_df.show(10)

from pyspark.ml.linalg import Vectors
from pyspark.ml.feature import VectorAssembler

assembler = VectorAssembler(inputCols = ['sourceVector', 'browserVector', 'sexVector', 'countryVector', 'purchase_todVector', 'purchase_sizeVector', 'age_binVector', 'waitedVector'], outputCol = 'features')

model_features = assembler.transform(model_df)

model_features.show(10)

train_data,test_data = model_features.randomSplit([0.7,0.3])

from pyspark.ml.classification import (LogisticRegression, GBTClassifier, DecisionTreeClassifier)
from pyspark.ml.evaluation import BinaryClassificationEvaluator
import matplotlib.pyplot as plt

lrc = LogisticRegression(featuresCol = 'features', labelCol = 'objective', maxIter=30)
dtc = DecisionTreeClassifier(featuresCol='features', labelCol='objective')
gbtc = GBTClassifier(featuresCol='features', labelCol='objective', maxIter=30)

lrc_model = lrc.fit(train_data)
dtc_model = dtc.fit(train_data)
gbtc_model = gbtc.fit(train_data)

## Examine Logistic model
print("Multinomial coefficients:", lrc_model.coefficientMatrix)
lrc_summary = lrc_model.summary
print('Logistic model training set areaUnderROC: ' + str(lrc_summary.areaUnderROC))

roc = lrc_summary.roc.toPandas()
plt.plot(roc['FPR'],roc['TPR'])
plt.ylabel('False Positive Rate')
plt.xlabel('True Positive Rate')
plt.title('Logistic Regression ROC Curve')
plt.show()

pr = lrc_summary.pr.toPandas()
plt.plot(pr['recall'],pr['precision'])
plt.ylabel('Precision')
plt.xlabel('Recall')
plt.title('Logistic Regression Precision Recall Curve')
plt.show()


####Now, analyze the Decision tree model
from sklearn.metrics import roc_curve, precision_recall_curve
print(dtc_model.featureImportances)
dt_predictions = dtc_model.transform(test_data)
dt_predictions.show(10)

label_obj = [float(row['objective']) for row in dt_predictions.select('objective','prediction').collect()]
prediction = [float(row['prediction']) for row in dt_predictions.select('objective','prediction').collect()]
roc = roc_curve(label_obj,prediction)
plt.plot(roc[0],roc[1])
plt.ylabel('False Positive Rate')
plt.xlabel('True Positive Rate')
plt.title('Decision Tree ROC Curve')
plt.show()

pr = precision_recall_curve(label_obj,prediction)
plt.plot(pr[0],pr[1])
plt.ylabel('False Positive Rate')
plt.xlabel('True Positive Rate')
plt.title('Decision Tree Precision Recall Curve')
plt.show()


def ExtractFeatureImportance(featureImp, dataset, featuresCol):
    list_extract = []
    for i in dataset.schema[featuresCol].metadata["ml_attr"]["attrs"]:
        list_extract = list_extract + dataset.schema[featuresCol].metadata["ml_attr"]["attrs"][i]
    varlist = pd.DataFrame(list_extract)
    varlist['score'] = varlist['idx'].apply(lambda x: featureImp[x])
    return (varlist.sort_values('score', ascending=False))

dataset_fi = ExtractFeatureImportance(dtc_model.featureImportances, dt_predictions, "features")
print(dataset_fi.head(10).to_string(index=False))



gbt_predictions = gbtc_model.transform(test_data)
label_obj = [float(row['objective']) for row in gbt_predictions.select('objective','prediction').collect()]
prediction = [float(row['prediction']) for row in gbt_predictions.select('objective','prediction').collect()]
roc = roc_curve(label_obj,prediction)
plt.plot(roc[0],roc[1])
plt.ylabel('False Positive Rate')
plt.xlabel('True Positive Rate')
plt.title('GBT ROC Curve')
plt.show()

pr = precision_recall_curve(label_obj,prediction)
plt.plot(pr[0],pr[1])
plt.ylabel('False Positive Rate')
plt.xlabel('True Positive Rate')
plt.title('GBT Precision Recall Curve')
plt.show()

dataset_fi = ExtractFeatureImportance(gbtc_model.featureImportances, gbt_predictions, "features")
print(dataset_fi.head(10).to_string(index=False))

print()
