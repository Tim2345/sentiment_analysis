import pandas as pd
import seaborn as sns

training_path = './data/training.1600000.processed.noemoticon.csv'
test_path = './data/testdata.manual.2009.06.14.csv'

training = pd.read_csv(training_path, header=None)
test = pd.read_csv(test_path, header=None)
# produce class distribution graphic
sns.countplot(training[0])
print(training[0].value_counts())

sns.countplot(test[0])
print(test[0].value_counts())




