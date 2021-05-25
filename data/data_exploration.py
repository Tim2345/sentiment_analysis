import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split

training_path = './data/training.1600000.processed.noemoticon.csv'
test_path = './data/testdata.manual.2009.06.14.csv'



training = pd.read_csv(training_path, header=None)
test = pd.read_csv(test_path, header=None)
# produce class distribution graphic
sns.countplot(training[0])
print(training[0].value_counts())

sns.countplot(test[0])
print(test[0].value_counts())

train_X, train_y, dev_X, dev_y = train_test_split(
    training.iloc(axis=1)[1:],
    training[0],
    train_size=0.9,
    random_state=42
)

