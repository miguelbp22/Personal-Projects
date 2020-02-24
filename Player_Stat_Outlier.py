import psycopg2
import pandas as pd
import pandas.io.sql as sqlio
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import os, os.path
import time
import onnxruntime as rt
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import auc, accuracy_score, confusion_matrix, classification_report
from sklearn.metrics import brier_score_loss
from sklearn.model_selection import train_test_split
from matplotlib import pyplot
import pickle 
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve, auc

sql = """select t.external_id, t.first_name, t.last_name, t.statistic, t.prediction, t.time_buckets, t.game_date, t.player_stat_bet_count, 
sum(t.player_stat_bet_count) over (partition by t.external_id, t.game_date, t.statistic, t.prediction order by t.time_buckets) as running_ct
from(
	select b.statistic, b.prediction, p.external_id, count(p.external_id) as player_stat_bet_count, p.first_name, p.last_name, cast(b.created_at as date) as game_date, 
	case 
		when (b.begin_period = 1 and b.begin_clock >= 540) then 1
		when (b.begin_period = 1 and b.begin_clock between 360 and 539) then 2
		when (b.begin_period = 1 and b.begin_clock between 180 and 359) then 3
		when (b.begin_period = 1 and b.begin_clock between 0 and 179) then 4
		when (b.begin_period = 2 and b.begin_clock >= 540) then 5
		when (b.begin_period = 2 and b.begin_clock between 360 and 539) then 6
		when (b.begin_period = 2 and b.begin_clock between 180 and 359) then 7
		when (b.begin_period = 2 and b.begin_clock between 0 and 179) then 8
		when (b.begin_period = 3 and b.begin_clock >= 540) then 9
		when (b.begin_period = 3 and b.begin_clock between 360 and 539) then 10
		when (b.begin_period = 3 and b.begin_clock between 180 and 359) then 11
		when (b.begin_period = 3 and b.begin_clock between 0 and 179) then 12
		when (b.begin_period = 4 and b.begin_clock >= 540) then 13 
		when (b.begin_period = 4 and b.begin_clock between 360 and 539) then 14 
		when (b.begin_period = 4 and b.begin_clock between 180 and 359) then 15
		when (b.begin_period = 4 and b.begin_clock between 0 and 179) then 16
		else 17 end as time_buckets 
	from bets b
	left join participants parts on b.participant_id = parts.id
	left join players p on parts.player_id = p.id
	where aasm_state = 'closed' and win is true and b.type = 'Basketball::Bet' and cast(b.created_at as date) >= '2019-11-25' and p.league_id = 1
	and b.entry_id not in (
		select e.id from entries e join users u on e.user_id = u.id
		where u.robot is true)
group by p.external_id, p.first_name, p.last_name, b.statistic, b.prediction, time_buckets, game_date
order by player_stat_bet_count desc) t
order by t.external_id, t.game_date, t.statistic asc, t.prediction asc, t.time_buckets asc"""

all_nba_bets = sqlio.read_sql_query(sql, connection)
connection.close()

#joining first and last name of players for simplicty's sake for the analysis part later on
all_nba_bets['full_name'] = all_nba_bets['first_name'].astype(str) + ' ' + all_nba_bets['last_name'].astype(str)
all_nba_bets = all_nba_bets.drop(['first_name', 'last_name'], axis = 1)

over_bets, under_bets = all_nba_bets[all_nba_bets['prediction'] == 0], all_nba_bets[all_nba_bets['prediction'] == 1]

over_bets['mean_bets'] = over_bets.groupby(['external_id', 'statistic', 'time_buckets'])['running_ct'].transform('mean')
over_bets['sd_bets'] = over_bets.groupby(['external_id', 'statistic', 'time_buckets'])['running_ct'].transform('std')

over_bets = over_bets.fillna(0)
over_bets['outlier'] = np.where(over_bets.running_ct > over_bets.mean_bets + (2.5*over_bets.sd_bets), 1, 0)

model_df = over_bets.loc[:, ['statistic', 'time_buckets', 'running_ct', 'mean_bets', 'sd_bets', 'outlier']]
model_df['statistic'] = model_df['statistic'].astype('object')
model_df = pd.get_dummies(model_df)

X = model_df.loc[:, ['time_buckets', 'running_ct', 'mean_bets', 'sd_bets', 'statistic_0', 'statistic_1', 'statistic_2', 'statistic_3', 'statistic_4']]
y = model_df.loc[:, 'outlier']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

logreg = LogisticRegression()
logreg.fit(X_train, y_train)

y_pred = logreg.predict_proba(X_test)
y_pred = y_pred[:,1] > 0.5

print("=== Confusion Matrix ===")
print(confusion_matrix(y_test, y_pred))
print('\n')
print("=== Classification Report ===")
print(classification_report(y_test, y_pred))

fpr, tpr, thresholds = roc_curve(y_test, y_pred)
# create plot
plt.plot(fpr, tpr, color='blue',
          label='ROC curve (area = %0.2f)' % auc(fpr, tpr))
plt.plot([0, 1], [0, 1], 'k--', label='Random guess')
_ = plt.xlabel('False Positive Rate')
_ = plt.ylabel('True Positive Rate')
_ = plt.title('ROC Curve')
_ = plt.xlim([-0.02, 1])
_ = plt.ylim([0, 1.02])
_ = plt.legend(loc="lower right")

under_bets['mean_bets'] = under_bets.groupby(['external_id', 'statistic', 'time_buckets'])['running_ct'].transform('mean')
under_bets['sd_bets'] = under_bets.groupby(['external_id', 'statistic', 'time_buckets'])['running_ct'].transform('std')

under_bets = under_bets.fillna(0)
under_bets['outlier'] = np.where(under_bets.running_ct > under_bets.mean_bets + (2.5*under_bets.sd_bets), 1, 0)

under_model_df = under_bets.loc[:, ['statistic', 'time_buckets', 'running_ct', 'mean_bets', 'sd_bets', 'outlier']]
under_model_df['statistic'] = under_model_df['statistic'].astype('object')
under_model_df = pd.get_dummies(under_model_df)

under_X = under_model_df.loc[:, ['time_buckets', 'running_ct', 'mean_bets', 'sd_bets', 'statistic_0', 'statistic_1', 'statistic_2', 'statistic_3', 'statistic_4']]
under_y = under_model_df.loc[:, 'outlier']
under_X_train, under_X_test, under_y_train, under_y_test = train_test_split(under_X, under_y, test_size=0.2, random_state=42)

under_logreg = LogisticRegression()
under_logreg.fit(under_X_train, under_y_train)

under_y_pred = under_logreg.predict_proba(under_X_test)
under_y_pred = under_y_pred[:,1] > 0.3

print("=== Confusion Matrix ===")
print(confusion_matrix(under_y_test, under_y_pred))
print('\n')
print("=== Classification Report ===")
print(classification_report(under_y_test, under_y_pred))

fpr, tpr, thresholds = roc_curve(under_y_test, under_y_pred)
# create plot
plt.plot(fpr, tpr, color='blue',
          label='ROC curve (area = %0.2f)' % auc(fpr, tpr))
plt.plot([0, 1], [0, 1], 'k--', label='Random guess')
_ = plt.xlabel('False Positive Rate')
_ = plt.ylabel('True Positive Rate')
_ = plt.title('ROC Curve')
_ = plt.xlim([-0.02, 1])
_ = plt.ylim([0, 1.02])
_ = plt.legend(loc="lower right")
