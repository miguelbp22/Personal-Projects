import pandas as pd
import numpy as np
import os.path
import time
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import RandomizedSearchCV
import os
from sklearn.metrics import accuracy_score
from sklearn.metrics import brier_score_loss
import onnxruntime as rt
import matplotlib.pyplot as plt
import seaborn as sns
import onnxmltools.convert.common.data_types as dt
from skl2onnx.common.data_types import FloatTensorType
import onnxmltools as ot
import skl2onnx

os.getcwd()
############################
#####DATAPRE-PROCESSING#####
############################
# loading column names of dataset and appending it with game_id
col_names = pd.read_csv('/Users/miguelbp/Downloads/training_files_playoffs/column_names.csv').values.flatten()
full_col_names = np.append(col_names, 'game_id')

# loading game info created with playoff series info
playoff_game_info = pd.read_csv('/Users/miguelbp/Documents/playoff_games_w_ids.csv')

# making a new dataframe
full_playoffs = pd.DataFrame(columns=full_col_names)

# creating a dataset from playoffs games, appening its game id to it to later join it with playoff game info df
path, dirs, files = next(os.walk("/Users/miguelbp/Downloads/training_files_playoffs/"))
full_playoffs_array = np.zeros((10000000, 58), dtype=object)
start_idx = 0
for num, file in enumerate(files):
    if not file.startswith('20'):
        pass
    else:
        game_id = str(file[11:-4])
        game = np.load(path + file, allow_pickle=True)['arr_0']

        game_id_full = np.full((game.shape[0], 1), game_id, dtype=object)
        game = np.concatenate((game, game_id_full), axis=1)

        end_idx = start_idx + game.shape[0]

        full_playoffs_array[start_idx:end_idx, :] = game

        start_idx = end_idx

        print(num)
# capping the array at the end of the relevant data
full_playoff_array = full_playoffs_array[0:end_idx, :]

# making a df out of all games and merging with the game info
full_playoffs = pd.DataFrame(data=full_playoff_array, columns=full_col_names)
full_playoff_df = pd.merge(full_playoffs, playoff_game_info, how='left', on='game_id')

# making a variable w the by-player state of the series
series_score = []
for row in full_playoff_df.itertuples():
    if row.is_home == 1:
        series_score.append(row.series_score_home)
    else:
        series_score.append(row.series_score_away)
full_playoff_df['series_score'] = series_score
full_playoff_df = full_playoff_df.drop(['series_score_home', 'series_score_away'], axis=1).reset_index(drop=True)

#full_playoff_df = pd.read_csv('/Users/miguelbp/Documents/full_playoff_df.csv')

##########################
#####ALL-STAT TUNNING#####
##########################

og_pts_idx = [10, 5, 0, 43, 11, 51, 12, 36, 35, 34, 31, 30, 28, 29, 19, 18, 21, 20, 23,
                22, 16, 45, 44, 47, 46, 49, 48, 39, 13, 37, 17, 38, 15, 40, 14, 41]
og_rebs_idx = [10, 7, 2, 43, 11, 51, 12, 36, 35, 34, 31, 30, 28, 29, 19, 18, 21, 20, 23,
                22, 16, 45, 44, 47, 46, 49, 48, 39, 13, 37, 17, 38, 15, 40, 14, 41]
og_assists_idx = [10, 6, 1, 43, 11, 51, 12, 36, 35, 34, 31, 30, 28, 29, 19, 18, 21, 20, 23,
                22, 16, 45, 44, 47, 46, 49, 48, 39, 13, 37, 17, 38, 15, 40, 14, 41]
og_stocks_idx = [10, 8, 3, 43, 11, 51, 12, 36, 35, 34, 31, 30, 28, 29, 19, 18, 21, 20, 23,
                22, 16, 45, 44, 47, 46, 49, 48, 39, 13, 37, 17, 38, 15, 40, 14, 41]
og_threes_idx = [10, 9, 4, 43, 11, 51, 12, 36, 35, 34, 31, 30, 28, 29, 19, 18, 21, 20, 23,
                22, 16, 45, 44, 47, 46, 49, 48, 39, 13, 37, 17, 38, 15, 40, 14, 41]
og_idx = [og_pts_idx, og_rebs_idx, og_assists_idx, og_stocks_idx, og_threes_idx]


pts_features = ['points_odds', 'points_temporal_odds', 'playing_time_percent',
                'point_differential', 'points', 'assists', 'blocks', 'steals',
                'offensive_rebounds', 'defensive_rebounds', 'two_point_makes',
                'two_point_attempts', 'three_point_makes', 'three_point_attempts',
                'freethrow_makes', 'freethrow_attempts', 'minutes', 'home_score', 'away_score', 'period',
                'game_clock', 'is_home', 'on_court', 'seconds_played', 'time_in',
                'in_play_points', 'in_play_assists', 'in_play_defensive_rebounds',
                'in_play_offensive_rebounds', 'in_play_steals', 'in_play_blocks',
                'in_play_turnovers', 'in_play_fouls', 'in_play_two_point_makes',
                'in_play_two_point_attempts', 'in_play_three_point_makes',
                'in_play_three_point_attempts', 'in_play_freethrow_makes',
                'in_play_freethrow_attempts', 'time_out', 'time_elapsed',
                'series_num', 'series_game_num',
                'series_score']
pts_target = ['observed_points']
rebs_features = ['rebounds_odds', 'rebounds_temporal_odds', 'playing_time_percent',
                'point_differential', 'points', 'assists', 'blocks', 'steals',
                'offensive_rebounds', 'defensive_rebounds', 'two_point_makes',
                'two_point_attempts', 'three_point_makes', 'three_point_attempts',
                'freethrow_makes', 'freethrow_attempts', 'minutes', 'home_score', 'away_score', 'period',
                'game_clock', 'is_home', 'on_court', 'seconds_played', 'time_in',
                'in_play_points', 'in_play_assists', 'in_play_defensive_rebounds',
                'in_play_offensive_rebounds', 'in_play_steals', 'in_play_blocks',
                'in_play_turnovers', 'in_play_fouls', 'in_play_two_point_makes',
                'in_play_two_point_attempts', 'in_play_three_point_makes',
                'in_play_three_point_attempts', 'in_play_freethrow_makes',
                'in_play_freethrow_attempts', 'time_out', 'time_elapsed',
                'series_num', 'series_game_num',
                'series_score']
rebs_target = ['observed_rebounds']
assists_features = ['assists_odds', 'assists_temporal_odds', 'playing_time_percent',
                'point_differential', 'points', 'assists', 'blocks', 'steals',
                'offensive_rebounds', 'defensive_rebounds', 'two_point_makes',
                'two_point_attempts', 'three_point_makes', 'three_point_attempts',
                'freethrow_makes', 'freethrow_attempts', 'minutes', 'home_score', 'away_score', 'period',
                'game_clock', 'is_home', 'on_court', 'seconds_played', 'time_in',
                'in_play_points', 'in_play_assists', 'in_play_defensive_rebounds',
                'in_play_offensive_rebounds', 'in_play_steals', 'in_play_blocks',
                'in_play_turnovers', 'in_play_fouls', 'in_play_two_point_makes',
                'in_play_two_point_attempts', 'in_play_three_point_makes',
                'in_play_three_point_attempts', 'in_play_freethrow_makes',
                'in_play_freethrow_attempts', 'time_out', 'time_elapsed',
                'series_num', 'series_game_num',
                'series_score']
assists_target = ['observed_assists']
stocks_features = ['stocks_odds', 'stocks_temporal_odds', 'playing_time_percent',
                'point_differential', 'points', 'assists', 'blocks', 'steals',
                'offensive_rebounds', 'defensive_rebounds', 'two_point_makes',
                'two_point_attempts', 'three_point_makes', 'three_point_attempts',
                'freethrow_makes', 'freethrow_attempts', 'minutes', 'home_score', 'away_score', 'period',
                'game_clock', 'is_home', 'on_court', 'seconds_played', 'time_in',
                'in_play_points', 'in_play_assists', 'in_play_defensive_rebounds',
                'in_play_offensive_rebounds', 'in_play_steals', 'in_play_blocks',
                'in_play_turnovers', 'in_play_fouls', 'in_play_two_point_makes',
                'in_play_two_point_attempts', 'in_play_three_point_makes',
                'in_play_three_point_attempts', 'in_play_freethrow_makes',
                'in_play_freethrow_attempts', 'time_out', 'time_elapsed',
                'series_num', 'series_game_num',
                'series_score']
stocks_target = ['observed_stocks']
threes_features = ['three_points_odds', 'three_points_temporal_odds', 'playing_time_percent',
                'point_differential', 'points', 'assists', 'blocks', 'steals',
                'offensive_rebounds', 'defensive_rebounds', 'two_point_makes',
                'two_point_attempts', 'three_point_makes', 'three_point_attempts',
                'freethrow_makes', 'freethrow_attempts', 'minutes', 'home_score', 'away_score', 'period',
                'game_clock', 'is_home', 'on_court', 'seconds_played', 'time_in',
                'in_play_points', 'in_play_assists', 'in_play_defensive_rebounds',
                'in_play_offensive_rebounds', 'in_play_steals', 'in_play_blocks',
                'in_play_turnovers', 'in_play_fouls', 'in_play_two_point_makes',
                'in_play_two_point_attempts', 'in_play_three_point_makes',
                'in_play_three_point_attempts', 'in_play_freethrow_makes',
                'in_play_freethrow_attempts', 'time_out', 'time_elapsed',
                'series_num', 'series_game_num',
                'series_score']
threes_target = ['observed_three_points']

stat_features = [pts_features, rebs_features, assists_features, stocks_features, threes_features]
stat_targets = [pts_target, rebs_target, assists_target, stocks_target, threes_target]
best_parameters = []

stats = ['points', 'rebounds', 'assists', 'stocks', 'three_points_made']

best_parameters = []

for num, stat in enumerate(stats):
    X = full_playoff_df.loc[:, stat_features[num]].reset_index(drop=True)
    y = full_playoff_df.loc[:, stat_targets[num]].reset_index(drop=True)

    X = X.astype(float)
    X = X.to_numpy()

    y = y.values
    y = np.where(y > .5, 1, 0)
    y = y.reshape(pts_y.shape[0],)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

    class_balance = np.count_nonzero(y == 0) / np.count_nonzero(y == 1)

    clf = XGBClassifier(n_jobs=12)
    param_grid = {
        'silent': [False],
        'scale_pos_weight': [1, class_balance],
        'objective': ['binary:logistic'],
        'learning_rate': [0.001, 0.01, 0.05, 0.1, 0.2],
        'colsample_bytree': [0.2, 0.3, 0.4, 0.5, 0.6],
        'subsample': [0.6, 0.7, 0.8],
        'n_estimators': [50, 100, 150, 200],
        'reg_alpha': [0.1, 0.3, 0.5],
        'max_depth': [1, 3, 5, 7],
        'gamma': [0, 1, 5, 10]
    }
    rs_clf = RandomizedSearchCV(clf, param_grid, n_iter=20,
                                pre_dispatch='2*n_jobs', verbose=1, cv=5,
                                scoring='neg_log_loss')
    print("Randomized search..")
    search_time_start = time.time()
    rs_clf.fit(X_train, y_train)
    print("Randomized search time:", time.time() - search_time_start)
    best_score = rs_clf.best_score_
    best_params = rs_clf.best_params_
    best_parameters.append(best_params)
    print("Best score: {}".format(best_score))
    print("Best params: ")
    for param_name in sorted(best_params.keys()):
        print('%s: %r' % (param_name, best_params[param_name]))
        

######################
####MODEL VIZ TOOL####
######################

def model_comp_viz(stat, full_game_id, player_name, player_id):
    idx = stats.index(stat)

    features = stat_features[idx]

    og_stat_idx = og_idx[idx]
    stat_idx = []
    for i in features[0:41]:
        stat_idx.append(col_names.tolist().index(i))

    playoff_game = np.load('/Users/miguelbp/Downloads/training_files_playoffs/%s.npz' % full_game_id, allow_pickle=True)['arr_0']

    player_stats = playoff_game[playoff_game[:, 25] == player_id]

    player_og_stats = player_stats[:, og_stat_idx]
    player_stats = player_stats[:, stat_idx]

    game_id = full_game_id[11:]
    player_game_info = playoff_game_info[playoff_game_info['game_id'] == game_id]
    player_game_info = player_game_info.loc[:, ['series_num', 'series_game_num', 'series_score_away']].values
    player_game_info = np.full((player_stats.shape[0], 3), player_game_info)
    player_stats = np.concatenate((player_stats, player_game_info), axis=1).astype(float)

    sess = rt.InferenceSession(
        '/Users/miguelbp/Documents/GitHub/playoff_models/%s_playoff_model.onnx' % stat)
    input_name = sess.get_inputs()[0]
    label_name = sess.get_outputs()[1]
    rows = player_stats.shape[0]
    xgb_model_preds = []
    for row in range(rows):
        pred_onx = \
            sess.run([label_name.name], {input_name.name: np.array([player_stats[row, :]], dtype=np.float32)})[0][0][
                1]
        xgb_model_preds.append(pred_onx)

    sess = rt.InferenceSession(
        '/Users/miguelbp/Documents/GitHub/ft-api/app/models/machine_learning/models/nba/temporal/%s.onnx' % stat)
    input_name = sess.get_inputs()[0]
    label_name = sess.get_outputs()[1]
    rows = player_og_stats.shape[0]
    old_model_preds = []
    for row in range(rows):
        pred_onx = \
        sess.run([label_name.name], {input_name.name: np.array([player_og_stats[row, :]], dtype=np.float32)})[0][0][1]
        old_model_preds.append(pred_onx)


    plt.plot(player_og_stats[:, 5], xgb_model_preds, 'o', color='blue',
             markersize=2, linewidth=1,
             markerfacecolor='blue',
             markeredgecolor='blue',
             markeredgewidth=1, label='Playoff Model Preds')
    plt.plot(player_og_stats[:, 5], player_og_stats[:, 1:2], 'o', color='red',
             markersize=2, linewidth=1,
             markerfacecolor='red',
             markeredgecolor='red',
             markeredgewidth=1, label='Historical Odds')
    plt.plot(player_og_stats[:, 5], old_model_preds, 'o', color='green',
             markersize=2, linewidth=1,
             markerfacecolor='green',
             markeredgecolor='green',
             markeredgewidth=1, label='Regular Season Model Odds')
    plt.legend()
    plt.title('%s %s Historic vs. Playoff Model Odds vs. Reg. Season Model Odds' % (player_name, stats[idx]))
    plt.show

#####################
#######POINTS########
#####################
X = full_playoff_df.loc[:, pts_features].reset_index(drop=True)
y = full_playoff_df.loc[:, pts_target].reset_index(drop=True)

X = X.astype(float)
X = X.to_numpy()

y = y.values
y = np.where(y > .5, 1, 0)
y = y.reshape(pts_y.shape[0],)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

points_xgb_model = XGBClassifier(verbosity = 2,
        scale_pos_weight=1,
        learning_rate=0.05,
        colsample_bytree = 0.6,
        subsample = 0.8,
        objective='binary:logistic',
        n_estimators=150, #1000
        reg_alpha = 0.3,
        max_depth=7,
        gamma=5,
        n_jobs=8)

points_xgb_model.fit(X_train, y_train)

y_preds = points_xgb_model.predict(X_test)
accuracy = accuracy_score(y_test, y_preds)
accuracy

brier = brier_score_loss(y_test, y_preds)
brier

sys.path.append("/Users/miguelbp/Documents/GitHub/data_science_library/scripts")
from model_tools import visualize_modified_brier_time_model

full_X_test = np.concatenate((X_test, player_X_test), axis = 1)
min_max_cols = []
min_max_cols.extend(pts_features)
min_max_cols.extend(['game_id', 'player_id'])

df = pd.DataFrame(data = full_X_test, columns = min_max_cols)

df['min'] = df.groupby(['game_id', 'player_id'])['points_temporal_odds'].transform('min')
df['max'] = df.groupby(['game_id', 'player_id'])['points_temporal_odds'].transform('max')
min_max = df[['min', 'max']].to_numpy()

visualize_modified_brier_time_model(y_preds, X_test, y_test, min_max, pts_features)

for num, col in enumerate(pts_features):
    print(col, ':', points_xgb_model.feature_importances_[num])

features = []
importance = []
for num, col in enumerate(pts_features):
    features.append(col)
    importance.append(points_xgb_model.feature_importances_[num])

feat_import = pd.DataFrame({'features': features, 'importance': importance}).sort_values('importance', ascending=False)

initial_type = [('float_input', dt.FloatTensorType([None, X_train.shape[1]]))]
onx = ot.convert_xgboost(points_xgb_model, initial_types=initial_type)
onx.graph.output[0].name = 'variable'
onx.graph.node[0].output[0] = 'variable'
onx.graph.output[1].name = 'output_probability'
onx.graph.node[0].output[1] = 'output_probability'
with open("points_playoff_model.onnx", "wb") as f:
    f.write(onx.SerializeToString())

sess = rt.InferenceSession("points_playoff_model.onnx")
input_name = sess.get_inputs()[0].name
label_name = sess.get_outputs()[0].name
pred_onnx = sess.run([label_name],
                    {input_name: X_test.astype(np.float32)})[0]


###################
####REBOUNDS#######
###################
X = full_playoff_df.loc[:, rebs_features].reset_index(drop=True)
y = full_playoff_df.loc[:, rebs_target].reset_index(drop=True)

X = X.astype(float)
X = X.to_numpy()

y = y.values
y = np.where(y > .5, 1, 0)
y = y.reshape(pts_y.shape[0],)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

rebounds_xgb_model = XGBClassifier(verbosity = 2,
        scale_pos_weight=1,
        learning_rate=0.1,
        colsample_bytree = 0.5,
        subsample = 0.6,
        objective='binary:logistic',
        n_estimators=200, #1000
        reg_alpha = 0.1,
        max_depth=7,
        gamma=1,
        n_jobs=8)

rebounds_xgb_model.fit(X_train, y_train)

rebounds_y_preds = rebounds_xgb_model.predict(X_test)
accuracy = accuracy_score(y_test, rebounds_y_preds)
accuracy

brier = brier_score_loss(y_test, rebounds_y_preds)
brier

full_X_test = np.concatenate((X_test, player_X_test), axis = 1)
min_max_cols = []
min_max_cols.extend(rebs_features)
min_max_cols.extend(['game_id', 'player_id'])

df = pd.DataFrame(data = full_X_test, columns = min_max_cols)

df['min'] = df.groupby(['game_id', 'player_id'])['rebounds_temporal_odds'].transform('min')
df['max'] = df.groupby(['game_id', 'player_id'])['rebounds_temporal_odds'].transform('max')
min_max = df[['min', 'max']].to_numpy()

visualize_modified_brier_time_model(rebounds_y_preds, X_test, y_test, min_max, rebs_features)

for num, col in enumerate(rebs_features):
    print(col, ':', rebounds_xgb_model.feature_importances_[num])

features = []
importance = []
for num, col in enumerate(rebs_features):
    features.append(col)
    importance.append(rebounds_xgb_model.feature_importances_[num])

feat_import = pd.DataFrame({'features': features, 'importance': importance}).sort_values('importance', ascending=False)

initial_type = [('float_input', dt.FloatTensorType([None, X_train.shape[1]]))]
onx = ot.convert_xgboost(rebounds_xgb_model, initial_types=initial_type)
onx.graph.output[0].name = 'variable'
onx.graph.node[0].output[0] = 'variable'
onx.graph.output[1].name = 'output_probability'
onx.graph.node[0].output[1] = 'output_probability'
with open("rebounds_playoff_model.onnx", "wb") as f:
    f.write(onx.SerializeToString())

sess = rt.InferenceSession("rebounds_playoff_model.onnx")
input_name = sess.get_inputs()[0].name
label_name = sess.get_outputs()[0].name
pred_onnx = sess.run([label_name],
                    {input_name: X_test.astype(np.float32)})[0]


###################
####ASSISTS########
###################
X = full_playoff_df.loc[:, assists_features].reset_index(drop=True)
y = full_playoff_df.loc[:, assists_target].reset_index(drop=True)

X = X.astype(float)
X = X.to_numpy()

y = y.values
y = np.where(y > .5, 1, 0)
y = y.reshape(pts_y.shape[0],)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

assists_xgb_model = XGBClassifier(verbosity = 2,
        scale_pos_weight=1,
        learning_rate=0.2,
        colsample_bytree = 0.6,
        subsample = 0.7,
        objective='binary:logistic',
        n_estimators=200, #1000
        reg_alpha = 0.5,
        max_depth=7,
        gamma=1,
        n_jobs=8)

assists_xgb_model.fit(X_train, y_train)

assists_y_preds = assists_xgb_model.predict(X_test)
accuracy = accuracy_score(y_test, assists_y_preds)
accuracy

brier = brier_score_loss(y_test, assists_y_preds)
brier

full_X_test = np.concatenate((X_test, player_X_test), axis = 1)
min_max_cols = []
min_max_cols.extend(assists_features)
min_max_cols.extend(['game_id', 'player_id'])

df = pd.DataFrame(data = full_X_test, columns = min_max_cols)

df['min'] = df.groupby(['game_id', 'player_id'])['assists_temporal_odds'].transform('min')
df['max'] = df.groupby(['game_id', 'player_id'])['assists_temporal_odds'].transform('max')
min_max = df[['min', 'max']].to_numpy()

visualize_modified_brier_time_model(assists_y_preds, X_test, y_test, min_max, assists_features)

for num, col in enumerate(assists_features):
    print(col, ':', assists_xgb_model.feature_importances_[num])

features = []
importance = []
for num, col in enumerate(assists_features):
    features.append(col)
    importance.append(assists_xgb_model.feature_importances_[num])
feat_import = pd.DataFrame({'features': features, 'importance': importance}).sort_values('importance', ascending=False)

initial_type = [('float_input', dt.FloatTensorType([None, X_train.shape[1]]))]
onx = ot.convert_xgboost(assists_xgb_model, initial_types=initial_type)
onx.graph.output[0].name = 'variable'
onx.graph.node[0].output[0] = 'variable'
onx.graph.output[1].name = 'output_probability'
onx.graph.node[0].output[1] = 'output_probability'
with open("assists_playoff_model.onnx", "wb") as f:
    f.write(onx.SerializeToString())

sess = rt.InferenceSession("assists_playoff_model.onnx")
input_name = sess.get_inputs()[0].name
label_name = sess.get_outputs()[0].name
pred_onnx = sess.run([label_name],
                    {input_name: X_test.astype(np.float32)})[0]


###################
#####STOCKS########
###################
X = full_playoff_df.loc[:, stocks_features].reset_index(drop=True)
y = full_playoff_df.loc[:, stocks_target].reset_index(drop=True)

X = X.astype(float)
X = X.to_numpy()

y = y.values
y = np.where(y > .5, 1, 0)
y = y.reshape(pts_y.shape[0],)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

stocks_xgb_model = XGBClassifier(verbosity = 2,
        scale_pos_weight=1,
        learning_rate=0.1,
        colsample_bytree = 0.3,
        subsample = 0.7,
        objective='binary:logistic',
        n_estimators=200, #1000
        reg_alpha = 0.5,
        max_depth=7,
        gamma=1,
        n_jobs=8)

stocks_xgb_model.fit(X_train, y_train)

stocks_y_preds = stocks_xgb_model.predict(X_test)
accuracy = accuracy_score(y_test, stocks_y_preds)
accuracy

brier = brier_score_loss(y_test, stocks_y_preds)
brier

full_X_test = np.concatenate((X_test, player_X_test), axis = 1)
min_max_cols = []
min_max_cols.extend(stocks_features)
min_max_cols.extend(['game_id', 'player_id'])

df = pd.DataFrame(data = full_X_test, columns = min_max_cols)

df['min'] = df.groupby(['game_id', 'player_id'])['stocks_temporal_odds'].transform('min')
df['max'] = df.groupby(['game_id', 'player_id'])['stocks_temporal_odds'].transform('max')
min_max = df[['min', 'max']].to_numpy()

visualize_modified_brier_time_model(stocks_y_preds, X_test, y_test, min_max, stocks_features)

features = []
importance = []
for num, col in enumerate(stocks_features):
    features.append(col)
    importance.append(stocks_xgb_model.feature_importances_[num])
feat_import = pd.DataFrame({'features': features, 'importance': importance}).sort_values('importance', ascending=False)

initial_type = [('float_input', dt.FloatTensorType([None, X_train.shape[1]]))]
onx = ot.convert_xgboost(stocks_xgb_model, initial_types=initial_type)
onx.graph.output[0].name = 'variable'
onx.graph.node[0].output[0] = 'variable'
onx.graph.output[1].name = 'output_probability'
onx.graph.node[0].output[1] = 'output_probability'
with open("stocks_playoff_model.onnx", "wb") as f:
    f.write(onx.SerializeToString())

sess = rt.InferenceSession("stocks_playoff_model.onnx")
input_name = sess.get_inputs()[0].name
label_name = sess.get_outputs()[0].name
pred_onnx = sess.run([label_name],
                    {input_name: X_test.astype(np.float32)})[0]


###################
#####THREES########
###################
X = full_playoff_df.loc[:, threes_features].reset_index(drop=True)
y = full_playoff_df.loc[:, threes_target].reset_index(drop=True)

X = X.astype(float)
X = X.to_numpy()

y = y.values
y = np.where(y > .5, 1, 0)
y = y.reshape(pts_y.shape[0],)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

threes_xgb_model = XGBClassifier(verbosity = 2,
        scale_pos_weight=1,
        learning_rate=0.2,
        colsample_bytree = 0.3,
        subsample = 0.6,
        objective='binary:logistic',
        n_estimators=200, #1000
        reg_alpha = 0.5,
        max_depth=7,
        gamma=1,
        n_jobs=8)

threes_xgb_model.fit(X_train, y_train)

threes_y_preds = threes_xgb_model.predict(X_test)
accuracy = accuracy_score(y_test, threes_y_preds)
accuracy

brier = brier_score_loss(y_test, threes_y_preds)
brier

full_X_test = np.concatenate((X_test, player_X_test), axis = 1)
min_max_cols = []
min_max_cols.extend(threes_features)
min_max_cols.extend(['game_id', 'player_id'])

df = pd.DataFrame(data = full_X_test, columns = min_max_cols)

df['min'] = df.groupby(['game_id', 'player_id'])['three_points_temporal_odds'].transform('min')
df['max'] = df.groupby(['game_id', 'player_id'])['three_points_temporal_odds'].transform('max')
min_max = df[['min', 'max']].to_numpy()

visualize_modified_brier_time_model(threes_y_preds, X_test, y_test, min_max, threes_features)

features = []
importance = []
for num, col in enumerate(threes_features):
    features.append(col)
    importance.append(threes_xgb_model.feature_importances_[num])
feat_import = pd.DataFrame({'features': features, 'importance': importance}).sort_values('importance', ascending=False)

initial_type = [('float_input', dt.FloatTensorType([None, X_train.shape[1]]))]
onx = ot.convert_xgboost(threes_xgb_model, initial_types=initial_type)
onx.graph.output[0].name = 'variable'
onx.graph.node[0].output[0] = 'variable'
onx.graph.output[1].name = 'output_probability'
onx.graph.node[0].output[1] = 'output_probability'
with open("threes_playoff_model.onnx", "wb") as f:
    f.write(onx.SerializeToString())

sess = rt.InferenceSession("threes_playoff_model.onnx")
input_name = sess.get_inputs()[0].name
label_name = sess.get_outputs()[0].name
pred_onnx = sess.run([label_name],
                    {input_name: X_test.astype(np.float32)})[0]


#############################
#####SINGLE-STAT TUNNING#####
#############################

X = full_playoff_df.loc[:, rebs_features].reset_index(drop=True)
y = full_playoff_df.loc[:, rebs_target].reset_index(drop=True)

X = X.astype(float)

X = X.to_numpy()
y = y.values

y = np.where(y > .5, 1, 0)
y = y.reshape(y.shape[0],)

# train-test 80-20 split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

class_balance = np.count_nonzero(pts_y == 0) / np.count_nonzero(pts_y == 1)

# setting up the classifier and grid
clf = XGBClassifier(n_jobs=8)
param_grid = {
    'silent': [False],
    'scale_pos_weight': [1, class_balance],
    'objective': ['binary:logistic'],
    'learning_rate': [0.001, 0.01, 0.05, 0.1, 0.2],
    'colsample_bytree': [0.2, 0.3, 0.4, 0.5, 0.6],
    'subsample': [0.6, 0.7, 0.8],
    'n_estimators': [50, 100, 150, 200],
    'reg_alpha': [0.1, 0.3, 0.5],
    'max_depth': [1, 3, 5, 7],
    'gamma': [0, 1, 5, 10]
}
rs_clf = RandomizedSearchCV(clf, param_grid, n_iter=20,
                         pre_dispatch='2*n_jobs', verbose=3, cv=5,
                            scoring='neg_log_loss')
print("Randomized search..")
search_time_start = time.time()
rs_clf.fit(X_train, y_train)
print("Randomized search time:", time.time() - search_time_start)
best_score = rs_clf.best_score_
best_params = rs_clf.best_params_
print("Best score: {}".format(best_score))
print("Best params: ")
for param_name in sorted(best_params.keys()):
    print('%s: %r' % (param_name, best_params[param_name]))
