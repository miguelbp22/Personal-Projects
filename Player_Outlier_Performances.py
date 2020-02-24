import psycopg2
import pandas as pd
import pandas.io.sql as sqlio
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import os, os.path
from datetime import datetime
import onnxruntime as rt
import xgboost as xgb
from sklearn.metrics import auc, accuracy_score, confusion_matrix, classification_report
from sklearn.metrics import brier_score_loss
from numpy import asarray
from numpy import savez_compressed
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
import time 
import sys
from matplotlib import pyplot
import pickle 


####################################################
####Determining distribution for each statistic#####
####################################################

#accesing each game file in the training files folder
path, dirs, files = next(os.walk("Downloads/training_files/"))

#creating 1-d numpy arrays for each of the observed statisitcs to determine their respective distributions 
all_game_points = np.zeros((10000000))
all_game_rebounds = np.zeros((10000000))
all_game_assists = np.zeros((10000000))
all_game_threes = np.zeros((10000000))
all_game_stocks = np.zeros((10000000))

#kickstarting an index for each statistic to keep track of the number of rows we fill
pts_start_idx = 0
rebs_start_idx = 0
asts_start_idx = 0
threes_start_idx = 0
stocks_start_idx = 0

#loop through the files, except for the csv file with the column names in there
for num, file in enumerate(files):
    if not file.startswith('20'):
        pass
    else:
        #making a dataframe for each game file to manipulate the data more easily
        game = np.load(path + file, allow_pickle = True)['arr_0']
        df = pd.DataFrame(game, columns = col_names).sort_values('time_elapsed').groupby('player_id').tail(1)
        
        #making new variables that are more in line with the statistics we measure
        points = df['in_play_points']
        rebounds = df['in_play_defensive_rebounds'] + df['in_play_offensive_rebounds']
        assists = df['in_play_assists']
        threes = df['in_play_three_point_makes']
        stocks = df['in_play_steals'] + df['in_play_blocks']
        
        #making an end index for each stat that will tell the program where to end filling the rows
        pts_end_idx = pts_start_idx + len(points)
        rebs_end_idx = rebs_start_idx + len(rebounds)
        asts_end_idx = asts_start_idx + len(assists)
        threes_end_idx = threes_start_idx + len(threes)
        stocks_end_idx = stocks_start_idx + len(stocks)
        
        #filling the previosuly-created array with the points data for each statistic
        all_game_points[pts_start_idx:pts_end_idx] = points
        all_game_rebounds[rebs_start_idx:rebs_end_idx] = rebounds
        all_game_assists[asts_start_idx:asts_end_idx] = assists
        all_game_threes[threes_start_idx:threes_end_idx] = threes
        all_game_stocks[stocks_start_idx:stocks_end_idx] = stocks
        
        #printing the file number to keep track of the progress
        print(num)
        
        #renaming the start index to be the end index from the previous iteration - start filling where we left off
        pts_start_idx = pts_end_idx
        rebs_start_idx = rebs_end_idx
        asts_start_idx = asts_end_idx
        threes_start_idx = threes_end_idx
        stocks_start_idx = stocks_end_idx

#once the loop is done, we get rid of all 0 rows
all_game_points = all_game_points[0:pts_end_idx]
all_game_rebounds = all_game_rebounds[0:rebs_end_idx]
all_game_assists = all_game_assists[0:asts_end_idx]
all_game_threes = all_game_threes[0:threes_end_idx]
all_game_stocks = all_game_stocks[0:stocks_end_idx]

#creating the cutoffs @ 2.5 std devs from the mean for each statistic
points_mean, points_std = np.mean(all_game_points), np.std(all_game_points)
points_cutoff = points_std * 2.5
points_upper_bound = points_mean + points_cutoff
rebounds_mean, rebounds_std = np.mean(all_game_rebounds), np.std(all_game_rebounds)
rebounds_cutoff = rebounds_std * 2.5
rebounds_upper_bound = rebounds_mean + rebounds_cutoff
assists_mean, assists_std = np.mean(all_game_assists), np.std(all_game_assists)
assists_cutoff = assists_std * 2.5
assists_upper_bound = assists_mean + assists_cutoff
threes_mean, threes_std = np.mean(all_game_threes), np.std(all_game_threes)
threes_cutoff = threes_std * 2.5
threes_upper_bound = threes_mean + threes_cutoff
stocks_mean, stocks_std = np.mean(all_game_stocks), np.std(all_game_stocks)
stocks_cutoff = stocks_std * 2.5
stocks_upper_bound = stocks_mean + stocks_cutoff

#comapring the cutoffs using 2.5 std devs from mean vs. iqr upper 
std_cutoffs = [points_cutoff, rebounds_cutoff, assists_cutoff, threes_cutoff, stocks_cutoff]
iqr_cutoffs = [0, 0, 0, 0, 0]

colors = ["#9b59b6", "#3498db", "#e74c3c", "#34495e", "#2ecc71"]

#making boxplot of each statistic to viauzlize distribution
for num, stat in enumerate(all_stats):
    
    stat = sorted(stat)
    q25, q75 = np.percentile(stat, 25), np.percentile(stat, 75)
    iqr = q75-q25
    cutoff = iqr*1.5
    upper = q75 + cutoff
    
    iqr_cutoffs[num] = upper
    print('Cutoff for %s: %d' % (stats[num], upper))
    
    plt.figure()
    sns.boxplot(x = stat, color = colors[num]).set_title("Distribution for %s" % stats[num])

#creating a histogram of the distributions for each statistic with both cutoffs to determine which is more appropriate
for num, stat in enumerate(all_stats):
    pyplot.figure(num)
    pyplot.hist(stat)
    pyplot.axvline(x = iqr_cutoffs[num], color = 'r', label = 'IQR cutoff')
    pyplot.axvline(x = std_cutoffs[num], color = 'black', label = 'STD cutoff')
    pyplot.legend()
    pyplot.title("Distribution of %s" % (stats[num]))

#Based on the images, std dev cutoff is more appropriate

######################################
#####Making the Outlier Data Sets#####
######################################

stats = ['points', 'rebounds', 'assists', 'threes', 'stocks']
path, dirs, files = next(os.walk("Downloads/training_files/"))

#these are from the previously run data and distributions(script above)
iqr_cutoffs = [31.5, 12.0, 7.5, 5.0, 5.0]

#some indexes had to be redetermined due to the stocks and rebounds /game having to be aggregated from blocks and steals and def and of rebounds, respectively
stat_per_game_idx = [12, 15, 13, 18, 14]


#differentiating the indexing elements from the training set that were integers vs string - splitting them up was necessary due to memory issues
non_string_idx = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,26,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56]
string_idx = [25, 27]


for i, stat in enumerate(stats):
    #to account for memory issues, one big array was made per stat as we looped through them instead of making several at once
    full_stat_outliers = np.zeros((100000000, 55))
    stat_ids = np.zeros((100000000, 3)).astype(object)
    
    start_idx = 0
    for num, file in enumerate(files):
        if not file.startswith('20'):
            pass
        else:
            #load each game and grab its game_id from the file name
            game = np.load(path + file, allow_pickle = True)['arr_0']
            game_id = str(file[11:-4])
            
            #for each game, had to make a new array that aggregated def + off rebounds and steals + stocks per game to be able to compare for cutoff
            temp_game = np.concatenate((game[:,0:14], (game[:, 14:15] + game[:, 15:16]), (game[:,16:17] + game[:,17:18]), game[:,18:]), axis = 1)
            #subsetting entries from the main game based on entries (same indexes) from the 'new game' depending on whether the stat/game in question cleared the threshold
            full_outliers = game[temp_game[:, stat_per_game_idx[i]] > iqr_cutoffs[i], :]
            
            #populating the integer and object arrays accordingly
            game_outliers = full_outliers[:, non_string_idx]
            outlier_info = full_outliers[:, string_idx]
            
            #make a repeat array of the game id the length of the size of the sample taken from each game - will be useful later for brier scoring
            game_id = np.repeat(game_id, len(game_outliers)).reshape(-1,1)
            
            #give the end_idx a value that will tell function from where to where the main array should be populated 
            end_idx = start_idx+len(game_outliers)
            
            #populating the respective arrays
            full_stat_outliers[start_idx:end_idx,:] = game_outliers
            stat_ids[start_idx:end_idx,:] = np.concatenate((game_id, outlier_info), axis = 1)
            
            #reassigning the start_idx for the next game to be the last row filled from the last game
            start_idx = end_idx
            
            #printing the file number to gauge process and end_idx, i.e. size of the subsample gathered for each game, as a sanity check
            print(num)
            print(end_idx)
    print('Last row for %s is: %d' % (stats[i], end_idx))
    
    #cutting off the zero rows from the original, large array
    full_stat_outliers = full_stat_outliers[0:end_idx, :]
    stat_ids = stat_ids[0:end_idx, :]
    
    #subsetting each dataset further by keeping only the records that are greater than the median
    stat_ids = stat_ids[full_stat_outliers[:,24] >= np.median(full_stat_outliers[:,24:25]), :]
    full_stat_outliers = full_stat_outliers[full_stat_outliers[:,24] >= np.median(full_stat_outliers[:,24:25]), :]
    
    #saving the arrays 
    savez_compressed('/Users/miguelbp/Documents/Outliers/%s_outliers.npz' % stats[i], full_stat_outliers)
    savez_compressed('/Users/miguelbp/Documents/Outliers/%s_outliers_info.npz' % stats[i], stat_ids)

###################################################
#####Sample of hyperparameter tunning for XGBM#####
###################################################

#hyperparamter tunning was done for all stats, but due to processing time bc of the large data set, it was done in parts
#this is the process for stocks and rebounds

path, dirs, files = next(os.walk("Documents/Outliers_StdDev/"))

data_files = ['players_rebs_out.npz', 'players_stocks_out.npz']
stats = ['Rebounds', 'Stocks']
#these are the indexes for the respective stats' target var (obs_idx) and the pred features (feat_idx)
obs_idx = [[54], [55]]
feat_idx = [[10, 7, 2, 43, 11, 51, 12, 36, 35, 34, 31, 30, 28, 29, 19, 18, 21, 20, 23, 
                22, 16, 45, 44, 47, 46, 49, 48, 39, 13, 37, 17, 38, 15, 40, 14, 41],
          [10, 8, 3, 43, 11, 51, 12, 36, 35, 34, 31, 30, 28, 29, 19, 18, 21, 20, 23, 
                22, 16, 45, 44, 47, 46, 49, 48, 39, 13, 37, 17, 38, 15, 40, 14, 41]]


for num, file in enumerate(data_files):
    
    data = np.load(path + file, allow_pickle=True)['arr_0']
    
    #the stat line for both statistics is over/under .5
    data_obs = data[:, obs_idx[num]] > 0.5
    
    #separating the independent and target variables
    data_out_X = data[:, feat_idx[num]]
    data_out_y = data_obs.astype(int)
    
    splitting into train/test - data is sorted by date, so using 80% of first games to train, 20% of last games to test
    train_pct_idx = int(0.8 * len(data_out_X))
    X_train, X_test = data_out_X[:train_pct_idx], data_out_X[train_pct_idx:]
    y_train, y_test = data_out_y[:train_pct_idx], data_out_y[train_pct_idx:]
    
    #initiating the classifier
    clf = xgb.XGBClassifier()
    
    #initiating the prarameter grid
    param_grid = {
        'silent': [False],
        'scale_pos_weight': [1],
        'objective': ['binary:logistic'],
        'learning_rate': [0.001, 0.01, 0.05, 0.1, 0.2],
        'colsample_bytree': [0.2, 0.3, 0.4, 0.5, 0.6],
        'subsample': [0.6, 0.7, 0.8],
        'n_estimators': [100, 250, 500],
        'reg_alpha': [0.1, 0.3, 0.5],
        'max_depth': [3, 6, 10, 15],
        'gamma': [0, 1, 5, 10]
        }
    
    #launching the randomized search 
    rs_clf = RandomizedSearchCV(clf, param_grid, n_iter=20,
                            n_jobs=8, verbose=2, cv=3,
                            scoring='neg_log_loss', refit=False, random_state=42)
    
    print("Randomized search for %s" % stats[num])
    search_time_start = time.time()
    rs_clf.fit(X_train, y_train)
    print("Randomized search time:", time.time() - search_time_start)

    best_score = rs_clf.best_score_
    best_params = rs_clf.best_params_
    print("Best score: {}".format(best_score))
    print("Best params: ")
    for param_name in sorted(best_params.keys()):
        print('%s: %r' % (param_name, best_params[param_name]))

###########################################################
#####post hyper-parameter tunning model implementation#####
###########################################################

outliers = [points_outliers, rebounds_outliers, assists_outliers, stocks_outliers, threes_outliers]

unique_players = tuple(np.unique(replay_game[:, 25]))

connection = psycopg2.connect(user = "miguel",
                                  password = "p666c5b625a6ac9f8c5b3dba9c3a335d7561b7983e25ba03c409cdd93d6a80156",
                                  host = "ec2-52-73-220-73.compute-1.amazonaws.com",
                                  port = "5432",
                                  database = "dfqr82vi93e42j")

sql = "select external_id, last_name from players where external_id in %(u_players)s"
param = {'u_players':unique_players}
last_names = sqlio.read_sql_query(sql, connection, params = param)
connection.close()

stats = ['Points', 'Rebounds', 'Assists', 'Stocks','Threes']

pts_features_idx = [10, 5, 0, 43, 11, 51, 12, 36, 35, 34, 31, 30, 28, 29, 19, 18, 21, 20, 23, 
                22, 16, 45, 44, 47, 46, 49, 48, 39, 13, 37, 17, 38, 15, 40, 14, 41]
rebs_features_idx = [10, 7, 2, 43, 11, 51, 12, 36, 35, 34, 31, 30, 28, 29, 19, 18, 21, 20, 23, 
                22, 16, 45, 44, 47, 46, 49, 48, 39, 13, 37, 17, 38, 15, 40, 14, 41]
asts_features_idx = [10, 6, 1, 43, 11, 51, 12, 36, 35, 34, 31, 30, 28, 29, 19, 18, 21, 20, 23, 
                22, 16, 45, 44, 47, 46, 49, 48, 39, 13, 37, 17, 38, 15, 40, 14, 41]
stocks_features_idx = [10, 8, 3, 43, 11, 51, 12, 36, 35, 34, 31, 30, 28, 29, 19, 18, 21, 20, 23, 
                22, 16, 45, 44, 47, 46, 49, 48, 39, 13, 37, 17, 38, 15, 40, 14, 41]
threes_features_idx = [10, 9, 4, 43, 11, 51, 12, 36, 35, 34, 31, 30, 28, 29, 19, 18, 21, 20, 23, 
                 22, 16, 45, 44, 47, 46, 49, 48, 39, 13, 37, 17, 38, 15, 40, 14, 41]

indexes = [pts_features_idx, rebs_features_idx, asts_features_idx, stocks_features_idx, threes_features_idx]


pts_obs_idx = 52
rebs_obs_idx = 54
asts_obs_idx = 53
stocks_obs_idx = 55
threes_obs_idx = 56

stat_obs_indexes = [pts_obs_idx, rebs_obs_idx, asts_obs_idx, stocks_obs_idx, threes_obs_idx]


models = [pts_xgb_model, rebs_xgb_model, asts_xgb_model, stocks_xgb_model, threes_xgb_model]

for i, player in enumerate(unique_players):
    for num, stat in enumerate(outliers):
        if player in stat:
            
            last_name = last_names.loc[last_names['external_id'] == player, 'last_name']
            player_actions = replay_game[replay_game[:, 25] == player][:,indexes[num]]
            
            print(player)
            player_preds = models[num].predict_proba(player_actions)
            
            plt.figure()
            plt.plot(player_actions[:,5], player_preds[:,1], 'o', color='blue',
             markersize=2, linewidth=1,
             markerfacecolor=colors[num],
             markeredgecolor=colors[num],
             markeredgewidth=1)
            plt.plot(player_actions[:,5], player_actions[:,1:2], 'o', color='red',
             markersize=2, linewidth=1,
             markerfacecolor='red',
             markeredgecolor='red',
             markeredgewidth=1, label = 'Historical Odds')
            plt.plot(player_actions[:,5], player_preds, 'o', color='green',
             markersize=2, linewidth=1,
             markerfacecolor='green',
             markeredgecolor='green',
             markeredgewidth=1, label = 'Old Model Odds')
            plt.title(stats[num]+' '+'Preds for' + ' '+ last_name)            
            plt.show()
