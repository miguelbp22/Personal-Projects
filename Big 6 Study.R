library(ggplot2)
library(corrplot)
library(cluster)
library(fpc)

project = read.csv("stats.csv", na.strings=c("NA", " ", "", "nan"))
sapply(project, function(x) sum(is.na(x)))
project = project[-c(1:80),]

#Over the past 8 years, there have been 35 different teams in the PL
length(unique(project$team))

#Subsetting so that only teams that have played at least 6 seasons in the PL are counted
a = as.data.frame(table(project$team))
a = a[a$Freq >=6,]
prem = subset(project, team %in% a$Var1)

######Best offensive teams in the past 8 years#####
#most wins
most_win = aggregate(wins~team, prem, sum)
most_win = most_win[order(-most_win$wins),]
ggplot(data=most_win[1:10,], aes(x=reorder(team, -wins), y=wins, fill=team)) +
  geom_bar(colour="black", stat="identity") +
  theme(legend.position="none") + labs(title = "Total Wins Over 8 Seasons", x = "Team", y = "Wins") +
  theme(legend.position="none")

#most goals
most_goal = aggregate(goals~team, prem, sum)
most_goal = most_goal[order(-most_goal$goals),]
ggplot(data=most_goal[1:10,], aes(x=reorder(team, -goals), y=goals, fill=team)) +
  geom_bar(colour="black", stat="identity") + labs(title = "Total Goals Over 8 Seasons", x = "Team", y = "Goals") +
  theme(legend.position="none")

#most passes
most_passes = aggregate(total_pass~team, prem, sum)
most_passes = most_passes[order(-most_passes$total_pass),]
ggplot(data=most_passes[1:10,], aes(x=reorder(team, -total_pass), y=total_pass, fill=team)) +
  geom_bar(colour="black", stat="identity") + labs(title = "Total Passes Over 8 Seasons", x = "Team", y = "Passes") +
  theme(legend.position="none")

#best finishing/most efficient team COME BACK TO THIS!!!!
finishes = aggregate(goals/total_scoring_att~team, prem, sum)
finishes = finishes[order(-finishes$`goals/total_scoring_att`),]
ggplot(data=finishes[1:10,], aes(x=reorder(team, -(`goals/total_scoring_att`)), y=(`goals/total_scoring_att`), fill=team)) +
  geom_bar(colour="black", stat="identity") + labs(title = "Goals/Attemps Over 8 Seasons", x = "Team", y = "Goals") +
  theme(legend.position="none")

#####Best defensive team in the past 8 years###
few_losses = aggregate(losses~team, prem, sum)
few_losses = few_losses[order(few_losses$losses),]
ggplot(data=few_losses[1:10,], aes(x=reorder(team, losses), y=losses, fill=team)) +
  geom_bar(colour="black", stat="identity") + labs(title = "Total Losses Over 8 Seasons", x = "Team", y = "Losses") +
  theme(legend.position="none")

#least goals conceded
goals_against = aggregate(goals_conceded~team, prem, sum)
goals_against = goals_against[order(goals_against$goals_conceded),]
ggplot(data=goals_against[1:10,], aes(x=reorder(team, goals_conceded), y=goals_conceded, fill=team)) +
  geom_bar(colour="black", stat="identity") + labs(title = "Total Goals Conceded Over 8 Seasons", x = "Team", y = "Goals Conceded") +
  theme(legend.position="none")
#Southampton has opnly been in Prem league for 6 seasons

#Most Clean Sheets
clean_sheets = aggregate(clean_sheet~team, prem, sum)
clean_sheets = clean_sheets[order(-clean_sheets$clean_sheet),]
ggplot(data=clean_sheets[1:10,], aes(x=reorder(team, -clean_sheet), y=clean_sheet, fill=team)) +
  geom_bar(colour="black", stat="identity") + labs(title = "Most Clean Sheets Over 8 Seasons", x = "Team", y = "Clean Sheets") +
  theme(legend.position="none")


#Clearly, the big 6 truly look like they're they big 6, so let's see how their numbers compare to the avg big 6 numbers
big_6 = c('Tottenham Hotspur', 'Arsenal', 'Manchester City', 'Manchester United', 'Chelsea', 'Liverpool')
big_6 = subset(project, team %in% big_6)

#####BY TEAM LOOKS#####

#MAN CITY LOOK
man_city = prem[prem$team == "Manchester City",]
ggplot(data = man_city) + geom_line(aes(x = (season), y = wins, group = 1, colour = "Wins")) + 
  geom_line(aes(x = (season), y = losses, group = 2, colour = "Losses")) + geom_line(data=avg_wins, aes(x=season, y = wins, group = 3, colour = "Avg. Big 6 Wins"), linetype = 3) + 
  geom_line(data=avg_losses, aes(x=season, y = losses, group = 4, colour = "Avg. Big 6 Losses"), linetype = 3) + 
  geom_vline(xintercept=which(man_city$season == '2016-2017')) + labs(title = "The Pep Effect", x = "Seasons", y = "Games") + scale_shape_identity()

#SPURS LOOK
spurs = project[project$team == 'Tottenham Hotspur',]
ggplot(data = spurs) + geom_line(aes(x = (season), y = wins, group = 1, colour = "Wins")) + 
  geom_line(aes(x = (season), y = losses, group = 2, colour = "Losses")) + geom_line(data=avg_wins, aes(x=season, y = wins, group = 3, colour = "Avg. Big 6 Wins"), linetype = 3) + 
  geom_line(data=avg_losses, aes(x=season, y = losses, group = 4, colour = "Avg. Big 6 Losses"), linetype = 3)+
  geom_vline(xintercept=which(spurs$season == '2014-2015'))+labs(title = "The Poch Effect", x = "Seasons", y = "Games")+theme(legend.title=element_blank())

#UNITED LOOK
united = project[project$team == 'Liverpool',]
ggplot(data = united) + geom_line(aes(x = (season), y = wins, group = 1, colour = "Wins")) + 
  geom_line(aes(x = (season), y = losses, group = 2, colour = "Losses")) + geom_line(data=avg_wins, aes(x=season, y = wins, group = 3, colour = "Avg. Big 6 Wins"), linetype = 3) + 
  geom_line(data=avg_losses, aes(x=season, y = losses, group = 4, colour = "Avg. Big 6 Losses"), linetype = 3)+
  geom_vline(xintercept=which(united$season == '2013-2014'))+
  geom_vline(xintercept=which(united$season == '2014-2015'))+
  geom_vline(xintercept=which(united$season == '2016-2017'))+labs(title = "The Search for Fergie's Replacement", x = "Seasons", y = "Games")+theme(legend.title=element_blank())

#ARSENAL LOOK
arsenal = project[project$team == 'Arsenal',]
ggplot(data = arsenal) + geom_line(aes(x = (season), y = wins, group = 1, colour = "Wins")) + 
  geom_line(aes(x = (season), y = losses, group = 2, colour = "Losses")) + geom_line(data=avg_wins, aes(x=season, y = wins, group = 3, colour = "Avg. Big 6 Wins"), linetype = 3) + 
  geom_line(data=avg_losses, aes(x=season, y = losses, group = 4, colour = "Avg. Big 6 Losses"), linetype = 3)+labs(title = "Wegner's Overstay", x = "Seasons", y = "Games")+theme(legend.title=element_blank())


#####K-MEANS####
full_stats = aggregate(.~team, prem, sum)

num_full_stats = full_stats[,-c(1,42)]
num_full_stats = scale(num_full_stats)

km.out = kmeans(num_full_stats, 2, nstart = 25)
clust_prem = data.frame(full_stats$team, km.out$cluster)
clust_prem



