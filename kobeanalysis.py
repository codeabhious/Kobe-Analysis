import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.patches import Circle, Rectangle, Arc
import seaborn as sns
sns.set_style("white")
sns.set_color_codes()
pd.options.mode.chained_assignment = None
from sklearn.cluster import KMeans
def handle_data(fp):
    #fp is the filepath on the computer
    #this function returns the data set which is needed to analyze shotlog csv
    #we make the data frame more readable by converting closest defender name to "first last" instead of "last, first"
    #we take the shot clock to be the game clock whenever the game clock is under 24 sec
    nba = pd.read_csv(fp)  #open the file 
    nba['CLOSEST_DEFENDER'] = nba['CLOSEST_DEFENDER'].apply(lambda x: x.lower())    #lower case everything
    nba['CLOSEST_DEFENDER'] = nba['CLOSEST_DEFENDER'].apply(lambda x: x.split(', ')) #make like the first name
    nba['CLOSEST_DEFENDER'] = nba['CLOSEST_DEFENDER'].apply(lambda x: x[1] + ' ' + x[0] if len(x) == 2 else None) #make like first name
    for item in range(len(nba['SHOT_CLOCK'])): #fill in N/A values
        if np.isnan(nba['SHOT_CLOCK'][item]):
            nba['SHOT_CLOCK'][item] = float(nba['GAME_CLOCK'][item][3:])            
    drops = ['FINAL_MARGIN','SHOT_RESULT','SHOT_NUMBER','player_id','CLOSEST_DEFENDER_PLAYER_ID'] #columns to get rid of 
    nba.drop(drops,axis=1,inplace=True) #get rid of these columns
    return nba

fp = "C:\\Users\\Abhijit\\Downloads\\shot_logs.csv\\shotlogs.csv"    # testing out handle_data function
y = handle_data(fp)

def shooterProfile(nba):
    #this function creates shooting profiles of each player who has played in 30 or more games
    playerlist = list(sorted(set(nba['player_name'])))
    
    newdata = pd.DataFrame(columns = ['Player','Mean Floor PPG','PPG Std','Mean Shot Dist','Shot Dist Std','Mean Shot Pcg. (3)','Shot Pcg Std (3)','Mean Shot Pcg. (2)','Shot Pcg Std (2)','2 Pt. Attempts/Game','2 Pt Std','3 Pt. Attempts/Game','3 PT. Std'])
    for player in playerlist:
        playerdata = nba.loc[nba['player_name']==player] #select appr. player
        playerdistavg = np.mean(playerdata['SHOT_DIST']) #calc shot dist
        playerdistvar = np.std(playerdata['SHOT_DIST']) #calc shot dist var
        shooting2 = []
        shooting3 = []
        threePA = []
        twoPA = []
        points = []
        gameslist = list(set(playerdata['GAME_ID']))
        for game in gameslist:
            playergame = playerdata.loc[playerdata['GAME_ID']==game] #get games for player
            shotstotal3 = playergame.loc[playergame['PTS_TYPE']==3].shape[0]
            shotsmade3 = playergame.loc[playergame['PTS']==3].shape[0]
            shotstotal2 = playergame.loc[playergame['PTS_TYPE']==2].shape[0]
            shotsmade2 = playergame.loc[playergame['PTS']==2].shape[0]
            if shotstotal2 > 0:
                shooting2.append(float(shotsmade2)/float(shotstotal2))
                twoPA.append(shotstotal2)#num 2 shot
            if shotstotal3 > 0:
                shooting3.append(float(shotsmade3)/float(shotstotal3)) 
                threePA.append(shotstotal3) #num 3 shot
            points.append(sum(playergame['PTS']))
        
        spcgm2 = np.mean(shooting2)
        spcgv2 = np.std(shooting2)
        spcgm3 = np.mean(shooting3)
        spcgv3 = np.std(shooting3)
        tpam = np.mean(threePA)
        tpav = np.std(threePA)
        twpam = np.mean(twoPA)
        twpav = np.std(twoPA)
        pointsm = np.mean(points)
        pointsv = np.std(points)
        if (len(gameslist)>=35) and (len(shooting2) > 0) and (len(shooting3) > 0):
            df = pd.DataFrame({'Player' : [player],'Mean Floor PPG': [pointsm],'PPG Std': [pointsv], 'Mean Shot Dist':[playerdistavg],'Shot Dist Std':[playerdistvar],'Mean Shot Pcg. (3)': [spcgm3],'Shot Pcg Std (3)':[spcgv3],'Mean Shot Pcg. (2)': [spcgm2],'Shot Pcg Std (2)':[spcgv2],'2 Pt. Attempts/Game':[twpam],'2 Pt Std':[twpav],'3 Pt. Attempts/Game':[tpam],'3 PT. Std':[tpav]})
            newdata =  newdata.append(df,ignore_index = True)
    return newdata
j = shooterProfile(y)
        
def handle_kobe_data(kobefilepath):
    kobedata = pd.read_csv(kobefilepath)
    kobedata['season'] = kobedata['season'].apply(lambda x: x[0:4]) #find specific start year 
    kobedata['season'] = kobedata['season'].apply(lambda x: int(x)) #make date into integer
    kobedata = kobedata.dropna(axis=0,how='any') #get rid of NaN
    kobedata['shot_type'] = kobedata['shot_type'].apply(lambda x: int(x[0])) #shot type shit
    kobedata['pts'] = kobedata['shot_type']*kobedata['shot_made_flag'] #add points on that shot
    #convert kobe csv to pandas data frame
    return kobedata

def kobeShooterProfile(kobedata):
    newdata = pd.DataFrame(columns = ['Player','Mean Floor PPG','PPG Std', 'Mean Shot Dist','Shot Dist Std','Mean Shot Pcg. (3)','Shot Pcg Std (3)','Mean Shot Pcg. (2)','Shot Pcg Std (2)','2 Pt. Attempts/Game','2 Pt Std','3 Pt. Attempts/Game','3 PT. Std'])
    yearlist = list(sorted(set(kobedata['season'])))
    for year in yearlist:
        kobeyear = kobedata.loc[kobedata['season']==year]
        kobedistavg = np.mean(kobeyear['shot_distance'])
        kobediststd = np.std(kobeyear['shot_distance'])
        shooting2 = []
        shooting3 = []
        threePA = []
        twoPA = []
        points = []
        gameslist = list(set(kobeyear['game_id']))        
        for game in gameslist: #specific game in a specific year 
            kobeyeargame = kobedata.loc[kobedata['game_id']==game]
            shotstotal3 = kobeyeargame.loc[kobeyeargame['shot_type']==3].shape[0]
            shotstotal2 = kobeyeargame.loc[kobeyeargame['shot_type']==2].shape[0]
            shotsmade3 = kobeyeargame.loc[kobeyeargame['pts']==3].shape[0]
            shotsmade2 = kobeyeargame.loc[kobeyeargame['pts']==2].shape[0]
            if shotstotal2 > 0:
                shooting2.append(float(shotsmade2)/float(shotstotal2))
                twoPA.append(shotstotal2)
            if shotstotal3 > 0:
                shooting3.append(float(shotsmade3)/float(shotstotal3))
                threePA.append(shotstotal3)
            points.append(sum(kobeyeargame['pts']))
        spcgm2 = np.mean(shooting2)
        spcgv2 = np.std(shooting2)
        spcgm3 = np.mean(shooting3)
        spcgv3 = np.std(shooting3)
        tpam = np.mean(threePA)
        tpav = np.std(threePA)
        twpam = np.mean(twoPA)
        twpav = np.std(twoPA)
        pointsm = np.mean(points)
        pointsv = np.std(points)
        df = pd.DataFrame({'Player': ['Kobe ' + str(year)],'Mean Floor PPG':[pointsm],'PPG Std':[pointsv],'Mean Shot Pcg. (3)': [spcgm3],'Shot Pcg Std (3)':[spcgv3],'Mean Shot Pcg. (2)':[spcgm2],'Shot Pcg Std (2)':[spcgv2], 'Mean Shot Dist':[kobedistavg],'Shot Dist Std':[kobediststd],'2 Pt. Attempts/Game':[twpam],'2 Pt Std':[twpav],'3 Pt. Attempts/Game':[tpam],'3 PT. Std':[tpav]})
        newdata = newdata.append(df,ignore_index = True)
    return newdata

kobefp = "C:\\Users\\Abhijit\\Documents\\kobedata.csv"
x = handle_kobe_data(kobefp)
l = kobeShooterProfile(x)
      
    
def normalizeKobeLeagueData(kobedata,leaguedata):
    for header in ['Mean Floor PPG','PPG Std', 'Mean Shot Dist','Shot Dist Std','Mean Shot Pcg. (3)','Shot Pcg Std (3)','Mean Shot Pcg. (2)','Shot Pcg Std (2)','2 Pt. Attempts/Game','2 Pt Std','3 Pt. Attempts/Game','3 PT. Std']:
        kobedata[header] = (kobedata[header]-float(min(min(kobedata[header]),min(leaguedata[header]))))/(float(max(max(kobedata[header]),max(leaguedata[header])))-float(min(min(kobedata[header]),min(leaguedata[header]))))
        leaguedata[header] = (leaguedata[header]-float(min(min(kobedata[header]),min(leaguedata[header]))))/(float(max(max(kobedata[header]),max(leaguedata[header])))-float(min(min(kobedata[header]),min(leaguedata[header]))))
    return kobedata,leaguedata

kobe,league = normalizeKobeLeagueData(l,j)

def computeSimMat(kobedata,leaguedata):
    league = leaguedata[~leaguedata['Player'].isin(['kobe bryant'])] #remove for any comparisons to kobe 
    totaldata = pd.concat([kobedata,league])
    simMat = np.zeros((totaldata.shape[0],totaldata.shape[0]))
    newdata = totaldata.drop('Player',1)
    mat1 = newdata.as_matrix()
    mat2 = np.transpose(mat1)
    for i in range(0,totaldata.shape[0]):
        for j in range(0,totaldata.shape[0]):
            if i!=j:
                simMat[i,j] = np.linalg.norm(mat1[i]- np.transpose(mat2[:,j]))
            else:
                simMat[i,j] = 0
            
    return simMat


def visualizeCluster(simmat,kobedata,leaguedata,kobeyear,playersofinterest):
    from sklearn import manifold
    league = leaguedata[~leaguedata['Player'].isin(['kobe bryant'])] #remove for any comparisons to kobe
    mds = manifold.MDS(n_components = 2,dissimilarity='precomputed')
    totaldata = pd.concat([kobedata,league]) #concatenate the data
    newdata = totaldata.drop('Player',1) #drop the names since they are useless
    transformdata = pd.DataFrame(mds.fit_transform(simmat)) #PCA transform
    transformdata['Player'] = totaldata['Player'].values #adding the class values
    transformdata['Player'] = transformdata['Player'].apply(lambda x: 'Other' if (kobeyear != x) and (x not in playersofinterest) else x)
    colors = ['blue','purple','yellow','orange','green','saddlebrown','black','lime','lightblue','cyan']
    i = 0
    for player in list(sorted(set(transformdata['Player']))):
        
        location = transformdata.loc[transformdata['Player']==player]
        if player == 'Other':
            plt.scatter(location[0],location[1], label = player, c = 'grey')
            #plt.scatter(location[0], label = player, c = 'grey')
        elif player in playersofinterest:
            plt.scatter(location[0],location[1], label = player, c = colors[i])
            #plt.scatter(location[0], label = player, c = colors[i])
            i += 1    
        else:
            plt.scatter(location[0], location[1], label = player, c = 'red')
            #plt.scatter(location[0], label = player, c = 'red')
        
    plt.legend()
    plt.show()
    return None


def estimateClusters(leaguedata):
    league = leaguedata[~leaguedata['Player'].isin(['kobe bryant'])]
    league = league.drop('Player',axis = 1)
    X = league.as_matrix()
    from sklearn import mixture
    import itertools
    lowest_bic = np.infty
    bic = []
    n_components_range = range(1, 7)
    cv_types = ['spherical', 'tied', 'diag', 'full']
    for cv_type in cv_types:
        for n_components in n_components_range:
            # Fit a Gaussian mixture with EM
            gmm = mixture.GaussianMixture(n_components=n_components,
                                      covariance_type=cv_type)
            gmm.fit(X)
            bic.append(gmm.bic(X))
            if bic[-1] < lowest_bic:
                lowest_bic = bic[-1]
                best_gmm = gmm
    bic = np.array(bic)
    color_iter = itertools.cycle(['navy', 'turquoise', 'cornflowerblue',
                              'darkorange'])
    clf = best_gmm
    bars = []
    # Plot the BIC scores
    spl = plt.subplot(2, 1, 1)
    for i, (cv_type, color) in enumerate(zip(cv_types, color_iter)):
        xpos = np.array(n_components_range) + .2 * (i - 2)
        bars.append(plt.bar(xpos, bic[i * len(n_components_range):
                                      (i + 1) * len(n_components_range)],
                            width=.2, color=color))
    plt.xticks(n_components_range)
    plt.ylim([bic.min() * 1.01 - .01 * bic.max(), bic.max()])
    plt.title('BIC score per model')
    xpos = np.mod(bic.argmin(), len(n_components_range)) + .65 +\
        .2 * np.floor(bic.argmin() / len(n_components_range))
    plt.text(xpos, bic.min() * 0.97 + .03 * bic.max(), '*', fontsize=14)
    spl.set_xlabel('Number of components')
    spl.legend([b[0] for b in bars], cv_types)
    plt.show()
    return None
#estimateClusters(league)

def KMeansCluster(leaguedata,k):
    from sklearn import cluster
    league = leaguedata[~leaguedata['Player'].isin(['kobe bryant'])]
    total = league.drop('Player',axis=1)
    X = total.as_matrix()
    kmeans = cluster.KMeans(n_clusters = k)
    kmeans.fit(X)
    predictions = kmeans.predict(X)
    clusters = {}
    i = 0
    for item in predictions:
        clusters[league['Player'].as_matrix()[i]] = item
        i += 1
    return clusters,kmeans
    
#clusters,kmeans = KMeansCluster(league,5)
#print(clusters)
def KobeCluster(kobedata,kmeans,kobeyear):
    kobepredict = []
    for year in kobeyear:
        kobe = kobedata.loc[kobedata['Player']==year]
        kobe = kobe.drop('Player',axis=1)
        kobemat = kobe.as_matrix()
        kobepredict.append(kmeans.predict(kobemat))
    return kobepredict

#print(KobeCluster(kobe,kmeans,['Kobe 2003']))
def KNNKobe(leaguedata,kobedata,kobeyear,k):
    from sklearn import neighbors
    league = leaguedata[~leaguedata['Player'].isin(['kobe bryant'])]
    total = league.drop('Player',axis=1)
    samples = total.as_matrix()
    kobedata = kobedata.loc[kobedata['Player']==kobeyear]
    kobe = kobedata.drop('Player',axis=1)
    kobe = kobe.as_matrix()
    neigh = neighbors.NearestNeighbors(n_neighbors=k)
    neigh.fit(samples)
    indexes = neigh.kneighbors(kobe,return_distance=False)
    similar = []
    for index in indexes[0]:
        similar.append(league['Player'].as_matrix()[index])
        
    return similar

#playersofinterest = KNNKobe(league,kobe,'Kobe 2014',5)
simMat = computeSimMat(kobe,league)
#playersofinterest = ['draymond green','harrison barnes','andrew bogut','klay thompson','stephen curry']
#print(playersofinterest)
#visualizeCluster(simMat,kobe,league,'Kobe 2014',playersofinterest)

#print(playersofinterest)
#visualizeCluster(simMat,kobe,league,'Kobe 2003',playersofinterest)
def EVPKobe(kobedata):
    newdata = pd.DataFrame(columns = ['EVS','Season'])
    yearlist = list(set(kobedata['season']))
    for year in yearlist:
        kobeyear = kobedata.loc[kobedata['season']==year]
        shooting3p = []
        shooting2p = []
        threePG = []
        twoPG = []
        gameslist = list(set(kobeyear['game_id']))        
        for game in gameslist: #specific game in a specific year 
            kobeyeargame = kobedata.loc[kobedata['game_id']==game]
            if kobeyeargame.loc[kobeyeargame['shot_type']==2].shape[0] > 0:
                shooting2p.append(float(kobeyeargame.loc[kobeyeargame['pts']==2].shape[0])/float(kobeyeargame.loc[kobeyeargame['shot_type']==2].shape[0]))
            if kobeyeargame.loc[kobeyeargame['shot_type']==3].shape[0] > 0:
                shooting3p.append(float(kobeyeargame.loc[kobeyeargame['pts']==3].shape[0])/float(kobeyeargame.loc[kobeyeargame['shot_type']==3].shape[0]))
            if kobeyeargame['shot_type'].shape[0] > 0:
                twoPG.append(float(kobeyeargame.loc[kobeyeargame['shot_type']==2].shape[0])/float(kobeyeargame['shot_type'].shape[0]))
                threePG.append(1-twoPG[-1])
        evs = np.mean(threePG)*np.mean(shooting3p)*3 + np.mean(twoPG)*np.mean(shooting2p)*2
        df = pd.DataFrame({'Season':[year],'EVS':[evs]})
        newdata = newdata.append(df,ignore_index = True)
    sns.pointplot(x='Season',y='EVS',hue=None,data=newdata,color = 'blue')
    sns.plt.title('Kobe Bryant: Expected Value Per Shot')
    sns.plt.show()
    return newdata
#EVPKobe(x)
def EVPPlayers(leaguedata,playerlist):
    newdata = pd.DataFrame(columns = ['EVS','Player'])
    for player in playerlist:
        league = leaguedata.loc[leaguedata['player_name']==player]
        shooting3p = []
        shooting2p = []
        threePG = []
        twoPG = []
        gameslist = list(set(league['GAME_ID']))
        for game in gameslist:
            leaguegame = league.loc[league['GAME_ID']==game]
            if leaguegame.loc[leaguegame['PTS_TYPE']==2].shape[0] > 0:
                shooting2p.append(float(leaguegame.loc[leaguegame['PTS']==2].shape[0])/float(leaguegame.loc[leaguegame['PTS_TYPE']==2].shape[0]))
            if leaguegame.loc[leaguegame['PTS_TYPE']==3].shape[0] > 0:
                shooting3p.append(float(leaguegame.loc[leaguegame['PTS']==3].shape[0])/float(leaguegame.loc[leaguegame['PTS_TYPE']==3].shape[0]))
            if leaguegame['PTS_TYPE'].shape[0] > 0:
                twoPG.append(float(leaguegame.loc[leaguegame['PTS_TYPE']==2].shape[0])/float(leaguegame['PTS_TYPE'].shape[0]))
                threePG.append(1-twoPG[-1])
        evs = np.mean(threePG)*np.mean(shooting3p)*3 + np.mean(twoPG)*np.mean(shooting2p)*2
        df = pd.DataFrame({'Player':[player],'EVS':[evs]})
        newdata = newdata.append(df,ignore_index = True)
    sns.barplot(x='Player',y = 'EVS',hue=None,data=newdata)
    sns.plt.title('Player(s) : Expected Value per Shot')
    sns.plt.show()
    return newdata
#EVPPlayers(y,['russell westbrook','lebron james','stephen curry','tim duncan','kyrie irving','pau gasol','lamarcus aldridge','james harden'])
            

def shotDistKobe(kobedata):
    newdata = pd.DataFrame(columns = ['Season','Zone %','Shot Range'])
    yearlist = list(set(kobedata['season']))
    for year in yearlist:
        kobeyear = kobedata.loc[kobedata['season']==year]
        distlist = list(set(kobeyear['shot_zone_range']))
        yearlength = float(kobeyear.shape[0])
        for zone in distlist:
            if zone != 'Back Court Shot':
                kobeyearzone = kobeyear.loc[kobeyear['shot_zone_range']==zone]
                zonecount = float(kobeyearzone.shape[0])/yearlength
                df = pd.DataFrame({'Season':[year],'Zone %':[zonecount],'Shot Range':[zone]})
                newdata = newdata.append(df,ignore_index = True)
    sns.pointplot(x='Season',y='Zone %',hue = 'Shot Range',data=newdata)
    sns.plt.title('Kobe Bryant: Shot Locations')
    sns.plt.show()
    return newdata
#shotDistKobe(x)
def shotZonePCGKobe(kobedata):
    newdata = pd.DataFrame(columns = ['Season','Shot %','Shot Range'])
    yearlist = list(set(kobedata['season']))
    for year in yearlist:
        kobeyear = kobedata.loc[kobedata['season']==year]
        distlist = list(set(kobeyear['shot_zone_range']))
        for zone in distlist:
            if zone != 'Back Court Shot':
                kobeyearzone = kobeyear.loc[kobeyear['shot_zone_range']==zone]
                zonecount = float(kobeyearzone.shape[0])
                zonemake = float(kobeyearzone.loc[kobeyearzone['shot_made_flag']==1].shape[0])
                df = pd.DataFrame({'Season':[year],'Shot %':[zonemake/zonecount],'Shot Range':[zone]})
                newdata = newdata.append(df,ignore_index = True)
    sns.pointplot(x='Season',y='Shot %',hue = 'Shot Range',data=newdata)
    sns.plt.title('Kobe Bryant: Shot Locations')
    sns.plt.show()
    return newdata
#shotZonePCGKobe(x)
def posOnFieldKobe(kobedata):
    newdata = pd.DataFrame(columns = ['Season','%','Shot Location'])
    yearlist = list(set(kobedata['season']))
    kobedata['shot_zone_area'] = kobedata['shot_zone_area'].apply(lambda x: 'Right' if 'R' in x else x)
    kobedata['shot_zone_area'] = kobedata['shot_zone_area'].apply(lambda x: 'Left' if 'L' in x else x)    
    for year in yearlist:
        kobeyear = kobedata.loc[kobedata['season']==year]
        loclist = list(set(kobeyear['shot_zone_area']))
        yearlength = float(kobeyear.shape[0])
        for location in loclist:
            if location != 'Back Court(BC)':
                kobeyearzone = kobeyear.loc[kobeyear['shot_zone_area']==location]
                loccount = float(kobeyearzone.shape[0])/yearlength
                df = pd.DataFrame({'Season':[year],'%':[loccount],'Shot Location':[location]})
                newdata = newdata.append(df,ignore_index = True)
    sns.pointplot(x='Season',y='%',hue='Shot Location',data=newdata)
    sns.plt.title('Kobe Bryant: Basic Shot Location')
    sns.plt.show()
    return newdata
#posOnFieldKobe(x)

def DarylMoreyKobe(kobedata,year):
    kobe = kobedata[['season','shot_zone_range','shot_made_flag','shot_zone_area','combined_shot_type']]
    kobeyear = kobe.loc[kobe['season']==year] #select specific year
    newdata = pd.DataFrame(columns = ['Event','Total'])
    newdata2 = pd.DataFrame(columns = ['Event','Total'])
    shotrange = list(sorted(set(kobeyear['shot_zone_range'])))
    shotrange.remove('Back Court Shot')
    for shotr in shotrange:
        rangedata = kobeyear.loc[kobeyear['shot_zone_range']==shotr]
        shotarea = list(sorted(set(rangedata['shot_zone_area'])))
        for shota in shotarea:
            areadata = rangedata.loc[rangedata['shot_zone_area']==shota]
            shottype = list(sorted(set(areadata['combined_shot_type'])))
            #print(shottype)
            #shottype.remove('Tip Shot')
            for shot in shottype:
                if shot != 'Tip Shot' and shot != 'Hook Shot':
                    shotdata = areadata.loc[areadata['combined_shot_type']==shot]
                    event = shot+shota+ shotr
                    df = pd.DataFrame({'Event':[event],'Total': [shotdata.shape[0]]})
                    df2 = pd.DataFrame({'Event':[event],'Total' :[sum(shotdata['shot_made_flag'])]})
                    newdata = pd.concat([df,newdata])
                    newdata2 = pd.concat([df2,newdata2])
                    
                    
    sns.barplot(x = 'Event', y = 'Total', data = newdata,color = 'red')
    plt.xticks(rotation=45,fontsize = 5)
    bottom_plot = sns.barplot(x = 'Event', y = 'Total',data=newdata2, color = "#0000A3")
    topbar = plt.Rectangle((0,0),1,1,fc="red", edgecolor = 'none')
    bottombar = plt.Rectangle((0,0),1,1,fc='#0000A3',  edgecolor = 'none')
    l = plt.legend([bottombar, topbar], ['Made', 'Total Taken'], loc=1, ncol = 2, prop={'size':16})
    l.draw_frame(False)
    #Optional code - Make plot look nicer
    sns.despine(left=True)
    sns.plt.title('Kobe Bryant Shot Tendencies:' + ' ' + str(year) + ' ' + 'Season')
    sns.plt.show()
    return newdata,newdata2
                
#DarylMoreyKobe(x,2014)
#print(p)
def DarylMoreyKobe2(kobedata,year):
    kobe = kobedata[['season','shot_zone_range','shot_made_flag','shot_zone_area','action_type']]
    kobe['action_type'] = kobe['action_type'].apply(lambda x: 'Drive' if 'Driv' in x else x)
    kobe['action_type'] = kobe['action_type'].apply(lambda x: 'Runner' if 'Run' in x else x)
    kobe['action_type'] = kobe['action_type'].apply(lambda x: 'Cut' if 'Cut' in x else x)
    kobe['action_type'] = kobe['action_type'].apply(lambda x: 'Dunk Shot' if 'Dunk' in x else x)
    kobe['action_type'] = kobe['action_type'].apply(lambda x: 'Bank Shot' if 'Bank' in x else x)
    kobe['action_type']= kobe['action_type'].apply(lambda x: 'Tip Shot' if 'Tip' in x else x)
    kobe['action_type'] = kobe['action_type'].apply(lambda x: 'Fadeaway Shot' if 'Fadeaway' in x else x)
    kobe['action_type'] = kobe['action_type'].apply(lambda x: 'Pull-Up Shot' if 'Pull' in x else x)
    kobe['action_type'] = kobe['action_type'].apply(lambda x: 'Turnaround Shot' if 'Turn' in x else x)
    kobe['action_type'] = kobe['action_type'].apply(lambda x: 'Hook Shot' if 'Hook' in x else x)
    kobe['action_type'] = kobe['action_type'].apply(lambda x: 'StepBack Shot' if 'Step' in x else x)
    kobe['action_type'] = kobe['action_type'].apply(lambda x: 'Jump Shot' if 'Jump' in x else x)
    kobe['action_type'] = kobe['action_type'].apply(lambda x: 'Layup Shot' if 'Lay' in x else x)
    kobe['action_type'] = kobe['action_type'].apply(lambda x: 'Layup Shot' if 'Finger' in x else x)
    kobeyear = kobe.loc[kobe['season']==year] #select specific year
    newdata = pd.DataFrame(columns = ['Event','Total'])
    newdata2 = pd.DataFrame(columns = ['Event','Total'])
    shottype = list(sorted(set(kobeyear['action_type'])))
    for shot in shottype:
        print(shot)
        if shot != 'Tip Shot' and shot != 'Hook Shot':
            shotdata = kobeyear.loc[kobeyear['action_type']==shot]
            event = shot 
            df = pd.DataFrame({'Event':[event],'Total': [shotdata.shape[0]]})
            df2 = pd.DataFrame({'Event':[event],'Total' :[sum(shotdata['shot_made_flag'])]})
            newdata = pd.concat([df,newdata])
            newdata2 = pd.concat([df2,newdata2])
                    
    
    sns.barplot(x = 'Event', y = 'Total', data = newdata,color = 'red')
    plt.xticks(rotation=45,fontsize = 5)
    bottom_plot = sns.barplot(x = 'Event', y = 'Total',data=newdata2, color = "#0000A3")
    topbar = plt.Rectangle((0,0),1,1,fc="red", edgecolor = 'none')
    bottombar = plt.Rectangle((0,0),1,1,fc='#0000A3',  edgecolor = 'none')
    l = plt.legend([bottombar, topbar], ['Made', 'Total Taken'], loc=1, ncol = 2, prop={'size':12})
    l.draw_frame(False)
    #Optional code - Make plot look nicer
    sns.despine(left=True)
    
    sns.plt.title('Kobe Bryant Scoring Style:' + ' ' + str(year) + ' ' + 'Season')
    sns.plt.show()
    return newdata,newdata2
#DarylMoreyKobe2(x,2014)
#print(p)
def draw_court(ax=None, color='black', lw=2, outer_lines=False):
    # If an axes object isn't provided to plot onto, just get current one
    if ax is None:
        ax = plt.gca()

    # Create the various parts of an NBA basketball court

    # Create the basketball hoop
    # Diameter of a hoop is 18" so it has a radius of 9", which is a value
    # 7.5 in our coordinate system
    hoop = Circle((0, 0), radius=7.5, linewidth=lw, color=color, fill=False)

    # Create backboard
    backboard = Rectangle((-30, -7.5), 60, -1, linewidth=lw, color=color)

    # The paint
    # Create the outer box 0f the paint, width=16ft, height=19ft
    outer_box = Rectangle((-80, -47.5), 160, 190, linewidth=lw, color=color,
                          fill=False)
    # Create the inner box of the paint, widt=12ft, height=19ft
    inner_box = Rectangle((-60, -47.5), 120, 190, linewidth=lw, color=color,
                          fill=False)

    # Create free throw top arc
    top_free_throw = Arc((0, 142.5), 120, 120, theta1=0, theta2=180,
                         linewidth=lw, color=color, fill=False)
    # Create free throw bottom arc
    bottom_free_throw = Arc((0, 142.5), 120, 120, theta1=180, theta2=0,
                            linewidth=lw, color=color, linestyle='dashed')
    # Restricted Zone, it is an arc with 4ft radius from center of the hoop
    restricted = Arc((0, 0), 80, 80, theta1=0, theta2=180, linewidth=lw,
                     color=color)

    # Three point line
    # Create the side 3pt lines, they are 14ft long before they begin to arc
    corner_three_a = Rectangle((-220, -47.5), 0, 140, linewidth=lw,
                               color=color)
    corner_three_b = Rectangle((220, -47.5), 0, 140, linewidth=lw, color=color)
    # 3pt arc - center of arc will be the hoop, arc is 23'9" away from hoop
    # I just played around with the theta values until they lined up with the 
    # threes
    three_arc = Arc((0, 0), 475, 475, theta1=22, theta2=158, linewidth=lw,
                    color=color)

    # Center Court
    center_outer_arc = Arc((0, 422.5), 120, 120, theta1=180, theta2=0,
                           linewidth=lw, color=color)
    center_inner_arc = Arc((0, 422.5), 40, 40, theta1=180, theta2=0,
                           linewidth=lw, color=color)

    # List of the court elements to be plotted onto the axes
    court_elements = [hoop, backboard, outer_box, inner_box, top_free_throw,
                      bottom_free_throw, restricted, corner_three_a,
                      corner_three_b, three_arc, center_outer_arc,
                      center_inner_arc]

    if outer_lines:
        # Draw the half court line, baseline and side out bound lines
        outer_lines = Rectangle((-250, -47.5), 500, 470, linewidth=lw,
                                color=color, fill=False)
        court_elements.append(outer_lines)

    # Add the court elements onto the axes
    for element in court_elements:
        ax.add_patch(element)

    return ax

def find_shootingPcts(kobe_df,kobeyear, gridNum,isClutch):
    np.seterr(divide='ignore', invalid='ignore')
    kobeyear_df = kobe_df.loc[kobe_df['season'] == kobeyear]
    if isClutch:
        kobeyear_df = kobeyear_df.loc[kobeyear_df['period'] >=4]
        kobeyear_df = kobeyear_df.loc[kobeyear_df['minutes_remaining']  <=5]
         #get kobe's year
    x = kobe_df.loc_x[kobe_df['loc_y']<425.1] #i want to make sure to only include shots I can draw
    y = kobe_df.loc_y[kobe_df['loc_y']<425.1]
    x_year = kobeyear_df.loc_x[kobeyear_df['loc_y']<425.1]
    y_year = kobeyear_df.loc_y[kobeyear_df['loc_y']<425.1]

    x_made = kobe_df.loc_x[(kobe_df['shot_made_flag']==1) & (kobe_df['loc_y']<425.1)]
    x_yearmade = kobeyear_df.loc_x[(kobeyear_df['shot_made_flag']==1) & (kobeyear_df['loc_y']<425.1)]
    y_made = kobe_df.loc_y[(kobe_df['shot_made_flag']==1) & (kobe_df['loc_y']<425.1)]
    y_yearmade = kobeyear_df.loc_y[(kobeyear_df['shot_made_flag']==1) & (kobeyear_df['loc_y']<425.1)]

    #compute number of shots made and taken from each hexbin location
    hb_shot = plt.hexbin(x, y, gridsize=gridNum, extent=(-250,250,425,-50));
    plt.close() #don't want to show this figure!
    hb_shotyear = plt.hexbin(x_year,y_year, gridsize = gridNum, extent = (-250,250,425,-50));
    plt.close()
    hb_made = plt.hexbin(x_made, y_made, gridsize=gridNum, extent=(-250,250,425,-50),cmap=plt.cm.Reds);
    plt.close()
    hb_madeyear = plt.hexbin(x_yearmade,y_yearmade, gridsize=gridNum, extent=(-250,250,425,-50),cmap=plt.cm.Reds);
    plt.close()
    #compute shooting percentage
    ShootingPctLocs = (hb_madeyear.get_array()/hb_shotyear.get_array())/(hb_made.get_array() / hb_shot.get_array())
    ShootingPctLocs[np.isnan(ShootingPctLocs)] = 0 #makes 0/0s=0
    shootingFreqLocs = hb_shotyear.get_array() / hb_shot.get_array()
    shootingFreqLocs[np.isnan(shootingFreqLocs)] = 0
    return (ShootingPctLocs, hb_shotyear,shootingFreqLocs)
#find_shootingPcts(x,2001,30)

def kobe_shooting_plot(kobe_df,kobeyear,gridNum,isClutch):
    

    #compute shooting percentage and # of shots
    (ShootingPctLocs, shotNumber,shootingFreqLocs) = find_shootingPcts(kobe_df,kobeyear, gridNum,isClutch)
    #draw figure and court
    fig = plt.figure()#(12,7)
    ax = plt.axes([0.1, 0.1, 0.8, 0.8]) #where to place the plot within the figure
    draw_court(outer_lines=False)
    ax.set_xlabel('')
    ax.set_ylabel('')
    ax.tick_params(labelbottom='off', labelleft='off')
    plt.xlim(-250,250)
    plt.ylim(400, -25)
    #cdict = {'red': (
    #}
    cmap = plt.cm.RdYlBu
    #mymap = mpl.colors.LinearSegmentedColormap('my_colormap', cdict, 1024)
    #cmap = mymap
    #draw circles
    for i, shots in enumerate(ShootingPctLocs):
        restricted = Circle(shotNumber.get_offsets()[i], radius=shootingFreqLocs[i]*50,
                            color=cmap(shots),alpha=0.8, fill=True)
        
        if restricted.radius > 240/gridNum: restricted.radius=240/gridNum
        ax.add_patch(restricted)

    #draw color bar
    ax2 = fig.add_axes([0.92, 0.1, 0.02, 0.8])
    cb = mpl.colorbar.ColorbarBase(ax2,cmap=cmap, orientation='vertical')
    cb.set_label('Efficiency Compared to Career Avg.')
    cb.set_ticks([0.0,1.0])
    cb.set_ticklabels(['Low %', 'High %'])
    if isClutch:
        plt.suptitle('Kobe Bryant: ' + str(kobeyear) + ' Clutch Shot Chart')
    else:
        plt.suptitle('Kobe Bryant: ' + str(kobeyear) + ' Shot Chart')
    plt.show()
    return ax
    
#kobe_shooting_plot(x,[2006,2007],30,False)    
     
def kobe_scatter(kobe_df,year):
    
    # create our jointplot
    kobeyear_df = kobe_df.loc[kobe_df['season'] == (year)]
    kobeyear_df = kobeyear_df[['loc_x','loc_y','shot_distance']]
    kobeyear_df = kobeyear_df[kobeyear_df.shot_distance >0]
    

    cmap=plt.cm.gist_heat_r
    joint_shot_chart = sns.jointplot(kobeyear_df.loc_x, kobeyear_df.loc_y, stat_func=None,
                                 kind='kde', space=0, color=cmap(.2), cmap=cmap)

    joint_shot_chart.fig.set_size_inches(12,11)

# A joint plot has 3 Axes, the first one called ax_joint 
# is the one we want to draw our court onto 
    ax = joint_shot_chart.ax_joint
    draw_court(ax)

# Adjust the axis limits and orientation of the plot in order
# to plot half court, with the hoop by the top of the plot
    ax.set_xlim(-250,250)
    ax.set_ylim(422.5, -47.5)

# Get rid of axis labels and tick marks
    ax.set_xlabel('')
    ax.set_ylabel('')
    ax.tick_params(labelbottom='off', labelleft='off')

# Add a title
    ax.set_title('Kobe Kernel Density Estimate Shot Chart:' + ' ' + str(year) + ' Season', y=1.2, fontsize=10)

    



    plt.show()
    return None

#kobe_scatter(x,2014)    
    

    
            
            
