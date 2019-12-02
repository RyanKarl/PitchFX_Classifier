import numpy as np
from pybaseball import statcast, statcast_pitcher, pitching_stats, playerid_lookup
import pandas as pd
import os
import math

def get_data(year = 2018, minimum_starts = 5):
    if not os.path.exists(str(year)):
        os.mkdir(str(year))
    if not os.path.exists(os.path.join(str(year), "Players_Stats_"+str(year)+".csv")):
        player_stats = pitching_stats(year, year)
        player_stats = player_stats[player_stats['GS']>minimum_starts]
        player_stats.to_csv(os.path.join(str(year), "Players_Stats_"+str(year)+".csv"))
    else:
        player_stats = pd.read_csv(os.path.join(str(year), "Players_Stats_"+str(year)+".csv"))
    out = None
    for name in player_stats['Name']:
        if not os.path.exists(os.path.join(str(year),'player')):
            os.mkdir(os.path.join(str(year),'player'))
        splitname = name.split(' ')
        # Database is really good and has some mistakes, so when we go to the lookup table for MLB Player IDs sometimes
        # it doesn't match up. This corrects the issues that I've found. Obviously this won't work for every year
        # out of the box because of this.
        splitname[0] = splitname[0].replace('.', '. ', 1)
        # print(splitname[0])
        if splitname[0] == 'J.A.':
            splitname[0] = 'J. A.'
        if name == 'Zack Wheeler':
            splitname[0] = 'Zach'
        if name == 'Matthew Boyd':
            splitname[0] = 'Matt'
        if name == 'C.J. Wilson':
            splitname[0] = 'c. j.'
        if name == 'R.A. Dickey':
            splitname[0] = 'R. A.'
        if name == 'Jon Niese':
            splitname[0] = 'Jonathon'
        if name == 'A.J. Burnett':
            splitname[0] = 'A. J.'
        if name == 'Jorge De La Rosa':
            splitname[0] = 'Jorge'
            splitname[1] = 'De La Rosa'
        if name == 'Rubby de la Rosa':
            splitname[0] = 'Rubby'
            splitname[1] = 'de la Rosa'
        if name == 'Cole DeVries':
            splitname[1] = 'De Vries'
        if name == 'Samuel Deduno':
            splitname[0] = 'Sam'
        if name == 'JC Ramirez':
            splitname[0] = 'J. C.'
        if name == 'Nathan Karns':
            splitname[0] = 'Nate'
        if name == 'Daniel Ponce de Leon':
            splitname[1] = 'Ponce de Leon'
        if name == 'Chi Chi Gonzalez':
            splitname[0] = 'Chi Chi'
            splitname[1] = 'Gonzalez'
        if name == 'Josh A. Smith':
            splitname[0] = 'Josh'
            splitname[1] = 'Smith'
        if name == 'Joel De La Cruz':
            splitname[1] = 'De La Cruz'

        if not os.path.exists(os.path.join(str(year), 'player', name+'-'+str(year)+'.csv')):
            player_id = playerid_lookup(splitname[1], splitname[0])
            print(year)
            player_id = player_id[player_id['mlb_played_first'] <= year]
            player_id = player_id[player_id['mlb_played_last'] >= year]

            print(player_id)
            print(len(player_id))
            if len(player_id) != 1:
                print(player_id)
                print(name)
                print("Concerning")


            player = statcast_pitcher(str(year)+'-1-01', str(year)+'-12-31', player_id['key_mlbam'].iloc(0)[0])
            player.to_csv(os.path.join(str(year), 'player', name+'-'+str(year)+'.csv'))
        else:
            player = pd.read_csv(os.path.join(str(year), 'player', name+'-'+str(year)+'.csv'))

        # ['SL' 'FF' 'CU' 'FT' 'CH' nan 'FC' 'KC' 'SI' 'PO' 'FS' 'EP' 'SC']
        player_row = pd.DataFrame({'Name':[name]})
        pitch_types = ['SL','FF','CU','FT','CH','FC','KC','SI','PO','FS','EP','SC','KN']
        soi = ['release_speed','release_pos_x','release_pos_z','pfx_x','pfx_z','vx0','vy0','vz0','ax','ay','az','effective_speed','release_spin_rate']
        for pitch in pitch_types:
            pitches = player[player['pitch_type'] == pitch]
            pitches = pitches[soi]
            for stat in soi:
                mean = np.mean(pitches[stat])
                if math.isnan(mean):
                    mean = 0
                std = np.std(pitches[stat])+0
                if math.isnan(std):
                    std = 0
                min = np.min(pitches[stat])+0
                if math.isnan(min):
                    min = 0
                max = np.max(pitches[stat])+0
                if math.isnan(max):
                    max = 0
                player_row[pitch+"_"+stat + '_std'] = std
                player_row[pitch+"_"+stat + '_mean'] = mean
                player_row[pitch + "_" + stat + '_min'] = min
                player_row[pitch + "_" + stat + '_max'] = max
        if out is None:
            out = player_row
        else:
            out = pd.concat([out,player_row])
    out
    out.to_csv(str(year)+".csv")

for x in range(2015,2017):
    get_data(x)
