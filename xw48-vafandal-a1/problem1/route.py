#!/usr/bin/env python
# Problem: Given a set of cities/towns with (latitude, longitude) information, and a set of major highway segments, come up with routing plans.
#
# In this problem, cities/towns/intersections are represented by nodes on graph. Each edge is related with: edge(highway) name, distance, and speed limit.
# the weight of each edge is designed according to routing modes. For instance, edge weight in 'distance' mode is the number of miles of current segment; edge
# weight in 'time' mode is the time spent on current segment.
#
# state space: each state could be interpreted as 'the city/town/intersection we are in'.
# successor function: 1) for bfs, dfs and idx, successor function of a node gives a set of neighbors of current city.
#                     2) for astar, we have two approaches in mind:
#                       a) take 'time' mode as an exmple. Since there's no priori of speed limit of upcoming highways, its difficult to come up with an admissible
#                       heuristic except h(current city) = distance(current city, end city)/MaximumSpeed. For 'segments', heuristics like 'h(s) = 1 if s is not end city'
#                       guarantees admissibility. However, in some cases, these heuristic functions coverage rather slow.
#                       b) Given (longitude, latitude) of cities, we can easily calculate lower bound of distance between two cities (see function distance_heuristic). This
#                       distance heuristic is guranteed to be admissible. What's more, it complies with common sense since roads are always designed to be short to minimize
#                       construction cost. We also noticed that, in most cases, distance reflects other measurements like time spent on road. Therefore, our implementation
#                       comes out with a set of candidate routing plans based on distance heuristic, and then chooses a best plan according to routing mode.
#
# During implementation, we found that some cities/towns/intersections on our graph do not appear in 'city_gps.txt', which means our heuristic function does not work for
# these nodes. It's problematic to add these nodes to priority queue of fringe. We managed to avoid adding these node to fringe set by following way: If such a city/intersection,
# say A, appears in successor function, we tries to recursively find neighbors of A that are located in 'city_gps.txt'. Adding these neighbors to successor set guarantees that
# all cities in fringe get a proper heuristic value.
#
# for implementation details, we use pandas package to handle routing data. This package provides easy and intuitive operations for our operation.
#
# Answer these questions:
# (1) Which search algorithm seems to work best for each routing options?
#  Our efficiency measurement is based on routing between Bloomington,_Indiana and Indianapolis,_Indiana.
#  Routing options      segments    distance    time        scenic
#   bfs                 0.16447     0.16302     0.17029     0.16892
#   dfs                 20.95988    21.29049    21.12573    20.90377  
#   ids                 0.28732     0.29473     0.26858     0.28188
#   astar               0.14253     0.14861     0.14451     0.14474  (when giving 1 candidate routing plan)
#
#   In above measurement, astar works better than other algorithms. Noticeably, the solution provided by dfs is rather long and is not optimal. It also consumes
#   too much time to finish. ids need more operations than bfs. 
#
# (2) Which algorithm is fastest in terms of the amount of computation time required by your program,
# and by how much, according to your experiments? (To measure time accurately, you may want to temporarily
# include a loop in your program that runs the routing a few hundred or thousand times.)
# Since our astar algorithm uses a heuristic for distance, we calculate computation time when # of candidate routing plans is set to 1 in distance mode. The 
# computation time is shown in (1). We can see that: astar is the fastest routing algorithm here. And it is almost twice faster than ids, 146 times faster than dfs,
# and 15.4% faster than bfs.
# 
# (3) Which algorithm requires the least memory, and by how much, according to your experiments?
#  This experiment is conducted under 'distance' routing option. The memory usage is extracted from 'top' command. 
#  We see that 'RES' segment of command output is: bfs (49820), dfs (65396), ids (52052), and astar (50752). In my opinion, the most part of memory usage is 
#  the routing data loaded into memory. 
#  bfs uses least memory in our experiment, dfs uses 31% more than bfs, ids uses 2.5% more than bfs, and astar uses 1.9% more than bfs.
#
# (4) Which heuristic function did you use, how good is it, and how might you make it better?
#  As discussed above, we use 'airline distance' based on the provided latitude and longitude. During our experiments, this heuristic function gives an answer in reasonal 
# amount of time. As discussed in (1), astar algorithm with this heuristic provides better performace than other algorithm. 
# 
# Till now, i have no idea about how to improve this heuristic. Because highways between two cities may follow the 'airline', other heuristic functions may not be admissible.
# 
# (5) Supposing you start in Bloomington, which city should you travel to if you want to take the longest possible drive (in miles)
# that is still the shortest path to that city? (In other words, which city is furthest from Bloomington?)
# How we solve this problem: Since bfs gurantees an optimal path, we extend bfs to find the furthest place from Bloomington. First we input a end city that does not exist,
# and each city is reached during bfs, we calculate the distance from Bloomington. At least, we choose the largest distance. 
#
# 


import sys
import numpy as np
import pandas as pd
from math import radians, cos, sin, asin, sqrt,degrees , atan2
from copy import deepcopy
import time
from threading import Thread



program_continues = 1
city_gps = np.array([0,0,0])
with open('./city-gps.txt','r') as gps :
    for line in gps :
        city_gps = np.vstack((city_gps,line.replace('\n','').split(' ')))
    city_gps =city_gps[1:]
city_gps_pd = pd.DataFrame(city_gps)

road_segments= np.array([0,0,0,0,0])
with open('./road-segments.txt','r') as seg :
    for line in seg:
        road_segments = np.vstack((road_segments,line.replace('\n','').split(' ')))
    road_segments =road_segments[1:]
road_segments_pd = pd.DataFrame(road_segments)

def successors(current_city , current_road = '---'):
    # try:
    #     return road_segments_pd[road_segments_pd[0]== current_city[1]].append(road_segments_pd[road_segments_pd[0].str.contains(current_city[4])])
    # except:
    #     return road_segments_pd[road_segments_pd[0] == current_city[1]].append(road_segments_pd[road_segments_pd[0].apply(lambda x: current_city[4] in x)])
    as_first_city = road_segments_pd[road_segments_pd[0] == current_city]
    as_second_city = road_segments_pd[road_segments_pd[1]== current_city]
    all_lines = as_first_city.append(as_second_city)
    # cross_roads1 = road_segments_pd[road_segments_pd[0].str.contains(current_road)]
    # cross_roads2 = road_segments_pd[road_segments_pd[1].str.contains(current_road)]
    # all_lines = all_lines.append(cross_roads1)
    # all_lines = all_lines.append(cross_roads2)
    all_lines = all_lines.drop_duplicates()
    return all_lines



def BFS(start_city_,end_city_):
    if start_city_ == end_city_:
        return [0,0,start_city_,end_city_]
    fringe = successors(start_city_)
    path_indices = []
    visited_cities = [start_city_]
    immidiate_next_idx_1 = fringe[0].isin([end_city_])[fringe[0].isin([end_city_]) == True].index     # check next cities
    immidiate_next_idx_2 = fringe[1].isin([end_city_])[fringe[1].isin([end_city_]) == True].index     # check next cities
    immidiate_next_idx =immidiate_next_idx_1.append(immidiate_next_idx_2)
    if len(immidiate_next_idx) > 0:
        path_indices.append(int(immidiate_next_idx.item()))
        return path_indices
    path_indices = [[kk] for kk in fringe.index.tolist()]
    while len(fringe) > 0:
        BFS_idx = 0 # pick the first
        row_index_ = fringe.index[BFS_idx]
        path_till_here = [it for it in path_indices if it[-1] == row_index_][0][:]
        print path_till_here
        current_row = fringe.iloc[BFS_idx]
        fringe = fringe.reset_index()
        fringe = fringe.drop(fringe.index[BFS_idx])
        fringe.index = fringe['index']
        del fringe['index']
        city_to_visit = current_row[1] if current_row[1] not in visited_cities else current_row[0]
        visited_cities.append(city_to_visit )
        for row_idx, path in successors(city_to_visit,current_row[4]).iterrows():
            current_path = path_till_here[:]
            next_city_to_check = ''
            if path[1] not in visited_cities:
                next_city_to_check = path[1]
            elif path[0] not in visited_cities:
                next_city_to_check = path[0]
            if next_city_to_check == end_city_:
                path_till_here.append(path.name)
                return path_till_here
            if next_city_to_check not in visited_cities and next_city_to_check != '':
                # visited_cities.append(next_city_to_check)
                fringe = fringe.append(path)
                current_path.append(path.name)
                path_indices.append(current_path)
    return False


def DFS(start_city_, end_city_,depth=-1):
    if start_city_ == end_city_:
        return [0,0,start_city_,end_city_]
    elif depth == 0 and start_city_ != end_city_:
        return False
    fringe = successors(start_city_)
    path_indices = []
    visited_cities = [start_city_]
    immidiate_next_idx_1 = fringe[0].isin([end_city_])[fringe[0].isin([end_city_]) == True].index  # check next cities
    immidiate_next_idx_2 = fringe[1].isin([end_city_])[fringe[1].isin([end_city_]) == True].index  # check next cities
    immidiate_next_idx = immidiate_next_idx_1.append(immidiate_next_idx_2)
    if len(immidiate_next_idx) > 0:
        path_indices.append(int(immidiate_next_idx.item()))
        return path_indices
    path_indices = [[kk] for kk in fringe.index.tolist()]
    while len(fringe) > 0:
        DFS_idx = -1
        row_index_ = fringe.index[DFS_idx]
        path_till_here = [it for it in path_indices if it[-1] == row_index_][0][:]
        print path_till_here
        current_row = fringe.iloc[DFS_idx]
        fringe = fringe.reset_index()
        fringe = fringe.drop(fringe.index[DFS_idx])
        fringe.index = fringe['index']
        del fringe['index']
        if len(path_till_here) < depth or depth == -1 :
            city_to_visit = current_row[1] if current_row[1] not in visited_cities else current_row[0]
            visited_cities.append(city_to_visit)
            for row_idx, path in successors(city_to_visit,current_row[4]).iterrows():
                current_path = path_till_here[:]
                next_city_to_check = ''
                if path[1] not in visited_cities:
                    next_city_to_check = path[1]
                elif path[0] not in visited_cities:
                    next_city_to_check = path[0]
                if next_city_to_check  == end_city_:
                    path_till_here.append(path.name)
                    return path_till_here
                if next_city_to_check not in visited_cities and next_city_to_check!='':
                    fringe = fringe.append(path)
                    current_path.append(path.name)
                    path_indices.append(current_path)
    return False

def IDS(start_city_, end_city_):
    depth = 0
    while True:
        print 'depth: %d'%depth
        tmp_result = DFS(start_city_, end_city_, depth=depth)
        if tmp_result:
            return tmp_result
        depth+=1





# def locaiton_estimator(city_name):
#     neighbor_list = neighbor_finder(city_name)
#     x_total = 0
#     y_total = 0
#     z_total = 0
#     for neighbor in neighbor_list:
#         city_row = city_gps_pd[city_gps_pd[0]==neighbor]
#         lat_tmp,long_tmp = map(radians, [city_row[1],city_row[2]])
#         x_total+= cos(lat_tmp)*cos(long_tmp)
#         y_total+=cos(lat_tmp)*sin(long_tmp)
#         z_total+=sin(lat_tmp)
#     x_total = x_total / len(neighbor_list)
#     y_total = y_total / len(neighbor_list)
#     z_total = z_total / len(neighbor_list)
#     estimated_lat = degrees(atan2(z_total, sqrt(x_total ** 2 + y_total ** 2)))
#     estimated_long = degrees(atan2(y_total,x_total))
#     return [estimated_lat,estimated_long]

def distance_heuristic(start_city_, end_city_):
    c1 = city_gps_pd[city_gps_pd[0]==start_city_]
    c2 = city_gps_pd[city_gps_pd[0]==end_city_]
    lat1,lon1,lat2,lon2 = map(radians, [c1[1],c1[2], c2[1],c2[2]])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
    c = 2 * asin(sqrt(a))
    r = 6371  # Radius of earth in kilometers. Use 3956 for miles
    return c * r

def neighbor_finder(city_name,level=0,threshold=1):
    # neighbor_names = pd.DataFrame({0 : []})
    neighbor_names = []
    all_neighbors = road_segments_pd[road_segments_pd[0] == city_name].append(road_segments_pd[road_segments_pd[1] == city_name])
    for row in all_neighbors.iterrows():
        if '&' in row[1][0] and row[1][0]!= city_name and level<threshold:
            neighbor_names = neighbor_names + [[row[1][1], row[0]] + item for item in neighbor_finder(row[1][1], level + 1, threshold)]
        elif '&' in row[1][1] and row[1][1]!= city_name and level<threshold:
            neighbor_names = neighbor_names + [[row[1][0], row[0]]+item for item in neighbor_finder(row[1][0], level + 1, threshold)]
        elif not '&' in row[1][0] or not '&' in row[1][1]:
            dest_name = row[1][1] if row[1][1]!= city_name else row[1][0]
            neighbor_names.append([dest_name,row[0]])
        # elif :
        #     dest_name = row[1][1] if row[1][1] != city_name else row[1][0]
        #     neighbor_names.append([dest_name,row[0]])
    return neighbor_names


def successors_astar(current_city , end_city_, path_to_here, current_road = '---'):
    as_first_city = road_segments_pd[road_segments_pd[0] == current_city]
    as_second_city = road_segments_pd[road_segments_pd[1]== current_city]
    all_lines = as_first_city.append(as_second_city)
    # cross_roads1 = road_segments_pd[road_segments_pd[0].str.contains(current_road)]
    # cross_roads2 = road_segments_pd[road_segments_pd[1].str.contains(current_road)]
    # all_lines = all_lines.append(cross_roads1)
    # all_lines = all_lines.append(cross_roads2)
    all_lines = all_lines.drop_duplicates()
    all_lines['2nd_city'] = all_lines.apply(lambda row: row[0] if (row[0] != current_city and current_road not in row[0]) else row[1], axis=1)
    all_lines['via'] = -1
    for row in all_lines.iterrows():
        city_neighbors = []
        if (row[1][0]==current_city or current_road in row[1][0]) and '&' in row[1][1] :
            city_neighbors = neighbor_finder(row[1][1])
        elif (row[1][1]==current_city or current_road in row[1][1]) and '&' in row[1][0]:
            city_neighbors = neighbor_finder(row[1][0])
        elif current_road in row[1][0] and '&' in row[1][1]:
            city_neighbors = neighbor_finder(row[1][1])
        elif current_road in row[1][1] and '&' in row[1][0]:
            city_neighbors = neighbor_finder(row[1][0])
        if len(city_neighbors) > 0 :
            all_lines = all_lines.drop([row[0]])
            for item in city_neighbors:
                if all([xx not in path_to_here for xx in item[1::2]]):
                    tmp_row = deepcopy(row[1])
                    tmp_row['2nd_city'] = item[-2]
                    tmp_row['via'] = tuple(item[1::2])
                    all_lines = all_lines.append(tmp_row)
        elif '&' in row[1]['2nd_city']:
            all_lines = all_lines.drop([row[0]])
    all_lines = all_lines.drop_duplicates()
    all_lines_pruned = pd.DataFrame({0: []})
    for row in all_lines.iterrows():
        if row[0] not in path_to_here:
            all_lines_pruned = all_lines_pruned.append(row[1])
    all_lines = all_lines_pruned
    try:
        directs =all_lines[all_lines['via']==-1]
        indirects = all_lines[all_lines['via']!=-1]
        indirects_tmp = indirects.reset_index()
        indirects_tmp['selection'] = indirects_tmp.apply(lambda row: row['index'] in row['via'],axis=1)
        indirects = indirects_tmp[indirects_tmp['selection']==False]
        indirects = indirects.set_index(indirects['index'])
        del indirects['selection']
        del indirects['index']
        all_lines = directs.append(indirects)
    except ValueError:
        pass
    all_lines['priority'] = all_lines.apply(lambda row: distance_heuristic(row['2nd_city'], end_city_), axis=1)
    return all_lines



def astar(start_city_, end_city_):
    global program_continues
    thread = Thread(target=threaded_astar_killer, args=())
    thread.start()
    if start_city_ == end_city_:
        return [0, 0, start_city_, end_city_]
    fringe = successors_astar(start_city_,end_city_,[0])
    path_indices = []
    visited_cities = [start_city_+'-1']
    all_pathes = []
    immidiate_next_idx_1 = fringe[0].isin([end_city_])[fringe[0].isin([end_city_]) == True].index  # check next cities
    immidiate_next_idx_2 = fringe[1].isin([end_city_])[fringe[1].isin([end_city_]) == True].index  # check next cities
    immidiate_next_idx = immidiate_next_idx_1.append(immidiate_next_idx_2)
    if len(immidiate_next_idx) > 0:
        path_indices.append(int(immidiate_next_idx.item()))
        all_pathes.append(path_indices)
    path_indices = [[kk] for kk in fringe.index.tolist()]
    while len(fringe) > 0 and program_continues:
        time.sleep(0.0001)
        current_row = fringe.loc[fringe['priority']==min(fringe['priority'])].iloc[0]
        tmp_df = fringe.reset_index()
        BFS_idx =tmp_df.ix[tmp_df['priority']==min(tmp_df['priority'])].index[0]
        row_index_ = tmp_df.iloc[BFS_idx]['index']
        if current_row['via'] == -1 :
            path_till_here = [it for it in path_indices if it[-1] == row_index_][0][:]
        else:
            indirect_path = list(current_row ['via']) + [current_row.name]
            path_till_here = [it for it in path_indices if all(x in it for x in indirect_path)][0][:]
        fringe = fringe.reset_index()
        fringe = fringe.drop(fringe.index[BFS_idx])
        fringe.index = fringe['index']
        del fringe['index']
        city_to_visit = current_row['2nd_city']
        visited_cities.append(city_to_visit+str(current_row['via']))
        for row_idx, path in successors_astar(city_to_visit, end_city_,path_till_here,current_row[4]).iterrows():
            current_path = path_till_here[:]
            next_city_to_check = ''
            if path['2nd_city']+str(path['via']) not in visited_cities:
                next_city_to_check = path['2nd_city']
            if next_city_to_check == end_city_:
                path_found = deepcopy(path_till_here)
                if path['via'] == -1 :
                    path_found.append(path.name)
                else:
                    path_found.append(path.name)
                    path_found += path['via']
                if path_found not in all_pathes:
                    all_pathes.append(path_found)
                    print "a new path was found! in total %d routes " %len(all_pathes)
                    print path_found
                    # if len(all_pathes) > 25 :
                    #     return all_pathes
            if next_city_to_check+str(path['via']) not in visited_cities and next_city_to_check != '' and next_city_to_check != end_city_:
                fringe = fringe.append(path)
                if path['via'] == -1 :
                    current_path.append(path.name)
                else:
                    current_path+= list(path['via']) + [path.name]
                path_indices.append(current_path)
    if len(all_pathes) > 0:
        return all_pathes
    else:
        return False

def segment_calculator(all_pathes):
    if all_pathes != False:
        all_segments = []
        for path in all_pathes :
            # for road in path:
            #     num_of_segs += int(road_segments_pd.ix[road][2])
            num_of_segs = len(path)
            all_segments.append(num_of_segs)
        return all_pathes[all_segments.index(min(all_segments))]
    else:
        return False

def distance_calculator(all_pathes):
    if all_pathes != False:
        all_distances = []
        for path in all_pathes :
            distance = 0
            for road in path:
                distance += int(road_segments_pd.ix[road][2])
            all_distances.append(distance)
        return all_pathes[all_distances.index(min(all_distances))]
    else:
        return False

def duration_calculator(all_pathes):
    if all_pathes != False:
        all_durations = []
        for path in all_pathes:
            total_duration = 0
            for road in path:
                total_duration += float(road_segments_pd.ix[road][2]) / float(road_segments_pd.ix[road][3])
            all_durations.append(total_duration)
        return all_pathes[all_durations.index(min(all_durations))]
    else:
        return False

def highway_calculator(all_pathes):
    if all_pathes != False:
        all_number_of_highways = []
        for path in all_pathes:
            number_of_highways = 0
            for road in path:
                if int(road_segments_pd.ix[road][3]) >= 55:
                    number_of_highways += 1
            all_number_of_highways.append(number_of_highways)
        return all_pathes[all_number_of_highways.index(max(all_number_of_highways))]
    else:
        return False

def build_output(start_city,path):
    if path!=False:
        total_distance = 0
        total_duration = 0
        for road in path:
            total_distance += int(road_segments_pd.ix[road][2])
            total_duration += float(road_segments_pd.ix[road][2]) / float(road_segments_pd.ix[road][3])
        city_names = [start_city]
        previous_city = start_city
        # previous_road = '-'
        for road in path:
            # if previous_road in road_segments_pd.ix[road][0]:
            #     next_city = road_segments_pd.ix[road][0]
            # elif previous_road in road_segments_pd.ix[road][1]:
            #     next_city = road_segments_pd.ix[road][1]
            if previous_city == road_segments_pd.ix[road][0]:
                next_city = road_segments_pd.ix[road][1]
            else:
                next_city = road_segments_pd.ix[road][0]
            city_names.append(next_city)
            # previous_road = road_segments_pd.ix[road][4]
            previous_city = next_city
        # city_names.append(end_city)
        print "%s %s"%(total_distance,total_duration),
        for city in city_names:
            print city,
    else:
        print "Sorry, No path found! :'[ "

def threaded_astar_killer():
    print "killer thread started"
    time.sleep(30)
    global program_continues
    program_continues = 0


if __name__ == '__main__':
    start_city = sys.argv[1]
    end_city = sys.argv[2]
    route_opt = sys.argv[3]
    assert route_opt in ['segments','distance','time','scenic'], "Routing option %s not defined"%route_opt
    route_alg = sys.argv[4]
    assert route_alg in ['bfs','dfs','ids','astar'], 'Algorithm %s not defined'%route_alg
    # start_city = "Bloomington,_Indiana"
    # end_city ="Indianapolis,_Indiana"
    # route_opt ="scenic"
    # route_alg = "dfs"
    route_options = {
        'segments' : segment_calculator,
        'distance' : distance_calculator,
        'time' : duration_calculator,
        'scenic' : highway_calculator
    }
    alg_options = {
        'bfs': BFS,
        'dfs': DFS,
        'ids': IDS,
        'astar': astar
    }
    pathes = alg_options[route_alg](start_city,end_city)
    if route_alg == 'astar':
        final_path = route_options[route_opt](pathes)
        build_output(start_city, final_path)
    else:
        build_output(start_city, pathes)
