def get_genre(item_id,item_attribute):
    attri_type=['unknown', 'Action', 'Adventure', 'Animation', 'Children\'s', 'Comedy', 'Crime',
                                  'Documentary', 'Drama', 'Fantasy', 'Film-Noir', 'Horror', 'Musical', 'Mystery', 
                                  'Romance', 'Sci-Fi', 'Thriller', 'War', 'Western']
    gentre_list=''
    for attri in attri_type:
        if item_attribute[attri]==1:
            gentre_list=gentre_list+attri+'|'
    return gentre_list
def construct_prompting(item_attribute, item_list, candidate_list): 
    # make history string
    history_string = "User history:\n" 
    for index in item_list:
        if(item_attribute[item_attribute['MovieID'] == index]['Title'].empty==True):
            continue
        title = item_attribute[item_attribute['MovieID'] == index]['Title'].values[0]
        genre = item_attribute[item_attribute['MovieID'] == index]['Genres'].values[0]
        #genre = item_attribute['genre'][index]
        history_string += "["
        history_string += str(index)
        history_string += "] "
        history_string += str(title) + ", "
        history_string += str(genre) + "\n"
    # make candidates
    candidate_string = "Candidates:\n" 
    for index in candidate_list:
        title = item_attribute[item_attribute['MovieID'] == index]['Title'].values[0]
        '''
        if(item_attribute[item_attribute['MovieID'] == index]['Genres'].empty==False):
            genre = item_attribute[item_attribute['MovieID'] == index]['Genres'].values[0]
        else :
            genre = ""'''
        genre = item_attribute[item_attribute['MovieID'] == index]['Genres'].values[0]
            #genre = item_attribute['genre'][index.item()]
        candidate_string += "["
        candidate_string += str(index)
        candidate_string += "] "
        candidate_string += str(title) + ", "
        candidate_string += str(genre) + "\n"
    # output format
    output_format = "Please output the index of user\'s favorite and least favorite movie only from candidate, but not user history. Please get the index from candidate, at the beginning of each line.\nOutput format:\nTwo numbers separated by '::'. Nothing else.Plese just give the index of candicates, remove [] (just output the digital value), please do not output other thing else, do not give reasoning.\n\n"
    # make prompt
    prompt = "You are a movie recommendation system and required to recommend user with movies based on user history that each movie with title(same topic/doctor), year(similar years), genre(similar genre).\n"
    prompt += history_string
    prompt += candidate_string
    prompt += output_format
    #print(prompt)
    return prompt