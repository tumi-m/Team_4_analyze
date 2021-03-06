import pandas as pd
import numpy as np


ebp_url = 'https://raw.githubusercontent.com/Explore-AI/Public-Data/master/Data/electrification_by_province.csv'
ebp_df = pd.read_csv(ebp_url)

for col, row in ebp_df.iloc[:,1:].iteritems():
    ebp_df[col] = ebp_df[col].str.replace(',','').astype(int)

ebp_df.head()


twitter_url = 'https://raw.githubusercontent.com/Explore-AI/Public-Data/master/Data/twitter_nov_2019.csv'
twitter_df = pd.read_csv(twitter_url)
twitter_df.head()

# gauteng ebp data as a list
gauteng = ebp_df['Gauteng'].astype(float).to_list()

# dates for twitter tweets
dates = twitter_df['Date'].to_list()

# dictionary mapping official municipality twitter handles to the municipality name
mun_dict = {
    '@CityofCTAlerts' : 'Cape Town',
    '@CityPowerJhb' : 'Johannesburg',
    '@eThekwiniM' : 'eThekwini' ,
    '@EMMInfo' : 'Ekurhuleni',
    '@centlecutility' : 'Mangaung',
    '@NMBmunicipality' : 'Nelson Mandela Bay',
    '@CityTshwane' : 'Tshwane'
}

# dictionary of english stopwords
stop_words_dict = {
    'stopwords':[
        'where', 'done', 'if', 'before', 'll', 'very', 'keep', 'something', 'nothing', 'thereupon', 
        'may', 'why', 'â€™s', 'therefore', 'you', 'with', 'towards', 'make', 'really', 'few', 'former', 
        'during', 'mine', 'do', 'would', 'of', 'off', 'six', 'yourself', 'becoming', 'through', 
        'seeming', 'hence', 'us', 'anywhere', 'regarding', 'whole', 'down', 'seem', 'whereas', 'to', 
        'their', 'various', 'thereafter', 'â€˜d', 'above', 'put', 'sometime', 'moreover', 'whoever', 'although', 
        'at', 'four', 'each', 'among', 'whatever', 'any', 'anyhow', 'herein', 'become', 'last', 'between', 'still', 
        'was', 'almost', 'twelve', 'used', 'who', 'go', 'not', 'enough', 'well', 'â€™ve', 'might', 'see', 'whose', 
        'everywhere', 'yourselves', 'across', 'myself', 'further', 'did', 'then', 'is', 'except', 'up', 'take', 
        'became', 'however', 'many', 'thence', 'onto', 'â€˜m', 'my', 'own', 'must', 'wherein', 'elsewhere', 'behind', 
        'becomes', 'alone', 'due', 'being', 'neither', 'a', 'over', 'beside', 'fifteen', 'meanwhile', 'upon', 'next', 
        'forty', 'what', 'less', 'and', 'please', 'toward', 'about', 'below', 'hereafter', 'whether', 'yet', 'nor', 
        'against', 'whereupon', 'top', 'first', 'three', 'show', 'per', 'five', 'two', 'ourselves', 'whenever', 
        'get', 'thereby', 'noone', 'had', 'now', 'everyone', 'everything', 'nowhere', 'ca', 'though', 'least', 
        'so', 'both', 'otherwise', 'whereby', 'unless', 'somewhere', 'give', 'formerly', 'â€™d', 'under', 
        'while', 'empty', 'doing', 'besides', 'thus', 'this', 'anyone', 'its', 'after', 'bottom', 'call', 
        'nâ€™t', 'name', 'even', 'eleven', 'by', 'from', 'when', 'or', 'anyway', 'how', 'the', 'all', 
        'much', 'another', 'since', 'hundred', 'serious', 'â€˜ve', 'ever', 'out', 'full', 'themselves', 
        'been', 'in', "'d", 'wherever', 'part', 'someone', 'therein', 'can', 'seemed', 'hereby', 'others', 
        "'s", "'re", 'most', 'one', "n't", 'into', 'some', 'will', 'these', 'twenty', 'here', 'as', 'nobody', 
        'also', 'along', 'than', 'anything', 'he', 'there', 'does', 'we', 'â€™ll', 'latterly', 'are', 'ten', 
        'hers', 'should', 'they', 'â€˜s', 'either', 'am', 'be', 'perhaps', 'â€™re', 'only', 'namely', 'sixty', 
        'made', "'m", 'always', 'those', 'have', 'again', 'her', 'once', 'ours', 'herself', 'else', 'has', 'nine', 
        'more', 'sometimes', 'your', 'yours', 'that', 'around', 'his', 'indeed', 'mostly', 'cannot', 'â€˜ll', 'too', 
        'seems', 'â€™m', 'himself', 'latter', 'whither', 'amount', 'other', 'nevertheless', 'whom', 'for', 'somehow', 
        'beforehand', 'just', 'an', 'beyond', 'amongst', 'none', "'ve", 'say', 'via', 'but', 'often', 're', 'our', 
        'because', 'rather', 'using', 'without', 'throughout', 'on', 'she', 'never', 'eight', 'no', 'hereupon', 
        'them', 'whereafter', 'quite', 'which', 'move', 'thru', 'until', 'afterwards', 'fifty', 'i', 'itself', 'nâ€˜t',
        'him', 'could', 'front', 'within', 'â€˜re', 'back', 'such', 'already', 'several', 'side', 'whence', 'me', 
        'same', 'were', 'it', 'every', 'third', 'together'
    ]
}

### START FUNCTION
def dictionary_of_metrics(items):
    #Converted the list of items into a numpy array called 'item_list'
    items_np = np.array(items)
    
    #Return a dictionary with the metrics 'mean','median','std','var','min','max' as keys, and with  'items_rounded' values
    metrics_dictionary = {'mean' : round(items_np.mean(), 2), 
                         'median':round(np.median(items_np,axis = None), 2),
                         'var' : round(items_np.var(ddof = 1), 2),
                         'std' : round(items_np.std(ddof = 1), 2),
                         'min' : round(items_np.min(), 2),
                         'max' : round(items_np.max(), 2)}

    return metrics_dictionary



### END FUNCTION

### START FUNCTION
def five_num_summary(items):
    # your code here
    return

### END FUNCTION

### START FUNCTION
def date_parser(dates):
    # your code here
    return

### END FUNCTION

### START FUNCTION
def filter_starts(sequence, sentence):
    return [word for word in sentence if word.startswith(sequence)]

def extract_municipality_hashtags(df):
    split_tweets = [tweet.split() for tweet in df["Tweets"]]
    
    hashtags = [filter_starts('#', tweet) for tweet in split_tweets]
    mentions = [[mun_dict[mention] for mention in filter_starts('@', tweet) if mention in mun_dict] for tweet in split_tweets]
    
    df['Municipality'] = [mun[0] if mun else np.nan for mun in mentions]
    df["Hashtags"] = [ht if ht else np.nan for ht in hashtags]
    return df
    
### END FUNCTION
    def number_of_tweets_per_day(df):
  """Function which takes a pandas dataframe as input of number of tweets per day and converts to new pandas dataframe into a new dataframe specified by the format yyyy-mm-dd

  Parameters :
    
    Dataframe input converted to yyyy-mm-dd format
    Groups by format and counts number of tweets

  Return   :

    DataFrame(df) : number of tweets per day organised in new dataframe grouped by day for dates 2019-11-20 to 2019-11-29
  
  """


  df['Date'] = pd.to_datetime(df['Date']).dt.strftime('%Y-%m-%d') #dataframe input converted to yyyy-mm-dd format
  new_df = df.groupby('Date').count() #groups by format and counts number of tweets
  return new_df
### END FUNCTION

### END FUNCTION
### START FUNCTION



### END FUNCTION

### START FUNCTION
def word_splitter(df):
    df = twitter_df.copy() #Made a copy of the main twitter_df and call it df
    tweets_dataseries = df['Tweets'] #Extract the tweets dataseries to a variable called tweets_dataseries
    tweets_dataseries_lower = tweets_dataseries.str.lower() #Lowercase the tweets_dataseries
    tweets_dataseries_split = tweets_dataseries_lower.str.split() #Split the lowercase tweets_dataseries
    df['Split Tweets'] = tweets_dataseries_split #Create a new column called 'Split Tweets', add it to the  dataframe
    return df

### END FUNCTION

### START FUNCTION
def stop_words_remover(df):
    """ Return a dataframe of Tweets without stop words
    Args : 

        token_tweets : splits(tokenizes) the tweets within the dataframe
        stops : stop words in the tokenized list
        df[] : modifies the input dataframe
    
    Return :
        dataframe : tweets without stop words

    Egs (for specific rows) :

        >>> stop_words_remover(twitter_df.copy()).loc[0, "Without Stop Words"] == ['@bongadlulane', 'send', 'email', 'mediadesk@eskom.co.za']"""
        

    token_tweets = df.Tweets.apply(lambda x: x.lower().split())
    stops = stop_words_dict['stopwords']
    df["Without Stop Words"] = token_tweets.apply(lambda x: [word for word in x if word not in stops ])
    return df

### END FUNCTION


if __name__ == "__main__":
    dictionary_of_metrics(gauteng)
    five_num_summary(gauteng)
    date_parser(dates[:3])
    extract_municipality_hashtags(twitter_df.copy())
    number_of_tweets_per_day(twitter_df.copy())
    word_splitter(twitter_df.copy())
    stop_words_remover(twitter_df.copy())