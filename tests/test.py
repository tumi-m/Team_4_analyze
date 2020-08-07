def number_of_tweets_per_day(df):
  """Function which list number of tweets per day by converting pandas dataframe into a new dataframe specified by yyyy-mm-dd"""
  df['Date'] = pd.to_datetime(df['Date']).dt.strftime('%Y-%m-%d')
  return df.groupby('Date').count()