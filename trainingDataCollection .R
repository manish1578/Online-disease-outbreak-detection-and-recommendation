library(rJava)
library(xlsx)
#library(twitteR)
library(readxl)
library(rtweet)
# plotting and pipes - tidyverse!
library(ggplot2)
library(dplyr)
# text mining library
library(tidytext)
library(data.table)
#Put your twitter keys
consumer_key<- ''
consumer_secret<-''
access_token<- ''
access_token_secret<- ''

twitter_token <- create_token(
  consumer_key = consumer_key,
  consumer_secret = consumer_secret,
  access_token = access_token,
  access_secret = access_token_secret)

#['#dengue', '#zika','#malaria','#chikungunya','dengue', 'zika','malaria','chikungunya', ]

malaria <- search_tweets(q = "#malaria OR malaria -filter:retweets" , lang = "en", retryonratelimit = TRUE)

malaria <- lat_lng(malaria)

fwrite(malaria,file='malaria_en.csv', row.names = FALSE)



dengue <- search_tweets(q = "#dengue OR dengue -filter:retweets" , lang = "en", retryonratelimit = TRUE)

dengue <- lat_lng(dengue)

fwrite(dengue,file='dengue_en.csv', row.names = FALSE)




chikungunya <- search_tweets(q = "#chikungunya OR chikungunya -filter:retweets" , lang = "en", retryonratelimit = TRUE)

chikungunya <- lat_lng(chikungunya)

fwrite(chikungunya,file='chikungunya_en.csv', row.names = FALSE)



zika <- search_tweets(q = "#zika OR zika -filter:retweets" , lang = "en", retryonratelimit = TRUE)

zika <- lat_lng(zika)

fwrite(zika,file='zika_en.csv', row.names = FALSE)





