library(readr)
library(ggplot2)
library(dplyr)
library(ggthemes)
data <- read_csv("C:/Users/Kamal/Desktop/Bootcamp/Projects/Project 2/tamingnews/analysis/data/data.csv")

df = as.data.frame(data)
df$Source <- as.factor(df$Source)
x <- as.factor(c('CNN', 'Fox News','Reuters'))
levels(df$Source) <- x


avGs <- df %>%
  group_by(Source) %>%
  summarise(mean(gMag),mean(gSent))

avLen <- df %>%
  group_by(Source) %>%
  summarise(len = mean(lenArticle))

df <- df %>%
  mutate(returnLen = lenGoogleAPIentities +
           lenGoogleAPItokens + 
           lenTextrazorAPIentailments+ 
           lenTextrazorAPIproperties+ 
           lenTextrazorAPItopics+ 
           lenTextrazorAPIsentences+ 
           lenTextrazorAPIrelations+ 
           lenTextRazorAPIcoarseTopics+
           lenTextrazorAPIentities)

'%!in%' <- function(x,y)!('%in%'(x,y))

gEntities1 <- df[df$gentity1 %!in% c("Trump", "Donald Trump"),] %>%
  group_by(Source, gentity1) %>%
  summarise(count = n()) %>%
  top_n(6)

gEntities2 <- df[df$gentity2 %!in% c("Trump", "Donald Trump"),] %>%
  group_by(Source, gentity2) %>%
  summarise(cumSalience = sum(gentity2salience)) %>%
  arrange(desc(cumSalience)) %>%
  top_n(6)

colnames(gEntities1) <- c('Source', 'Entity', 'Count')
colnames(gEntities2) <- c('Source', 'Entity', 'Count')
gEntities <- rbind(gEntities1, gEntities2)
gEntities <- gEntities %>%
  group_by(Source, Entity) %>%
  summarise(Count = sum(Count)) %>%
  arrange(desc(Count)) %>%
  top_n(6)


tEntities1 <- df[df$tentity1 %!in% c("Trump", "Donald Trump"),] %>%
  group_by(Source, tentity1) %>%
  summarise(cumSalience = sum(tentity1salience)) %>%
  arrange(desc(cumSalience)) %>%
  top_n(6)

tEntities2 <- df[df$tentity2 %!in% c("Trump", "Donald Trump"),] %>%
  group_by(Source, tentity2) %>%
  summarise(cumSalience = sum(tentity2salience)) %>%
  arrange(desc(cumSalience)) %>%
  top_n(6)

colnames(tEntities1) <- c('Source', 'Entity', 'Count')
colnames(tEntities2) <- c('Source', 'Entity', 'Count')
tEntities <- rbind(tEntities1, tEntities2)
tEntities <- tEntities %>%
  group_by(Source, Entity) %>%
  summarise(Count = sum(Count)) %>%
  arrange(desc(Count)) %>%
  top_n(6)


gtopic1 <- df[df$gtopic1 %!in% c("Trump", "Donald Trump"),] %>%
  group_by(Source, gtopic1) %>%
  summarise(num = n()) %>%
  arrange(desc(num)) %>%
  top_n(6)

gtopic2 <- df[df$gtopic2 %!in% c("Trump", "Donald Trump"),] %>%
  group_by(Source, gtopic2) %>%
  summarise(num = n()) %>%
  arrange(desc(num)) %>%
  top_n(6)

colnames(gtopic1) <- c('Source', 'Topic', 'Count')
colnames(gtopic2) <- c('Source', 'Topic', 'Count')
gtopic <- rbind(gtopic1, gtopic2)
gtopic <- gtopic %>%
  group_by(Source, Topic) %>%
  summarise(Count = sum(Count)) %>%
  arrange(desc(Count)) %>%
  top_n(6)


gSent <- ggplot(df) + geom_density(aes(x = gSent, fill = Source),bw = "bcv", alpha = .35) + 
  ggtitle("Article Sentiment ") +
  theme_classic() + xlab('Sentiment') + xlim(-1,1) +ylab("")
gSent + theme_light() + scale_color_brewer(palette="Set1") + theme(legend.position = c(.9,.9))

gMag <- ggplot(df) + geom_density(aes(x = gMag, fill = Source),bw = "bcv", alpha = .35) + 
  ggtitle("Magnitude of Sentiment") + 
  theme_classic() + xlab('Magnitude') + 
  ylab("") + xlim(c(-5, 45))
gMag + theme_light() + scale_color_brewer(palette="Set1")+ theme(legend.position = c(.9,.9))

lenG <- ggplot(avLen) + 
  geom_bar(aes(x = Source, y = len, fill = Source), stat = 'identity', alpha = .65,width = .5) + 
  ggtitle("Average Length of Article") + 
  theme_classic() + xlab('') + ylab("Number of Characters")
lenG+ theme_light() + scale_color_brewer(palette="Set1")+ theme(legend.position = c(.9,.9))

returnLenG <- ggplot(df) + 
  geom_point(aes(lenArticle, returnLen, color = Source), fill = 'blue', alpha = .35) +
  theme_classic()+ 
  xlab("Number of Character in an Article") + 
  ylab("Number of Character in Response")
returnLenG + theme_light() + scale_color_brewer(palette="Set1")+ theme(legend.position = c(.9,.9))

gEntities$Entity <- as.factor(gEntities$Entity)
levels(gEntities$Entity)[levels(gEntities$Entity) == "United States"] <- "US"
gEntitiesGcnn <- ggplot(gEntities[gEntities$Source %in% c("CNN"),]) + 
  geom_bar(aes(x = reorder(Entity, -Count), y = Count), stat = 'identity', alpha = .65, fill = 'blue')+
  theme_classic()+ ggtitle("Main Entities on CNN (Google NLP API)") +
  xlab("") + 
  ylab("")+ theme(axis.text.x = element_text(angle = 60, hjust = 1))
gEntitiesGcnn+ theme_light() + scale_color_brewer(palette="Set2")


gEntitiesGreuters <- ggplot(gEntities[gEntities$Source %in% c("Reuters"),]) + 
  geom_bar(aes(x = reorder(Entity, -Count), y = Count), stat = 'identity', fill = 'blue', alpha = .65)+
  theme_classic()+ ggtitle("Main Entities on Reuters (Google NLP API)") +
  xlab("") + 
  ylab("")+ theme(axis.text.x = element_text(angle = 60, hjust = 1))
gEntitiesGreuters+ theme_light() + scale_color_brewer(palette="Set1")


gEntitiesGfox <- ggplot(gEntities[gEntities$Source %in% c("Fox News"),]) + 
  geom_bar(aes(x = reorder(Entity, -Count), y = Count), stat = 'identity', fill = 'blue', alpha = .65)+
  theme_classic()+ ggtitle("Main Entities on Fox News (Google NLP API)") +
  xlab("") + 
  ylab("")+ theme(axis.text.x = element_text(angle = 60, hjust = 1))
gEntitiesGfox+ theme_light() + scale_color_brewer(palette="Set1")

tEntities$Entity <- as.factor(tEntities$Entity)
levels(tEntities$Entity)[levels(tEntities$Entity) == "Patient Protection and Affordable Care Act"] <- "ACA"
levels(tEntities$Entity)[levels(tEntities$Entity) == "North American Free Trade Agreement"] <- "NAFTA"
levels(tEntities$Entity)[levels(tEntities$Entity) == "Republican Party (United States)"] <- "Republican Party"
levels(tEntities$Entity)[levels(tEntities$Entity) == "Dodd-Frank Wall Street Reform and Consumer Protection Act"] <- "Dodd-Frank Act"
levels(tEntities$Entity)[levels(tEntities$Entity) == "Supreme Court of the United States"] <- "US Supreme Court"
tEntitiesGcnn <- ggplot(tEntities[tEntities$Source %in% c("CNN"),]) + 
  geom_bar(aes(x = reorder(Entity, -Count), y = Count), stat = 'identity', fill = 'blue', alpha = .65)+
  theme_classic()+ ggtitle("Main Entities on CNN (TextRazor NLP API)") +
  xlab("") + 
  ylab("")+ theme(axis.text.x = element_text(angle = 60, hjust = 1))
tEntitiesGcnn+ theme_light() + scale_color_brewer(palette="Set1")

tEntities$Entity <- as.factor(tEntities$Entity)
levels(tEntities$Entity)[2] <- "Dodd-Frank Act"
tEntitiesGreuters <- ggplot(tEntities[tEntities$Source %in% c("Reuters"),]) + 
  geom_bar(aes(x = reorder(Entity, -Count), y = Count), stat = 'identity', fill = 'blue', alpha = .65)+
  theme_classic()+ ggtitle("Main Entities on Reuters (TextRazor NLP API)") +
  xlab("") + 
  ylab("")+ theme(axis.text.x = element_text(angle = 60, hjust = 1))
tEntitiesGreuters+ theme_light() + scale_color_brewer(palette="Set1")


tEntitiesGfox <- ggplot(tEntities[tEntities$Source %in% c("Fox News"),]) + 
  geom_bar(aes(x = reorder(Entity, -Count), y = Count), stat = 'identity', fill = 'blue', alpha = .65)+
  theme_classic()+ ggtitle("Main Entities on Fox News (TextRazor NLP API)") +
  xlab("") + 
  ylab("")+ theme(axis.text.x = element_text(angle = 60, hjust = 1))
tEntitiesGfox+ theme_light() + scale_color_brewer(palette="Set1")


gtopic$Topic <- as.factor(gtopic$Topic)
levels(gtopic$Topic)[levels(gtopic$Topic) == "President of the United States"] <- "US President"
levels(gtopic$Topic)[levels(gtopic$Topic) == "Republican Party (United States)"] <- "Republican Party"
levels(gtopic$Topic)[levels(gtopic$Topic) == "Democratic Party (United States)"] <- "Democratic Party"
levels(gtopic$Topic)[levels(gtopic$Topic) == "Illegal immigration to the United States"] <- "Illegal Immigration"
levels(gtopic$Topic)[levels(gtopic$Topic) == "Supreme Court of the United States"] <- "Supreme Court"
levels(gtopic$Topic)[levels(gtopic$Topic) == "United States Senate"] <- "Senate"
gtopicGcnn <- ggplot(gtopic[gtopic$Source %in% c("CNN"),]) + 
  geom_bar(aes(x = reorder(Topic, -Count), y = Count), stat = 'identity', fill = 'blue', alpha = .65)+
  theme_classic()+ ggtitle("Main Topics on CNN (Google NLP API)") +
  xlab("") + 
  ylab("")+ theme(axis.text.x = element_text(angle = 60, hjust = 1))
gtopicGcnn+ theme_light() + scale_color_brewer(palette="Set1")

levels(gtopic$Topic)[levels(gtopic$Topic) == "Patient Protection and Affordable Care Act"] <- "ACA"
levels(gtopic$Topic)[levels(gtopic$Topic) == "Federal Reserve System"] <- "Federal Reserve"
gtopicGreuters <- ggplot(gtopic[gtopic$Source %in% c("Reuters"),]) + 
  geom_bar(aes(x = reorder(Topic, -Count), y = Count), stat = 'identity', fill = 'blue', alpha = .65)+
  theme_classic()+ ggtitle("Main Topics on Reuters (Google NLP API)") +
  xlab("") + 
  ylab("")+ theme(axis.text.x = element_text(angle = 60, hjust = 1))
gtopicGreuters+ theme_light() + scale_color_brewer(palette="Set1")


gtopicGfox <- ggplot(gtopic[gtopic$Source %in% c("Fox News"),]) + 
  geom_bar(aes(x = reorder(Topic, -Count), y = Count), stat = 'identity',fill = 'blue', alpha = .65)+
  theme_classic()+ ggtitle("Main Topics on Fox News (Google NLP API)") + 
  xlab("") + 
  ylab("") + theme(axis.text.x = element_text(angle = 60, hjust = 1))
gtopicGfox + theme_light() + scale_color_brewer(palette="Set1")
