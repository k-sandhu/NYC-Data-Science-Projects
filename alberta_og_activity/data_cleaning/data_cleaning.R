library(dplyr)
library(ggplot2)
library(geosphere)
library(rgdal)
library(sp)


getData <- function(type,folder){
  file_list <- list.files(folder,recursive = TRUE)
  file_list = paste0(folder,file_list)
  
  df <- data.frame()
  temp_df <- data.frame()
  
  for (file in file_list){
    if(type == 'drilling_activity'){
      temp_df <- cleanDrillingActivity(read.delim(file))
    }else if(type == 'well_licences'){
      temp_df <- cleanWellLic(read.delim(file))      
    }else if(type == 'active_facility'){
      temp_df <- cleanActiveFacility(read.delim(file,skip = 4))
    }else if(type == 'horizontal_well_list'){
      temp_df <- cleanHorizontalWellList(read.delim(file,sep = ","))
    }else if(type == 'con_well'){
      temp_df <- cleanConWell(read.delim(file, skip = 19))
    }else if(type == 'pipelines'){
      temp_df <- cleanPipelines(read.delim(file,skip = 8,header = F))
    }else if(type == 'pipelinesConst'){
      temp_df <- cleanPipelinesConst(read.delim(file,skip = 6,header = F))
    }
    if(!is.null(temp_df)){
      df <- rbind(df, temp_df)
    }
  }
  df = data.frame(lapply(df, trimws))
  rm(temp_df)
  return(df)
  
}

cleanWellLic <- function(file){
  
  file = trimws(as.character(file[,1]), which = c("both"))
  file = file[14:length(file)]
  
  cutAt = 3
  cutAt = grep('----',file, ignore.case = T)[1]
  #cat("cutAt is: ",cutAt,"\n" )
  if(cutAt < 4 | is.null(cutAt)){
    return(NULL)
  }else{
    file = file[1:(cutAt - 2)]
  }
  #print(file)
  currentPermit = list(
    WELL_NAME = character(), LICENCE_NUMBER = character(),MINERAL_RIGHTS = character(),GROUND_ELEVATION = character(),
    UNIQUE_IDENTIFIER = character(), SURFACE_COORDINATES  = character(), BOARD_FIELD_CENTRE = character(),PROJECTED_DEPTH = character(),
    LAHEE_CLASSIFICATION = character(), FIELD  = character(), TERMINATING_ZONE = character(), 
    DRILLING_OPERATION = character(), WELL_PURPOSE = character(), WELL_TYPE = character(),SUBSTANCE = character(),
    LICENSEE = character(), SURFACE_LOCATION = character()
  )
  
  k = 1
  i = 1
  while(i <= length(file)){
    
    #37,  10,  21, to the end #ignore first 4
    currentLine = file[i]
    i = i + 1
    currentPermit$WELL_NAME[k] = substr(currentLine,1,37)
    currentPermit$LICENCE_NUMBER[k] = substr(currentLine,38,47)
    currentPermit$MINERAL_RIGHTS[k] = substr(currentLine,48,68)
    currentPermit$GROUND_ELEVATION[k] = substr(currentLine,69,nchar(currentLine))
    
    currentLine = file[i]
    i = i + 1
    currentPermit$UNIQUE_IDENTIFIER[k] = substr(currentLine,1,23)
    currentPermit$SURFACE_COORDINATES[k] = substr(currentLine,24,47)
    currentPermit$BOARD_FIELD_CENTRE[k] = substr(currentLine,48,68)
    currentPermit$PROJECTED_DEPTH[k] = substr(currentLine,69,nchar(currentLine))
    
    #37, 31, to the end #ignore first 4
    currentLine = file[i]
    i = i + 1
    currentPermit$LAHEE_CLASSIFICATION[k] = substr(currentLine,1,37)
    currentPermit$FIELD[k] = substr(currentLine,38,67)
    currentPermit$TERMINATING_ZONE[k] = substr(currentLine,69,nchar(currentLine))
    
    #37, 31, to the end #ignore first 4
    currentLine = file[i]
    i = i + 1
    currentPermit$DRILLING_OPERATION[k] = substr(currentLine,1,37)
    currentPermit$WELL_PURPOSE[k] = substr(currentLine,38,46)
    currentPermit$WELL_TYPE[k] = substr(currentLine,47,68)
    currentPermit$SUBSTANCE[k] = substr(currentLine,69,nchar(currentLine))
    
    #68, to the end #ignore first 4
    currentLine = file[i]
    i = i + 2
    currentPermit$LICENSEE[k] = substr(currentLine,1,67)
    currentPermit$SURFACE_LOCATION[k] = substr(currentLine,68,nchar(currentLine))
    
    k = k + 1
  }
  df = as.data.frame(currentPermit)

  df = mutate(df, LSD = as.numeric(substr(SURFACE_LOCATION,1,3)), Section = as.numeric(substr(SURFACE_LOCATION,5,6)),Township = as.numeric(substr(SURFACE_LOCATION,8,10)), Range = as.numeric(substr(SURFACE_LOCATION,12,13)), Meridian = as.numeric(substr(SURFACE_LOCATION,15,16)))
  df1 = as.data.frame(getCo(df$LSD,df$Section,df$Township,df$Range,df$Meridian))
  df$lat = df1$lat
  df$long = df1$lon
  rm(df1)
  
  return(df)
}

cleanActiveFacility <- function(file){
  
  return(as.data.frame(file))
}

cleanHorizontalWellList <- function(file){

  
  return(file)
}

cleanConWell <- function(file){
  temp_df <- list(WellLocation=character(),Licence=character(),LicenseeCode=character(),
                  LicenseeName=character(),ConfidentialType=character(),ConfBelowFrmtn=character(),
                  ConfReleaseDate=character())
  
  
  for(i in 1:dim(file)[1]){
    current_line = as.character(file[i,1])

    temp_df$WellLocation[i] = substr(current_line,1,19)                     
    temp_df$Licence[i] = substr(current_line,22,28)  
    temp_df$LicenseeCode[i] = substr(current_line,31,35)
    temp_df$LicenseeName[i] = substr(current_line,36,66)  
    temp_df$ConfidentialType[i] = substr(current_line,67,84)  
    temp_df$ConfBelowFrmtn[i] = substr(current_line,85,103)  
    temp_df$ConfReleaseDate[i] = substr(current_line,104,122)  
  }  
  
  df = mutate(as.data.frame(temp_df), LSD = as.numeric(substr(WellLocation,4,5)), Section = as.numeric(substr(WellLocation,7,8)),Township = as.numeric(substr(WellLocation,10,12)), Range = as.numeric(substr(WellLocation,14,15)), Meridian = as.numeric(substr(WellLocation,17,17)))
  df1 = as.data.frame(getCo(df$LSD,df$Section,df$Township,df$Range,df$Meridian))
  df$lat = df1$lat
  df$long = df1$lon
  rm(df1)
  return(df)
}

cleanPipelines <- function(file){
  skip = trimws(as.character(file,"both") == "NO APPROVALS")
  
  temp_df = list(PermitLicenseNumber = character(), Permittee = character(), 
                 SubtCode = character(), PipelineFromLoc= character(),
                 ODMax= character(), ApprovalDate= character(),
                 OperatingKM= character())

  i = 1
  while(isTRUE(i < (dim(file)[1]-5))){
    current_line = as.character(file[i,1])
    
    temp_df$PermitLicenseNumber[i] = substr(current_line,8,13)                     
    temp_df$Permittee[i] = substr(current_line,18,55)
    temp_df$SubtCode[i] = substr(current_line,57,64)
    temp_df$PipelineFromLoc[i] = substr(current_line,71,84)
    temp_df$ODMax[i] = substr(current_line,91,97)
    temp_df$ApprovalDate[i] = substr(current_line,98,128)  
    temp_df$OperatingKM[i] = substr(current_line,129,160)
    i = i + 1
  }
  return(as.data.frame(temp_df))
}

cleanPipelinesConst <- function(file){
  #file = trimws(as.character(file,"both"))
  line1 <- as.character(file[1,])
  print(as.character(file[1,]))
  pos = gregexpr(' ', line1)
  print(pos[[1]])
  
  temp_df = list(BaID = character(), Licencee = character(), 
                 Licence = character(), FieldCentre = character(),
                 ActivityStartDate = character(), Number = character(),
                 From = character(), To = character(), Length = character())
  
  i = 2
  while (isTRUE(i < (dim(file)[1] - 5))){
    current_line = as.character(file[i,1])
    
    temp_df$BaID[i] = substr(current_line,1,pos[[1]][1])                     
    temp_df$Licencee[i] = substr(current_line,pos[[1]][1],pos[[1]][2])
    temp_df$Licence[i] = substr(current_line,pos[[1]][2],pos[[1]][3])
    temp_df$FieldCentre[i] = substr(current_line,pos[[1]][3],pos[[1]][4])
    temp_df$ActivityStartDate[i] = substr(current_line,pos[[1]][4],pos[[1]][5]-12)
    temp_df$Number[i] = substr(current_line,pos[[1]][5],pos[[1]][6])  
    temp_df$From[i] = substr(current_line,pos[[1]][6],pos[[1]][7])
    temp_df$To[i] = substr(current_line,pos[[1]][7],pos[[1]][8])
    temp_df$Length[i] = substr(current_line,pos[[1]][8],nchar(current_line))
    i = i + 1
  }
  return(as.data.frame(temp_df))
}

getCo <- function(LSD,Section,Township,Range,Meridian){
  
  co = c(-110,49)
  distNorth = Township * 9.7 + (Section%/%6) * 1.6 + LSD%/%4 * 0.402
  distWest = (Range * 9.7 + ifelse((Section%/%6)%%2 == 0, 
                                   Section * 1.6, 
                                   (6 - Section) * 1.6) +
                ifelse((LSD%/%4)%%2 == 0, 
                       LSD * 0.402, 
                       (4 - LSD) * 0.402))
  
  d = sqrt(distNorth^2 + distWest^2)
  
  b = atan(distWest/distNorth)/0.0174532925
  cod = destPoint(co, 360 - b, d * 1000)
  
  return(cod)
}

#Create Datasets
well_licences_df <- getData('well_licences',"C:/Users/sandh/Dropbox/FRM/NY Data Science/Projects/Project 1/data/well_licences/")
horizontal_well_list_df <- getData('horizontal_well_list',"C:/Users/sandh/Dropbox/FRM/NY Data Science/Projects/Project 1/data/horizontal_well_list/")
active_facility_df <- getData('active_facility',"C:/Users/sandh/Dropbox/FRM/NY Data Science/Projects/Project 1/data/active_facility/")
con_well_df <- getData('con_well',"C:/Users/sandh/Dropbox/FRM/NY Data Science/Projects/Project 1/data/con_well/")
pipelines <- getData('pipelines',"C:/Users/sandh/Dropbox/FRM/NY Data Science/Projects/Project 1/data/pipelines/")


#Well Licences
well_licences_df$GROUND_ELEVATION = as.numeric(unlist(strsplit(as.character(well_licences_df$GROUND_ELEVATION), "*M")))
well_licences_df$PROJECTED_DEPTH = as.numeric(unlist(strsplit(as.character(well_licences_df$PROJECTED_DEPTH), "*M")))


#Active Facility
active_facility_df$LSD <- as.numeric(active_facility_df$LSD)
active_facility_df$SEC <- as.numeric(active_facility_df$SEC)
active_facility_df$TWP <- as.numeric(active_facility_df$TWP)
active_facility_df$RNG <- as.numeric(active_facility_df$RNG)
active_facility_df$MER <- as.numeric(active_facility_df$MER)
df = as.data.frame(getCo(active_facility_df$LSD,active_facility_df$SEC,
                         active_facility_df$TWP,active_facility_df$RNG,
                         active_facility_df$MER))
active_facility_df$lat = df$lat
active_facility_df$long = df$lon
rm(df)


#Horizontal Well List
horizontal_well_list_df$SPUD.DT1 = lapply(horizontal_well_list_df$SPUD.DT,function(x) as.Date(x))
horizontal_well_list_df$FIN.DRL1 = lapply(horizontal_well_list_df$FIN.DRL,function(x) as.Date(x,  "%Y%d%m"))
horizontal_well_list_df$UPDATED1 = lapply(horizontal_well_list_df$UPDATED,function(x) as.Date(x,  "%Y%d%m"))
horizontal_well_list_df$TVD = lapply(horizontal_well_list_df$TVD, as.numeric)
horizontal_well_list_df$TD = lapply(horizontal_well_list_df$TD, as.numeric)

#Confidential Wells
con_well_df <- con_well_df %>%
  filter(lat != 'NaN')

#Pipelines
pipelines <- pipelines %>% 
  filter(PermitLicenseNumber != "")
pipelines <-  mutate(as.data.frame(pipelines), LSD = 0, 
                     Section = as.numeric(substr(as.character(PipelineFromLoc),1,2)),
                     Township = as.numeric(substr(as.character(PipelineFromLoc),4,6)), 
                     Range = as.numeric(substr(as.character(PipelineFromLoc),8,9)), 
                     Meridian = as.numeric(substr(as.character(PipelineFromLoc),11,11)))
df = as.data.frame(getCo(pipelines$LSD,pipelines$Section,pipelines$Township,pipelines$Range,pipelines$Meridian))
pipelines$lat = df$lat
pipelines$long = df$lon
rm(df)
pipelines$long = mutate(pipelines, long = as.numeric(getCo(LSD,Section,Township,Range,Meridian)[1]))
pipelines$ODMax = as.numeric(pipelines$ODMax)
pipelines$OperatingKM = as.numeric(pipelines$OperatingKM)
colnames(pipelines)[6] <- c("Date")


######################## Pipeline Construction ###################################################

pipelinesConst <- getData('pipelinesConst',"C:/Users/sandh/Dropbox/FRM/NY Data Science/Projects/Project 1/data/pipelinesConst/")
pipelinesConst1 <- pipelinesConst %>% 
  filter(BaID != ' ')
pipelinesConst1 <-  mutate(as.data.frame(pipelinesConst1), 
                          fLSD = as.numeric(substr(as.character(From),1,2)), 
                     fSection = as.numeric(substr(as.character(From),4,5)),
                     fTownship = as.numeric(substr(as.character(From),7,9)), 
                     fRange = as.numeric(substr(as.character(From),11,12)), 
                     fMeridian = as.numeric(substr(as.character(From),14,14)))
df = as.data.frame(getCo(pipelinesConst1$fLSD,
                         pipelinesConst1$fSection,
                         pipelinesConst1$fTownship,
                         pipelinesConst1$fRange,
                         pipelinesConst1$fMeridian))
pipelinesConst1$fLat = df$lat
pipelinesConst1$fLong = df$lon
rm(df)
pipelinesConst1 <-  mutate(as.data.frame(pipelinesConst1), 
                           tLSD = as.numeric(substr(as.character(To),1,2)), 
                           tSection = as.numeric(substr(as.character(To),4,5)),
                           tTownship = as.numeric(substr(as.character(To),7,9)), 
                           tRange = as.numeric(substr(as.character(To),11,12)), 
                           tMeridian = as.numeric(substr(as.character(To),14,14)))
df = as.data.frame(getCo(pipelinesConst1$tLSD,
                         pipelinesConst1$tSection,
                         pipelinesConst1$tTownship,
                         pipelinesConst1$tRange,
                         pipelinesConst1$tMeridian))
pipelinesConst1$tLat = df$lat
pipelinesConst1$tLong = df$lon
rm(df)
pipelinesConst1 <- pipelinesConst1[,-c(10,11,12,13,14,17,18,19,20,21)]
pipelinesConst1 <- pipelinesConst1 %>%
  filter(fLong != ' ')

#Eliminate all pipelines with same start and end locations as they could be duplicates,
#
n <- pipelinesConst1 %>% group_by(From,To) %>% distinct()
n1 <- ungroup(n)
pipelinesConst1 <- left_join(n1,pipelinesConst1,by = c("From","To"))

#Get counties of the start and ending locations
ab <- readOGR("D:/data/CanadaMap/shapefile","CAN_adm3")
i = which(ab@data$NAME_1 == "Alberta")
ab <- ab[i,]

pipelinesConst_spf <- pipelinesConst
pipelinesConst_spt <- pipelinesConst

coordinates(pipelinesConst_spf) <- ~fLong+fLat
coordinates(pipelinesConst_spt) <- ~tLong+tLat


crs(pipelinesConst_spf) <- ab@proj4string
crs(pipelinesConst_spt) <- ab@proj4string

zf <- over(pipelinesConst_spf,ab)
zt <- over(pipelinesConst_spt,ab)

counties_drillingf <- as.data.frame(zf[,c(9)])
colnames(counties_drillingf) <- c("County.From")
counties_drillingt <- as.data.frame(zt[,c(9)])
colnames(counties_drillingt) <- c("County.To")


pipelinesConst_1 <- cbind(pipelinesConst,counties_drillingf)
pipelinesConst_1 <- cbind(pipelinesConst_1,counties_drillingt)

pipelinesConst_1$County.Name.f <- pipelinesConst_1$ID_2
pipelinesConst_1$County.Name.t <- pipelinesConst_1$NAME_2
pipelinesConst_1 <- pipelinesConst_1[,-c(1)]


write.csv(well_licences_df, file = "C:/Users/sandh/Dropbox/FRM/NY Data Science/Projects/Project 1/well_licences_df.csv")
write.csv(horizontal_well_list_df, file = "C:/Users/sandh/Dropbox/FRM/NY Data Science/Projects/Project 1/horizontal_well_list_df.csv")
write.csv(active_facility_df, file = "C:/Users/sandh/Dropbox/FRM/NY Data Science/Projects/Project 1/active_facility_df.csv")
write.csv(con_well_df, file = "C:/Users/sandh/Dropbox/FRM/NY Data Science/Projects/Project 1/con_well_df.csv")
write.csv(pipelines,  file = "C:/Users/sandh/Dropbox/FRM/NY Data Science/Projects/Project 1/shinydashboard/OandGDashboard/www/pipelines.csv")
write.csv(pipelinesConst_1,  file = "C:/Users/sandh/Dropbox/FRM/NY Data Science/Projects/Project 1/shinydashboard/OandGDashboard/www/pipelinesConst.csv")

