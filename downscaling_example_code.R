# Example of spatial and temporal downscaling on ERA5 temperature
# Authors: Isabel Rushby and Laura Dawkins
library(RNetCDF)
library(fields)
library(maps)
library(mgcv)
library(scales)
library(lubridate)
library(parallel)
library(dplyr)
library(tidyr)
library(data.table)
library(ggplot2)

DIR = '/scratch/irushby/'
TEMP_HIGH_FILE = 'England_7_era5_temp_high.nc'
TEMP_LOW_FILE = 'England_7_era5_temp_low.nc'
country='England'
month = 7

#load high res data
nc <- open.nc(paste0(DIR,TEMP_HIGH_FILE))
print.nc(nc)
hightemp <- var.get.nc(nc,'t2m')
highlon <- var.get.nc(nc,'longitude')
highlat <- var.get.nc(nc,'latitude')
hightime <- var.get.nc(nc,'time')
close.nc(nc)

y <- cbind(expand.grid(highlon=highlon,highlat=highlat))
y = mutate(y, hightemp = as.vector(hightemp[,,1]))
y_region = filter(y,hightemp < 1000) # filter masked values
#load low res data
nc <- open.nc(paste0(DIR, TEMP_LOW_FILE))
print.nc(nc)
lowtemp <- var.get.nc(nc,'t2m')
lowlon <- var.get.nc(nc,'longitude')
lowlat <- var.get.nc(nc,'latitude')
lowtime <- var.get.nc(nc,'time')
close.nc(nc)

x <- cbind(expand.grid(lowlon=lowlon,lowlat=lowlat))
df = data.frame(y_region)
#match high and low res lons and lats
df2 = mutate(df, index=apply(rdist.earth(dplyr::select(df,-hightemp),x), 1, which.min), lowlon=x[index,1],lowlat=x[index,2])
# create df of all high res coords
region_highlon = expand.grid(highlon = df2$highlon,hightime = hightime)
region_highlat = expand.grid(highlat = df2$highlat,hightime = hightime)
region_lowlat = expand.grid(lowlat = df2$lowlat,hightime = hightime)
region_lowlon = expand.grid(lowlon = df2$lowlon,hightime = hightime)
region_df = data.frame(region_highlon$highlon, region_highlat$highlat, region_highlat$hightime, region_lowlon$lowlon,region_lowlat$lowlat)
region_hightemp = hightemp[which(highlon %in% df$highlon),which(highlat %in% df$highlat),]
# filter out masked values
region_hightemp = data.frame(region_hightemp=as.vector(region_hightemp))%>%dplyr::filter(region_hightemp < 1000)
region_df = bind_cols(region_df, region_hightemp)
x = mutate(x, lowtemp = as.vector(lowtemp[,,1]))%>%filter(lowtemp < 1000)#filter masked values
#create df of all low res coords
x_lowlon = x$lowlon
x_lowlat = x$lowlat
x_lowlontime = expand.grid(lowlon = x_lowlon, lowtime)
x_lowlontime = expand.grid(lowlon = x_lowlon,lowtime= lowtime)
x_lowlattime = expand.grid(lowlat = x_lowlat,lowtime= lowtime)
x_region_df = data.frame(lowlon = x_lowlontime$lowlon, lowlat = x_lowlattime$lowlat, lowtime=x_lowlattime$lowtime)
region_lowtemp = lowtemp[which(lowlon %in% x_lowlon),which(lowlat %in% x_lowlat),]
region_lowtemp = data.frame(region_lowtemp=as.vector(region_lowtemp))%>%dplyr::filter(region_lowtemp < 1000) # filter low res masked values
x_region_df = bind_cols(x_region_df, region_lowtemp)
region_df = region_df %>%mutate(date = as.Date(region_highlat.hightime/24,origin='1900/01/01'))
x_region_df = x_region_df %>%mutate(date = as.Date(lowtime/24,origin='1900/01/01'))
region_df = region_df %>% transmute(highlon = region_highlon.highlon, highlat = region_highlat.highlat,
                                    hightime = region_highlat.hightime, lowlon = region_lowlon.lowlon,
                                    lowlat = region_lowlat.lowlat, hightemp = region_hightemp,
                                    date= as.POSIXct(hightime*3600,origin='1900/01/01',tz='GMT'),
                                    timeofday=hour(date),dayofyear = yday(date),year=year(date))
x_region_df = x_region_df %>%mutate(dayofyear=yday(date),year=year(date))%>%dplyr::select(-date,-lowtime)
#combine high and low res values
region_df_2 = mutate(region_df,timeofday=hour(date),dayofyear = yday(date),year=year(date))%>%group_by(lowlon,lowlat,dayofyear,year)%>%dplyr::select(-hightime,-date)
region_df_2 = nest(region_df_2)
region_df_2 = left_join(region_df_2, x_region_df, by=c('lowlon','lowlat','dayofyear','year'))
region_df_2 = region_df_2%>% unnest(data)%>%ungroup()%>%mutate(date = as.Date(dayofyear-1, origin=paste(as.character(year),"-01-01",sep="")))
region_df_2 = region_df_2 %>% mutate(hightemp = hightemp - 273.15,lowtemp=region_lowtemp)%>%mutate(z = hightemp - lowtemp)%>%filter(lowlon > -12)

nc <- 6   ## cluster size
cl <- makeCluster(nc)

data.northwest = filter(region_df_2, lowlat > 52.5, lowlon < -1)
#fit model
northwest_fit <- bam(z ~ ti(highlon,highlat,k=8,bs="tp")  + ti(timeofday,bs="cc",k=10) +#+ ti(region_lowtemp,bs="tp",k=9)
                       ti(dayofyear,bs="cc",k=5) +
                       ti(highlon,highlat,timeofday,d=c(2,1),bs=c("tp","cc")) +
                       ti(highlon,highlat,dayofyear)#,region_lowtemp,d=c(2,1,1),bs=c("tp","cc","tp"))
                     ,data=data.northwest,cluster=cl)
save(northwest_fit, file=paste0(DIR,country,'_',month,'_downscale_temp_northwest.RData'))
png(paste(DIR,'example/','gam_diagnostics_test_fit_northwest',country,'_',month,'_all.png',sep=""),width = 800, height = 800)
gam.check(northwest_fit)
dev.off()
png(paste(DIR,'example/','gam_relationships_test_fit_northwest',country,'_',month,'_all.png',sep=""),width = 800, height = 800)
plot(northwest_fit,pages=1,residuals=TRUE,all.terms=TRUE,shade=TRUE,shade.col=2)
dev.off()

all_pred = data.northwest %>%mutate(pred=predict.gam(northwest_fit,data.northwest),temppred=pred+lowtemp)

all_pred_no_missing = filter(all_pred,temppred!='NA')

data.allregions.day1 = filter(all_pred_no_missing,date=="1979-07-04",timeofday==10)
# Spatial plots
palette <- colorRampPalette(c('yellow','green','turquoise','blue','purple'), space="rgb")
brks <- seq(-30,60,length=90)
png(paste(DIR,'example/','gam_map_high_test_all_',month,'_day1.png',sep=""),width = 800, height = 800)
quilt.plot(data.allregions.day1$highlon,data.allregions.day1$highlat,data.allregions.day1$hightemp,
           nx=length(unique(data.allregions.day1$highlon)),ny=length(unique(data.allregions.day1$highlat)),main='High Res ERA5 Temp: 1979-07-04 10:00',col=palette(89),breaks=brks)
map('world',add=T)
dev.off()
png(paste(DIR,'example/','gam_map_low_test_all_',month,'_day1.png',sep=""),width = 800, height = 800)
quilt.plot(data.allregions.day1$lowlon,data.allregions.day1$lowlat,data.allregions.day1$lowtemp,nx=length(unique(data.allregions.day1$lowlon)),ny=length(unique(data.allregions.day1$lowlat)),main='Low Res ERA5 Temp: 1979-07-04',col=palette(89),breaks=brks)
map('world',add=T)
dev.off()
png(paste(DIR,'example/','gam_map_high_pred_test_all_',month,'_day1.png',sep=""),width = 800, height = 800)
quilt.plot(data.allregions.day1$highlon,data.allregions.day1$highlat,data.allregions.day1$temppred,nx=length(unique(data.allregions.day1$highlon)),ny=length(unique(data.allregions.day1$highlat)),main='High Res Predicted ERA5 Temp: 1979-07-04 10:00',col=palette(89),breaks=brks)
map('world',add=T)

# plots of daily cycle
plot2 = ggplot(filter(test.data.allregions,lowlon==-2.5, between(lowlat,52.2,52.3),date=='1979-07-04'))+geom_point(aes(timeofday,hightemp,col='True'))+geom_point(aes(timeofday,temppred,col='Pred'))+facet_wrap(highlon~highlat, labeller = label_both)+geom_hline(aes(yintercept=lowtemp,color='Low Res Grid Square Daily Mean'))+xlab('Time of Day')+ylab('Temperature (C)')+scale_color_discrete(name="")+ggtitle('Date: 1979-07-04')
ggsave(paste0(DIR,'example/',country,'_',month,'_gam_hourly_pred_day1location1.png'),plot=plot2)
plot5 = ggplot(filter(test.data.allregions,lowlon==-2.5, between(lowlat,52.2,52.3),date=='2011-07-01'))+geom_point(aes(timeofday,hightemp,col='True'))+geom_point(aes(timeofday,temppred,col='Pred'))+facet_wrap(highlon~highlat, labeller = label_both)+geom_hline(aes(yintercept=lowtemp,color='Low Res Grid Square Daily Mean'))+xlab('Time of Day')+ylab('Temperature (C)')+scale_color_discrete(name="")+ggtitle('Date: 2011-07-01')
ggsave(paste0(DIR,'example/',country,'_',month,'_gam_hourly_pred_day2location1.png'),plot=plot5)
plot6 = ggplot(filter(test.data.allregions,lowlon==0, between(lowlat,52.7,52.8),date=='2011-07-01'))+geom_point(aes(timeofday,hightemp,col='True'))+geom_point(aes(timeofday,temppred,col='Pred'))+facet_wrap(highlon~highlat, labeller = label_both)+geom_hline(aes(yintercept=lowtemp,color='Low Res Grid Square Daily Mean'))+xlab('Time of Day')+ylab('Temperature (C)')+scale_color_discrete(name="")+ggtitle('Date: 2011-07-01')
ggsave(paste0(DIR,'example/',country,'_',month,'_gam_hourly_pred_day2location2.png'),plot=plot6)
# plot predicted mean against actual mean to check is conserved
all_pred_no_missing = group_by(all_pred_no_missing,lowlon,lowlat,date)%>%mutate(lowtemppredmean = mean(temppred))
df_10_cells = group_by(test.data.allregions,lowlon,lowlat)%>%nest()
df_10_cells = head(df_10_cells) %>%unnest()
plot7 = ggplot(df_10_cells)+geom_point(aes(lowtemp,lowtemppredmean,col=interaction(as.factor(round(lowlon,digits=2)),as.factor(round(lowlat,digits=2)))))+geom_abline(intercept=0,slope=1,col='black')+scale_color_discrete(name='Low Res Grid Square')+xlab('True Daily Mean')+ylab('Pred Daily Mean')
ggsave(paste0(DIR,'example/','compare_daily_mean_',country,'_',month,'_all.png'),plot=plot7)

