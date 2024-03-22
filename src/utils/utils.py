# imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def plot_results(forecast, test): 
    """
    plot resutls of trained model
    """
    fig, axs = plt.subplots(5,2, figsize=(8,24))
    axs = axs.flatten()

    for i, dma in enumerate(test.columns): 
        # plot results
        axs[i].plot(forecast[dma], color='orange', label='forecast', linewidth=2)
        axs[i].plot(test[dma], color='blue', label='observed', linewidth=2)
        axs[i].fill_between(forecast.index, forecast['lower_'+dma], forecast['upper_'+dma] ,color='green',alpha=0.2)

        axs[i].set_title(dma)
        axs[i].set_ylabel('demand')
        axs[i].set_xlabel('date')
        leg = axs[i].legend(loc='upper center', bbox_to_anchor=(0.5, -0.1),
            fancybox=False, shadow=False, ncol=6, edgecolor='w', borderpad=0.5)
        leg.get_frame().set_linewidth(0.5)
    
    plt.show()

def extract_features(df): 
    """
    extract time features of data
    """
    # extract time features
    df['year'] = [d.year for d in df.index]
    df['month'] = [d.month for d in df.index]
    df['day'] = [d.weekday for d in df.index]
    df['date'] = [d.date() for d in df.index]
    df['time'] = [d.time() for d in df.index]
    df['hour'] = [d.hour() for d in df.index]

    # make list of special days for the DMAs region
    official_holidays = ["2021-01-01","2021-01-06","2021-04-04","2021-04-05","2021-04-25","2021-05-01","2021-06-02","2021-08-15","2021-11-01","2021-12-05","2021-12-25","2021-12-26","2022-01-01","2022-01-06","2022-04-17","2022-04-18","2022-04-25","2022-05-01","2022-06-22","2022-08-15","2022-11-01","2022-12-08","2022-12-25","2022-12-26"]

    legally_not_recongnized_holidays = ["2021-04-23","2021-05-23","2022-04-23","2022-06-05"]

    event_day = ["2021-03-28","2021-05-09","2021-10-31","2021-11-28","2021-05-12","2021-12-12","2021-12-19","2021-12-31","2022-03-27","2022-04-10","2022-05-08","2022-05-09","2022-10-30","2022-11-27","2022-12-04","2022-12-11","2022-12-18","2022-12-31"]

    # make columns for special days
    df['official_holiday'] = 0
    df['legally_not_recongnized_holidays'] = 0
    df['event_day'] = 0
    df['weekend'] = 0

    # add indicator variable for special days
    for i in df.index:
        if str(i)[:10] in official_holidays:
            df['official_holiday'][i] = 1

    for i in df.index:
        if str(i)[:10] in legally_not_recongnized_holidays:
            df['legally_not_recongnized_holidays'][i] = 1

    for i in df.index:
        if str(i)[:10] in event_day:
            df['event_day'][i] = 0

    # add variable for weekend days
    for i in df.index:
        if i.weekday() == 5 or i.weekday() == 6:
            df['weekend'][i] = 1

    #Vector for days
    day_arr = []
    for i in range(0,len(df)):
        day_vec = np.array([0,0,0,0,0,0,0])
        day_vec[df['day'].iloc[i]] = 1
        day_arr.append(day_vec)
    df['day_arr']=day_arr

    return df

