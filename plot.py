import matplotlib.pyplot as plt
import numpy as np

def plot_model_predictions(m, y, date, data_ind = None):
    plt.plot(y.T.cpu().reshape(y.shape[0]), color = BLUE, label = "Ground Truth") # blue line is Ground Truth
    plt.plot(m.T.cpu().detach().numpy().reshape(y.shape[0]), color = ORANGE, label = "Prediction") #orange line is model prediction
    plt.xlabel("Timestamp")
    plt.ylabel("Resistance")
    if data_ind is None:
        plt.title("Ground Truth (Blue) vs Prediction (Orange) on " + str(date).replace("_", " "))
    else:
        plt.title("Ground Truth (Blue) vs Prediction (Orange) on " + str(date).replace("_", " ") + " with sensor " + str(data_ind))
    plt.legend(loc="lower right", ncol=1)
    plt.show()

def plot_results(m, y, event_time, date, data_inds_pair = None):

    if event_time is not None:
        for e_id in range(len(event_time)):
            if event_time[e_id] == 1:
                plt.axvline(x = e_id, color = GREY1)
    
    plt.axhline(y = 0, ls = '--', color = "black")

    plt.plot(y.T.cpu().reshape(y.shape[0]) - m.T.cpu().detach().numpy().reshape(y.shape[0]), color = "red") # red line is Ground Truth
    
    curr_ylim = plt.ylim()
    
    if np.abs(curr_ylim[0]) > np.abs(curr_ylim[1]):
        plt.ylim(curr_ylim[0], np.abs(curr_ylim[0]))
    else:
        plt.ylim(-np.abs(curr_ylim[1]), curr_ylim[1])

    #print(plt.ylim())
    #plt.ylim(-0.5, 0.5)
    plt.xlabel("Timestamp")
    plt.ylabel("Resistance(Denoised)")
    if data_inds_pair is None:
        plt.title(str(date).replace("_", " "))
    else:
        plt.title(str(date).replace("_", " ") + " with sensor pair" + str(data_inds_pair))
    plt.show()

def plot_results_small_windows(m, y, event_time, date, data_inds_pair = None, plot_result = True):

    events = []
    one_event = []
    for e_id in range(len(event_time)):
        if event_time[e_id] == 1:
            if len(one_event) == 0 or one_event[-1] + 1 == e_id:
                #print(e_id)
                one_event.append(e_id)
            else:
                
                events.append(one_event)
                one_event = []
    print(events)
    EVENTS = events
    print("There are " + str(len(events)) + " stoat events!")

    success_num = 0
    slopes = []
    e_ind = 0
    for e in events:
        e_ind+=1
        
        sensor_signal_fragment = y[e[0]:e[-1]].T.cpu().reshape(-1) - m[e[0]:e[-1]].T.cpu().detach().numpy().reshape(-1)
        x_coor = list(range(sensor_signal_fragment.shape[0]))


        slope, intecept = np.polyfit(list(range(sensor_signal_fragment.shape[0])), sensor_signal_fragment, 1)
        trendpoly = np.poly1d([slope, intecept]) 
        #print(slope)
        
        if plot_result:
        
            plt.plot(sensor_signal_fragment, color = "red")
            
            plt.plot(x_coor, trendpoly(x_coor))
            
            plt.xlabel("Timestamp")
            plt.ylabel("Resistance(Denoised)")
            if data_inds_pair is None:
                plt.title(str(date).replace("_", " ") + ", event " + str(e_ind))
            else:
                plt.title(str(date).replace("_", " ") + ", event " + str(e_ind) + " with sensor pair" + str(data_inds_pair))
            
            plt.show()
        slopes.append(slope)
        if slope > 0:
            success_num +=1

    print("Success rate is " + str(success_num/len(events)))
    return success_num, len(events), slopes
    

def plot_results_small_windows_with_events(m, y, event_time, date, data_inds_pair = None, plot_result = True):

    events = []
    one_event = []
    for e_id in range(len(event_time)):
        if event_time[e_id] == 1:
            if len(one_event) == 0 or one_event[-1] + 1 == e_id:
                #print(e_id)
                one_event.append(e_id)
            else:
                
                events.append(one_event)
                one_event = []
    print("There are " + str(len(events)) + " stoat events!")

    success_num = 0
    slopes = []
    e_ind = 0
    for e in events:
        e_ind+=1
        
        sensor_signal_fragment = y[e[0]:e[-1]].T.cpu().reshape(-1) - m[e[0]:e[-1]].T.cpu().detach().numpy().reshape(-1)
        x_coor = list(range(sensor_signal_fragment.shape[0]))


        slope, intecept = np.polyfit(list(range(sensor_signal_fragment.shape[0])), sensor_signal_fragment, 1)
        trendpoly = np.poly1d([slope, intecept]) 
        #print(slope)
        
        if plot_result:
        
            plt.plot(sensor_signal_fragment, color = "red")
            
            plt.plot(x_coor, trendpoly(x_coor))
            
            plt.xlabel("Timestamp")
            plt.ylabel("Resistance(Denoised)")
            if data_inds_pair is None:
                plt.title(str(date).replace("_", " ") + ", event " + str(e_ind))
            else:
                plt.title(str(date).replace("_", " ") + ", event " + str(e_ind) + " with sensor pair" + str(data_inds_pair))
            
            plt.show()
        slopes.append(slope)
        if slope > 0:
            success_num +=1

    print("Success rate is " + str(success_num/len(events)))
    return success_num, len(events), slopes, events

GREY1 = "#b8b8b8"
GREY2 = "#707070"
BLUE = "#1a80bb"
YELLOW = "#ddaa33"
ORANGE="#ea801c"
PURPLE = "#882255"
RED =  "#c46666"