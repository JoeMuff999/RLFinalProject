import matplotlib.pyplot as plt
import numpy as np
# from scipy.interpolate import interp1d
# from scipy.interpolate import spline

def read_np_array():
    file = '/Users/joey/Documents/UT/Sp22/RL_Final_Project/data/20220509-023522/rewards.npy'
    data = np.load(file)
    xaxis = [i for i in range(len(data))]
    # for d in data:
    plt.plot(xaxis, data)
    plt.show()

def rewards_over_time():
    file = open('../data/total_rewards-1000000.txt')
    data = []
    for idx, line in enumerate(file):
        line = line.replace('[', '')
        line = line.replace(']', '')
        data.append([])
        for elem in line.split('  '):
            data[idx].append(round(float(elem), 2))
    xaxis = [i for i in range(len(data[0]))]
    for d in data:
        plt.plot(xaxis, d)
    plt.show()

def rewards_over_time_confidence(file_name, time_steps):
    time_steps = time_steps
    file = open(file_name, 'r')
    data = []
    for idx, line in enumerate(file):
        line = line.replace('[ ', '')
        line = line.replace(' ]', '')
        data.append([])
        line = line.replace('  ', ',')
        line = line.replace(' ', ',')
        line = line.replace('   ', ',')
        line = line.replace('    ', ',')

        # print(line[:100])
        for elem in line.split(','):
            # print("hi " + elem)
            try:
                float(elem)
            except:
                continue
            data[idx].append(round(float(elem), 2))
    # print(len(data))
    xaxis = [i for i in range(len(data[0]))]
    smart_data = np.array([0.0 for _ in range(len(data[0]))])
    confidence_intervals = np.array([0.0 for _ in range(len(data[0]))])
    for t in range(len(data[0])):
        sum_across_runs = 0.0
        vals = [data[idx][t] for idx in range(len(data))]
        # for data_idx in range(len(data)):
        #     # print(len(data[idx]))
        #     # assert len(data[idx]) == 499999, len(data[data_idx])
        #     sum_across_runs += data[data_idx][t]

        smart_data[t] = sum(vals)/len(data)
        confidence_intervals[t] = 1.96 * np.std(vals)/np.sqrt(len(vals))
    
    # cubic_interploation_model_ci = interp1d(xaxis, confidence_intervals, kind = "cubic")
    # confidence_intervals = cubic_interploation_model_ci(xaxis)
    # cubic_interploation_model_data = interp1d(xaxis, smart_data, kind = "cubic")
    # smart_data = cubic_interploation_model_data(smart_data)
    fig, ax = plt.subplots()
    ax.plot(xaxis, smart_data)
    ax.fill_between(xaxis, (smart_data-confidence_intervals), (smart_data+confidence_intervals), color='b', alpha=.1)
    # print(smart_data[:100])
    # plt.plot(xaxis, smart_data)
    plt.show()

def plot_all(files, labels, time_steps):
    all_data = []
    for file_name in files:
        time_steps = time_steps
        file = open(file_name, 'r')
        data = []
        for idx, line in enumerate(file):
            line = line.replace('[ ', '')
            line = line.replace(' ]', '')
            data.append([])
            line = line.replace('  ', ',')
            line = line.replace(' ', ',')
            line = line.replace('   ', ',')
            line = line.replace('    ', ',')

            # print(line[:100])
            for elem in line.split(','):
                # print("hi " + elem)
                try:
                    float(elem)
                except:
                    continue
                data[idx].append(round(float(elem), 2))
        all_data.append(data)
    
    # print(len(data))
    all_smart_data = np.array([])
    fig, ax = plt.subplots()
    # datas = [len(d) for d in all_data]
    # print(datas)
    for idx, d in enumerate(all_data):
        xaxis = [i for i in range(len(d[idx]))]

        smart_data = np.array([0.0 for _ in range(len(d[0]))])
        confidence_intervals = np.array([0.0 for _ in range(len(d[0]))])
        for t in range(len(d[0])):
            sum_across_runs = 0.0
            vals = [d[idx][t] for idx in range(len(d))]
            # for data_idx in range(len(data)):
            #     # print(len(data[idx]))
            #     # assert len(data[idx]) == 499999, len(data[data_idx])
            #     sum_across_runs += data[data_idx][t]

            smart_data[t] = sum(vals)/len(d)
            confidence_intervals[t] = 1.96 * np.std(vals)/np.sqrt(len(vals))
        # while len(smart_data) != time_steps:
            # smart_data = np.append(smart_data, smart_data[len(smart_data)-1])

        line, = ax.plot(xaxis, smart_data, label=labels[idx])   
        print(line)
        ax.fill_between(xaxis, (smart_data-confidence_intervals), (smart_data+confidence_intervals), color=line.get_color(), alpha=.1)

    plt.ylabel('Average Reward')
    plt.xlabel('Time Step')

    plt.title('Average Reward vs. Time Steps')

    plt.legend()
    plt.savefig('../data/figures/combined_plot.png', dpi=216)
    plt.show()

# rewards_over_time()
# read_np_array()
files = ['../data/total_rewards-500000.txt', '../data/total_rewards_TRPO-500000.txt', '../data/total_rewards_RNOB-500000.txt', '../data/total_rewards_RWB-500000.txt']
labels = ['PPO', 'TRPO', 'REINFORCE', 'REINFORCE w/ Baseline']
plot_all(files, labels, 500000)
# rewards_over_time_confidence('../data/total_rewards_TRPO-500000.txt', 500000)

# rewards_over_time_confidence('../data/total_rewards_RNOB-500000.txt', 500000)
# rewards_over_time_confidence('../data/total_rewards_RWB-500000.txt', 50000)
