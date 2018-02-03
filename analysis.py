import torch
from logger import Logger
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np

values = [1, 2, 3, 4, 5, 6, 7, 8, 16, 32, 64, 128, 256, 512, 1024]
repeats = 1000


def handle_gradients(results, file_name):
    logger = Logger('./analysis_logs/{}'.format(file_name))
    all_magnitudes = {}
    for key in values:
        exp_list = results[key]

        magnitudes = []
        for i in tqdm(range(repeats)):
            magnitude = torch.norm(exp_list[i])
            magnitudes.append(magnitude)
            logger.histo_summary('batchsize {} gradients'.format(key), exp_list[i].numpy(), i)

        all_magnitudes[key] = magnitudes
        logger.histo_summary('all gradients', torch.stack(exp_list).numpy(), key)

    torch.save(all_magnitudes, "./results/magnitudes_{}".format(file_name))


def process_gradients():
    epoch2_results = torch.load("./results/expbs1024_epoch_2_model.pth.result")
    epoch30_results = torch.load("./results/expbs1024_epoch_30_model.pth.result")
    handle_gradients(epoch2_results, "analysis_epoch2")
    handle_gradients(epoch30_results, "analysis_epoch30")


def handle_magnitude(magnitudes, file_name):
    logger = Logger('./analysis_logs/{}'.format(file_name))
    avg_norms = []
    std_norms = []
    old_key = -1
    for key in values:
        assert old_key < key  # make sure it's doing this in order
        mags = np.asarray(magnitudes[key])
        avg_norms.append(np.mean(mags))
        std_norms.append(np.std(mags))
        old_key = key

    for i in range(len(values)):
        key = values[i]
        mags = np.asarray(magnitudes[key])
        normalized_mags = (avg_norms[len(avg_norms)-1]/avg_norms[i])*mags
        logger.histo_summary("distribution of normalized (to batch 1024) gradient magnitudes", normalized_mags, key)


    return avg_norms, std_norms


def analyze_magnitude():
    epoch2_mags = torch.load("./results/magnitudes_analysis_epoch2")
    epoch30_mags = torch.load("./results/magnitudes_analysis_epoch30")

    avg2, std2 = handle_magnitude(epoch2_mags, "norm_dist2")
    avg30, std30 = handle_magnitude(epoch30_mags, "norm_dist30")

    normalized_avg2 = [x * (avg2[14] / avg2[i]) for i, x in enumerate(avg2)]
    normalized_std2 = [x * (avg2[14] / avg2[i]) for i, x in enumerate(std2)]

    normalized_avg30 = [x * (avg30[14] / avg30[i]) for i, x in enumerate(avg30)]
    normalized_std30 = [x * (avg30[14] / avg30[i]) for i, x in enumerate(std30)]

    plt.errorbar(values, avg2, std2, linestyle='None', marker='^', label='Epoch 2')
    plt.errorbar(values, avg30, std30, linestyle='None', marker='^', color='g', label='Epoch 30')
    plt.legend(loc='upper center')
    plt.show()

    plt.errorbar(values, normalized_avg2, normalized_std2, linestyle='None', marker='^', label='Epoch 2')
    plt.errorbar(values, normalized_avg30, normalized_std30, linestyle='None', marker='^', color='g', label='Epoch 30')
    plt.legend(loc='upper left')
    plt.show()

# ============================================== Main =================================================== #
analyze_magnitude()
# process_gradients()

print "done"