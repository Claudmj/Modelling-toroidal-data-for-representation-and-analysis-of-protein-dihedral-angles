import json
import numpy as np

class Utils:
    @staticmethod
    def load_entries(file_name: str):
        with open(file_name, "rt") as file:
            entries = json.load(file)

        return entries

    @staticmethod
    def make_metric_result_dict():
        result_dict = {}

        result_dict['ll'] = []
        result_dict['aic'] = []
        result_dict['bic'] = []
        result_dict['test_ll'] = []
        result_dict['test_aic'] = []
        result_dict['test_bic'] = []
        result_dict['time'] = []

        return result_dict

    @staticmethod
    def add_to_metric_result_dict(result_dict, means):
        result_dict['ll'].append(means[0])
        result_dict['aic'].append(means[1])
        result_dict['bic'].append(means[2])
        result_dict['test_ll'].append(means[3])
        result_dict['test_aic'].append(means[4])
        result_dict['test_bic'].append(means[5])
        result_dict['time'].append(means[6])

        return result_dict

    @staticmethod
    def make_fold_result_dict():
        result_dict = {}

        result_dict['ll'] = []
        result_dict['aic'] = []
        result_dict['bic'] = []
        result_dict['test_ll'] = []
        result_dict['test_aic'] = []
        result_dict['test_bic'] = []
        result_dict['phi_mean'] = []
        result_dict['phi_conc'] = []
        result_dict['psi_mean'] = []
        result_dict['psi_conc'] = []
        result_dict['weights'] = []
        result_dict['corr'] = []
        result_dict['skewness'] = []
        result_dict['time'] = []

        return result_dict

    @staticmethod
    def add_to_fold_result_dict(result_dict, sample_mcmc, time_difference):
        result_dict['ll'].append(sample_mcmc.loglikelihood)
        result_dict['aic'].append(sample_mcmc.aic)
        result_dict['bic'].append(sample_mcmc.bic)
        result_dict['test_ll'].append(sample_mcmc.test_loglikelihood)
        result_dict['test_aic'].append(sample_mcmc.test_aic)
        result_dict['test_bic'].append(sample_mcmc.test_bic)
        result_dict['phi_mean'].append(sample_mcmc.estimated_params['phi_loc'])
        result_dict['phi_conc'].append(sample_mcmc.estimated_params['phi_conc'])
        result_dict['psi_mean'].append(sample_mcmc.estimated_params['psi_loc'])
        result_dict['psi_conc'].append(sample_mcmc.estimated_params['psi_conc'])
        result_dict['weights'].append(sample_mcmc.estimated_params['mix_weights'])
        # if sample_mcmc.estimated_params['corr_scale'] is not None:
        #     result_dict['corr'].append(sample_mcmc.estimated_params['corr_scale'])
        # if sample_mcmc.estimated_params['skewness'] is not None:
        #     result_dict['skewness'].append(sample_mcmc.estimated_params['skewness'])
        result_dict['time'].append(time_difference)

        return result_dict

    @staticmethod
    def summary_stats_result_dict(title, result_dict, output_file_path):
        text = f"{title}\n"
        text += "\n"
        text += f"Fold, ll, aic, bic, test_ll, test_aic, test_bic, phi_mean, phi_conc, psi_mean, psi_conc, weights, time, corr, skewness\n"

        for i in range(len(result_dict['ll'])):
            text += (
                f"{i + 1}, {result_dict['ll'][i]}, {result_dict['aic'][i]}, {result_dict['bic'][i]}, {result_dict['test_ll'][i]},"
                f" {result_dict['test_aic'][i]}, {result_dict['test_bic'][i]}, {result_dict['phi_mean'][i]}, {result_dict['phi_conc'][i]},"
                f" {result_dict['psi_mean'][i]}, {result_dict['psi_conc'][i]}, {result_dict['weights'][i]}, {result_dict['time'][i]}")
            # if result_dict['corr'] is not None:
            #     text += f", {result_dict['corr'][i]}"
            # else:
            #     text += ", "

            # if result_dict['skewness'] is not None:
            #     text += f", {result_dict['skewness'][i]}"
            # else:
            #     text += ", "
            text += "\n"

        series = np.array([result_dict['ll'], result_dict['aic'], result_dict['bic'], result_dict['test_ll'], result_dict['test_aic'], result_dict['test_bic'], result_dict['time']])
        means = np.mean(series, axis=1)
        std = np.std(series, axis=1)

        mean_text = ', '.join(map(str, means)) + "\n"
        std_text = ', '.join(map(str, std)) + "\n"

        text += "\n"
        text += "Kfold stats \n"
        text += f"Statistic, ll, aic, bic, test_ll, test_aic, test_bic, time\n"
        text += f"Mean, {mean_text}"
        text += f"Std, {std_text}"

        with open(output_file_path, "wt") as file:
            file.write(text)

        return means