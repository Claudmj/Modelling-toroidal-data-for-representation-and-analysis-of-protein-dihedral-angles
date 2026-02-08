import matplotlib.pyplot as plt
import seaborn as sns

class Plots:
    @staticmethod
    def contour_plot(x, y, title, x_label, y_label):
        sns.jointplot(x=x, y=y, kind='kde')
        plt.grid()
        plt.title(title)
        plt.xlabel(x_label)
        plt.ylabel(y_label)
        plt.show()


    @staticmethod
    def scatter_plot(x, y, title, x_label, y_label):
        sns.jointplot(x=x, y=y, kind='scatter')
        plt.grid()
        plt.title(title)
        plt.xlabel(x_label)
        plt.ylabel(y_label)
        plt.show()


    @staticmethod
    def scatter_and_fitted(x, y, title, x_label, y_label, mu, nu, responsibilities):
        labels = responsibilities.argmax(axis=1)
        plt.scatter(x=x, y=y, c=labels, alpha=0.5)
        plt.scatter(mu, nu, marker='+', c="r")
        plt.grid()
        plt.title(title)
        plt.xlabel(x_label)
        plt.ylabel(y_label)
        plt.show()

    @staticmethod
    def elbow_plot_single(times, aic, bic, save_path):
        fig, ax = plt.subplots()

        # First y-axis: Time
        ax.grid()
        time_plot, = ax.plot(times, label="Time", marker='o', color='blue')
        ax.set_ylabel("Time (s)", color="black")
        ax.tick_params(axis='y', labelcolor="black")

        # Second y-axis: AIC and BIC
        ax2 = ax.twinx()
        aic_plot, = ax2.plot(aic, label="AIC", marker='o', color='green')
        bic_plot, = ax2.plot(bic, label="BIC", marker='o', color='red')
        ax2.set_ylabel("AIC and BIC", color="black")
        ax2.tick_params(axis='y', labelcolor="black")

        # Formatting
        ax.set_xlim(-1, 9)
        ax.set_xticks(range(-1, 9))
        ax.set_xticklabels(range(1, 11))
        ax.set_xlabel("Number of components")

        # Combine legends from both axes
        lines = ax.lines + ax2.lines
        labels = [line.get_label() for line in lines]
        ax.legend(lines, labels, loc="upper left")

        # Layout and save
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()


    @staticmethod
    def contour_and_fitted(x, y, title, x_label, y_label, mu, nu, responsibilities):
        labels = responsibilities.argmax(axis=1)
        sns.kdeplot(x=x, y=y, c=labels, alpha=0.5)
        plt.scatter(mu, nu, marker='+', c="r")
        plt.grid()
        plt.title(title)
        plt.xlabel(x_label)
        plt.ylabel(y_label)
        plt.show()

    @staticmethod
    def elbow_plot(times, aic, bic, distributions, save_path):
        fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=False)

        for i, distribution in enumerate(distributions):
            ax = axes[i]
            ax.grid()
            time_plot, = ax.plot(times[i], label="Time", marker='o', color='blue')
            ax.set_ylabel("Time (s)", color="black")
            ax.tick_params(axis='y', labelcolor="black")

            ax2 = ax.twinx()
            aic_plot, = ax2.plot(aic[i], label="AIC", marker='o', color='green')
            bic_plot, = ax2.plot(bic[i], label="BIC", marker='o', color='red')
            ax2.set_ylabel("AIC and BIC", color="black")
            ax2.tick_params(axis='y', labelcolor="black")

            ax.set_title(f"{distribution}")
            ax.set_xlim(-1, 9)
            ax.set_xticks(range(-1, 9))
            ax.set_xticklabels(range(1, 11))
            ax.set_xlabel("Number of components")

            lines = ax.lines + ax2.lines
            labels = [line.get_label() for line in lines]
            ax.legend(lines, labels, loc="upper left")

        plt.tight_layout()
        plt.savefig(save_path)

    @staticmethod
    def elbow_plot_deprecated(times, loglikelihoods, distributions, title):
        fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=False)

        for i, (time, loglikelihood) in enumerate(zip(times, loglikelihoods)):
            ax = axes[i]
            ax.grid()
            time_plot, = ax.plot(time, label="Time", marker='o', color='blue')
            ax.set_ylabel("Time", color="black")
            ax.tick_params(axis='y', labelcolor="black")

            ax2 = ax.twinx()
            loglikelihood_plot, = ax2.plot(loglikelihood, label="Log Likelihood", marker='s', color='red')
            ax2.set_ylabel("Log Likelihood", color="black")
            ax2.tick_params(axis='y', labelcolor="black")

            ax.set_title(f"{distributions[i]}")
            ax.set_xlim(1, 10)
            ax.set_xticks(range(1, 11))
            ax.set_xlabel("Number of components")

            lines = [time_plot, loglikelihood_plot]
            labels = [line.get_label() for line in lines]
            ax.legend(lines, labels, loc="upper left")

        plt.tight_layout()
        plt.show()