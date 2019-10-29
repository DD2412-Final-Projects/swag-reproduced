import matplotlib as mpl
import matplotlib.pyplot as plt


if __name__ == "__main__":

    SGD_max_conf = None
    SGD_conf_acc_diff = None

    SWA_max_conf = SGD_max_conf
    SWA_conf_acc_diff = SGD_conf_acc_diff

    SWAG_diag_max_conf = SGD_max_conf
    SWAG_diag_conf_acc_diff = SGD_conf_acc_diff

    SWAG_max_conf = SGD_max_conf
    SWAG_conf_acc_diff = SGD_conf_acc_diff

    ax = plt.axes()

    ax.axhline(linestyle="--", color="b", label="Ideal")  # ideal
    ax.plot(SGD_max_conf, SGD_conf_acc_diff, "-o", color="r", label="SGD")  # SGD
    ax.plot(SWA_max_conf, SWA_conf_acc_diff, "-o", color="g", label="SWA")  # SWA
    ax.plot(SWAG_diag_max_conf, SWAG_diag_conf_acc_diff, "-o", color="y", label="SWAG-Diag")  # SWAG-Diag
    ax.plot(SWAG_max_conf, SWAG_conf_acc_diff, "-o", color="m", label="SWAG")  # SWAG

    plt.title("Reliability diagram")
    plt.xlabel("Confidence (max in bin)")
    plt.ylabel("Confidence - Accuracy (mean in bin)")
    plt.xscale('logit')
    plt.xlim(0.01, 0.9999999)
    ax.xaxis.set_minor_formatter(mpl.ticker.FormatStrFormatter(""))
    ax.tick_params(axis='both', which='major', labelsize=8)
    ax.tick_params(axis='both', which='minor', labelsize=6)

    # Shrink current axis's height by 10% on the bottom
    box = ax.get_position()
    ax.set_position([box.x0, box.y0 + box.height * 0.15,
                     box.width, box.height * 0.85])

    # Put a legend below current axis
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15),
              fancybox=True, shadow=True, ncol=5)
    plt.grid()
    plt.show()
