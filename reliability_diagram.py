import matplotlib as mpl
import matplotlib.pyplot as plt


if __name__ == "__main__":

    SGD_max_conf = [0.692955493927002, 0.928031861782074, 0.9869927167892456, 0.9979840517044067, 0.9996472597122192, 0.9999258518218994, 0.9999830722808838, 0.9999967813491821, 0.9999995231628418, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
    SGD_conf_acc_diff = [0.09352872249484062, 0.34402215814590453, 0.44052620565891265, 0.358126888513565, 0.26105671846866607, 0.1718211196660996, 0.1239618804454804, 0.10199133419990536, 0.09999856138229368, 0.07999979686737058, 0.014000000000000012, 0.014000000000000012, 0.02400000000000002, 0.02400000000000002, 0.020000000000000018, 0.014000000000000012, 0.020000000000000018, 0.02400000000000002, 0.02400000000000002, 0.02200000000000002]

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
