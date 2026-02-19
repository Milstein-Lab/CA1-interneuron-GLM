from Exploring_BTSP import plot_jeff2, get_W
import pickle
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit
import scipy

plt.rcParams.update({
    # overall defaults
    "font.size": 12,

    # axes titles/labels
    "axes.titlesize": 16,
    "axes.labelsize": 14,

    # tick labels
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,

    # legend
    "legend.fontsize": 16,
    "legend.title_fontsize": 16,

    # figure suptitle (plt.suptitle)
    "figure.titlesize": 18,
})

plt.rcParams.update({
    "legend.fontsize": 9,        # <-- increase this
    "legend.title_fontsize": 9,  # <-- only matters if you set legend title
})



def plot_summary(backwards_experiment_per_hz_dict, backwards_model_per_hz_dict, title=None):
    fig, axs  = plt.subplots(2,4, figsize=(13,8))

    fig.suptitle(title)

    axs_list = [axs[0,0], axs[0,1], axs[1,0], axs[1,1]]

    model_t0_list = []


    apparent_tau_model_list_backwards = []
    apparent_tau_model_list_forwards = []

    # mse_list = []

    for i, hz in enumerate(backwards_experiment_per_hz_dict):
        back_dict = backwards_experiment_per_hz_dict[hz]

        model_dict = backwards_model_per_hz_dict[hz]


        model_at_t0 = model_dict['y_mod'][4]
        model_t0_list.append(model_at_t0)

        print(f"model_dict['x_s'] {model_dict['x_s'][4]}")

        ax = axs_list[i]

        
        label_exp = "Experimental"
        if hz==20:
            ax.plot(back_dict["x_exp_s_backward"], back_dict["y_exp_s_backward"], 'o', color='k', label=label_exp if hz ==20 else None)
            label_tau = f'Exp back (τ={back_dict["TAU_BACK_PAPER"]:.2f}'
            ax.plot(back_dict["xx_b"], back_dict["yy_b"], '-', lw=2.0, color='red', label=label_tau if hz ==20 else None)
            ax.plot(back_dict["xx_f"], back_dict["yy_f"], '-', lw=2.0, color='red', label=f'Exp fwd (τ={back_dict["TAU_FWD_PAPER"]:.2f}s)')

        ax.plot(model_dict["x_s"], model_dict["y_mod"], 'o', label=f"Model Pred. @ {hz}Hz", color='blue')
        ax.plot(model_dict["xx_b"], model_dict["model_backwards_line"], '--', lw=2.0, color='purple', label=f"Model back (τ={model_dict['taumb']:.2f}s)")
        ax.plot(model_dict["xx_f"], model_dict["model_forwards_line"], '--', lw=2.0, color='purple', label=f"Model fwd (τ={model_dict['taumf']:.2f}s)")

        apparent_tau_model_list_backwards.append(model_dict['taumb'])
        apparent_tau_model_list_forwards.append(model_dict['taumf'])


        ax.set_xlabel("Time from plateau (s)")
        ax.set_ylabel("EPSP Amplitude (relative %)") #("ΔW")
        ax.set_ylim(0, 2.50)
        custom_tick_locations = [0.0, 0.5, 1.0, 1.5, 2.0, 2.5]
        custom_labels = ['100', '150', '200', '250', '300', '350']
        ax.set_yticks(custom_tick_locations)
        ax.set_yticklabels(custom_labels)
        # ax.legend(loc="upper left")
        ax.legend(
    loc="upper center",
    bbox_to_anchor=(0.5, -0.28),
    ncol=2,
    frameon=False
)

    model_t0_array = np.array(model_t0_list)
    model_t0_array_scaled = (model_t0_array*100) + 100

    axs[1,2].plot(range(len(model_t0_list)),model_t0_array_scaled, color='k', marker='o')
    axs[1,2].set_ylim(100, np.max(model_t0_array_scaled)+10)
    axs[1,2].set_xticks(range(4), ["10Hz", "20Hz", "40Hz", "100Hz"])
    axs[1,2].set_ylabel("EPSP Amplitude (relative %)")
    axs[1,2].set_title("Weight Change @ dt=0")
    axs[1,2].set_xlabel("Frequency")


    hzs = np.array([10, 20, 40, 100])

    axs[0,2].plot(hzs, apparent_tau_model_list_backwards, '-o', color='k')
    axs[0,2].set_ylabel("Backward Apparent Tau (s)")
    axs[0,2].set_ylim(0, 2)

    axs[0, 3].axis('off')

    axs[1,3].plot(hzs, apparent_tau_model_list_forwards, '-o', color='k')
    axs[1,3].set_ylabel("Forward Apparent Tau (s)")
    axs[1,3].set_ylim(0, 2)

    axs[0,3].set_xlim(left=0)
    axs[1,3].set_xlim(left=0)

    axs[0,3].set_xticks([0, 20, 40, 60, 80, 100])
    axs[1,3].set_xticks([0, 20, 40, 60, 80, 100])
    axs[0,3].set_xlabel("Presynaptic frequency (Hz)")
    axs[1,3].set_xlabel("Presynaptic frequency (Hz)")


    # plt.tight_layout()

    fig.tight_layout(rect=[0., 0.01, 0.96,0.98])  # leave 12% of figure height at bottom
    fig.subplots_adjust(hspace=0.8)          # ↑ increase this to add more vertical gap
    # plt.tight_layout()
    # plt.show()
    
    plt.show()


def plot_summary_ching_lung_data_jeff_model(ching_lung_data, backwards_model_per_hz_dict, title=None):
    fig, axs  = plt.subplots(2,4, figsize=(13,8))

    fig.suptitle(title)

    axs_list = [axs[0,0], axs[0,1], axs[1,0], axs[1,1]]

    model_t0_list = []


    apparent_tau_model_list_backwards = []
    apparent_tau_model_list_forwards = []

    # mse_list = []

    for i, hz in enumerate(backwards_experiment_per_hz_dict):
        # back_dict = backwards_experiment_per_hz_dict[hz]

        model_dict = backwards_model_per_hz_dict[hz]


        model_at_t0 = model_dict['y_mod'][4]
        model_t0_list.append(model_at_t0)

        print(f"model_dict['x_s'] {model_dict['x_s'][4]}")

        ax = axs_list[i]

        
        # label_exp = "Experimental"
        # if hz==20:
        #     ax.plot(back_dict["x_exp_s_backward"], back_dict["y_exp_s_backward"], 'o', color='k', label=label_exp if hz ==20 else None)
        #     label_tau = f'Exp back (τ={back_dict["TAU_BACK_PAPER"]:.2f}'
        #     ax.plot(back_dict["xx_b"], back_dict["yy_b"], '-', lw=2.0, color='red', label=label_tau if hz ==20 else None)
        #     ax.plot(back_dict["xx_f"], back_dict["yy_f"], '-', lw=2.0, color='red', label=f'Exp fwd (τ={back_dict["TAU_FWD_PAPER"]:.2f}s)')

        ax.plot(model_dict["x_s"], model_dict["y_mod"], 'o', label=f"Model Pred. @ {hz}Hz", color='blue')
        ax.plot(model_dict["xx_b"], model_dict["model_backwards_line"], '--', lw=2.0, color='purple', label=f"Model back (τ={model_dict['taumb']:.2f}s)")
        ax.plot(model_dict["xx_f"], model_dict["model_forwards_line"], '--', lw=2.0, color='purple', label=f"Model fwd (τ={model_dict['taumf']:.2f}s)")

        apparent_tau_model_list_backwards.append(model_dict['taumb'])
        apparent_tau_model_list_forwards.append(model_dict['taumf'])

        # mse = np.mean(np.square(back_dict['y_exp_s_backward']-model_dict['y_mod']))

        # mse_list.append(mse)

        # ax.set_title(f"{hz}Hz MSE={mse:.4f}")
        ax.set_xlabel("Time from plateau (s)")
        ax.set_ylabel("EPSP Amplitude (relative %)") #("ΔW")
        ax.set_ylim(0, 2.50)
        custom_tick_locations = [0.0, 0.5, 1.0, 1.5, 2.0, 2.5]
        custom_labels = ['100', '150', '200', '250', '300', '350']
        ax.set_yticks(custom_tick_locations)
        ax.set_yticklabels(custom_labels)
        ax.legend(loc="upper left")

    # axs[0,2].bar(range(len(mse_list)), mse_list)
    # axs[0,2].set_title("MSE Experiment vs. Model")
    # axs[0,2].set_ylabel("MSE")
    # axs[0,2].set_xticks(range(4), ["10Hz", "20Hz", "40Hz", "100Hz"])

    # scaled_vals = [(val*100)+100 for val in model_t0_list]

    model_t0_array = np.array(model_t0_list)

    model_t0_scaled_array = model_t0_array*100 +100

    axs[1,2].plot(range(len(model_t0_list)), model_t0_scaled_array, color='k', marker='o')
    axs[1,2].set_ylim(100, np.max(model_t0_scaled_array))
    axs[1,2].set_xticks(range(4), ["10Hz", "20Hz", "40Hz", "100Hz"])
    # axs[1,2].set_ylabel("Weight Change")
    axs[1,2].set_title("Weight Change @ dt=0")
    axs[1,2].set_xlabel("Frequency")



    # long_range = np.arange(101)
    # hzs=[10, 20, 40, 100]
    # for i, hz in enumerate(hzs):
    #     axs[0,3].plot(long_range[hz], apparent_tau_model_list_backwards[i], marker='o', color='k')
    #     axs[0,3].set_ylabel("Backward Apparent Tau")


    #     axs[1,3].plot(long_range[hz], apparent_tau_model_list_forwards[i], marker='o', color='k')
    #     axs[1,3].set_ylabel("Forward Apparent Tau")

    hzs = np.array([10, 20, 40, 100])

    axs[0,3].plot(hzs, apparent_tau_model_list_backwards, '-o', color='k')
    axs[0,3].set_ylabel("Backward Apparent Tau")
    axs[0,3].set_ylim(0, 1.5)

    axs[1,3].plot(hzs, apparent_tau_model_list_forwards, '-o', color='k')
    axs[1,3].set_ylabel("Forward Apparent Tau")
    axs[1,3].set_ylim(0, 1.5)

    axs[0,3].set_xlim(left=0)
    axs[1,3].set_xlim(left=0)

    axs[0,3].set_xticks([0, 20, 40, 60, 80, 100])
    axs[1,3].set_xticks([0, 20, 40, 60, 80, 100])
    axs[0,3].set_xlabel("Presynaptic frequency (Hz)")
    axs[1,3].set_xlabel("Presynaptic frequency (Hz)")


    plt.tight_layout()
    plt.show()

def multiply_et_is(spike_train, is_pre_conv, tau_ms_et, tau_ms_is, lam_et, lam_is, dt_ms=1.0):

    ET_over_time = np.empty_like(spike_train)
    IS_over_time = np.empty_like(spike_train)
    ET = 0.0
    IS = 0.0
    one_over_tau_et = 1.0 / tau_ms_et
    one_over_tau_is = 1.0 / tau_ms_is
    for t in range(len(spike_train)):
        et_spike = spike_train[t]
        is_spike = is_pre_conv[t]
        dEdt = (-ET + lam_et * et_spike) * one_over_tau_et
        dIdt = (-IS + lam_is * is_spike) * one_over_tau_is       
        ET+= (dEdt * dt_ms)   
        IS+= (dIdt * dt_ms)
        # ET = np.clip(ET, 0.0, 1.0)     
        ET = min(1.0, ET)
        IS = min(1.0, IS)
        ET_over_time[t] = ET
        IS_over_time[t] = IS
    return ET_over_time, IS_over_time

def plot_et_is(x_arr_int, params, title="Jeff Model Fit Jeff Data", plat_len=None):
    et_time_1000_dict = {}

    hz_list = [10, 20, 40, 100]

    for hz_used in hz_list:

        delta_w_list, et_time_1000, IS = get_W(x_arr_int, post_ms=10000, pre_ms=10000, hz_used=hz_used, plateau_length=plat_len,tau_et =params["tau_et"],tau_is = params["tau_is"],lam_et=params["lam_et"],lam_is=params["lam_is"],dt_ms=1.0, eta_ms = params["eta_ms"], plot_intermediates=False, special_case=True)

        et_time_1000_dict[hz_used] = et_time_1000

    # order = [10, 20, 40, 100, "20Hz_full_kernel"]

    fig, axs = plt.subplots(1, 4, figsize=(20,4))

    fig.suptitle(title)

    # pick keys that actually exist, in the desired order, cap at 5
    # keys = [k for k in order if k in et_time_1000_dict][:5]
    keys = hz_list

    pre_ms = 10000
    post_ms = 10000

    T_ms = int(pre_ms + post_ms)
    t_ms = np.arange(-pre_ms, post_ms, dtype=int)

    dw_per_hz_special_0 = []

    for i, k in enumerate(keys):
        if k!="20Hz_full_kernel":
            y = np.asarray(et_time_1000_dict[k], float)   # y-only series is fine
            W = np.trapz(y*IS, dx=1.0)
            dW = params["eta_ms"] * W
            dW_scaled = (dW+1) *100
            dw_per_hz_special_0.append(dW_scaled)
            # ax = axs[i]
            axs[i].plot(t_ms[5000:15000], y[5000:15000], lw=2, label="ET", color='b')
            axs[i].plot(t_ms[5000:15000], IS[5000:15000], label="IS", color='r')
            axs[i].set_title(f"{k}Hz dT(ET-IS)=0s, dW={dW:.3f}")
            axs[i].set_xlabel("Time (ms)")
            axs[i].set_ylabel("A.U.")
            axs[i].legend()


    plt.tight_layout()
    plt.show()





def plot_et_is_all_is_len(x_arr_int, params, title="Jeff Model Fit Jeff Data"):
    et_time_1000_dict = {}
    is_time_dict = {}

    plat_len_list = [50,100,300,500,700]

    for plat_len in plat_len_list:

        delta_w_list, et_time_1000, IS = get_W(x_arr_int, post_ms=10000, pre_ms=10000, hz_used=20, plateau_length=plat_len,tau_et =params["tau_et"],tau_is = params["tau_is"],lam_et=params["lam_et"],lam_is=params["lam_is"],dt_ms=1.0, eta_ms = params["eta_ms"], plot_intermediates=False, special_case=True)

        et_time_1000_dict[plat_len] = et_time_1000

        is_time_dict[plat_len] = IS

    # order = [10, 20, 40, 100, "20Hz_full_kernel"]

    fig, axs = plt.subplots(1, 5, figsize=(20,4))

    fig.suptitle(title)

    # pick keys that actually exist, in the desired order, cap at 5
    # keys = [k for k in order if k in et_time_1000_dict][:5]
    keys = plat_len_list

    pre_ms = 10000
    post_ms = 10000

    T_ms = int(pre_ms + post_ms)
    t_ms = np.arange(-pre_ms, post_ms, dtype=int)

    dw_per_hz_special_0 = []

    mask = (t_ms >= -5000) & (t_ms <= 8000)


    for i, k in enumerate(keys):
        IS = is_time_dict[k]
        y = np.asarray(et_time_1000_dict[k], float)   # y-only series is fine
        W = np.trapz(y*IS, dx=1.0)
        dW = params["eta_ms"] * W
        dW_scaled = (dW+1) *100
        dw_per_hz_special_0.append(dW_scaled)
        # ax = axs[i]
        # axs[i].plot(t_ms, np.roll(y, -3000), lw=2, color='b') #[5000:15000] #[5000:20000]
        # axs[i].plot(t_ms, y, lw=2, label="ET", color='b') #[2000:20000] #[5000:20000]
        # axs[i].plot(t_ms, np.roll(y, 3000), lw=2, color='b') #[8000:20000] #[5000:20000]

        axs[i].plot(t_ms[mask], np.roll(y, -3000)[mask], lw=2, color='b')
        axs[i].plot(t_ms[mask], y[mask],               lw=2, label="ET", color='b')
        axs[i].plot(t_ms[mask], np.roll(y,  3000)[mask], lw=2, color='b')
        axs[i].plot(t_ms[mask], IS[mask], label="IS", color='r')


        # axs[i].plot(t_ms[5000:15000], IS[5000:15000], label="IS", color='r')
        axs[i].set_title(f"{k}ms Plateau Len. \n dW={dW:.3f} @ dT=0s")
        axs[i].set_xlabel("Time (ms)")
        axs[i].set_ylabel("A.U.")
        axs[i].legend()


    plt.tight_layout()
    plt.show()






def exp_with_fixed_c(x, A, tau, c_fixed):
    x = np.asarray(x, float)
    return float(c_fixed) + A*np.exp(x/float(tau))

def exp_c0(x, A, tau):
    return exp_with_fixed_c(x, A, tau, c_fixed=0.0)

def exp_back_c0(x, A, tau):   # for x <= 0
    return A*np.exp(x/tau)

def exp_fwd_c0(x, A, tau):    # for x >= 0
    return A*np.exp(-x/tau)



def plot_model_kernel_grid_multi(params, plat_lens=(50,100,300,500,700),
                                 x_min=-4000, x_max=4000, step=100, flip_x=True):
    model_x = np.arange(x_min, x_max + step, step).astype(int)
    x_plot = -model_x if flip_x else model_x

    plt.figure(figsize=(7, 5))

    for plat_len in plat_lens:
        delta_w_list, _, _ = get_W(
            model_x,
            post_ms=10000, pre_ms=10000,
            hz_used=20,
            plateau_length=plat_len,
            tau_et=params["tau_et"], tau_is=params["tau_is"],
            lam_et=params["lam_et"], lam_is=params["lam_is"],
            dt_ms=1.0, eta_ms=params["eta_ms"],
            plot_intermediates=False
        )

        dw = np.asarray(delta_w_list, float)
        if dw.shape[0] != model_x.shape[0]:
            dw = dw[:model_x.shape[0]]

        y = dw * 100 + 100
        plt.plot(x_plot, y, marker='o', lw=1.2, label=f"IS={plat_len}ms")

    plt.axvline(0, lw=1, alpha=0.6)
    plt.xlabel("time from plateau (ms)")
    plt.ylabel("EPSP Amplitude (Relative %)")
    plt.title("Model Kernel on Fixed Δt Grid")
    plt.legend()
    plt.tight_layout()
    plt.show()



def comparing_is_size_kernel(params):

    plt.figure()

    model_x_int = np.arange(0, 4000, 100)

    plat_len_list = [50, 100, 300, 500, 700]

    for plat_len in plat_len_list:

       
        delta_w_list, et_time_1000, IS = get_W(np.array(model_x_int),
                post_ms=10000, pre_ms=10000,
                hz_used=20,               
                plateau_length=plat_len,
                tau_et=params["tau_et"], tau_is=params["tau_is"],
                lam_et=params["lam_et"], lam_is=params["lam_is"],
                dt_ms=1.0, eta_ms=params["eta_ms"],
                plot_intermediates=False
            )
        # plt.plot([val*-1 for val in model_x_int], [val*100 +100 for val in delta_w_list], 'o', color='b', label='Model')

        plt.plot(model_x_int, [val*100 +100 for val in delta_w_list], 'o', label=f'IS={plat_len}ms')

        delta_w_list, et_time_1000, IS = get_W(np.array(model_x_int),
                post_ms=10000, pre_ms=10000,
                hz_used=20,               
                plateau_length=plat_len,
                tau_et=params["tau_et"], tau_is=params["tau_is"],
                lam_et=params["lam_et"], lam_is=params["lam_is"],
                dt_ms=1.0, eta_ms=params["eta_ms"],
                plot_intermediates=False
            )

        plt.plot(model_x_int*-1, [val*100 +100 for val in delta_w_list], 'o', label=f'IS={plat_len}ms')

    plt.legend()
    plt.tight_layout()
    plt.show()


def spike_train_by_rate(rate_hz=20, n=10, t0_ms=0.0, jitter_ms=0.0, min_isi_ms=0.0,
                        rng=None, rounding="nearest"):
    """
    Generate n integer spike times (ms) for a constant firing rate.
      - rate_hz: spikes/s  (ISI = 1000 / rate_hz ms)
      - t0_ms: start time (ms)
      - jitter_ms: Gaussian jitter std (ms)
      - min_isi_ms: enforce minimum ISI AFTER rounding (ms)
      - rounding: 'nearest' | 'floor' | 'ceil'
    Returns: np.ndarray[int64] of shape (n,)
    """
    if rate_hz <= 0:
        raise ValueError("rate_hz must be > 0")
    rng = np.random.default_rng() if rng is None else rng

    isi_ms = 1000.0 / float(rate_hz)
    times_f = t0_ms + np.arange(n, dtype=float) * isi_ms
    if jitter_ms > 0:
        times_f = times_f + rng.normal(0.0, jitter_ms, size=n)

    if rounding == "nearest":
        times = np.rint(times_f)
    elif rounding == "floor":
        times = np.floor(times_f)
    elif rounding == "ceil":
        times = np.ceil(times_f)
    else:
        raise ValueError("rounding must be 'nearest', 'floor', or 'ceil'")
    times = times.astype(np.int64)

    min_isi_int = int(np.ceil(min_isi_ms))
    if min_isi_int > 0:
        for i in range(1, n):
            min_allowed = times[i-1] + min_isi_int
            if times[i] < min_allowed:
                times[i] = min_allowed

    return times

def get_W_new(x_arr_int, post_ms=10000, pre_ms=10000, hz_used=10,
              plateau_length=300, tau_et=1500., tau_is=500.,
              lam_et=200., lam_is=4, dt_ms=1.0, eta_ms=0.0001,
              plot_intermediates=True):

    spike_times = spike_train_by_rate(rate_hz=hz_used, n=10, t0_ms=0).astype(int)
    mid_spike = spike_times[4]

    T_ms = int(pre_ms + post_ms)
    t_ms = np.arange(-pre_ms, post_ms, dtype=int)
    center_idx = pre_ms

    # align mid_spike to t=0 (plateau start/center reference)
    offset = center_idx - mid_spike
    spike_idx = offset + spike_times

    spike_train = np.zeros(T_ms)
    valid = (spike_idx >= 0) & (spike_idx < T_ms)
    spike_train[spike_idx[valid]] = 1.0

    is_pre_conv = np.zeros(T_ms)
    is_pre_conv[center_idx:center_idx + plateau_length] = 1.0

    ET, IS = multiply_et_is(spike_train, is_pre_conv, tau_et, tau_is, lam_et, lam_is, dt_ms=dt_ms)

    def shift_zeros(a, shift):
        out = np.zeros_like(a)
        if shift > 0:
            out[shift:] = a[:-shift]
        elif shift < 0:
            out[:shift] = a[-shift:]
        else:
            out[:] = a
        return out

    delta_w_list = []
    for dt in x_arr_int:
        shift = int(round(dt / dt_ms))   # safe if dt_ms != 1
        ET_shifted = shift_zeros(ET, shift)
        W = np.trapz(ET_shifted * IS, dx=dt_ms)
        delta_w_list.append(eta_ms * W)

    # pick something sensible to return for ET_to_plot
    ET_to_plot = ET

    if plot_intermediates:
        plt.plot(t_ms, spike_train, label="pre spikes")
        plt.plot(t_ms, is_pre_conv, label="plateau")
        plt.legend(); plt.show()

        plt.plot(t_ms, ET, label="ET")
        plt.plot(t_ms, IS, label="IS")
        plt.legend(); plt.show()

    return delta_w_list, ET_to_plot, IS

def comparing_is_size_kernel_two_sided(params):
    plt.figure()

    model_x = np.arange(-4000, 4000 + 100, 100).astype(int)   # two-sided grid
    plat_len_list = [50, 100, 300, 500, 700]

    for plat_len in plat_len_list:
        delta_w_list, _, _ = get_W_new(
            model_x,
            post_ms=10000, pre_ms=10000,
            hz_used=20,
            plateau_length=plat_len,
            tau_et=params["tau_et"], tau_is=params["tau_is"],
            lam_et=params["lam_et"], lam_is=params["lam_is"],
            dt_ms=1.0, eta_ms=params["eta_ms"],
            plot_intermediates=False
        )

        dw = np.asarray(delta_w_list, float)

        # safety if get_W returns 1 extra point sometimes
        if dw.shape[0] != model_x.shape[0]:
            dw = dw[:model_x.shape[0]]

        y = dw * 100 + 100
        plt.plot(model_x, y, 'o-', lw=1.2, label=f'IS={plat_len}ms')

    plt.axvline(0, lw=1, alpha=0.6)
    plt.xlabel("time from plateau (ms)")
    plt.ylabel("EPSP Amplitude (Relative %)")
    plt.legend()
    plt.tight_layout()
    plt.show()
        
#         ax.set_xlabel('time from plateau (ms)')
#         ax.set_ylabel('EPSP Amplitude (Relative %)')


#             print(f"np.array(delta_w_list) for the yyyyyyyykkkkkkk len(np.array(delta_w_list)) {len(np.array(delta_w_list))} {np.array(delta_w_list)}")

#             yb_mod = np.array(delta_w_list)[:-1][xk <= 0]
#             yf_mod = np.array(delta_w_list)[:-1][xk >= 0]

#             p0b = [yb_mod.max(), 1000.0]
#             (A_b_mod, tau_b_mod), _ = curve_fit(exp_back_c0, xb, yb_mod, p0=p0b,
#                                         bounds=([-np.inf, 1e-6], [np.inf, np.inf]),
#                                         maxfev=20000)
#             xx_b = np.linspace(xb.min(), 0.0, 400)
#             yy_b = exp_back_c0(xx_b, A_b_mod, tau_b_mod)

#             # fit forward (x >= 0)
#             p0f = [yf_mod.max(), 1000.0]
#             (A_f_mod, tau_f_mod), _ = curve_fit(exp_fwd_c0, xf, yf_mod, p0=p0f,
#                                         bounds=([-np.inf, 1e-6], [np.inf, np.inf]),
#                                         maxfev=20000)
#             xx_f = np.linspace(0.0, xf.max(), 400)
#             yy_f = exp_fwd_c0(xx_f, A_f_mod, tau_f_mod)

#             # ax.plot(xk*-1, (yk*100)+100, 'o', color='k', label='Exp.')
#             ax.plot(xx_b*-1, (yy_b*100)+100, '-', color='purple', label=f'Tau Model Fwd={tau_b_mod/1000:.2f}s', linestyle='--')
#             ax.plot(xx_f*-1, (yy_f*100)+100, '-', color='purple', label=f'Tau Model Bwd={tau_f_mod/1000:.2f}s', linestyle='--')
#             ax.legend()

#             model_apparent_taus_dict["20Hzfull_fwd"] = tau_b_mod/1000
#             fwd_tau_model_dict["20Hzfull_fwd"] = tau_b_mod/1000
#             model_apparent_taus_dict["20Hzfull_back"] = tau_f_mod/1000


            
#             dw_0_dict_experiment[key] = yy_b[-1]
#             dw_0_dict_model[key] = delta_w_list[-1]

#             print(f"len(delta_w_list) 20 {len(delta_w_list)} len(yy_b) Tau Exp Fwd={tau_b/1000:.2f} {len(yk)}")
#             print(f"delta_w_list[:-1] {delta_w_list[:-1]} yk {yk}")

#             mse = np.mean(np.square(delta_w_list[:-1] - yk))
#             mse_dict[key] = mse

#             ax.set_title(f"20 Hz Full Kernel \n MSE={mse:.4f}")

#             ax.set_ylim(95, 400)

#             ax.legend(
#     loc="upper center",
#     bbox_to_anchor=(0.5, -0.28),
#     ncol=2,
#     frameon=False
# )






















def data_over_experiment(cleaned_data_dict, params, title=None, plat_len=None):
    dw_0_dict_experiment = {}
    dw_0_dict_model = {}

    mse_dict = {}

    tau_experiment_dict = {}
    model_apparent_taus_dict = {}

    fwd_tau_model_dict = {}

    fig, axs = plt.subplots(2,4, figsize=(15,6))

    fig.suptitle(title, y=0.98)

    key_list = [axs[0,0], axs[0,1], axs[0,2], axs[0,3], axs[1,0], axs[1,1], axs[1,2], axs[1,3]]


    for i, key in enumerate(cleaned_data_dict):

        ax = key_list[i]

        if key == '20Hz_full_kernel':


            xk = np.asarray(cleaned_data_dict['20Hz_full_kernel']['x'], float)
            yk = np.asarray(cleaned_data_dict['20Hz_full_kernel']['y'], float)

            xb, yb = xk[xk <= 0], yk[xk <= 0]   # backward arm
            xf, yf = xk[xk >= 0], yk[xk >= 0]   # forward arm

            p0b = [yb.max(), 1000.0]
            (A_b, tau_b), _ = curve_fit(exp_back_c0, xb, yb, p0=p0b,
                                        bounds=([-np.inf, 1e-6], [np.inf, np.inf]),
                                        maxfev=20000)
            xx_b = np.linspace(xb.min(), 0.0, 400)
            yy_b = exp_back_c0(xx_b, A_b, tau_b)

            # fit forward (x >= 0)
            p0f = [yf.max(), 1000.0]
            (A_f, tau_f), _ = curve_fit(exp_fwd_c0, xf, yf, p0=p0f,
                                        bounds=([-np.inf, 1e-6], [np.inf, np.inf]),
                                        maxfev=20000)
            xx_f = np.linspace(0.0, xf.max(), 400)
            yy_f = exp_fwd_c0(xx_f, A_f, tau_f)

            ax.plot(xk*-1, (yk*100)+100, 'o', color='k', label='Exp.')
            ax.plot(xx_b*-1, (yy_b*100)+100, '-', color='r', label=f'Tau Exp Fwd={tau_b/1000:.2f}s')
            ax.plot(xx_f*-1, (yy_f*100)+100, '-', color='r', label=f'Tau Exp Bwd={tau_f/1000:.2f}s')

            tau_experiment_dict["20Hzfull_fwd"] = tau_b
            tau_experiment_dict["20Hzfull_back"] = tau_f
            

            model_x = xk.tolist()
            model_x.append(0)
            model_x_int = [int(val) for val in model_x]
            delta_w_list, et_time_1000, IS = get_W(np.array(model_x_int),
                    post_ms=10000, pre_ms=10000,
                    hz_used=20,               
                    plateau_length=plat_len,
                    tau_et=params["tau_et"], tau_is=params["tau_is"],
                    lam_et=params["lam_et"], lam_is=params["lam_is"],
                    dt_ms=1.0, eta_ms=params["eta_ms"],
                    plot_intermediates=False
                )
            ax.plot([val*-1 for val in model_x_int], [val*100 +100 for val in delta_w_list], 'o', color='b', label='Model')
            
            ax.set_xlabel('time from plateau (ms)')
            ax.set_ylabel('EPSP Amplitude (Relative %)')


            print(f"np.array(delta_w_list) for the yyyyyyyykkkkkkk len(np.array(delta_w_list)) {len(np.array(delta_w_list))} {np.array(delta_w_list)}")

            yb_mod = np.array(delta_w_list)[:-1][xk <= 0]
            yf_mod = np.array(delta_w_list)[:-1][xk >= 0]

            p0b = [yb_mod.max(), 1000.0]
            (A_b_mod, tau_b_mod), _ = curve_fit(exp_back_c0, xb, yb_mod, p0=p0b,
                                        bounds=([-np.inf, 1e-6], [np.inf, np.inf]),
                                        maxfev=20000)
            xx_b = np.linspace(xb.min(), 0.0, 400)
            yy_b = exp_back_c0(xx_b, A_b_mod, tau_b_mod)

            # fit forward (x >= 0)
            p0f = [yf_mod.max(), 1000.0]
            (A_f_mod, tau_f_mod), _ = curve_fit(exp_fwd_c0, xf, yf_mod, p0=p0f,
                                        bounds=([-np.inf, 1e-6], [np.inf, np.inf]),
                                        maxfev=20000)
            xx_f = np.linspace(0.0, xf.max(), 400)
            yy_f = exp_fwd_c0(xx_f, A_f_mod, tau_f_mod)

            # ax.plot(xk*-1, (yk*100)+100, 'o', color='k', label='Exp.')
            ax.plot(xx_b*-1, (yy_b*100)+100, '-', color='purple', label=f'Tau Model Fwd={tau_b_mod/1000:.2f}s', linestyle='--')
            ax.plot(xx_f*-1, (yy_f*100)+100, '-', color='purple', label=f'Tau Model Bwd={tau_f_mod/1000:.2f}s', linestyle='--')
            ax.legend()

            model_apparent_taus_dict["20Hzfull_fwd"] = tau_b_mod/1000
            fwd_tau_model_dict["20Hzfull_fwd"] = tau_b_mod/1000
            model_apparent_taus_dict["20Hzfull_back"] = tau_f_mod/1000


            
            dw_0_dict_experiment[key] = yy_b[-1]
            dw_0_dict_model[key] = delta_w_list[-1]

            print(f"len(delta_w_list) 20 {len(delta_w_list)} len(yy_b) Tau Exp Fwd={tau_b/1000:.2f} {len(yk)}")
            print(f"delta_w_list[:-1] {delta_w_list[:-1]} yk {yk}")

            mse = np.mean(np.square(delta_w_list[:-1] - yk))
            mse_dict[key] = mse

            ax.set_title(f"20 Hz Full Kernel \n MSE={mse:.4f}")

            ax.set_ylim(95, 400)

            ax.legend(
    loc="upper center",
    bbox_to_anchor=(0.5, -0.28),
    ncol=2,
    frameon=False
)



            

            
            
        else:

            x = np.asarray(cleaned_data_dict[key]['x'] * -1, float)  
            y = np.asarray(cleaned_data_dict[key]['y'], float)

            p0 = [y.max(), 1000.0]                       
            bounds = ([-np.inf, 1e-6], [np.inf, np.inf]) 

            [A_fit, tau_fit], pcov = curve_fit(exp_c0, x, y, p0=p0, bounds=bounds, maxfev=20000)

            xx = np.linspace(x.min(), 0, 400)
            yy = exp_c0(xx, A_fit, tau_fit)

            model_x = list(cleaned_data_dict[key]['x'])

            model_x.append(0)

            print(f"model_x {model_x}")

            delta_w_list, et_time_1000, IS = get_W(model_x,
                    post_ms=10000, pre_ms=10000,
                    hz_used=key,               # <-- only this changes
                    plateau_length=plat_len,
                    tau_et=params["tau_et"], tau_is=params["tau_is"],
                    lam_et=params["lam_et"], lam_is=params["lam_is"],
                    dt_ms=1.0, eta_ms=params["eta_ms"],
                    plot_intermediates=False
                )
            
            print(f"delta_w_list {delta_w_list}")
            print(f"y {y}")
            print(f"len(delta_w_list) {len(delta_w_list)} len(y) {len(y)}")
                    
            # axs[i].plot(cleaned_data_dict[key]['x'] * -1, delta_w_list, 'o', label='Model', color='b')

            x_list = list(x)
            x_list.append(0.)

            ax.plot(x_list, [(dw*100 )+100 for dw in delta_w_list], 'o', label='Model', color='b')

            p0 = [np.array(delta_w_list).max(), 1000.0]  
            model_x_array = np.array(model_x)*-1  
            (A_model, tau_model), _ = curve_fit(exp_c0, model_x_array, delta_w_list, p0=p0,
                                        bounds=([-np.inf, 1e-6], [np.inf, np.inf]),
                                        maxfev=20000)
            xx_mod = np.linspace(model_x_array.min(), 0, 400)
            yy_mod = exp_c0(xx_mod, A_model, tau_model)
            yy_mod = np.array(yy_mod)
            ax.plot(xx_mod, yy_mod*100 +100, '-', label=f'Tau Model.={tau_model/1000:.2f}s', color='purple', linestyle='--')

            model_apparent_taus_dict[key] = tau_model/1000

            print(f"cleaned_data_dict[20Hz_full_kernel].keys() {cleaned_data_dict['20Hz_full_kernel']['x']}")

            if 0 not in cleaned_data_dict[key]['x']:
                model_x.append(0)
                model_x.append(-200)
                model_x.append(-600)
                model_x.append(-1700)
            else:
                model_x.append(-200)
                model_x.append(-600)
                model_x.append(-1700)


            mse = np.mean(np.square(delta_w_list[:-1]-y))
            mse_dict[key] = mse

            y_plot = (y*100) +100
            yy_plot = (yy*100) +100
        
            ax.plot(x, y_plot, 'o', label='Exp.', color='k')
            tau_experiment_dict[key] = tau_fit
            ax.plot(xx, yy_plot, '-', label=f'Tau Exp.={tau_fit/1000:.2f}s', color='r')

            ax.set_xlabel('time from plateau (ms)')
            ax.set_ylabel('EPSP Amplitude (Relative %)')
            ax.set_title(f"{key} Hz \n MSE={mse:.4f}")
#             ax.legend(
#     loc="upper center",
#     bbox_to_anchor=(0.5, -0.28),   # (x, y) in axes coords; y<0 pushes it below
#     ncol=2,                        # adjust columns to fit
#     frameon=False
# )

            dw_0_dict_experiment[key] = yy[-1]
            dw_0_dict_model[key] = delta_w_list[-1]

            ax.set_ylim(95, 400)






############################ fitting forward just for numbers purposes ########################



            xk = np.asarray(cleaned_data_dict['20Hz_full_kernel']['x'], float)
            # yk = np.asarray(cleaned_data_dict['20Hz_full_kernel']['y'], float)

            xb = xk[xk <= 0]
            xf = xk[xk >= 0]


            model_x = xk.tolist()
            model_x.append(0)
            model_x_int = [int(val) for val in model_x]
            delta_w_list, et_time_1000, IS = get_W(np.array(model_x_int),
                    post_ms=10000, pre_ms=10000,
                    hz_used=key,               # <-- only this changes
                    plateau_length=plat_len,
                    tau_et=params["tau_et"], tau_is=params["tau_is"],
                    lam_et=params["lam_et"], lam_is=params["lam_is"],
                    dt_ms=1.0, eta_ms=params["eta_ms"],
                    plot_intermediates=False
                )

            yb_mod = np.array(delta_w_list)[:-1][xk <= 0]

            xb = xk[xk <= 0]

            p0b = [yb_mod.max(), 1000.0]
            (A_b_mod, tau_b_mod), _ = curve_fit(exp_back_c0, xb, yb_mod, p0=p0b,
                                        bounds=([-np.inf, 1e-6], [np.inf, np.inf]),
                                        maxfev=20000)
            xx_b = np.linspace(xb.min(), 0.0, 400)
            yy_b = exp_back_c0(xx_b, A_b_mod, tau_b_mod)

            # # fit forward (x >= 0)
            # p0f = [yf_mod.max(), 1000.0]
            # xf = xk[xk >= 0]
            # (A_f_mod, tau_f_mod), _ = curve_fit(exp_fwd_c0, xf, yf_mod, p0=p0f,
            #                             bounds=([-np.inf, 1e-6], [np.inf, np.inf]),
            #                             maxfev=20000)
            # xx_f = np.linspace(0.0, xf.max(), 400)
            # yy_f = exp_fwd_c0(xx_f, A_f_mod, tau_f_mod)
            

            # ax.plot(xk*-1, (yk*100)+100, 'o', color='k', label='Exp.')
            ax.plot(xb*-1, (yb_mod*100)+100, 'o', color='b', label='Model Fwd. Pred')
            ax.plot(xx_b*-1, (yy_b*100)+100, '-', color='purple', label=f'Tau Model Fwd={tau_b_mod/1000:.2f}s', linestyle='--')
            # ax.plot(xx_f*-1, (yy_f*100)+100, '-', color='purple', label=f'Tau Model Bwd={tau_f_mod/1000:.2f}s', linestyle='--')
            # ax.legend()
            ax.legend(
    loc="upper center",
    bbox_to_anchor=(0.5, -0.28),
    ncol=2,
    frameon=False
)

            fwd_tau_model_dict[key] = tau_b_mod/1000

        


    ax =  key_list[i+1]

    # --- Model points (blue) ---

    model_data_list = []

    model_labeled = False
    for i, key in enumerate(dw_0_dict_model):
        if key == "20Hz_full_kernel":
            continue

        value = (dw_0_dict_model[key] * 100) + 100
        ax.plot(
            i, value, 'o',
            color='b',
            label="Model" if not model_labeled else None
        )
        model_labeled = True


        model_data_list.append(value)

    ax.plot(model_data_list, color='b')


    # --- Experimental extrapolated from tau (red) + full kernel (magenta) ---
    exp_labeled = False
    kernel_labeled = False

    exp_list = []

    for i, key in enumerate(dw_0_dict_experiment):
        value = (dw_0_dict_experiment[key] * 100) + 100

        if key == "20Hz_full_kernel":
            ax.plot(
                1.1, value, 'o',
                color='magenta',
                label="20Hz Full Kernel Model" if not kernel_labeled else None
            )
            kernel_labeled = True
        else:
            ax.plot(
                i, value, 'o',
                color='r',
                label="Experimental Extrapolated" if not exp_labeled else None
            )
            exp_labeled = True
            exp_list.append(value)

    ax.plot(exp_list, color='r')

    # --- x-axis formatting ---
    xticks = list(range(len(dw_0_dict_model) - 1))  # excludes 20Hz_full_kernel
    xlabels = [f'{k} Hz' for k in dw_0_dict_model.keys() if k != "20Hz_full_kernel"]

    ax.set_xticks(xticks, xlabels)
    ax.set_xlabel("Presynaptic Frequency")
    ax.set_ylabel("dW (%EPSP) @ t=0")
    ax.set_ylim(100,400)

    ax.legend()


    ax =  key_list[i+2]




    model_labeled = False
    kernel_labeled = False

    fwd_model_preds = []

    # # Build x positions for non-kernel keys so ticks/points align
    keys = [k for k in fwd_tau_model_dict.keys() if k != "20Hz_full_kernel"]
    xpos = {k: i for i, k in enumerate(keys)}

    for key, value in fwd_tau_model_dict.items():
        if key == "20Hzfull_fwd":
            # ax.plot(
            #     1.1, value, 'o',  # or pick a specific x you want
            #     color='magenta',
            #     label="20Hz Full Kernel Model" if not kernel_labeled else None
            # )
            kernel_labeled = True
            continue

        ax.plot(
            xpos[key], value, 'o',
            color='b',
            label="Fwd Model Prediction" if not model_labeled else None
        )
        model_labeled = True

        fwd_model_preds.append(value)

    ax.set_xticks(range(len(keys)), [f"{k} Hz" for k in keys])
    ax.set_xlabel("Presynaptic Frequency")
    ax.set_ylabel("Apparent Tau (s)")
    ax.set_title("Apparent Fwd. Tau")
    ax.plot(fwd_model_preds, color='b')
    ax.set_ylim(0,2)
    ax.legend()


    ax =  key_list[i+3]



    full_keys = {"20Hzfull_fwd", "20Hzfull_back"}
    freq_keys = [k for k in tau_experiment_dict.keys() if k not in full_keys]  # [10,20,40,100]
    xpos = {k: i for i, k in enumerate(freq_keys)}

    # model_labeled = False
    # kernel_fwd_labeled = False
    # kernel_back_labeled = False

    model_labeled = False
    kernel_fwd_labeled = False
    kernel_back_labeled = False
    kernel_fwd_model_labeled = False
    kernel_back_model_labeled = False

    back_x = []
    back_exp_y = []
    back_mod_y = []

    for key, tau_ms in tau_experiment_dict.items():
        value = tau_ms / 1000.0  # ms -> s

        if key in full_keys:
            base_x = xpos[20]

            # small jitter so both points are visible
            x = base_x + (-0.1) #if key == "20Hzfull_fwd" else 0.08)

            if key == "20Hzfull_fwd":
                # key_list[i+2].plot(
                #     x, model_apparent_taus_dict[key], 'o',
                #     # color='green',
                #     label="Full Kernel Fwd. Model" if not kernel_fwd_model_labeled else None
                # )
                kernel_fwd_model_labeled = True
            elif key == "20Hzfull_back":
                ax.plot(
                    0.9, model_apparent_taus_dict[key], 'o',
                    color='blue',
                    label="Full Kernel Back. Model" if not kernel_back_model_labeled else None
                )
                kernel_back_model_labeled = True

            if key == "20Hzfull_fwd":
                label = "Full Kernel Fwd. Experiment" if not kernel_fwd_labeled else None
                kernel_fwd_labeled = True
                color="orange"

                key_list[i+2].plot(base_x+0.1, value, 'o', color=color, label=label)
                key_list[i+2].legend()

                # ax.plot(xpos[20], model_apparent_taus_dict[key], 'o',color='green', label="Full Kernel Fwd. Model" if not model_labeled else None)
            else:
                label = "Full Kernel Back. Experiment" if not kernel_back_labeled else None
                kernel_back_labeled = True
                color="orange"

                ax.plot(base_x+0.1, value, 'o', color=color, label=label)

                # ax.plot(xpos[20], model_apparent_taus_dict[key], 'o',color='orchid', label="Full Kernel Back. Model" if not model_labeled else None)

            continue
        
        x = xpos[key]
        back_x.append(x)
        back_exp_y.append(value)
        back_mod_y.append(model_apparent_taus_dict[key])


        ax.plot(
            xpos[key], value,
            color='red',
            label="Backwards Arm Experiment" if not model_labeled else None
        )
        ax.plot(
            xpos[key], model_apparent_taus_dict[key], 
            color='purple',
            label="Backwards Arm Model" if not model_labeled else None
        )
        model_labeled = True

    order = np.argsort(back_x)
    bx = np.array(back_x)[order]
    by_exp = np.array(back_exp_y)[order]
    by_mod = np.array(back_mod_y)[order]

    ax.plot(bx, by_exp, marker='o', color='red')
    ax.plot(bx, by_mod, marker='o', color='purple')

    ax.set_title("Apparent Bwd.Tau")

    ax.set_xticks(range(len(freq_keys)), [f"{k} Hz" for k in freq_keys])
    ax.set_xlabel("Presynaptic Frequency")
    ax.set_ylabel("Apparent Tau (s)")
    ax.set_ylim(0, 2.)
    ax.legend()
    

    fig.tight_layout(rect=[0., 0.01, 1.0,1.08])  # leave 12% of figure height at bottom
    fig.subplots_adjust(hspace=0.9)          # ↑ increase this to add more vertical gap
    # plt.tight_layout()
    plt.show()






def main():
    # if which_one == "jeff_data_jeff_model":
    save_path = "/Users/michaelfinch/CA1-interneuron-GLM/ching_lung/jeffs_data_csv.pkl"
    with open(save_path, 'rb') as f:
        jeffs_data_dict = pickle.load(f)

    plot_full_intermediates = False
    jeffs_params = {
        "tau_et": 1307.3706852728226,
        "tau_is": 517.0085321387273,
        "lam_et": 119.87951416286887,
        "lam_is": 49.984596145977214,
        "eta_ms": 0.003982527270809444
    }

    params_CL =   {"tau_et": 1474.7515259682878,
    "tau_is": 739.2145185588507,
    "lam_et": 161.24381107148224,
    "lam_is": 39.57778784397287,
    "eta_ms": 0.002976991280586468}

    x_arr_int = jeffs_data_dict[20]['x']
    plot_et_is_all_is_len(x_arr_int, params_CL, title="Simple Model Trained on Your Experimental Data")

    comparing_is_size_kernel_two_sided(params_CL)
    # plot_model_kernel_grid_multi(params_CL, plat_lens=(50,100,300,500,700), x_min=-4000, x_max=4000, step=100, flip_x=True)




    backwards_experiment_per_hz_dict, backwards_model_per_hz_dict = plot_jeff2(
        jeffs_params, jeffs_data_dict, plot_full_intermediates
    )

    plot_summary(backwards_experiment_per_hz_dict, backwards_model_per_hz_dict, title="Simple Model Trained on Science Data")

    # you had hz + params undefined here; keep minimal but make it run:
    hz = 20
    x_arr_int = jeffs_data_dict[hz]['x']
    plot_et_is(x_arr_int, jeffs_params, title="Simple Model Trained on Science Data", plat_len=300)


    save_path = "/Users/michaelfinch/CA1-interneuron-GLM/ching_lung/pickle_of_all_experimental_data.pkl"
    with open(save_path, 'rb') as f:
        cleaned_data_dict = pickle.load(f)


    data_over_experiment(cleaned_data_dict, jeffs_params, title="Simple Model Trained on Science Data Predicts Your Experimental Data", plat_len=300)

    plot_et_is(x_arr_int, params_CL, title="Simple Model Trained on Your Experimental Data", plat_len=300)

    data_over_experiment(cleaned_data_dict, params_CL, title="Simple Model Trained on Your Experimental Data Predicts Your Experimental Data", plat_len=300)


    


    # plateau_length_dose_response_list = [50, 100, 300, 500, 700]

    # for plat_length in plateau_length_dose_response_list:

    #     plot_et_is(x_arr_int, jeffs_params, title=f"Simple Model Trained on Your Experimental Data Predicts Your Experimental Data Plateau Length = {plat_length}ms", plat_len=plat_length)

    #     data_over_experiment(cleaned_data_dict, params_CL, title=f"Simple Model Trained on Your Experimental Data Predicts Your Experimental Data Plateau Length = {plat_length}ms", plat_len=plat_length)

        


    # plot_summary_ching_lung_data_jeff_model(ching_lung_data, backwards_model_per_hz_dict, title="Jeff Model Plot Over Ching Lungs Labs Data")

    # elif which_one == "ching_lung_data_jeff_model":

    #     save_path = "/Users/michaelfinch/CA1-interneuron-GLM/ching_lung/pickle_of_all_experimental_data.pkl"
    #     with open(save_path, 'rb') as f:
    #         cleaned_data_dict = pickle.load(f)

    # else:
    #     raise ValueError(f"Unknown which_one={which_one}")

if __name__ == "__main__":
    main()



# if __name__ == "__main__":

#     if which_one=="jeff_data_jeff_model":
#         save_path = "/Users/michaelfinch/CA1-interneuron-GLM/ching_lung/jeffs_data_csv.pkl"
#         with open(save_path, 'rb') as f:
#             jeffs_data_dict = pickle.load(f)

#         plot_full_intermediates=False
#         jeffs_params = {
#         "tau_et": 1307.3706852728226,
#         "tau_is": 517.0085321387273,
#         "lam_et": 119.87951416286887,
#         "lam_is": 49.984596145977214,
#         "eta_ms": 0.003982527270809444}


#         backwards_experiment_per_hz_dict, backwards_model_per_hz_dict = plot_jeff2(jeffs_params, jeffs_data_dict, plot_full_intermediates)

    
#         plot_summary(backwards_experiment_per_hz_dict, backwards_model_per_hz_dict)


#         x_arr_int = jeffs_data_dict[hz]['x']
#         plot_et_is(x_arr_int, jeffs_params)

    



