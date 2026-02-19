import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import pandas as pd
import pickle
from scipy.optimize import curve_fit
import scipy

def str_true_false_to_bool(s):
    """
    Accepts 'true' or 'false' (any case, with spaces). 
    Returns True/False. Raises ValueError otherwise.
    """
    if isinstance(s, bool):
        return s
    if not isinstance(s, str):
        raise ValueError("Expected a string 'true' or 'false'")
    t = s.strip().lower()
    if t == "true":
        return True
    if t == "false":
        return False
    raise ValueError(f"Unrecognized boolean string: {s!r}")



def plot_ching_lung(context, params):

    plot_full_intermediates = str_true_false_to_bool(context.plot_full_intermediates)
    special_case = str_true_false_to_bool(context.special_case)

    loss, products_dict, et_time_1000_dict, IS = objective(context.cleaned_data_dict, params, export=True, plot=plot_full_intermediates, special_case=special_case)

    print(f"context.cleaned_data_dict {context.cleaned_data_dict}")

    order = [10, 20, 40, 100, "20Hz_full_kernel"]

    fig, axs = plt.subplots(1, 4, figsize=(15,3))

    # pick keys that actually exist, in the desired order, cap at 5
    keys = [k for k in order if k in et_time_1000_dict][:5]

    pre_ms = 10000
    post_ms = 10000

    T_ms = int(pre_ms + post_ms)
    t_ms = np.arange(-pre_ms, post_ms, dtype=int)

    dw_per_hz_special_0 = []

    for i, k in enumerate(keys):
        print(f"k {k}")
        if k!="20Hz_full_kernel":
            y = np.asarray(et_time_1000_dict[k], float)   # y-only series is fine
            W = np.trapz(y*IS, dx=1.0)
            dW = params["eta_ms"] * W
            dW_scaled = (dW+1) *100
            dw_per_hz_special_0.append(dW_scaled)
            # ax = axs[i]
            axs[i].plot(t_ms[5000:15000], y[5000:15000], lw=2, label="ET")
            axs[i].plot(t_ms[5000:15000], IS[5000:15000], label="IS")
            axs[i].set_title(f"{k} Hz dT(ET to IS)=0.0s, dW={dW:.3f}")
            axs[i].set_xlabel("Time (ms)")
            axs[i].set_ylabel("A.U.")
            axs[i].legend()

    # # hide the 6th (unused) panel to match the 5-panel layout
    # axs[5].axis('off')

    plt.tight_layout()
    plt.show()

    

    plt.figure(figsize=(5.2, 3.2))
    plt.plot(dw_per_hz_special_0, marker='o', lw=2)
    plt.xticks(range(len(dw_per_hz_special_0)), ["10Hz", "20Hz", "40Hz", "100Hz"])
    plt.xlabel('Stimulation Frequency Model Fit to Fig 2D')
    plt.ylabel('EPSP Normalized (relative %)')
    plt.ylim(100, 350)
    plt.tight_layout()
    plt.show()


    
    # taus   = [1.44, 1.75, 1.80, 1.03]        # seconds
    # titles = ("10 Hz", "20 Hz", "40 Hz", "100 Hz")

    max_dict = plot_fixed_data(context, products_dict, fixed_c=0.00)

    order_keys = [10, 20, 40, 100, '20 forward']
    labels = ["10Hz", "20Hz Backward", "40Hz", "100Hz", "20Hz Forward"]

    # Pull in that order and convert to %
    max_values_pct = [max_dict[k] * 100.0 for k in order_keys]

    plt.figure()
    plt.plot(max_values_pct, marker='o', color='k', linestyle="None")
    plt.plot(max_values_pct, color='r')
    plt.ylabel("Max Potentiation %")
    plt.ylim(100, 250)  # same as your original style
    plt.xticks(np.arange(len(labels)), labels, rotation=0)
    plt.tight_layout()
    plt.show()


    print(f"products_dict.keys {products_dict.keys} products_dict {products_dict}")

    ### c=1 search over tau

    print(f"context.cleaned_data_dict.keys() {context.cleaned_data_dict.keys()}")


    results = fit_and_plot_btsp_paper_style_from_cleaned(
    cleaned=context.cleaned_data_dict,
    products_dict=products_dict,
    titles=("10 Hz","40 Hz","20 Hz (merged)","100 Hz"),
    panels=(10, 40, 20, 100),

    tau_mode_exp="fit",
    tau_mode_model="fit",
    tau_bounds_exp={10:(1.1,1.9), 40:(1.4,2.3), 20:(1.3,2.2), 100:(0.8,1.3)},
    tau_bounds_model={10:(1.1,1.9), 40:(1.4,2.3), 20:(1.3,2.2), 100:(0.8,1.3)},
    n_grid=600,

    c_mode="fixed",
    c_value=0.00,          # match fixed_c

    exclude_window_s=0.0,  # match trimming
    weight_half_life=None, # disable weights to match plain LS
    xlim=(-4.5, 0.5),
    y_max_map={10:2.0, 40:2.2, 20:2.7, 100:1.7},
)


    tau_b, tau_f = fit_and_plot_full_kernel_paper_style(
        cleaned=context.cleaned_data_dict,
        label="20Hz_full_kernel",
        title="20 Hz (full kernel)",
        tau_bounds_back=(1.2,1.7),
        tau_bounds_fwd=(0.55,0.95),
        exclude_window_s=0.0,
        weight_half_life=0.5,
        n_grid=600
    )

    print("Full-kernel τ (paper-style): backward=", tau_b, " forward=", tau_f)




def plot_ching_lung2(cleaned_data_dict, params, plot_full_intermediates=False, special_case=True):

    # plot_full_intermediates = str_true_false_to_bool(context.plot_full_intermediates)
    # special_case = str_true_false_to_bool(context.special_case)

    loss, products_dict, et_time_1000_dict, IS = objective(cleaned_data_dict, params, export=True, plot=plot_full_intermediates, special_case=special_case)

    # print(f"context.cleaned_data_dict {context.cleaned_data_dict}")

    order = [10, 20, 40, 100, "20Hz_full_kernel"]

    fig, axs = plt.subplots(1, 4, figsize=(15,3))

    # pick keys that actually exist, in the desired order, cap at 5
    keys = [k for k in order if k in et_time_1000_dict][:5]

    pre_ms = 10000
    post_ms = 10000

    T_ms = int(pre_ms + post_ms)
    t_ms = np.arange(-pre_ms, post_ms, dtype=int)

    dw_per_hz_special_0 = []

    for i, k in enumerate(keys):
        print(f"k {k}")
        if k!="20Hz_full_kernel":
            y = np.asarray(et_time_1000_dict[k], float)   # y-only series is fine
            W = np.trapz(y*IS, dx=1.0)
            dW = params["eta_ms"] * W
            dW_scaled = (dW+1) *100
            dw_per_hz_special_0.append(dW_scaled)
            # ax = axs[i]
            axs[i].plot(t_ms[5000:15000], y[5000:15000], lw=2, label="ET")
            axs[i].plot(t_ms[5000:15000], IS[5000:15000], label="IS")
            axs[i].set_title(f"{k} Hz dT(ET to IS)=0.0s, dW={dW:.3f}")
            axs[i].set_xlabel("Time (ms)")
            axs[i].set_ylabel("A.U.")
            axs[i].legend()

    # # hide the 6th (unused) panel to match the 5-panel layout
    # axs[5].axis('off')

    plt.tight_layout()
    plt.show()

    

    plt.figure(figsize=(5.2, 3.2))
    plt.plot(dw_per_hz_special_0, marker='o', lw=2)
    plt.xticks(range(len(dw_per_hz_special_0)), ["10Hz", "20Hz", "40Hz", "100Hz"])
    plt.xlabel('Stimulation Frequency Model Fit to Fig 2D')
    plt.ylabel('EPSP Normalized (relative %)')
    plt.ylim(100, 350)
    plt.tight_layout()
    plt.show()


    
    # taus   = [1.44, 1.75, 1.80, 1.03]        # seconds
    # titles = ("10 Hz", "20 Hz", "40 Hz", "100 Hz")

    max_dict = plot_fixed_data(cleaned_data_dict, products_dict, fixed_c=0.00)

    order_keys = [10, 20, 40, 100, '20 forward']
    labels = ["10Hz", "20Hz Backward", "40Hz", "100Hz", "20Hz Forward"]

    # Pull in that order and convert to %
    max_values_pct = [max_dict[k] * 100.0 for k in order_keys]

    plt.figure()
    plt.plot(max_values_pct, marker='o', color='k', linestyle="None")
    plt.plot(max_values_pct, color='r')
    plt.ylabel("Max Potentiation %")
    plt.ylim(100, 250)  # same as your original style
    plt.xticks(np.arange(len(labels)), labels, rotation=0)
    plt.tight_layout()
    plt.show()


    print(f"products_dict.keys {products_dict.keys} products_dict {products_dict}")

    ### c=1 search over tau


    results = fit_and_plot_btsp_paper_style_from_cleaned(
    cleaned=cleaned_data_dict,
    products_dict=products_dict,
    titles=("10 Hz","40 Hz","20 Hz (merged)","100 Hz"),
    panels=(10, 40, 20, 100),

    tau_mode_exp="fit",
    tau_mode_model="fit",
    tau_bounds_exp={10:(1.1,1.9), 40:(1.4,2.3), 20:(1.3,2.2), 100:(0.8,1.3)},
    tau_bounds_model={10:(1.1,1.9), 40:(1.4,2.3), 20:(1.3,2.2), 100:(0.8,1.3)},
    n_grid=600,

    c_mode="fixed",
    c_value=0.00,          # match fixed_c

    exclude_window_s=0.0,  # match trimming
    weight_half_life=None, # disable weights to match plain LS
    xlim=(-4.5, 0.5),
    y_max_map={10:2.0, 40:2.2, 20:2.7, 100:1.7},
)


    tau_b, tau_f = fit_and_plot_full_kernel_paper_style(
        cleaned=cleaned_data_dict,
        label="20Hz_full_kernel",
        title="20 Hz (full kernel)",
        tau_bounds_back=(1.2,1.7),
        tau_bounds_fwd=(0.55,0.95),
        exclude_window_s=0.0,
        weight_half_life=0.5,
        n_grid=600
    )

    print("Full-kernel τ (paper-style): backward=", tau_b, " forward=", tau_f)






def plot_jeff(context, params):

    # ---- FALSE path: test the model at 10/20/40/100 on Jeff's 20Hz x-grid ----
    plot_full_intermediates = str_true_false_to_bool(context.plot_full_intermediates)

    hz_tested_list = [10, 20, 40, 100]

    # Reuse the same x time grid (ms) for ALL test frequencies
    x_20 = context.jeffs_data_dict[20]['x']          # <-- single grid
    y_20_exp = context.jeffs_data_dict[20]['y']      # experimental only exists for 20 Hz

    products_dict_jeff = {}
    et_by_hz = {}

    for hz_used in hz_tested_list:
        delta_w_list, et_time_1000, IS = get_W(
            x_20,
            post_ms=10000, pre_ms=10000,
            hz_used=hz_used,               # <-- only this changes
            plateau_length=300,
            tau_et=params["tau_et"], tau_is=params["tau_is"],
            lam_et=params["lam_et"], lam_is=params["lam_is"],
            dt_ms=1.0, eta_ms=params["eta_ms"],
            plot_intermediates=False
        )
        print(f"delta_w_list (hz={hz_used}): {delta_w_list}")

        products_dict_jeff[hz_used] = {
            "x": x_20,
            "y": np.array(delta_w_list),
            "params_dict": params,
        }
        et_by_hz[hz_used] = et_time_1000  # optional: keep ET if you want to inspect it

    # --- Plot: model predictions for all; overlay experimental points only for 20 Hz ---
    fig, axs = plt.subplots(2, 2, figsize=(9, 6), sharex=True, sharey=True)
    axs = axs.flat

    # --- Plot + tau fits with c=0 for back/fwd, model vs experimental (20 Hz only) ---
    fig, axs = plt.subplots(2, 2, figsize=(10, 7), sharex=True, sharey=True)
    axs = axs.flat

    def split_back_fwd(x_s, y):
        """Split at 0 s: backward (x<=0), forward (x>=0). Returns sorted splits."""
        x_s = np.asarray(x_s, float); y = np.asarray(y, float)
        m = np.isfinite(x_s) & np.isfinite(y)
        x_s, y = x_s[m], y[m]
        # sort by time
        o = np.argsort(x_s); x_s, y = x_s[o], y[o]
        mb = x_s <= 0
        mf = x_s >= 0
        xb, yb = x_s[mb], y[mb]
        xf, yf = x_s[mf], y[mf]
        return xb, yb, xf, yf

    def fit_back_fwd_with_c0(xb, yb, xf, yf):
        """Fit c=0. Forward is fit in z=-x so it’s an increasing exponential."""
        tau_b = tau_f = np.nan
        A_b = A_f = np.nan
        # backward
        if (yb > 0).sum() >= 2:
            A_b, tau_b = fit_exp_fixed_c(xb, yb, c_fixed=0.0)
        # forward (fit in z = -x for positive slope)
        if (yf > 0).sum() >= 2:
            zf = -xf
            A_f, tau_f = fit_exp_fixed_c(zf, yf, c_fixed=0.0)
        return (A_b, tau_b), (A_f, tau_f)
    
    # Toggle this ON to use paper taus for experimental curves; OFF to fit them
    FORCE_PAPER_TAU = True
    TAU_BACK_PAPER  = 1.31  # s
    TAU_FWD_PAPER   = 0.69  # s

    def _apex_value(x_s, y):
        x_s = np.asarray(x_s, float); y = np.asarray(y, float)
        m = np.isfinite(x_s) & np.isfinite(y)
        x_s, y = x_s[m], y[m]
        o = np.argsort(x_s); x_s, y = x_s[o], y[o]
        if np.any(x_s == 0.0):
            return float(y[x_s == 0.0][0])
        return float(np.interp(0.0, x_s, y))
    

    ymod_list = []

    for i, hz in enumerate(hz_tested_list):
        ax = axs[i]
        x_ms = np.asarray(products_dict_jeff[hz]["x"], float)
        y_mod = np.asarray(products_dict_jeff[hz]["y"], float)

        # seconds from plateau (negative = backward)
        x_s = -x_ms / 1000.0

        # MODEL points
        ax.plot(x_s, y_mod, 'o', label="Model", color='tab:blue')

        print(f"y_mod {y_mod}")
        ymod_list.append(y_mod[4])

        # MODEL fits (c=0), back & fwd
        xb_m, yb_m, xf_m, yf_m = split_back_fwd(x_s, y_mod)
        (Amb, taumb), (Amf, taumf) = fit_back_fwd_with_c0(xb_m, yb_m, xf_m, yf_m)

        if np.isfinite(taumb):
            xx = np.linspace(xb_m.min(), xb_m.max(), 400)
            ax.plot(xx, exp_with_fixed_c(xx, Amb, taumb, c_fixed=0.0),
                    '--', lw=2.0, color='purple', label=f"Model back (τ={taumb:.2f}s)")
        if np.isfinite(taumf):
            xx = np.linspace(xf_m.min(), xf_m.max(), 400)
            ax.plot(xx, exp_with_fixed_c(-xx, Amf, taumf, c_fixed=0.0),
                    '--', lw=2.0, color='purple', label=f"Model fwd (τ={taumf:.2f}s)")

        # EXPERIMENTAL points (from 20 Hz dataset)
        x_exp_s  = -x_20 / 1000.0
        y_exp_dw = y_20_exp
        ax.plot(x_exp_s, y_exp_dw, 'o', color='k', label="Experimental")

        # EXPERIMENTAL curves
        xb_e, yb_e, xf_e, yf_e = split_back_fwd(x_exp_s, y_exp_dw)
        if FORCE_PAPER_TAU:
            # Anchor both arms at the experimental apex at x=0
            y0 = _apex_value(x_exp_s, y_exp_dw)

            if xb_e.size:
                xx_b = np.linspace(xb_e.min(), 0.0, 400)
                yy_b = y0 * np.exp(xx_b / TAU_BACK_PAPER)
                ax.plot(xx_b, yy_b, '-', lw=2.0, color='red',
                        label=f'Exp back (τ={TAU_BACK_PAPER:.2f}s)')

            if xf_e.size:
                xx_f = np.linspace(0.0, xf_e.max(), 400)
                yy_f = y0 * np.exp(-xx_f / TAU_FWD_PAPER)
                ax.plot(xx_f, yy_f, '-', lw=2.0, color='red',
                        label=f'Exp fwd (τ={TAU_FWD_PAPER:.2f}s)')
        else:
            (Aeb, taueb), (Aef, tauef) = fit_back_fwd_with_c0(xb_e, yb_e, xf_e, yf_e)
            if np.isfinite(taueb):
                xx = np.linspace(xb_e.min(), xb_e.max(), 400)
                ax.plot(xx, exp_with_fixed_c(xx, Aeb, taueb, c_fixed=0.0),
                        '-', lw=2.0, color='red', label=f"Exp back (τ={taueb:.2f}s)")
            if np.isfinite(tauef):
                xx = np.linspace(xf_e.min(), xf_e.max(), 400)
                ax.plot(xx, exp_with_fixed_c(-xx, Aef, tauef, c_fixed=0.0),
                        '-', lw=2.0, color='red', label=f"Exp fwd (τ={tauef:.2f}s)")

        ax.set_title(f"{hz} Hz Pre Stim Simulated Model")
        ax.set_xlabel("Time from plateau (s)")
        ax.set_ylabel("EPSP Amplitude (relative %)") #("ΔW")
        ax.set_ylim(0, 2.50)
        custom_tick_locations = [0.0, 0.5, 1.0, 1.5, 2.0, 2.5]
        custom_labels = ['100', '150', '200', '250', '300', '350']
        ax.set_yticks(custom_tick_locations)
        ax.set_yticklabels(custom_labels)
        ax.legend(fontsize=7, loc="upper left")

    plt.tight_layout()
    plt.show()

    plt.figure()
    plt.plot(ymod_list, marker="o")
    plt.xlabel("Frequencies of Presynaptic Stimulation - Model")
    plt.ylabel("% LTP at 0ms from Plateau")
    plt.xticks(np.arange(4), ["10Hz", "20Hz", "40Hz", "100Hz"])
    plt.ylim(100, 350)
    plt.title("Model Fit to 2017 Kernel Imposed Presynaptic Spiking @ Diff Freq.")
    plt.show()


    # loss, products_dict_jeff, _, _ = objective(context.jeffs_data_dict, params, export=True, plot=plot_full_intermediates)

    # print(f"products_dict_jeff {products_dict_jeff}")



    # tau_backward_model, tau_forward_model = plot_full_kernel_bidirectional(context.jeffs_data_dict, products_dict_jeff,taus_bidirectional,label=20,title="20 Hz (full kernel)")
    # plot_full_kernel_bidirectional(context.jeffs_data_dict, products_dict_jeff, label=20, title="20 Hz (full kernel - 2017)", c_fixed=0.0, tau_bounds_back=(0.6, 2.5), tau_bounds_fwd=(0.4, 1.5), n_grid=600, align_model="none",color_model="blue",color_fit_exp="red", color_fit_model="purple")

    # build products_dict_* for all Hz (your code already does this)
    # products_dict_jeff[hz] = {"x": x_arr_int, "y": np.array(delta_w_list), "params_dict": params}



    tau_results, max_value_list = plot_kernels_grid(
        context.jeffs_data_dict,          # cleaned (experiment)
        products_dict_jeff,               # model output you computed
        hz_list=hz_tested_list,
        c_fixed=0.0,                      # or 1.0 if plotting normalized EPSP
        tau_bounds_back=(0.6, 2.5),
        tau_bounds_fwd=(0.4, 1.5),
        n_grid=600,
        align_model="none"                # or "lsq" if you want visual alignment
    )

    plt.figure()
    plt.plot(max_value_list, marker='o', color='k', linestyle="None")
    plt.plot(max_value_list, color='r')
    plt.ylabel("Max Potentiation %")
    plt.ylim(175,225)
    plt.xticks(np.arange(4), ["10Hz", "20Hz", "40Hz", "100Hz"])
    plt.show()

   


def plot_jeff2(params, jeffs_data_dict, plot_full_intermediates, handin_hz=20):

    # ---- FALSE path: test the model at 10/20/40/100 on Jeff's 20Hz x-grid ----
    plot_full_intermediates = str_true_false_to_bool(plot_full_intermediates)

    hz_tested_list = [10, 20, 40, 100]

    


    products_dict_jeff = {}
    et_by_hz = {}

    x_20 = jeffs_data_dict[handin_hz]['x']          # <-- single grid
    y_20_exp = jeffs_data_dict[handin_hz]['y']      # experimental only exists for 20 Hz

    for hz_used in hz_tested_list:
            
    #     if actually_jeff:
    #         x_20 = jeffs_data_dict[20]['x']          # <-- single grid
    #         y_20_exp = jeffs_data_dict[20]['y']      # experimental only exists for 20 Hz

    #     else:
    #         x_20 = jeffs_data_dict[hz_used]['x']          # <-- single grid
    #         y_20_exp = jeffs_data_dict[hz_used]['y']      # experimental only exists for 20 Hz


        delta_w_list, et_time_1000, IS = get_W(
            x_20,
            post_ms=10000, pre_ms=10000,
            hz_used=hz_used,               # <-- only this changes
            plateau_length=300,
            tau_et=params["tau_et"], tau_is=params["tau_is"],
            lam_et=params["lam_et"], lam_is=params["lam_is"],
            dt_ms=1.0, eta_ms=params["eta_ms"],
            plot_intermediates=False
        )
        print(f"delta_w_list (hz={hz_used}): {delta_w_list}")

        products_dict_jeff[hz_used] = {
            "x": x_20,
            "y": np.array(delta_w_list),
            "params_dict": params,
        }
        et_by_hz[hz_used] = et_time_1000  # optional: keep ET if you want to inspect it


    def split_back_fwd(x_s, y):
        """Split at 0 s: backward (x<=0), forward (x>=0). Returns sorted splits."""
        x_s = np.asarray(x_s, float); y = np.asarray(y, float)
        m = np.isfinite(x_s) & np.isfinite(y)
        x_s, y = x_s[m], y[m]
        # sort by time
        o = np.argsort(x_s); x_s, y = x_s[o], y[o]
        mb = x_s <= 0
        mf = x_s >= 0
        xb, yb = x_s[mb], y[mb]
        xf, yf = x_s[mf], y[mf]
        return xb, yb, xf, yf

    def fit_back_fwd_with_c0(xb, yb, xf, yf):
        """Fit c=0. Forward is fit in z=-x so it’s an increasing exponential."""
        tau_b = tau_f = np.nan
        A_b = A_f = np.nan
        # backward
        if (yb > 0).sum() >= 2:
            A_b, tau_b = fit_exp_fixed_c(xb, yb, c_fixed=0.0)
        # forward (fit in z = -x for positive slope)
        if (yf > 0).sum() >= 2:
            zf = -xf
            A_f, tau_f = fit_exp_fixed_c(zf, yf, c_fixed=0.0)
        return (A_b, tau_b), (A_f, tau_f)
    
    # Toggle this ON to use paper taus for experimental curves; OFF to fit them
    FORCE_PAPER_TAU = True
    TAU_BACK_PAPER  = 1.31  # s
    TAU_FWD_PAPER   = 0.69  # s

    def _apex_value(x_s, y):
        x_s = np.asarray(x_s, float); y = np.asarray(y, float)
        m = np.isfinite(x_s) & np.isfinite(y)
        x_s, y = x_s[m], y[m]
        o = np.argsort(x_s); x_s, y = x_s[o], y[o]
        if np.any(x_s == 0.0):
            return float(y[x_s == 0.0][0])
        return float(np.interp(0.0, x_s, y))
    



    # --- Plot: model predictions for all; overlay experimental points only for 20 Hz ---
    fig, axs = plt.subplots(2, 2, figsize=(9, 6), sharex=True, sharey=True)
    axs = axs.flat

    # # --- Plot + tau fits with c=0 for back/fwd, model vs experimental (20 Hz only) ---
    # fig, axs = plt.subplots(2, 2, figsize=(10, 7), sharex=True, sharey=True)
    # axs = axs.flat

    

    ymod_list = []

    backwards_model_per_hz_dict = {}
    backwards_experiment_per_hz_dict = {}

    for i, hz in enumerate(hz_tested_list):
        ax = axs[i]
        x_ms = np.asarray(products_dict_jeff[hz]["x"], float)
        y_mod = np.asarray(products_dict_jeff[hz]["y"], float)

        # seconds from plateau (negative = backward)
        x_s = -x_ms / 1000.0

        # MODEL points
        ax.plot(x_s, y_mod, 'o', label="Model", color='tab:blue')

        print(f"y_mod {y_mod}")
        ymod_list.append(y_mod[4])

        # MODEL fits (c=0), back & fwd
        xb_m, yb_m, xf_m, yf_m = split_back_fwd(x_s, y_mod)
        # (Amb, taumb), (Amf, taumf) = fit_back_fwd_with_c0(xb_m, yb_m, xf_m, yf_m)



        p0b = [yb_m.max(), 1000.0]
        (A_b, tau_b), _ = curve_fit(exp_back_c0, xb_m, yb_m, p0=p0b,
                                    bounds=([-np.inf, 1e-6], [np.inf, np.inf]),
                                    maxfev=20000)
        xx_b = np.linspace(xb_m.min(), 0.0, 400)
        yy_b = exp_back_c0(xx_b, A_b, tau_b)

        # fit forward (x >= 0)
        p0f = [yf_m.max(), 1000.0]
        (A_f, tau_f), _ = curve_fit(exp_fwd_c0, xf_m, yf_m, p0=p0f,
                                    bounds=([-np.inf, 1e-6], [np.inf, np.inf]),
                                    maxfev=20000)
        xx_f = np.linspace(0.0, xf_m.max(), 400)
        yy_f = exp_fwd_c0(xx_f, A_f, tau_f)

        ax.plot(x_ms*-1, (y_mod*100)+100, 'o', color='k', label='Exp.')
        ax.plot(xx_b*-1, (yy_b*100)+100, '-', color='r', label=f'Tau Exp Fwd={tau_b/1000:.2f}s')
        ax.plot(xx_f*-1, (yy_f*100)+100, '-', color='r', label=f'Tau Exp Bwd={tau_f/1000:.2f}s')




        # [A_alt, tau_alt], _ = scipy.optimize.curve_fit(exp_with_fixed_c, x, y, p0=[1., -1000.])


        
        


        # if np.isfinite(taumb):
        #     xx_b = np.linspace(xb_m.min(), xb_m.max(), 400)
        #     model_backwards_line = exp_with_fixed_c(xx_b, Amb, taumb, c_fixed=0.0)
        #     ax.plot(xx_b, model_backwards_line,
        #             '--', lw=2.0, color='purple', label=f"Model back (τ={taumb:.2f}s)")
        # if np.isfinite(taumf):
        #     xx_f = np.linspace(xf_m.min(), xf_m.max(), 400)
        #     model_forwards_line = exp_with_fixed_c(-xx_f, Amf, taumf, c_fixed=0.0)
        #     ax.plot(xx_f, model_forwards_line,
        #             '--', lw=2.0, color='purple', label=f"Model fwd (τ={taumf:.2f}s)")
        # else:
        #     xx_f=0
        #     model_forwards_line=0
        #     taumf=0
            

        model_backwards_dict = {"xx_f":xx_f,
                                "xx_b":xx_b,
                                "model_backwards_line":yy_b,
                                "model_forwards_line":yy_f,
                                "Amb":A_b,
                                "Amf":A_f,
                                "taumb":tau_b,
                                "taumf":tau_f,
                                "x_s":x_s,
                                "y_mod":y_mod}
        

        backwards_model_per_hz_dict[hz] = model_backwards_dict

        # EXPERIMENTAL points (from 20 Hz dataset)
        x_exp_s  = -x_20 / 1000.0
        y_exp_dw = y_20_exp
        
        
        x_exp_s_backward = x_exp_s
        y_exp_s_backward = y_exp_dw



        ax.plot(x_exp_s, y_exp_dw, 'o', color='k', label="Experimental")

        # EXPERIMENTAL curves
        xb_e, yb_e, xf_e, yf_e = split_back_fwd(x_exp_s, y_exp_dw)
        if FORCE_PAPER_TAU:
            # Anchor both arms at the experimental apex at x=0
            y0 = _apex_value(x_exp_s, y_exp_dw)

            if xb_e.size:
                xx_b = np.linspace(xb_e.min(), 0.0, 400)
                yy_b = y0 * np.exp(xx_b / TAU_BACK_PAPER)
                ax.plot(xx_b, yy_b, '-', lw=2.0, color='red',
                        label=f'Exp back (τ={TAU_BACK_PAPER:.2f}s)')

            if xf_e.size:
                xx_f = np.linspace(0.0, xf_e.max(), 400)
                yy_f = y0 * np.exp(-xx_f / TAU_FWD_PAPER)
                ax.plot(xx_f, yy_f, '-', lw=2.0, color='red',
                        label=f'Exp fwd (τ={TAU_FWD_PAPER:.2f}s)')
            else:
                xx_f=0
                yy_f=0
                TAU_FWD_PAPER=0
        else:
            # (Aeb, taueb), (Aef, tauef) = fit_back_fwd_with_c0(xb_e, yb_e, xf_e, yf_e)
            [A_alt, tau_alt], _ = scipy.optimize.curve_fit(exp_with_fixed_c, x, y, p0=[1., -1000.])
            if np.isfinite(taueb):
                xx = np.linspace(xb_e.min(), xb_e.max(), 400)
                ax.plot(xx, exp_with_fixed_c(xx, Aeb, taueb, c_fixed=0.0),
                        '-', lw=2.0, color='red', label=f"Exp back (τ={taueb:.2f}s)")
            if np.isfinite(tauef):
                xx = np.linspace(xf_e.min(), xf_e.max(), 400)
                ax.plot(xx, exp_with_fixed_c(-xx, Aef, tauef, c_fixed=0.0),
                        '-', lw=2.0, color='red', label=f"Exp fwd (τ={tauef:.2f}s)")
                

        experiment_backwards_dict = {"xx_b":xx_b,
                                     "xx_f":xx_f,
                                "yy_b":yy_b,
                                "yy_f":yy_f,
                                # "Amb":Amb,
                                "TAU_BACK_PAPER":TAU_BACK_PAPER,
                                "TAU_FWD_PAPER":TAU_FWD_PAPER,
                                "x_exp_s_backward":x_exp_s_backward,
                                "y_exp_s_backward":y_exp_s_backward}
        
        backwards_experiment_per_hz_dict[hz] = experiment_backwards_dict

        ax.set_title(f"{hz} Hz Pre Stim Simulated Model")
        ax.set_xlabel("Time from plateau (s)")
        ax.set_ylabel("EPSP Amplitude (relative %)") #("ΔW")
        ax.set_ylim(0, 2.50)
        custom_tick_locations = [0.0, 0.5, 1.0, 1.5, 2.0, 2.5]
        custom_labels = ['100', '150', '200', '250', '300', '350']
        ax.set_yticks(custom_tick_locations)
        ax.set_yticklabels(custom_labels)
        ax.legend(fontsize=7, loc="upper left")

    plt.tight_layout()
    plt.show()

    plt.figure()
    plt.plot(ymod_list, marker="o")
    plt.xlabel("Frequencies of Presynaptic Stimulation - Model")
    plt.ylabel("% LTP at 0ms from Plateau")
    plt.xticks(np.arange(4), ["10Hz", "20Hz", "40Hz", "100Hz"])
    plt.ylim(100, 350)
    plt.title("Model Fit to 2017 Kernel Imposed Presynaptic Spiking @ Diff Freq.")
    plt.show()

    ymod_list = []

    for i, hz in enumerate(hz_tested_list):
        ax = axs[i]
        x_ms = np.asarray(products_dict_jeff[hz]["x"], float)
        y_mod = np.asarray(products_dict_jeff[hz]["y"], float)

        # seconds from plateau (negative = backward)
        x_s = -x_ms / 1000.0

        # MODEL points
        ax.plot(x_s, y_mod, 'o', label="Model", color='tab:blue')

        print(f"y_mod {y_mod}")
        ymod_list.append(y_mod[4])

        # MODEL fits (c=0), back & fwd
        xb_m, yb_m, xf_m, yf_m = split_back_fwd(x_s, y_mod)
        (Amb, taumb), (Amf, taumf) = fit_back_fwd_with_c0(xb_m, yb_m, xf_m, yf_m)

        if np.isfinite(taumb):
            xx = np.linspace(xb_m.min(), xb_m.max(), 400)
            ax.plot(xx, exp_with_fixed_c(xx, Amb, taumb, c_fixed=0.0),
                    '--', lw=2.0, color='purple', label=f"Model back (τ={taumb:.2f}s)")
        if np.isfinite(taumf):
            xx = np.linspace(xf_m.min(), xf_m.max(), 400)
            ax.plot(xx, exp_with_fixed_c(-xx, Amf, taumf, c_fixed=0.0),
                    '--', lw=2.0, color='purple', label=f"Model fwd (τ={taumf:.2f}s)")

        # EXPERIMENTAL points (from 20 Hz dataset)
        x_exp_s  = -x_20 / 1000.0
        y_exp_dw = y_20_exp
        
        
        x_exp_s_backward = x_exp_s
        y_exp_s_backward = y_exp_dw



        ax.plot(x_exp_s_backward, y_exp_s_backward, 'o', color='k', label="Experimental")

        # EXPERIMENTAL curves
        xb_e, yb_e, xf_e, yf_e = split_back_fwd(x_exp_s, y_exp_dw)
        if FORCE_PAPER_TAU:
            # Anchor both arms at the experimental apex at x=0
            y0 = _apex_value(x_exp_s, y_exp_dw)

            if xb_e.size:
                xx_b = np.linspace(xb_e.min(), 0.0, 400)
                yy_b = y0 * np.exp(xx_b / TAU_BACK_PAPER)
                ax.plot(xx_b, yy_b, '-', lw=2.0, color='red',
                        label=f'Exp back (τ={TAU_BACK_PAPER:.2f}s)')

            if xf_e.size:
                xx_f = np.linspace(0.0, xf_e.max(), 400)
                yy_f = y0 * np.exp(-xx_f / TAU_FWD_PAPER)
                ax.plot(xx_f, yy_f, '-', lw=2.0, color='red',
                        label=f'Exp fwd (τ={TAU_FWD_PAPER:.2f}s)')
        else:
            (Aeb, taueb), (Aef, tauef) = fit_back_fwd_with_c0(xb_e, yb_e, xf_e, yf_e)
            if np.isfinite(taueb):
                xx = np.linspace(xb_e.min(), xb_e.max(), 400)
                ax.plot(xx[:5], exp_with_fixed_c(xx, Aeb, taueb, c_fixed=0.0)[:5],
                        '-', lw=2.0, color='red', label=f"Exp back (τ={taueb:.2f}s)")
            if np.isfinite(tauef):
                xx = np.linspace(xf_e.min(), xf_e.max(), 400)
                ax.plot(xx[:5], exp_with_fixed_c(-xx, Aef, tauef, c_fixed=0.0)[:5],
                        '-', lw=2.0, color='red', label=f"Exp fwd (τ={tauef:.2f}s)")
                
        
        print(f"xx {xx}")

        ax.set_title(f"{hz} Hz Pre Stim Simulated Model")
        ax.set_xlabel("Time from plateau (s)")
        ax.set_ylabel("EPSP Amplitude (relative %)") #("ΔW")
        ax.set_ylim(0, 2.50)
        custom_tick_locations = [0.0, 0.5, 1.0, 1.5, 2.0, 2.5]
        custom_labels = ['100', '150', '200', '250', '300', '350']
        ax.set_yticks(custom_tick_locations)
        ax.set_yticklabels(custom_labels)
        ax.legend(fontsize=7, loc="upper left")

    plt.tight_layout()
    plt.show()


    # loss, products_dict_jeff, _, _ = objective(context.jeffs_data_dict, params, export=True, plot=plot_full_intermediates)

    # print(f"products_dict_jeff {products_dict_jeff}")



    # tau_backward_model, tau_forward_model = plot_full_kernel_bidirectional(context.jeffs_data_dict, products_dict_jeff,taus_bidirectional,label=20,title="20 Hz (full kernel)")
    # plot_full_kernel_bidirectional(context.jeffs_data_dict, products_dict_jeff, label=20, title="20 Hz (full kernel - 2017)", c_fixed=0.0, tau_bounds_back=(0.6, 2.5), tau_bounds_fwd=(0.4, 1.5), n_grid=600, align_model="none",color_model="blue",color_fit_exp="red", color_fit_model="purple")

    # build products_dict_* for all Hz (your code already does this)
    # products_dict_jeff[hz] = {"x": x_arr_int, "y": np.array(delta_w_list), "params_dict": params}



    # tau_results, max_value_list = plot_kernels_grid(
    #     jeffs_data_dict,          # cleaned (experiment)
    #     products_dict_jeff,               # model output you computed
    #     hz_list=hz_tested_list,
    #     c_fixed=0.0,                      # or 1.0 if plotting normalized EPSP
    #     tau_bounds_back=(0.6, 2.5),
    #     tau_bounds_fwd=(0.4, 1.5),
    #     n_grid=600,
    #     align_model="none"                # or "lsq" if you want visual alignment
    # )

    # plt.figure()
    # plt.plot(max_value_list, marker='o', color='k', linestyle="None")
    # plt.plot(max_value_list, color='r')
    # plt.ylabel("Max Potentiation %")
    # plt.ylim(175,225)
    # plt.xticks(np.arange(4), ["10Hz", "20Hz", "40Hz", "100Hz"])
    # plt.show()

    return backwards_experiment_per_hz_dict, backwards_model_per_hz_dict



import numpy as np
import matplotlib.pyplot as plt

def plot_jeff3(
    params,
    jeffs_data_dict,
    plot_full_intermediates=False,
    hz_tested_list=(10, 20, 40, 100),
    exp_key=None,                 # <-- any key in jeffs_data_dict (e.g. 20, "20Hz_full_kernel", etc.)
    x_units="s",
    force_paper_tau=True,
    tau_back_paper=1.31,
    tau_fwd_paper=0.69,
    actually_jeff=True,          # kept for compatibility; exp_key controls which exp dataset is shown
):
    """
    Plots model predictions for hz_tested_list on the SAME x-grid (from exp_key),
    and overlays experimental points/curves from that same exp_key.

    Requirements:
      - jeffs_data_dict[exp_key] must have {'x': array(ms), 'y': array(values)}
      - get_W, fit_exp_fixed_c, exp_with_fixed_c, str_true_false_to_bool must exist in your namespace
    """

    plot_full_intermediates = str_true_false_to_bool(plot_full_intermediates)

    # -----------------------
    # pick experimental dataset key
    # -----------------------
    if exp_key is None:
        # default: use 20 if it exists, else first key
        exp_key = 20 if 20 in jeffs_data_dict else list(jeffs_data_dict.keys())[0]

    if exp_key not in jeffs_data_dict:
        raise KeyError(f"exp_key={exp_key!r} not found in jeffs_data_dict keys: {list(jeffs_data_dict.keys())}")

    x_exp_ms = np.asarray(jeffs_data_dict[exp_key]["x"], float)
    y_exp    = np.asarray(jeffs_data_dict[exp_key]["y"], float)

    # this is the SINGLE grid used for ALL model runs
    x_grid_ms = x_exp_ms.copy()

    # -----------------------
    # helpers
    # -----------------------
    def split_back_fwd(x_s, y):
        x_s = np.asarray(x_s, float); y = np.asarray(y, float)
        m = np.isfinite(x_s) & np.isfinite(y)
        x_s, y = x_s[m], y[m]
        o = np.argsort(x_s); x_s, y = x_s[o], y[o]
        mb = x_s <= 0
        mf = x_s >= 0
        return x_s[mb], y[mb], x_s[mf], y[mf]

    def fit_back_fwd_with_c0(xb, yb, xf, yf):
        tau_b = tau_f = np.nan
        A_b = A_f = np.nan
        if (np.isfinite(yb) & (yb > 0)).sum() >= 2:
            A_b, tau_b = fit_exp_fixed_c(xb, yb, c_fixed=0.0)
        if (np.isfinite(yf) & (yf > 0)).sum() >= 2:
            zf = -xf
            A_f, tau_f = fit_exp_fixed_c(zf, yf, c_fixed=0.0)
        return (A_b, tau_b), (A_f, tau_f)

    def apex_value_at_zero(x_s, y):
        x_s = np.asarray(x_s, float); y = np.asarray(y, float)
        m = np.isfinite(x_s) & np.isfinite(y)
        x_s, y = x_s[m], y[m]
        o = np.argsort(x_s); x_s, y = x_s[o], y[o]
        if x_s.size == 0:
            return np.nan
        if np.any(x_s == 0.0):
            return float(y[x_s == 0.0][0])
        return float(np.interp(0.0, x_s, y))

    # seconds from plateau (negative = backward)
    if x_units == "s":
        x_grid_s = -x_grid_ms / 1000.0
        xlabel = "Time from plateau (s)"
    else:
        x_grid_s = -x_grid_ms
        xlabel = "Time from plateau (ms)"

    # -----------------------
    # run model on shared grid for each hz
    # -----------------------
    products_dict = {}
    for hz_used in hz_tested_list:
        delta_w_list, et_time_1000, IS = get_W(
            x_grid_ms,
            post_ms=10000, pre_ms=10000,
            hz_used=hz_used,
            plateau_length=300,
            tau_et=params["tau_et"], tau_is=params["tau_is"],
            lam_et=params["lam_et"], lam_is=params["lam_is"],
            dt_ms=1.0, eta_ms=params["eta_ms"],
            plot_intermediates=False
        )
        products_dict[hz_used] = {
            "x_ms": x_grid_ms,
            "y": np.asarray(delta_w_list, float),
        }

    # -----------------------
    # plot
    # -----------------------
    fig, axs = plt.subplots(2, 2, figsize=(9, 6), sharex=True, sharey=True)
    axs = np.array(axs).ravel()

    backwards_model_per_hz_dict = {}
    backwards_experiment_per_hz_dict = {}

    # precompute experimental split / apex once (since it's shared)
    xb_e, yb_e, xf_e, yf_e = split_back_fwd(x_grid_s, y_exp)
    y0_exp = apex_value_at_zero(x_grid_s, y_exp)

    for i, hz in enumerate(hz_tested_list):
        ax = axs[i]
        y_mod = products_dict[hz]["y"]

        # model points
        ax.plot(x_grid_s, y_mod, "o", label="Model", color="tab:blue")

        # model fits
        xb_m, yb_m, xf_m, yf_m = split_back_fwd(x_grid_s, y_mod)
        (Amb, taumb), (Amf, taumf) = fit_back_fwd_with_c0(xb_m, yb_m, xf_m, yf_m)

        xx_b = np.array([])
        model_backwards_line = np.array([])
        if np.isfinite(taumb) and xb_m.size:
            xx_b = np.linspace(xb_m.min(), 0.0, 400)
            model_backwards_line = exp_with_fixed_c(xx_b, Amb, taumb, c_fixed=0.0)
            ax.plot(xx_b, model_backwards_line, "--", lw=2.0, color="purple",
                    label=f"Model back (τ={taumb:.2f}s)")

        xx_f = np.array([])
        model_forwards_line = np.array([])
        if np.isfinite(taumf) and xf_m.size:
            xx_f = np.linspace(0.0, xf_m.max(), 400)
            model_forwards_line = exp_with_fixed_c(-xx_f, Amf, taumf, c_fixed=0.0)
            ax.plot(xx_f, model_forwards_line, "--", lw=2.0, color="purple",
                    label=f"Model fwd (τ={taumf:.2f}s)")

        backwards_model_per_hz_dict[hz] = dict(
            xx_b=xx_b, xx_f=xx_f,
            model_backwards_line=model_backwards_line,
            model_forwards_line=model_forwards_line,
            Amb=Amb, Amf=Amf, taumb=taumb, taumf=taumf,
            x_s=x_grid_s, y_mod=y_mod
        )

        # experimental points (same for every panel)
        ax.plot(x_grid_s, y_exp, "o", color="k", label=f"Experimental ({exp_key})")

        # experimental curves (same for every panel)
        if force_paper_tau:
            if xb_e.size:
                xx = np.linspace(xb_e.min(), 0.0, 400)
                yy = y0_exp * np.exp(xx / tau_back_paper)
                ax.plot(xx, yy, "-", lw=2.0, color="red",
                        label=f"Exp back (τ={tau_back_paper:.2f}s)")
            if xf_e.size:
                xx = np.linspace(0.0, xf_e.max(), 400)
                yy = y0_exp * np.exp(-xx / tau_fwd_paper)
                ax.plot(xx, yy, "-", lw=2.0, color="red",
                        label=f"Exp fwd (τ={tau_fwd_paper:.2f}s)")
            exp_tau_back = tau_back_paper
            exp_tau_fwd  = tau_fwd_paper
        else:
            (Aeb, taueb), (Aef, tauef) = fit_back_fwd_with_c0(xb_e, yb_e, xf_e, yf_e)
            exp_tau_back, exp_tau_fwd = taueb, tauef
            if np.isfinite(taueb) and xb_e.size:
                xx = np.linspace(xb_e.min(), 0.0, 400)
                ax.plot(xx, exp_with_fixed_c(xx, Aeb, taueb, c_fixed=0.0),
                        "-", lw=2.0, color="red", label=f"Exp back (τ={taueb:.2f}s)")
            if np.isfinite(tauef) and xf_e.size:
                xx = np.linspace(0.0, xf_e.max(), 400)
                ax.plot(xx, exp_with_fixed_c(-xx, Aef, tauef, c_fixed=0.0),
                        "-", lw=2.0, color="red", label=f"Exp fwd (τ={tauef:.2f}s)")

        backwards_experiment_per_hz_dict[hz] = dict(
            x_exp_s=x_grid_s, y_exp=y_exp,
            y0_exp=y0_exp,
            exp_tau_back=exp_tau_back,
            exp_tau_fwd=exp_tau_fwd
        )

        ax.set_title(f"{hz} Hz Pre Stim Simulated Model")
        ax.set_xlabel(xlabel)
        ax.set_ylabel("EPSP Amplitude (relative %)")
        ax.axvline(0, color="0.7", lw=1)
        ax.legend(fontsize=7, loc="upper left")

    plt.tight_layout()
    plt.show()

    return products_dict, backwards_model_per_hz_dict, backwards_experiment_per_hz_dict


# Example usage:
# plot_jeff3(params, jeffs_data_dict, exp_key="20Hz_full_kernel", hz_tested_list=(10,20,40,100))
# plot_jeff3(params, jeffs_data_dict, exp_key=20, hz_tested_list=(10,20,40,100), force_paper_tau=True)



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

def get_W(x_arr_int, post_ms=10000, pre_ms=10000, hz_used=10, plateau_length = 300,tau_et =1500.,tau_is = 500.,lam_et=200.,lam_is=4,dt_ms=1.0, eta_ms = 0.0001, plot_intermediates=True, special_case=False):
    
    if special_case:
        time_to_is=0
    else:
        time_to_is=x_arr_int[0] 

    spike_times = spike_train_by_rate(rate_hz=hz_used, n=10, t0_ms=0).astype(int)
    mid_spike = spike_times[4]

    T_ms = int(pre_ms + post_ms)
    t_ms = np.arange(-pre_ms, post_ms, dtype=int)
    center_idx = pre_ms  

    offset = center_idx - time_to_is - mid_spike
    spike_idx = offset + spike_times

    spike_train = np.zeros(T_ms)
    spike_train[spike_idx] = 1.0

    is_pre_conv = np.zeros(T_ms)
    is_pre_conv[center_idx:center_idx+plateau_length] =1

    if plot_intermediates:
        plt.plot(t_ms,spike_train, label='pre spikes')
        plt.plot(t_ms,is_pre_conv, label="plateau")
        plt.title("Inputs")
        plt.xlabel("Time (ms)")
        plt.legend()
        plt.show()
    
    # ET = multiply_et(spike_train, tau_ms=tau_et, lam=lam_et, dt_ms=dt_ms)

    # IS= multiply_et(is_pre_conv, tau_ms=tau_is, lam=lam_is, dt_ms=1.0)

    ET, IS = multiply_et_is(spike_train, is_pre_conv, tau_et, tau_is, lam_et, lam_is, dt_ms=dt_ms)

    if special_case:
        ET_to_plot = ET


    if plot_intermediates:
        plt.plot(t_ms, ET, label="ET")
        plt.plot(t_ms, IS, label="IS")
        plt.title(f"tau_et {tau_et}, tau_is {tau_is}, lam_et {lam_et}, lam_is {lam_is}")
        plt.legend()
        plt.show()

    x_arr_diff = np.diff(x_arr_int)
    cumulative_rolls = np.cumsum(x_arr_diff)
    cumulative_rolls = np.concatenate(([0], cumulative_rolls))

    delta_w_list = []

    for idx, roll_time in enumerate(np.abs(cumulative_rolls)):

        ET_rolled = np.roll(ET, roll_time)

        
        if idx == 3 and hz_used!=100 and not special_case:
            ET_to_plot = ET_rolled
        elif idx == 2 and hz_used==100 and not special_case:
            ET_to_plot = ET_rolled


        if plot_intermediates:
            plt.plot(t_ms, ET_rolled, label="ET Rolled")
            plt.plot(t_ms, IS, label="IS")
            plt.title(f" Rolled {roll_time} ms to the right")
            plt.show()

        W = np.trapz(ET_rolled*IS, dx=dt_ms)
        dW = eta_ms * W

        # W = integrate_weight(ET_rolled, IS, eta=eta_ms, Wmin=0.0, Wmax=5.0, dt_ms=dt_ms)
        # # run once with NO clipping and w0=0
        # W = integrate_weight(ET_rolled, IS, eta=eta_ms, Wmin=-np.inf, Wmax=np.inf, dt_ms=dt_ms)


        # if plot_intermediates:
        #     plt.plot(W)
        #     plt.title("W")
        #     plt.show()

        delta_w_list.append(dW)

    # print(f"delta_w_list {delta_w_list}")

    return delta_w_list, ET_to_plot, IS

def fit_A_c_fixed_tau(x, y, tau):
    Phi = np.column_stack([np.exp(x/tau), np.ones_like(x)])
    A, c = np.linalg.lstsq(Phi, y, rcond=None)[0]
    return float(A), float(c)

def exp_rise(x, A, c, tau):
    return c + A*np.exp(x/tau)

def fit_tau_A_c(x, y, tau_min=0.1, tau_max=5.0, n_grid=600):
    taus = np.linspace(tau_min, tau_max, n_grid)
    best = (np.inf, None, None, None)
    for tau in taus:
        A, c = fit_A_c_fixed_tau(x, y, tau)
        sse = np.sum((y - exp_rise(x, A, c, tau))**2)
        if sse < best[0]:
            best = (sse, tau, A, c)
    _, tau_hat, A_hat, c_hat = best
    return tau_hat, A_hat, c_hat

def load_two_numeric_cols(path):
    p = Path(path).expanduser()
    # 1) Try auto-detect delimiter, handle BOM
    df = pd.read_csv(
        p, sep=None, engine="python", header=None,
        encoding="utf-8-sig",  # handles BOM if present
        comment="#",           # ignore commented lines
        skip_blank_lines=True
    )
    if df.empty:
        raise ValueError(f"Read empty DataFrame from {p}. Check delimiter/encoding.")

    # 2) If there are more than 2 cols, keep the last two
    if df.shape[1] >= 2:
        df = df.iloc[:, -2:]
    else:
        raise ValueError(f"Only {df.shape[1]} column(s) found in {p}, expected 2.")

    # 3) Coerce to numeric, drop any non-numeric rows
    df.columns = ["x", "y"]
    df["x"] = pd.to_numeric(df["x"], errors="coerce")
    df["y"] = pd.to_numeric(df["y"], errors="coerce")
    df = df.dropna(subset=["x", "y"]).reset_index(drop=True)

    if df.empty:
        raise ValueError(f"All rows non-numeric after coercion in {p}.")
    return df





def objective(cleaned_data_dict, params_dict, export=False, plot=None, special_case=False):

    final_calc_data_list = []
    final_experimental_data_list = []

    products_dict = {}

    et_time_1000_dict = {}
    

    for hz in cleaned_data_dict:

        x_arr_int = cleaned_data_dict[hz]['x']
        y_arr_scaled = cleaned_data_dict[hz]['y']

        # hz_used = hz_list[i]
        # df_used = df_list[i]

        # x_arr = df_used["x"].to_numpy() *-1000
        # x_arr_int = [int(x) for x in x_arr]

        # print(f"string_list[i] {string_list[i]} len(x_arr_int) {len(x_arr_int)}")

        if hz == "20Hz_full_kernel":
            hz_used = 20
        else:
            hz_used = hz

        delta_w_list, et_time_1000, IS = get_W(x_arr_int, post_ms=10000, pre_ms=10000, hz_used=hz_used, plateau_length = 300,tau_et =params_dict["tau_et"],tau_is = params_dict["tau_is"],lam_et=params_dict["lam_et"],lam_is=params_dict["lam_is"],dt_ms=1.0, eta_ms = params_dict["eta_ms"], plot_intermediates=plot, special_case=special_case)

        et_time_1000_dict[hz] = et_time_1000

        # y_arr = df_used["y"].to_numpy()
        # y_arr_scaled = (y_arr-1)

        final_calc_data_list.append(delta_w_list)
        final_experimental_data_list.append(y_arr_scaled)

        if export:
            products_dict[hz] = {"x":x_arr_int,
                        "y":np.array(delta_w_list),
                        "params_dict":params_dict}


    calculated_array = np.hstack(final_calc_data_list)
    experimental_array = np.hstack(final_experimental_data_list)

    loss = float(np.mean((calculated_array - experimental_array) ** 2))

    # print(f"len(calculated_array) {len(calculated_array)} len(experimental_array) {len(experimental_array)}")
    # print(f"len(calculated_array) {calculated_array} len(experimental_array) {experimental_array}")


    if export:
        return loss, products_dict, et_time_1000_dict, IS
    return loss


def plot_tau_buckets(taus_experimental, taus_model):
    order = [10, 20, 40, 100, "forward"]
    labels = ["10Hz Backward", "20Hz Backward", "40Hz Backward", "100Hz Backward", "20Hz Forward"]
    x = np.arange(len(order))

    fig, ax = plt.subplots(figsize=(8, 4.5))

    # small horizontal offsets so exp/model don't overlap
    off_exp, off_mod = -0.12, +0.12
    jitter = 0.07  # spread multiple points within a series

    def _points(vals, x0, offset, color, marker):
        vals = list(vals) if vals is not None else []
        if not vals:
            return
        if len(vals) == 1:
            xs = [x0 + offset]
        else:
            xs = x0 + offset + np.linspace(-jitter, jitter, len(vals))
        ax.plot(xs, vals, marker=marker, ls="None", color=color, ms=7)

    for xi, key in enumerate(order):
        _points(taus_experimental.get(key, []), xi, off_exp, "red", "o")
        _points(taus_model.get(key, []),        xi, off_mod, "tab:blue", "s")

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=15)
    ax.set_ylabel("Time constant τ (s)")
    ax.grid(axis="y", alpha=0.3)

    # legend handles
    h1 = ax.plot([], [], "ro", ls="None", ms=7, label="Experimental")[0]
    h2 = ax.plot([], [], "s",  color="tab:blue", ls="None", ms=7, label="Model")[0]
    ax.legend(handles=[h1, h2], frameon=False, loc="upper left")

    plt.tight_layout()
    plt.show()

def _solve_ls(Phi, y, w=None, ridge=0.0):
    """
    Solve min ||W^{1/2}(Phi*[A,c] - y)||^2 + ridge*||[A,c]||^2.
    """
    if w is not None:
        W = np.diag(w.astype(float))
        AtA = Phi.T @ W @ Phi
        Aty = Phi.T @ W @ y
    else:
        AtA = Phi.T @ Phi
        Aty = Phi.T @ y
    if ridge > 0.0:
        AtA = AtA + ridge * np.eye(AtA.shape[0])
    coef = np.linalg.solve(AtA, Aty)
    return float(coef[0]), float(coef[1])  # A, c

def exp_rise_fixed_tau(x, A, c, tau):
    return c + A * np.exp(x / tau)

def fit_free_tau_grid_flexible(
    x, y,
    tau_min=0.1, tau_max=5.0, n_grid=600,
    weights=None, ridge=0.0
):
    """
    Grid-search τ; for each τ, solve (possibly weighted/ridge) LS for A,c.
    Returns (tau_hat, A_hat, c_hat).
    """
    x = np.asarray(x, float); y = np.asarray(y, float)
    taus = np.linspace(tau_min, tau_max, n_grid)
    best = (np.inf, None, None, None)
    Phi_base = np.column_stack([np.exp(x / 1.0), np.ones_like(x)])  # placeholder scale

    for tau in taus:
        # reuse structure: same as [exp(x/tau), 1]
        Phi = np.column_stack([np.exp(x / tau), np.ones_like(x)])
        A, c = _solve_ls(Phi, y, w=weights, ridge=ridge)
        sse = np.sum(((y - (c + A*np.exp(x/tau)))**2) if weights is None
                     else weights * (y - (c + A*np.exp(x/tau)))**2)
        if sse < best[0]:
            best = (sse, tau, A, c)
    _, tau_hat, A_hat, c_hat = best
    return tau_hat, A_hat, c_hat


def make_weights_by_distance(x, center=0.0, half_life=1.5):
    """
    x in seconds (negative to 0). Weight = exp(-|x-center|/half_life).
    """
    x = np.asarray(x, float)
    return np.exp(-np.abs(x - center) / float(half_life))


def make_capped_weights(y, cap=0.5):
    """
    Smaller weights for very large values. w = 1 / (1 + (y/cap)^2)
    """
    y = np.asarray(y, float)
    return 1.0 / (1.0 + (y / float(cap))**2)


def _fit_tau_A_fixed_c(x, y, tau_min, tau_max, c, n_grid):
    """
    Grid-search τ for y ≈ c + A*exp(x/τ) with closed-form A(τ) = <phi, y-c>/<phi, phi>.
    Returns (tau*, A*). Works on either x or z=-x (see calls below).
    """
    x = np.asarray(x, float); y = np.asarray(y, float)
    taus = np.linspace(float(tau_min), float(tau_max), int(n_grid))
    best = (np.inf, None, None)  # (sse, tau, A)
    r = y - float(c)
    for tau in taus:
        phi = np.exp(x / float(tau))
        den = float(np.dot(phi, phi))
        if den <= 0:
            continue
        A = float(np.dot(phi, r) / den)
        yhat = float(c) + A*phi
        sse = float(np.dot(y - yhat, y - yhat))
        if sse < best[0]:
            best = (sse, float(tau), A)
    return best[1], best[2]

def _exp_fixed_c(x, A, tau, c):
    return float(c) + A*np.exp(x/float(tau))

def _plot_one_kernel_panel(ax, cleaned, products_dict, label,
                           c_fixed=0.0,
                           tau_bounds_back=(0.6, 2.5),
                           tau_bounds_fwd=(0.4, 1.5),
                           n_grid=600,
                           align_model="none",
                           color_model="tab:blue",
                           color_fit_exp="red",
                           color_fit_model="purple",
                           title=None):
    x_exp_ms = np.asarray(cleaned[20]["x"])
    y_exp    = np.asarray(cleaned[20]["y"], dtype=float)
    x_mod_ms = np.asarray(products_dict[label]["x"])
    y_mod    = np.asarray(products_dict[label]["y"], dtype=float)

    max_val_stim = np.max(y_mod)

    emap = {int(x): float(y) for x, y in zip(x_exp_ms, y_exp)}
    mmap = {int(x): float(y) for x, y in zip(x_mod_ms, y_mod)}
    common = sorted(set(emap).intersection(mmap))
    if len(common) < 3:
        ax.set_title(f"{label} Hz (not enough common points)")
        return None

    x_ms = np.array(common, dtype=int)
    y_e  = np.array([emap[v] for v in common], dtype=float)
    y_m  = np.array([mmap[v] for v in common], dtype=float)

    x_s = (-x_ms) / 1000.0
    order = np.argsort(x_s)
    xs, ye, ym = x_s[order], y_e[order], y_m[order]

    if align_model == "lsq":
        Phi = np.column_stack([ym, np.ones_like(ym)])
        (a, b), *_ = np.linalg.lstsq(Phi, ye, rcond=None)
        ym_plot = a*ym + b
    else:
        ym_plot = ym

    back = xs <= 0
    fwd  = xs >= 0

    tau_b_exp = tau_f_exp = np.nan
    tau_b_mod = tau_f_mod = np.nan

    # backward
    if np.sum(back) >= 3:
        xb, yeb = xs[back], ye[back]
        tau_b_exp, A_b_exp = _fit_tau_A_fixed_c(xb, yeb, *tau_bounds_back, c_fixed, n_grid)
        xxb = np.linspace(xb.min(), 0.0, 300)
        yyb_exp = _exp_fixed_c(xxb, A_b_exp, tau_b_exp, c_fixed)

        # model backward
        ymb = ym_plot[back]
        tau_b_mod, A_b_mod = _fit_tau_A_fixed_c(xb, ymb, *tau_bounds_back, c_fixed, n_grid)
        yyb_mod = _exp_fixed_c(xxb, A_b_mod, tau_b_mod, c_fixed)
    else:
        xxb = np.array([]); yyb_exp = yyb_mod = np.array([])

    # forward (fit in z=-x)
    if np.sum(fwd) >= 3:
        xf, yef = xs[fwd], ye[fwd]
        zf = -xf
        tau_f_exp, A_f_exp = _fit_tau_A_fixed_c(zf, yef, *tau_bounds_fwd, c_fixed, n_grid)
        xxf = np.linspace(0.0, xf.max(), 300)
        yyf_exp = _exp_fixed_c(-xxf, A_f_exp, tau_f_exp, c_fixed)

        ymf = ym_plot[fwd]
        tau_f_mod, A_f_mod = _fit_tau_A_fixed_c(zf, ymf, *tau_bounds_fwd, c_fixed, n_grid)
        yyf_mod = _exp_fixed_c(-xxf, A_f_mod, tau_f_mod, c_fixed)
    else:
        xxf = np.array([]); yyf_exp = yyf_mod = np.array([])

    # plot
    ax.plot(xs, ye, "ko", ms=4, label="Experimental")
    ax.plot(xs, ym_plot, "o", ms=4, color=color_model, label="Model")

    if xxb.size:
        ax.plot(xxb, yyb_exp, "-", color=color_fit_exp, lw=2,
                label=f"Exp back (τ≈{tau_b_exp:.2f}s)")
        ax.plot(xxb, yyb_mod, "--", color=color_fit_model, lw=1.8,
                label=f"Model back (τ≈{tau_b_mod:.2f}s)")
    if xxf.size:
        ax.plot(xxf, yyf_exp, "-", color=color_fit_exp, lw=2,
                label=f"Exp fwd (τ≈{tau_f_exp:.2f}s)")
        ax.plot(xxf, yyf_mod, "--", color=color_fit_model, lw=1.8,
                label=f"Model fwd (τ≈{tau_f_mod:.2f}s)")

    ax.set_xlabel("Time from plateau (s)")
    ax.set_ylabel("ΔW" if c_fixed == 0.0 else "Normalized EPSP")
    ax.set_title(title or f"{label} Hz (full kernel)")
    ax.set_xlim(xs.min() - 0.05, xs.max() + 0.05)
    ax.set_ylim(min(ye.min(), ym_plot.min()) - 0.05,
                max(ye.max(), ym_plot.max()) + 0.2)
    ax.legend(frameon=False, fontsize=7, loc="best")

    return {"tau_back_exp": tau_b_exp, "tau_fwd_exp": tau_f_exp,
            "tau_back_mod": tau_b_mod, "tau_fwd_mod": tau_f_mod}, max_val_stim

# ---------- grid function: one subplot per Hz ----------
def plot_kernels_grid(cleaned, products_dict, hz_list,
                      c_fixed=0.0,
                      tau_bounds_back=(0.6, 2.5),
                      tau_bounds_fwd=(0.4, 1.5),
                      n_grid=600,
                      align_model="none"):
    """
    Make a grid of subplots—one per Hz in hz_list—each showing the bidirectional kernel
    with τ searched (c fixed). Returns a dict of τs per Hz.
    """
    hz_list = list(hz_list)
    n = len(hz_list)
    ncols = 2 if n >= 2 else 1
    nrows = int(np.ceil(n / ncols))

    fig, axes = plt.subplots(nrows, ncols, figsize=(6.5*ncols, 4.5*nrows), squeeze=False)
    results = {}

    max_value_list = []

    for i, hz in enumerate(hz_list):
        r = i // ncols
        c = i % ncols
        ax = axes[r, c]
        out, max_value = _plot_one_kernel_panel(
            ax, cleaned, products_dict, label=hz,
            c_fixed=c_fixed,
            tau_bounds_back=tau_bounds_back,
            tau_bounds_fwd=tau_bounds_fwd,
            n_grid=n_grid,
            align_model=align_model,
            title=f"{hz} Hz (full kernel)"
        )
        results[hz] = out
        max_value_list.append(max_value*100)


    # hide any unused axes
    for j in range(i+1, nrows*ncols):
        r = j // ncols
        c = j % ncols
        axes[r, c].axis("off")

    plt.tight_layout()
    plt.show()
    return results, max_value_list



def plot_full_kernel_bidirectional(
    cleaned, products_dict, *,
    label="20Hz_full_kernel",
    title="20 Hz (full kernel)",
    c_fixed=0.0,                 # << baseline fixed here (use 1.0 if data is normalized EPSP)
    tau_bounds_back=(0.6, 2.5),  # search range (seconds) for backward side
    tau_bounds_fwd=(0.4, 1.5),   # search range (seconds) for forward side
    n_grid=600,
    align_model="none",          # 'none' | 'lsq'
    color_model="tab:blue",
    color_fit_exp="red",
    color_fit_model="purple",
):
    """
    Fits y ≈ c_fixed + A * exp(x/τ) with c fixed, by τ grid-search and closed-form A.
    Does this separately for EXPERIMENT and MODEL, backward (x<=0) and forward (x>=0).

    Returns: (tau_back_exp, tau_fwd_exp, tau_back_model, tau_fwd_model)
    """

    if label not in cleaned or label not in products_dict:
        raise KeyError(f"Need both cleaned and products_dict entries for {label!r}")

    # ------------- helpers -------------
    def _fit_tau_A_fixed_c(x, y, tau_min, tau_max, c, n_grid):
        """
        Grid-search τ for y ≈ c + A*exp(x/τ) with closed-form A(τ) = <phi, y-c>/<phi, phi>.
        """
        x = np.asarray(x, float); y = np.asarray(y, float)
        taus = np.linspace(float(tau_min), float(tau_max), int(n_grid))
        best_sse, best_tau, best_A = np.inf, None, None
        r = y - float(c)

        for tau in taus:
            phi = np.exp(x / float(tau))
            den = float(np.dot(phi, phi))
            if den <= 0:  # safety
                continue
            A_hat = float(np.dot(phi, r) / den)
            y_hat = float(c) + A_hat * phi
            sse = float(np.dot(y - y_hat, y - y_hat))
            if sse < best_sse:
                best_sse, best_tau, best_A = sse, float(tau), A_hat
        return best_tau, best_A

    def _exp_fixed_c(x, A, tau, c):
        return float(c) + A * np.exp(x / float(tau))

    # ------------- pull data -------------
    # IMPORTANT: since you want c=0, we assume y is already Δ (baseline-subtracted).
    # If your y was normalized (baseline ~1), set c_fixed=1.0 instead.
    x_exp_ms = np.asarray(cleaned[label]["x"])
    y_exp    = np.asarray(cleaned[label]["y"], dtype=float)

    x_mod_ms = np.asarray(products_dict[label]["x"])
    y_mod    = np.asarray(products_dict[label]["y"], dtype=float)

    max_model = np.max(y_mod)

    # align on common x (ms), then convert to seconds, negative=before, positive=after
    def _align_by_x_ms(xe, ye, xm, ym):
        emap = {int(x): float(y) for x, y in zip(xe, ye)}
        mmap = {int(x): float(y) for x, y in zip(xm, ym)}
        common = sorted(set(emap).intersection(mmap))
        x = np.array(common, dtype=int)
        yea = np.array([emap[v] for v in common], dtype=float)
        yma = np.array([mmap[v] for v in common], dtype=float)
        return x, yea, yma

    x_ms, y_e, y_m = _align_by_x_ms(x_exp_ms, y_exp, x_mod_ms, y_mod)
    x_s = (-x_ms) / 1000.0  # seconds; 0 at plateau; <0 backward, >0 forward

    back = x_s <= 0
    fwd  = x_s >= 0

    # optional linear alignment of MODEL to EXP (shared a,b across both sides)
    y_m_plot = y_m.copy()
    if align_model == "lsq":
        Phi = np.column_stack([y_m_plot, np.ones_like(y_m_plot)])
        (a, b), *_ = np.linalg.lstsq(Phi, y_e, rcond=None)
        y_m_plot = a * y_m_plot + b

    # sort for display
    order = np.argsort(x_s)
    xs, ye, ym = x_s[order], y_e[order], y_m_plot[order]

    # ------------- EXPERIMENT fits (fixed c, search τ) -------------
    xb = x_s[back]; yeb = y_e[back]
    xf = x_s[fwd];  yef = y_e[fwd]
    # forward: fit on z = -x (rise away from 0), then evaluate with -x
    zf = -xf

    tau_b_exp, A_b_exp = _fit_tau_A_fixed_c(xb, yeb, *tau_bounds_back, c_fixed, n_grid)
    tau_f_exp, A_f_exp = _fit_tau_A_fixed_c(zf, yef, *tau_bounds_fwd,  c_fixed, n_grid)

    xxb = np.linspace(xb.min(), 0.0, 300)
    xxf = np.linspace(0.0, xf.max(), 300)
    yyb_exp = _exp_fixed_c(xxb, A_b_exp, tau_b_exp, c_fixed)
    yyf_exp = _exp_fixed_c(-xxf, A_f_exp, tau_f_exp, c_fixed)

    # ------------- MODEL fits (fixed c, search τ) -------------
    ymb = y_m_plot[back]
    ymf = y_m_plot[fwd]
    tau_b_mod, A_b_mod = _fit_tau_A_fixed_c(xb, ymb, *tau_bounds_back, c_fixed, n_grid)
    tau_f_mod, A_f_mod = _fit_tau_A_fixed_c(zf, ymf, *tau_bounds_fwd,  c_fixed, n_grid)

    yyb_mod = _exp_fixed_c(xxb, A_b_mod, tau_b_mod, c_fixed)
    yyf_mod = _exp_fixed_c(-xxf, A_f_mod, tau_f_mod, c_fixed)

    # ------------- plot -------------
    plt.figure(figsize=(6.4, 4.8))
    plt.plot(xs, ye, "ko", ms=5, label="Experimental")
    plt.plot(xs, ym, "o", ms=5, color=color_model, label="Model")

    # exp fits
    plt.plot(xxb, yyb_exp, "-", color=color_fit_exp, lw=2,
             label=fr"Exp back (τ≈{tau_b_exp:.2f}s, c={c_fixed:g})")
    plt.plot(xxf, yyf_exp, "-", color=color_fit_exp, lw=2,
             label=fr"Exp fwd (τ≈{tau_f_exp:.2f}s, c={c_fixed:g})")

    # model fits
    plt.plot(xxb, yyb_mod, "--", color=color_fit_model, lw=1.8,
             label=fr"Model back (τ≈{tau_b_mod:.2f}s)")
    plt.plot(xxf, yyf_mod, "--", color=color_fit_model, lw=1.8,
             label=fr"Model fwd (τ≈{tau_f_mod:.2f}s)")

    plt.xlabel("Time from plateau (s)")
    plt.ylabel("ΔW" if c_fixed == 0.0 else "Normalized EPSP")
    plt.title(title)
    plt.xlim(xs.min() - 0.05, xs.max() + 0.05)
    ymin = min(ye.min(), ym.min())
    ymax = max(ye.max(), ym.max(), yyb_exp.max(), yyf_exp.max(), yyb_mod.max(), yyf_mod.max())
    plt.ylim(0, 3.0)
   
    plt.legend(frameon=False, fontsize=8)
    plt.tight_layout()
    plt.show()

    return tau_b_exp, tau_f_exp, tau_b_mod, tau_f_mod





# ---------- helpers ----------
def fit_A_c_fixed_tau(x, y, tau):
    """Least-squares fit of A and c in y = c + A*exp(x/tau), with tau fixed."""
    Phi = np.column_stack([np.exp(x / tau), np.ones_like(x)])
    coef, *_ = np.linalg.lstsq(Phi, y, rcond=None)
    A, c = coef[0], coef[1]
    return float(A), float(c)

def exp_rise_fixed_tau(x, A, c, tau):
    return c + A*np.exp(x/tau)

import numpy as np
import matplotlib.pyplot as plt

# -------------------- core helpers --------------------

def _to_seconds_backward(x_ms):
    """ms distances before plateau -> negative seconds (…,-3,-2,-1,0)."""
    return -(np.asarray(x_ms, dtype=float) / 1000.0)

def _distance_weights(x_s, half_life=0.8):
    """
    Exponential weights that DOWN-weight points near 0 s.
    Larger half_life -> gentler down-weighting.
    """
    x = np.asarray(x_s, dtype=float)
    # distance from 0:
    d = np.abs(x)
    # exp(-d / half_life)
    w = np.exp(-d / float(half_life))
    # invert so points farther from 0 carry MORE weight:
    w = 1.0 - w
    # normalize
    if np.any(w > 0):
        w = w / w.max()
    else:
        w = np.ones_like(w)
    return w

def _weighted_A_given_tau(x_s, y, tau, weights=None, c=1.0):
    """
    Solve y ~ c + A * exp(x/tau) for A (c fixed).
    Weighted least squares with diagonal W.
    """
    x_s = np.asarray(x_s, dtype=float)
    y   = np.asarray(y, dtype=float)
    phi = np.exp(x_s / float(tau))
    r   = y - float(c)
    if weights is None:
        num = np.dot(phi, r)
        den = np.dot(phi, phi)
    else:
        w   = np.asarray(weights, dtype=float)
        num = np.dot(w * phi, r)
        den = np.dot(w * phi, phi)
    if den <= 0:
        return 0.0
    return float(num / den)

def _sse_for_tau(x_s, y, tau, weights=None, c=1.0):
    A = _weighted_A_given_tau(x_s, y, tau, weights=weights, c=c)
    yhat = c + A * np.exp(x_s / float(tau))
    if weights is None:
        err = y - yhat
        return float(np.dot(err, err))
    else:
        w = np.asarray(weights, dtype=float)
        err = y - yhat
        return float(np.dot(w * err, err))

def fit_tau_A_fixed_c1_weighted(x_s, y, tau_min, tau_max, n_grid=400,
                                weights=None, c=1.0):
    """
    Grid-search τ with c fixed to 1.0 (100% baseline), weighted SSE.
    Returns (tau_hat, A_hat).
    """
    taus = np.linspace(tau_min, tau_max, int(n_grid))
    best = (np.inf, None)
    for tau in taus:
        sse = _sse_for_tau(x_s, y, tau, weights=weights, c=c)
        if sse < best[0]:
            best = (sse, tau)
    tau_hat = best[1]
    A_hat   = _weighted_A_given_tau(x_s, y, tau_hat, weights=weights, c=c)
    return float(tau_hat), float(A_hat)

def exp_rise_fixed_c1(x, A, tau):
    return 1.0 + A * np.exp(x / float(tau))

# # -------------------- paper-style fits on your 'cleaned' --------------------

# def fit_and_plot_btsp_paper_style_from_cleaned(
#     cleaned,
#     titles=("10 Hz", "20 Hz", "40 Hz", "100 Hz"),
#     # τ bounds per condition (seconds) — tuned to be close to the paper
#     tau_bounds={10:(1.1,1.9), 20:(1.3,2.2), 40:(1.4,2.3), 100:(0.8,1.3)},
#     # exclude points within this many seconds of 0 (the steep region)
#     exclude_window_s=0.25,
#     # distance weighting half-life (seconds); bigger = gentler
#     weight_half_life=0.8,
#     # resolution of the τ grid
#     n_grid=500
# ):
#     """
#     Fits backward panels (10/20/40/100 Hz) with y = 1 + A*exp(x/τ), x≤0,
#     using: (i) fixed baseline c=1, (ii) trimming |x|<exclude_window_s,
#     (iii) distance-based weights, (iv) condition-specific τ bounds.
#     Plots experimental points and the fitted curve per panel.
#     Returns dict of τ estimates per panel.
#     """
#     panels = [10, 20, 40, 100]
#     tau_out = {}

#     fig, axes = plt.subplots(2, 2, figsize=(8, 6), sharex=True, sharey=True)
#     axes = axes.ravel()

#     for ax, hz, ttl in zip(axes, panels, titles):
#         if hz not in cleaned:
#             raise KeyError(f"`cleaned` missing key {hz!r}")

#         x_ms = np.asarray(cleaned[hz]["x"])
#         y    = np.asarray(cleaned[hz]["y"], dtype=float) + 1.0

#         x_s = _to_seconds_backward(x_ms)

#         keep = np.where(np.abs(x_s) >= float(exclude_window_s))[0]
#         if keep.size < 2:
#             keep = np.arange(x_s.size)
#         xs = x_s[keep]
#         ys = y[keep]

#         w = _distance_weights(xs, half_life=weight_half_life)

#         tmin, tmax = tau_bounds.get(hz, (0.5, 2.5))

#         # fit τ (with c fixed = 1)
#         tau_hat, A_hat = fit_tau_A_fixed_c1_weighted(
#             xs, ys, tau_min=tmin, tau_max=tmax, n_grid=n_grid, weights=w, c=1.0
#         )
#         tau_out[hz] = tau_hat

#         # curve only over data span
#         xx = np.linspace(xs.min(), xs.max(), 400)
#         yy = exp_rise_fixed_c1(xx, A_hat, tau_hat)

#         # plot
#         order = np.argsort(x_s)
#         ax.plot(x_s[order], y[order], "ko", ms=5, label="Experiment")
#         ax.plot(xx, yy, "-", color="red", lw=2,
#                 label=fr"Fit (τ≈{tau_hat:.2f}s)")

#         # cosmetics
#         ax.set_xlim(-4.1, 0.05)
#         ax.set_ylim(1.0, 3.6)
#         ax.set_xticks([-4, -3, -2, -1, 0])
#         ax.set_title(ttl, loc="left", fontsize=11, pad=2)
#         ax.legend(loc="lower right", fontsize=8, frameon=False)

#     fig.text(0.5, 0.02, "Time from plateau (s)", ha="center")
#     fig.text(0.02, 0.5, "Normalized EPSP (Δ units)", va="center", rotation="vertical")
#     plt.tight_layout(rect=(0.06, 0.06, 1, 1))
#     plt.show()
#     return tau_out



# ---------- helpers ----------
def _to_seconds_backward(x_ms):
    x = np.asarray(x_ms, dtype=float) / 1000.0
    return -x if np.all(x >= 0) else x  # make backward times negative if given as positive lead times

def _distance_weights(x_s, half_life=0.8):
    if (half_life is None) or (half_life <= 0):
        return np.ones_like(x_s, dtype=float)
    decay = np.log(0.5) / float(half_life)
    w = np.exp(decay * np.abs(x_s))
    s = w.sum()
    return w if s == 0 else w / s

def _fit_A_given_tau_c(x, y, w, tau, c):
    """Closed-form weighted least squares for A with fixed tau and c."""
    b = np.exp(x / tau)
    yc = y - c
    denom = np.sum(w * b * b)
    if denom <= 1e-16:
        return 0.0
    return float(np.sum(w * b * yc) / denom)

def _fit_A_c_given_tau(x, y, w, tau):
    """Weighted linear LS for [A, c] with fixed tau; design=[exp(x/tau), 1]."""
    b = np.exp(x / tau)
    # Solve (X^T W X) beta = X^T W y, with X=[b, 1]
    Xw = np.stack([b * np.sqrt(w), np.sqrt(w)], axis=1)
    yw = y * np.sqrt(w)
    # normal equations via lstsq (stable)
    beta, *_ = np.linalg.lstsq(Xw, yw, rcond=None)
    A_hat, c_hat = map(float, beta)
    return A_hat, c_hat

def _sse_weighted(y_true, y_pred, w):
    r = y_true - y_pred
    return float(np.sum(w * r * r))

# ---------- main flexible fitter ----------
# def fit_and_plot_btsp_paper_style_from_cleaned(cleaned, products_dict, titles=("10 Hz", "40 Hz", "20 Hz", "100 Hz"),panels=(10, 40, 20, 100),tau_mode="fit", tau_bounds={10:(1.1,1.9), 40:(1.4,2.3), 20:(1.3,2.2), 100:(0.8,1.3)}, tau_fixed=None,n_grid=500,c_mode="fixed", c_value=1.0, exclude_window_s=0.0,weight_half_life=0.8,xlim=(-4.5, 0.5),ylim=(0.0, 3.0),plot_color="red"):
    
#     y_max_map = {10: 1.3, 40: 2.2, 20: 2.7, 100: 1.7}
    
#     """
#     Fits y = c + A * exp(x/τ) on backward-time panels.
#       - tau_mode:
#           "fit"   → grid search τ within per-panel bounds
#           "fixed" → use tau_fixed (scalar or dict)
#       - c_mode:
#           "fixed" → hold c at c_value (scalar or dict)
#           "fit"   → fit c jointly with A for each τ candidate

#     Returns dict: {panel: {"tau": τ, "A": A, "c": c}}
#     """
#     results = {}

#     fig, axes = plt.subplots(2, 2, figsize=(8, 6), sharex=True, sharey=False)
#     axes = axes.ravel()

#     # normalize scalar/dict inputs
#     def _get_per_panel(val, default=None):
#         if isinstance(val, dict):
#             return val
#         elif val is None:
#             return {} if default is None else {p: default for p in panels}
#         else:
#             return {p: val for p in panels}

#     tau_fixed_map = _get_per_panel(tau_fixed)
#     c_value_map   = _get_per_panel(c_value, default=1.0)

#     count = 0

#     for ax, hz, ttl in zip(axes, panels, titles):
#         if hz not in cleaned:
#             raise KeyError(f"`cleaned` missing key {hz!r}")

#         x_ms = np.asarray(cleaned[hz]["x"])
#         y    = np.asarray(cleaned[hz]["y"], dtype=float)

#         x_ms_model = np.asarray(products_dict[hz]["x"])
#         y_model    = np.asarray(products_dict[hz]["y"], dtype=float)

#         x_s = _to_seconds_backward(x_ms)

#         x_s_model = _to_seconds_backward(x_ms_model)

#         keep = np.where(np.abs(x_s) >= float(exclude_window_s))[0]
#         if keep.size < 2:
#             keep = np.arange(x_s.size)
#         xs = x_s[keep]
#         ys = y[keep]

#         w = _distance_weights(xs, half_life=weight_half_life)

#         w_model = _distance_weights(x_s_model, half_life=weight_half_life)

#         best = {"tau": None, "A": None, "c": None, "sse": np.inf}
#         best_model = {"tau": None, "A": None, "c": None, "sse": np.inf}

#         if tau_mode == "fixed":
#             if hz not in tau_fixed_map:
#                 raise ValueError(f"tau_mode='fixed' but tau_fixed missing for panel {hz}")
#             tau_candidates = np.atleast_1d(float(tau_fixed_map[hz]))
#         else:
#             tmin, tmax = tau_bounds.get(hz, (0.5, 2.5))
#             tau_candidates = np.linspace(max(1e-6, tmin), max(tmin, tmax), int(n_grid))

#         for tau in tau_candidates:
#             if c_mode == "fixed":
#                 c = float(c_value_map.get(hz, 1.0))
#                 A = _fit_A_given_tau_c(xs, ys, w, tau, c)
#                 A_model = _fit_A_given_tau_c(x_s_model, y_model, w_model, tau, c)
#             else:  # c_mode == "fit"
#                 A, c = _fit_A_c_given_tau(xs, ys, w, tau)
#                 A_model, c_model = _fit_A_c_given_tau(x_s_model, y_model, w_model, tau)

#             yhat = c + A * np.exp(xs / tau)
#             yhat_model = c_model + A_model * np.exp(x_s_model / tau)
#             sse = _sse_weighted(ys, yhat, w)
#             if sse < best["sse"]:
#                 best.update({"tau": float(tau), "A": float(A), "c": float(c), "sse": sse})

#             sse_model = _sse_weighted(y_model, yhat_model, w_model)
#             if sse_model < best_model["sse"]:
#                 best_model.update({"tau": float(tau), "A": float(A), "c": float(c), "sse": sse})

#         results[hz] = {k: best[k] for k in ("tau", "A", "c")}

#         xx = np.linspace(xs.min(), xs.max(), 400)
#         yy = best["c"] + best["A"] * np.exp(xx / best["tau"])
#         yy_model = best["c"] + best["A"] * np.exp(xx / best["tau"])
#         order = np.argsort(x_s)
#         ax.plot(x_s[order], y[order], "ko", ms=5, label="Experiment", color='k')
#         ax.plot(x_s[order], y[order], "ko", ms=5, label="Model", color='r')
#         ax.plot(xx, yy, "-", color='b', lw=2, label=fr"Experimental Fit ($\tau\approx{best['tau']:.2f}$ s, $c\approx{best['c']:.2f}$)")
#         ax.plot(xx, yy_model, "-", color='r', lw=2, label=fr"Model Fit ($\tau\approx{best['tau']:.2f}$ s, $c\approx{best['c']:.2f}$)")

#         ax.set_xlim(*xlim)
#         ax.set_ylim(0,y_max_map[hz])
#         ax.set_xticks([-4, -3, -2, -1, 0])
#         ax.set_title(ttl, loc="left", fontsize=11, pad=2)
#         ax.legend(loc="lower right", fontsize=8, frameon=False)

#         count+=1

#     fig.text(0.5, 0.02, "Time from plateau (s)", ha="center")
#     fig.text(0.02, 0.5, "Response (units)", va="center", rotation="vertical")
#     plt.tight_layout(rect=(0.06, 0.06, 1, 1))
#     plt.show()
#     return results


import numpy as np
import matplotlib.pyplot as plt

# assumes you already have:
# _to_seconds_backward, _distance_weights, _fit_A_given_tau_c, _fit_A_c_given_tau, _sse_weighted

def fit_and_plot_btsp_paper_style_from_cleaned(
    cleaned, products_dict,
    titles=("10 Hz", "40 Hz", "20 Hz", "100 Hz"),
    panels=(10, 40, 20, 100),

    # τ control (independent for exp vs model)
    tau_mode_exp="fit",                      # "fit" | "fixed"
    tau_mode_model="fit",                    # "fit" | "fixed"
    tau_bounds_exp={10:(1.1,1.9), 40:(1.4,2.3), 20:(1.3,2.2), 100:(0.8,1.3)},
    tau_bounds_model=None,                   # if None, reuse tau_bounds_exp
    tau_fixed_experiment=None,               # scalar or {panel: tau}
    tau_fixed_model=None,                    # scalar or {panel: tau}
    n_grid=500,

    # c control (applies to both series for simplicity)
    c_mode="fixed",                          # "fixed" | "fit"
    c_value=1.0,                             # scalar or {panel: c}

    # plotting / trimming
    exclude_window_s=0.0,
    weight_half_life=0.8,
    xlim=(-4.5, 0.5),
    y_max_map={10:2.0, 40:2.2, 20:2.7, 100:1.7},
    draw_model_curve_on_exp_span=False,      # if True, draw model curve across exp span
):
    """Overlay experiment vs model with independent τ control and add a 5th τ-summary subpanel."""
    if tau_bounds_model is None:
        tau_bounds_model = tau_bounds_exp

    def _per_panel(val, panels, default=None):
        if isinstance(val, dict):
            return val
        if val is None:
            return {} if default is None else {p: default for p in panels}
        return {p: val for p in panels}

    c_map        = _per_panel(c_value, panels, default=1.0)
    tau_fix_exp  = _per_panel(tau_fixed_experiment, panels) if tau_mode_exp   == "fixed" else {}
    tau_fix_mod  = _per_panel(tau_fixed_model,      panels) if tau_mode_model == "fixed" else {}

    results = {}   # {panel: {"exp": {...}, "model": {...}}}

    # --- make a 2x3 grid: 4 single-panel fits + 1 tau summary + 1 empty ---
    fig, axes = plt.subplots(2, 3, figsize=(12, 6))
    # first four axes used for the per-Hz fits
    fit_axes = axes.ravel()[:4]
    tau_ax   = axes.ravel()[4]
    # disable the last extra axis
    axes.ravel()[5].set_visible(False)

    for ax, hz, ttl in zip(fit_axes, panels, titles):
        # --- load data ---
        x_ms_exp = np.asarray(cleaned[hz]["x"])
        y_exp    = np.asarray(cleaned[hz]["y"], dtype=float)
        x_ms_mod = np.asarray(products_dict[hz]["x"])
        y_mod    = np.asarray(products_dict[hz]["y"], dtype=float)

        x_exp = _to_seconds_backward(x_ms_exp)
        x_mod = _to_seconds_backward(x_ms_mod)

        # --- trimming (independent) ---
        k_exp = np.where(np.abs(x_exp) >= float(exclude_window_s))[0]
        if k_exp.size < 2: k_exp = np.arange(x_exp.size)
        xs_e, ys_e = x_exp[k_exp], y_exp[k_exp]

        k_mod = np.where(np.abs(x_mod) >= float(exclude_window_s))[0]
        if k_mod.size < 2: k_mod = np.arange(x_mod.size)
        xs_m, ys_m = x_mod[k_mod], y_mod[k_mod]

        w_e = np.ones_like(xs_e)
        w_m = np.ones_like(xs_m)

        # w_e = _distance_weights(xs_e, half_life=weight_half_life)
        # w_m = _distance_weights(xs_m, half_life=weight_half_life)

        # --- τ candidates (separate) ---
        if tau_mode_exp == "fixed":
            if hz not in tau_fix_exp:
                raise ValueError(f"tau_mode_exp='fixed' but tau_fixed_experiment missing for {hz}")
            tau_candidates_exp = np.atleast_1d(float(tau_fix_exp[hz]))
        else:
            tmin, tmax = tau_bounds_exp.get(hz, (0.5, 2.5))
            tau_candidates_exp = np.linspace(max(1e-6, tmin), max(tmin, tmax), int(n_grid))

        if tau_mode_model == "fixed":
            if hz not in tau_fix_mod:
                raise ValueError(f"tau_mode_model='fixed' but tau_fixed_model missing for {hz}")
            tau_candidates_mod = np.atleast_1d(float(tau_fix_mod[hz]))
        else:
            tminm, tmaxm = tau_bounds_model.get(hz, (0.5, 2.5))
            tau_candidates_mod = np.linspace(max(1e-6, tminm), max(tminm, tmaxm), int(n_grid))

        # --- fit EXP ---
        best_e = {"tau": None, "A": None, "c": None, "sse": np.inf}
        for tau in tau_candidates_exp:
            if c_mode == "fixed":
                c = float(c_map.get(hz, 1.0))
                A = _fit_A_given_tau_c(xs_e, ys_e, w_e, tau, c)
            else:
                A, c = _fit_A_c_given_tau(xs_e, ys_e, w_e, tau)
            sse = _sse_weighted(ys_e, c + A*np.exp(xs_e/tau), w_e)
            if sse < best_e["sse"]:
                best_e.update({"tau": float(tau), "A": float(A), "c": float(c), "sse": sse})

        # --- fit MODEL ---
        best_m = {"tau": None, "A": None, "c": None, "sse": np.inf}
        for tau in tau_candidates_mod:
            if c_mode == "fixed":
                c = float(c_map.get(hz, 1.0))
                A = _fit_A_given_tau_c(xs_m, ys_m, w_m, tau, c)
            else:
                A, c = _fit_A_c_given_tau(xs_m, ys_m, w_m, tau)
            sse = _sse_weighted(ys_m, c + A*np.exp(xs_m/tau), w_m)
            if sse < best_m["sse"]:
                best_m.update({"tau": float(tau), "A": float(A), "c": float(c), "sse": sse})

        results[hz] = {"exp": {k: best_e[k] for k in ("tau","A","c")},
                       "model": {k: best_m[k] for k in ("tau","A","c")}}

        # --- plot points ---
        ax.plot(np.sort(x_exp), y_exp[np.argsort(x_exp)], "ko", ms=5, label="Experiment")
        ax.plot(np.sort(x_mod), y_mod[np.argsort(x_mod)],
                marker="o", ms=5, mfc="none", mec="tab:blue", ls="", label="Model")

        # --- plot curves ---
        xx_e = np.linspace(xs_e.min(), xs_e.max(), 400)
        yy_e = best_e["c"] + best_e["A"]*np.exp(xx_e / best_e["tau"])
        ax.plot(xx_e, yy_e, "-", color="red", lw=2,
                label=fr"Exp fit ($\tau\approx{best_e['tau']:.2f}$, $c\approx{best_e['c']:.2f}$)")

        xx_m = xx_e if draw_model_curve_on_exp_span else np.linspace(xs_m.min(), xs_m.max(), 400)
        yy_m = best_m["c"] + best_m["A"]*np.exp(xx_m / best_m["tau"])
        ax.plot(xx_m, yy_m, "--", color="tab:blue", lw=2,
                label=fr"Model fit ($\tau\approx{best_m['tau']:.2f}$, $c\approx{best_m['c']:.2f}$)")

        # cosmetics per-panel
        ax.set_xlim(*xlim)
        ax.set_ylim(0, y_max_map[hz])
        ax.set_xticks([-4, -3, -2, -1, 0])
        ax.set_title(ttl, loc="left", fontsize=11, pad=2)
        ax.legend(loc="upper left", fontsize=5, frameon=False)

    # --- 5th subpanel: τ summary lines ---
    # collect taus in numeric-Hz order (not subplot order)
    hz_sorted = sorted(panels)
    tau_exp   = [results[h]["exp"]["tau"]   for h in hz_sorted]
    tau_mod   = [results[h]["model"]["tau"] for h in hz_sorted]

    tau_ax.plot(hz_sorted, tau_exp, "-o", color="red",  label="Exp τ")
    tau_ax.plot(hz_sorted, tau_mod, "-o", color="tab:blue", label="Model τ", linestyle="--")
    tau_ax.set_xlabel("Frequency (Hz)")
    tau_ax.set_ylabel("Time constant τ (s)")
    tau_ax.set_title("τ summary")
    tau_ax.set_ylim(0,3)
    tau_ax.legend(frameon=False)

    # global labels
    fig.text(0.40, 0.04, "xlabel = Time from plateau (s)", ha="center")
    fig.text(0.06, 0.50, "Weight Change", va="center", rotation="vertical")

    plt.tight_layout(rect=(0.06, 0.06, 1, 1))
    plt.show()
    return results








def fit_and_plot_full_kernel_paper_style(
    cleaned, label="20Hz_full_kernel", title="20 Hz (full kernel)",
    tau_bounds_back=(1.2, 1.7), tau_bounds_fwd=(0.55, 0.95),
    exclude_window_s=0.25, weight_half_life=0.8, n_grid=500
):
    """
    Splits the full kernel into backward (x<=0) and forward (x>=0) sides,
    fits y = 1 + A*exp(x/τ) with c=1 on backward and y = 1 + A*exp((-x)/τ)
    with c=1 on forward, using trimming, weights, and tight τ bounds.
    Returns (tau_back, tau_fwd).
    """
    if label not in cleaned:
        raise KeyError(f"`cleaned` missing key {label!r}")

    x_ms = np.asarray(cleaned[label]["x"])
    y    = np.asarray(cleaned[label]["y"], dtype=float) + 1.0
    x_s  = _to_seconds_backward(x_ms)  # negative before, positive after

    # split
    back_mask = x_s <= 0
    fwd_mask  = x_s >= 0

    # ---------- backward ----------
    xb = x_s[back_mask]; yb = y[back_mask]
    kb = np.where(np.abs(xb) >= float(exclude_window_s))[0]
    xb, yb = (xb if kb.size < 2 else xb[kb]), (yb if kb.size < 2 else yb[kb])
    wb = _distance_weights(xb, half_life=weight_half_life)

    tau_b, A_b = fit_tau_A_fixed_c1_weighted(
        xb, yb, tau_min=tau_bounds_back[0], tau_max=tau_bounds_back[1],
        n_grid=n_grid, weights=wb, c=1.0
    )
    xxb = np.linspace(xb.min(), xb.max(), 400)
    yyb = exp_rise_fixed_c1(xxb, A_b, tau_b)

    # ---------- forward ----------
    xf = x_s[fwd_mask]; yf = y[fwd_mask]
    kf = np.where(np.abs(xf) >= float(exclude_window_s))[0]
    xf, yf = (xf if kf.size < 2 else xf[kf]), (yf if kf.size < 2 else yf[kf])
    # for forward, model y = 1 + A*exp((-x)/τ). Define z = -x (<= 0) and reuse the same fitter.
    zf = -xf
    wf = _distance_weights(zf, half_life=weight_half_life)

    tau_f, A_f = fit_tau_A_fixed_c1_weighted(
        zf, yf, tau_min=tau_bounds_fwd[0], tau_max=tau_bounds_fwd[1],
        n_grid=n_grid, weights=wf, c=1.0
    )
    xxf = np.linspace(xf.min(), xf.max(), 400)
    yyf = 1.0 + A_f * np.exp((-xxf) / float(tau_f))

    # ---------- plot ----------
    plt.figure(figsize=(6.2, 4.6))
    order = np.argsort(x_s)
    plt.plot(x_s[order], y[order], "ko", ms=5, label="Experiment")
    plt.plot(xxb, yyb, "-", color="red", lw=2,
             label=fr"Backward fit (τ≈{tau_b:.2f}s)")
    plt.plot(xxf, yyf, "-", color="red", lw=2,
             label=fr"Forward fit (τ≈{tau_f:.2f}s)")

    plt.xlabel("Time from plateau (s)")
    plt.ylabel("Normalized EPSP (Δ units)")
    plt.title(title)
    plt.xlim(x_s.min() - 0.05, x_s.max() + 0.05)
    plt.ylim(y.min() - 0.05, y.max() + 0.2)
    plt.legend(frameon=False, fontsize=9)
    plt.tight_layout()
    plt.show()

    return tau_b, tau_f


def plot_full_kernel_panel(
    cleaned, products_dict,
    label="20Hz_full_kernel",
    title="20 Hz (full kernel)",
    color_model="tab:blue"):
    if label not in cleaned or label not in products_dict:
        raise KeyError(f"Need both cleaned and products_dict entries for {label!r}")

    # +1 offset back to normalized scale
    x_exp_ms = np.asarray(cleaned[label]["x"])
    y_exp    = np.asarray(cleaned[label]["y"], dtype=float) + 1.0
    x_mod_ms = np.asarray(products_dict[label]["x"])
    y_mod    = np.asarray(products_dict[label]["y"], dtype=float) + 1.0

    # align by common x
    x_ms, y_e, y_m = _align_by_x_ms(x_exp_ms, y_exp, x_mod_ms, y_mod)

    # flip sign so negative = before, positive = after plateau
    x_s = (-x_ms) / 1000.0

    order = np.argsort(x_s)
    x_sorted  = x_s[order]
    y_e_sorted = y_e[order]
    y_m_sorted = y_m[order]

    plt.figure(figsize=(5, 4))
    # experiment: black line + points
    plt.plot(x_sorted, y_e_sorted, "k-", lw=1.5, label="Experiment")
    plt.plot(x_sorted, y_e_sorted, "ko", ms=5, label="_nolegend_")
    # model: blue line + points
    plt.plot(x_sorted, y_m_sorted, "-", lw=2, color=color_model, label="Model")
    plt.plot(x_sorted, y_m_sorted, "o", ms=5, color=color_model, label="_nolegend_")

    plt.xlabel("Time from plateau (s)")
    plt.ylabel("Normalized EPSP (Δ units)")
    plt.title(title)
    plt.legend(frameon=False)
    plt.xlim(x_sorted[0] - 0.05, x_sorted[-1] + 0.05)
    plt.tight_layout()
    plt.show()

def fit_exp_zerobase(x, y):
    x = np.asarray(x, float)
    y = np.asarray(y, float)

    # keep strictly positive y for the log
    m = np.isfinite(x) & np.isfinite(y) & (y > 0)
    if m.sum() < 2:
        raise ValueError("Need at least two positive points to fit.")

    # sort by x so the curve looks nice
    idx = np.argsort(x[m])
    xs, ys = x[m][idx], y[m][idx]

    # ln y = ln A + (1/tau) * x
    slope, intercept = np.polyfit(xs, np.log(ys), 1)
    A   = float(np.exp(intercept))
    tau = float(1.0 / slope)
    return A, tau

def exp_model(x, A, tau):
    x = np.asarray(x, float)
    return A * np.exp(x / float(tau))



def exp_with_c(x, A, tau, c):
    return c + A * np.exp(x / tau)

def fit_exp_with_c(x, y, k_tail=3, tau_positive=True):
    x = np.asarray(x, float); y = np.asarray(y, float)
    m = np.isfinite(x) & np.isfinite(y)
    x, y = x[m], y[m]
    idx = np.argsort(x); x, y = x[idx], y[idx]

    # inits
    c0 = float(np.mean(y[-k_tail:])) if y.size >= k_tail else float(np.median(y))
    A0 = max(1e-8, float(y[0] - c0))
    tau0 = max(1e-6, x.ptp() if x.ptp() > 0 else 1.0)

    # bounds (keep it simple): A>=0, tau>0 if desired, c free
    if tau_positive:
        bounds = ([0.0, 1e-9, -np.inf], [np.inf, np.inf, np.inf])
    else:
        bounds = (-np.inf, np.inf)

    popt, pcov = curve_fit(exp_with_c, x, y, p0=(A0, tau0, c0),
                        bounds=bounds, maxfev=20000)
    A, tau, c = map(float, popt)

    # quick R^2
    yhat = exp_with_c(x, A, tau, c)
    resid = y - yhat
    ss_res = float(resid @ resid)
    ss_tot = float(((y - y.mean()) @ (y - y.mean())))
    R2 = 1.0 - ss_res/ss_tot if ss_tot > 0 else np.nan
    return A, tau, c, R2


def fit_exp_fixed_c(x, y, c_fixed):
    """
    Fit y ≈ c_fixed + A * exp(x/τ) with c fixed (paper-style).
    Returns A, tau. Works for backward or forward (use z=-x for forward).
    """
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    y0 = y - float(c_fixed)

    m = np.isfinite(x) & np.isfinite(y0) & (y0 > 0)
    if m.sum() < 2:
        raise ValueError("Need at least two positive (y - c_fixed) points.")

    idx = np.argsort(x[m])
    xs, ys = x[m][idx], y0[m][idx]

    # log-linear: ln(y-c) = ln A + (1/τ) x
    s, b = np.polyfit(xs, np.log(ys), 1)
    A   = float(np.exp(b))
    tau = float(1.0 / s)
    return A, tau

def exp_with_fixed_c(x, A, tau, c_fixed):
    return float(c_fixed) + A*np.exp(x/float(tau))




def exp_with_fixed_c(x, A, tau, c_fixed):
    x = np.asarray(x, float)
    return float(c_fixed) + A*np.exp(x/float(tau))

def exp_c0(x, A, tau):
    return exp_with_fixed_c(x, A, tau, c_fixed=0.0)

def exp_back_c0(x, A, tau):   # for x <= 0
    return A*np.exp(x/tau)

def exp_fwd_c0(x, A, tau):    # for x >= 0
    return A*np.exp(-x/tau)






def fit_exp_fixed_c_grid(x, y, c_fixed, tau_min=0.5, tau_max=2.5, n_grid=2000):
    """
    Fit y ≈ c_fixed + A * exp(x/τ) with c fixed by grid-searching τ in y-space
    and solving A by least squares for each τ. Returns (A_best, tau_best).
    """
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    c = float(c_fixed)

    m = np.isfinite(x) & np.isfinite(y)
    x = x[m]; y = y[m]

    taus = np.linspace(tau_min, tau_max, int(n_grid))
    best = (np.inf, None, None)  # (sse, A, tau)

    y_c = y - c
    for tau in taus:
        e = np.exp(x / tau)
        # Solve A in least squares sense: y_c ≈ A * e
        denom = np.dot(e, e)
        if denom <= 0.0:
            continue
        A = np.dot(e, y_c) / denom
        resid = y_c - A * e
        sse = float(np.dot(resid, resid))
        if sse < best[0]:
            best = (sse, float(A), float(tau))

    if best[1] is None:
        raise ValueError("Could not fit A,tau (check data / tau range).")
    return best[1], best[2]



def plot_fixed_data(cleaned_data_dict, products_dict, fixed_c=0.01, fit_grid=True):

    # === your plotting ===
    fig, axs = plt.subplots(2, 3, figsize=(11, 6))
    axs = axs.flat
    panel_i = 0

    tau_dict_experimental = {}
    tau_dict_model = {}

    max_dict = {}

    # ---- 10 / 40 / 100 Hz (backward; x ≤ 0) ----
    for hz in [10, 40, 100]:
        d  = cleaned_data_dict[hz]
        dm = products_dict[hz]

        x  = np.asarray(d['x'],  float) / -1000.0
        y  = np.asarray(d['y'],  float)
        xm = np.asarray(dm['x'], float) / -1000.0
        ym = np.asarray(dm['y'], float)

        m  = np.isfinite(x)  & np.isfinite(y)
        mm = np.isfinite(xm) & np.isfinite(ym)
        xs, ys   = x[m],  y[m]
        xms, yms = xm[mm], ym[mm]
        o  = np.argsort(xs);  xs, ys   = xs[o],  ys[o]
        om = np.argsort(xms); xms,yms  = xms[om],yms[om]

        max_dict[hz] = np.max(ym)

        if fit_grid:
            A,  tau= fit_exp_fixed_c_grid(xs,  ys, fixed_c)
            Am, taum= fit_exp_fixed_c_grid(xms, yms, fixed_c)
        else:
            A,  tau= fit_exp_fixed_c(xs,  ys, fixed_c)
            Am, taum= fit_exp_fixed_c(xms, yms, fixed_c)


        tau_dict_experimental[hz] = tau
        tau_dict_model[hz] = taum

        x_fit   = np.linspace(xs.min(),  xs.max(),  200)
        x_fit_m = np.linspace(xms.min(), xms.max(), 200)

        # if fit_grid:
        #     y_fit   = fit_exp_fixed_c_grid(x_fit,   A,  tau,  fixed_c)
        #     y_fit_m = fit_exp_fixed_c_grid(x_fit_m, Am, taum, fixed_c)
        # else:
        y_fit   = exp_with_c(x_fit,   A,  tau,  fixed_c)
        y_fit_m = exp_with_c(x_fit_m, Am, taum, fixed_c)

        ax = axs[panel_i]; panel_i += 1
        ax.plot(xs,  ys,  'o', color='k', label='data (exp)')
        ax.plot(x_fit,  y_fit,  '-', lw=2, color='r',
                label=f'fit exp+c τ={tau:.2f}')
        ax.plot(xms, yms, 'o', color='b', label='data (model)')
        ax.plot(x_fit_m, y_fit_m, '-', lw=2, color='purple',
                label=f'fit model+c τ={taum:.2f}')
        ax.set_ylim(0, 3)
        ax.set_xlabel('Time from plateau (s)')
        ax.set_ylabel('ΔW')
        ax.set_title(f'{hz} Hz')
        ax.legend(fontsize=7, loc='upper left')

    # ---- 20 Hz backward (merge) ----
    x20  = np.asarray(cleaned_data_dict[20]['x'], float) / -1000.0
    y20  = np.asarray(cleaned_data_dict[20]['y'], float)
    xm20 = np.asarray(products_dict[20]['x'], float) / -1000.0
    ym20 = np.asarray(products_dict[20]['y'], float)
    max_dict[20] = np.max(ym20)

    m   = np.isfinite(x20)  & np.isfinite(y20)
    mm  = np.isfinite(xm20) & np.isfinite(ym20)
    xs20, ys20   = x20[m],  y20[m]
    xms20, yms20 = xm20[mm], ym20[mm]
    o  = np.argsort(xs20);  xs20,  ys20  = xs20[o],  ys20[o]
    om = np.argsort(xms20); xms20, yms20 = xms20[om], yms20[om]

    xfk = np.asarray(cleaned_data_dict["20Hz_full_kernel"]['x'], float) / -1000.0
    yfk = np.asarray(cleaned_data_dict["20Hz_full_kernel"]['y'], float)
    xm_fk = np.asarray(products_dict["20Hz_full_kernel"]['x'], float) / -1000.0
    ym_fk = np.asarray(products_dict["20Hz_full_kernel"]['y'], float)

    max_dict["20 forward"] = np.max(ym_fk)


    mk   = np.isfinite(xfk) & np.isfinite(yfk)
    mmk  = np.isfinite(xm_fk) & np.isfinite(ym_fk)
    xs_fk, ys_fk   = xfk[mk],   yfk[mk]
    xms_fk, yms_fk = xm_fk[mmk], ym_fk[mmk]
    ok  = np.argsort(xs_fk);   xs_fk,  ys_fk  = xs_fk[ok],  ys_fk[ok]
    omk = np.argsort(xms_fk);  xms_fk, yms_fk = xms_fk[omk], yms_fk[omk]

    xb, yb     = xs_fk[:5],   ys_fk[:5]
    xmb, ymb   = xms_fk[:5],  yms_fk[:5]
    xfwd, yfwd   = xs_fk[-5:],   ys_fk[-5:]
    xmfwd, ymfwd = xms_fk[-5:],  ym_fk[-5:]

    xs20m  = np.concatenate([xs20,  xb])
    ys20m  = np.concatenate([ys20,  yb])
    xms20m = np.concatenate([xms20, xmb])
    yms20m = np.concatenate([yms20, ymb])
    o  = np.argsort(xs20m);  xs20m,  ys20m  = xs20m[o],  ys20m[o]
    om = np.argsort(xms20m); xms20m, yms20m = xms20m[om], yms20m[om]
    
    if fit_grid:
        A20,  tau20 = fit_exp_fixed_c_grid(xs20m,  ys20m, fixed_c)
        Am20, taum20 = fit_exp_fixed_c_grid(xms20m, yms20m, fixed_c)
    else:
        A20,  tau20 = fit_exp_fixed_c(xs20m,  ys20m, fixed_c)
        Am20, taum20 = fit_exp_fixed_c(xms20m, yms20m, fixed_c)

    x_fit20   = np.linspace(xs20m.min(),  xs20m.max(),  200)
    x_fit20_m = np.linspace(xms20m.min(), xms20m.max(), 200)

    # if fit_grid:
    #     y_fit20   = fit_exp_fixed_c_grid(x_fit20,   A20,  tau20,  fixed_c)
    #     y_fit20_m = fit_exp_fixed_c_grid(x_fit20_m, Am20, taum20, fixed_c)
    # else:
    y_fit20   = exp_with_c(x_fit20,   A20,  tau20,  fixed_c)
    y_fit20_m = exp_with_c(x_fit20_m, Am20, taum20, fixed_c)

    

    ax20 = axs[panel_i]; panel_i += 1
    ax20.plot(xs20m,  ys20m,  'o', color='k', label='20 Hz data (exp, merged)')
    ax20.plot(x_fit20, y_fit20, '-', lw=2, color='r', label=f'fit exp+c={fixed_c} τ={tau20:.2f}')
    ax20.plot(xms20m, yms20m, 'o', color='b', label='20 Hz data (model, merged)')
    ax20.plot(x_fit20_m, y_fit20_m, '-', lw=2, color='purple', label=f'fit model+c={fixed_c} τ={taum20:.2f}')
    ax20.set_ylim(0, 3)
    ax20.set_xlabel('Time from plateau (s)')
    ax20.set_ylabel('ΔW')
    ax20.set_title('20 Hz — backward (merged)')
    ax20.legend(fontsize=7, loc='upper left')

    # ---- forward arm (x ≥ 0) : fit in z = -x so it’s a rise ----
    z_fwd   = -xfwd
    zm_fwd  = -xmfwd

    if fit_grid:
        Af,  tauf = fit_exp_fixed_c_grid(z_fwd,  yfwd, fixed_c)
        Amf, taumf = fit_exp_fixed_c_grid(zm_fwd, ymfwd, fixed_c)
    else:
        Af,  tauf = fit_exp_fixed_c(z_fwd,  yfwd, fixed_c)
        Amf, taumf = fit_exp_fixed_c(zm_fwd, ymfwd, fixed_c)

    

    x_fit_f   = np.linspace(xfwd.min(),  xfwd.max(),  200)
    x_fit_fm  = np.linspace(xmfwd.min(), xmfwd.max(), 200)

    # if fit_grid:
    #     y_fit_f   = fit_exp_fixed_c_grid(-x_fit_f,  Af,  tauf,  fixed_c)   # note the minus here
    #     y_fit_fm  = fit_exp_fixed_c_grid(-x_fit_fm, Amf, taumf, fixed_c)
    # else:
    y_fit_f   = exp_with_c(-x_fit_f,  Af,  tauf,  fixed_c)   # note the minus here
    y_fit_fm  = exp_with_c(-x_fit_fm, Amf, taumf, fixed_c)

    

    # store taus
    tau_dict_experimental['20Hz Forward'] = abs(tauf)
    tau_dict_model['20Hz Forward']        = abs(taumf)
    tau_dict_experimental['20Hz Backward'] = tau20
    tau_dict_model['20Hz Backward']        = taum20

    ax5 = axs[4]
    ax5.plot(xfwd,  yfwd,  'o', color='k', label='forward data (exp)')
    ax5.plot(x_fit_f,  y_fit_f,  '-', lw=2, color='r',      label=f'forward fit c={fixed_c} (τ={tauf:.2f})')
    ax5.plot(xmfwd, ymfwd, 'o', color='b', label='forward data (model)')
    ax5.plot(x_fit_fm, y_fit_fm, '-', lw=2, color='purple', label=f'forward fit model c={fixed_c} (τ={taumf:.2f})')
    ax5.set_title('20 Hz — forward arm')
    ax5.set_ylim(0, 3)
    ax5.set_xlabel('Time from plateau (s)')
    ax5.set_ylabel('ΔW')
    ax5.legend(fontsize=7, loc='upper right')

    # ---- panel 6: τ summary ----
    axs[5].plot(
        [tau_dict_experimental[10],
        tau_dict_experimental["20Hz Backward"],
        tau_dict_experimental[40],
        tau_dict_experimental[100],
        tau_dict_experimental["20Hz Forward"]],
        color='r', marker='o', label='Experimental τ')
    axs[5].plot(
        [tau_dict_model[10],
        tau_dict_model["20Hz Backward"],
        tau_dict_model[40],
        tau_dict_model[100],
        tau_dict_model["20Hz Forward"]],
        color='purple', marker='o', label='Model τ')
    axs[5].set_ylabel("Tau (sec)")
    axs[5].set_xticks(np.arange(5))
    axs[5].set_xticklabels(["10Hz","20Hz Back (merged)","40Hz","100Hz","20Hz Fwd"], rotation=45, ha='right')
    axs[5].set_ylim(0, 3)
    axs[5].legend(fontsize=7)

    plt.tight_layout()
    plt.show()

    return max_dict


