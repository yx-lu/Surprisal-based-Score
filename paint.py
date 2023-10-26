import numpy as np
import copy
from matplotlib import pyplot as plt
import calc
from scipy.stats import beta
from matplotlib.colors import LogNorm
import matplotlib.gridspec as gridspec
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
import matplotlib.ticker as ticker
from matplotlib.lines import Line2D

small_font = 32
medium_font = 40
big_font = 48


def paint_correct_rate_fix_lam(lam_A, lam_B, n):  # Paint the precise correct rate of our method and baseline fixing lambda and n
    plt.cla()
    print('paint_correct_rate_fix_lam', lam_A, lam_B, n)
    plt.figure(figsize=(22.8, 9.6))

    x = np.linspace(0, 1, 400)
    y = np.linspace(0, 1, 400)
    X, Y = np.meshgrid(x, y)
    Z_ours = copy.deepcopy(X)
    Z_ours_evil = copy.deepcopy(X)
    Z_baseline = copy.deepcopy(X)

    for i in range(len(X)):
        print(i)
        for j in range(len(X[i])):
            if X[i][j] == 0.: X[i][j] += calc.eps
            if X[i][j] == 1.: X[i][j] -= calc.eps
            if Y[i][j] == 0.: Y[i][j] += calc.eps
            if Y[i][j] == 1.: Y[i][j] -= calc.eps
            Z_ours[i][j], Z_baseline[i][j] = calc.calc_correct_rate(float(X[i][j]), float(Y[i][j]), lam_A, lam_B, n)
            #Z_ours_evil[i][j] = calc.calc_correct_rate(float(X[i][j]), float(Y[i][j]), lam_A, lam_B, n, evil=True)[0]

    # Ours
    ax = plt.subplot(1, 2, 1)  #figsize=(8, 8), dpi=100
    CS = ax.contourf(X, Y, Z_ours, [0., .1, .2, .3, .4, .5, .6, .7, .8, .9, 1.])
    cbar = plt.colorbar(CS)
    cbar.set_label('Accuracy', fontdict={'size': medium_font})
    cbar.ax.tick_params(labelsize=small_font)
    CS = ax.contour(X, Y, Z_ours, [0., .1, .2, .3, .4, .5, .6, .7, .8, .9, 1.])
    #plt.clabel(CS, inline=True, fontsize=10, colors='black')
    ax.set_xlabel('$p_A$', fontsize=medium_font)
    ax.set_ylabel('$p_B$', fontsize=medium_font)
    ax.set_title('Our method', fontsize=big_font)
    plt.xticks(size=small_font)
    plt.yticks(size=small_font)

    # Baseline
    ax = plt.subplot(1, 2, 2)  #figsize=(8, 8), dpi=100
    CS = ax.contourf(X, Y, Z_baseline, [0., .1, .2, .3, .4, .5, .6, .7, .8, .9, 1.])
    cbar = plt.colorbar(CS)
    cbar.set_label('Accuracy', fontdict={'size': medium_font})
    cbar.ax.tick_params(labelsize=small_font)
    CS = ax.contour(X, Y, Z_baseline, [0., .1, .2, .3, .4, .5, .6, .7, .8, .9, 1.])
    #plt.clabel(CS, inline=True, fontsize=10, colors='black')
    ax.set_xlabel('$p_A$', fontsize=medium_font)
    ax.set_ylabel('$p_B$', fontsize=medium_font)
    ax.set_title('Baseline', fontsize=big_font)
    plt.xticks(size=small_font)
    plt.yticks(size=small_font)

    #plt.suptitle('Comparasion of our method and baseline under $λ_A=%.2f$ and $λ_B=%.2f$' % (lam_A, lam_B), fontsize=30)
    plt.tight_layout()
    plt.savefig('./fig/correct_rate_fix_lam/%s_%d_%.1f_%.1f.pdf' % (calc.bias_type, n, lam_A, lam_B))


def paint_correct_rate_fix_p(p_A, p_B, n):  # Paint the precise correct rate of our method and baseline fixing p and n
    plt.cla()
    print('paint_correct_rate_fix_p', p_A, p_B, n)
    plt.figure(figsize=(25, 10.4))

    x = np.linspace(0, 1, 400)
    y = np.linspace(0, 1, 400)
    X, Y = np.meshgrid(x, y)
    Z_ours = copy.deepcopy(X)
    Z_ours_evil = copy.deepcopy(X)
    Z_baseline = copy.deepcopy(X)

    for i in range(len(X)):
        print(i)
        for j in range(len(X[i])):
            if X[i][j] == 0.: X[i][j] += calc.eps
            if X[i][j] == 1.: X[i][j] -= calc.eps
            if Y[i][j] == 0.: Y[i][j] += calc.eps
            if Y[i][j] == 1.: Y[i][j] -= calc.eps

            Z_ours[i][j], Z_baseline[i][j] = calc.calc_correct_rate(p_A, p_B, float(X[i][j]), float(Y[i][j]), n)
            Z_ours_evil[i][j] = calc.calc_correct_rate(p_A, p_B, float(X[i][j]), float(Y[i][j]), n, True)[0]

    # Ours
    ax = plt.subplot(1, 2, 1)  #figsize=(8, 8), dpi=100
    CS = ax.contourf(X, Y, Z_ours, [0., .1, .2, .3, .4, .5, .6, .7, .8, .9, 1.])
    cbar = plt.colorbar(CS)
    cbar.set_label('Accuracy', fontdict={'size': medium_font})
    cbar.ax.tick_params(labelsize=small_font)
    CS = ax.contour(X, Y, Z_ours, [0., .1, .2, .3, .4, .5, .6, .7, .8, .9, 1.])
    #plt.clabel(CS, inline=True, fontsize=10, colors='black')
    ax.set_xlabel('$λ_A$', fontsize=medium_font)
    ax.set_ylabel('$λ_B$', fontsize=medium_font)
    ax.set_title('Our method', fontsize=big_font)
    plt.xticks(size=small_font)
    plt.yticks(size=small_font)

    # Baseline
    ax = plt.subplot(1, 2, 2)  #figsize=(8, 8), dpi=100
    CS = ax.contourf(X, Y, Z_baseline, [0., .1, .2, .3, .4, .5, .6, .7, .8, .9, 1.])
    cbar = plt.colorbar(CS)
    cbar.set_label('Accuracy', fontdict={'size': medium_font})
    cbar.ax.tick_params(labelsize=small_font)
    CS = ax.contour(X, Y, Z_baseline, [0., .1, .2, .3, .4, .5, .6, .7, .8, .9, 1.])
    #plt.clabel(CS, inline=True, fontsize=10, colors='black')
    ax.set_xlabel('$λ_A$', fontsize=medium_font)
    ax.set_ylabel('$λ_B$', fontsize=medium_font)
    ax.set_title('Baseline', fontsize=big_font)
    plt.xticks(size=small_font)
    plt.yticks(size=small_font)

    #plt.suptitle('Comparasion of our method and baseline under $p_A=%.2f$ and $p_B=%.2f$' % (p_A, p_B), fontsize=30)
    plt.tight_layout()
    plt.savefig('./fig/correct_rate_fix_p/%s_%d_%.1f_%.1f.pdf' % (calc.bias_type, n, p_A, p_B))


def paint_correct_rate_int_p(n):
    assert (0)
    plt.figure(figsize=(19.2, 9.6))

    x = np.linspace(0, 1, 100)
    y = np.linspace(0, 1, 100)
    X, Y = np.meshgrid(x, y)
    Z_ours = copy.deepcopy(X)
    Z_ours_evil = copy.deepcopy(X)
    Z_baseline = copy.deepcopy(X)

    for i in range(len(X)):
        print(i)
        for j in range(len(X[i])):
            if X[i][j] < calc.eps: X[i][j] = calc.eps
            if X[i][j] > 1. - calc.eps: X[i][j] = 1. - calc.eps
            if Y[i][j] < calc.eps: Y[i][j] = calc.eps
            if Y[i][j] > calc.eps: Y[i][j] = 1. - calc.eps

            Z_ours[i][j] = 0.
            Z_baseline[i][j] = 0.
            Z_ours_evil[i][j] = 0.
            sli = 10
            step = 1. / sli
            for tmp_A in range(sli):
                for tmp_B in range(sli):
                    p_A = float(tmp_A) / sli
                    p_B = float(tmp_B) / sli
                    prob = beta.cdf(p_A + step, .2, .2) - beta.cdf(p_A, .2, .2)  # Beta(3,3)
                    prob *= beta.cdf(p_B + step, .2, .2) - beta.cdf(p_B, .2, .2)

                    tmp_ours, tmp_baseline = calc.calc_correct_rate(p_A + step / 2, p_B + step / 2, float(X[i][j]), float(Y[i][j]), n)
                    Z_ours[i][j] += tmp_ours * prob
                    Z_baseline[i][j] += tmp_baseline * prob

    # Ours
    ax = plt.subplot(1, 2, 1)  #figsize=(8, 8), dpi=100
    CS = ax.contourf(X, Y, Z_ours, [.5, .55, .6, .65, .7, .75, .8, .85, .9, .95, 1.])
    cbar = plt.colorbar(CS)
    cbar.set_label('Accuracy', fontdict={'size': 20})
    cbar.ax.tick_params(labelsize=16)
    CS = ax.contour(X, Y, Z_ours, [.5, .55, .6, .65, .7, .75, .8, .85, .9, .95, 1.])
    #plt.clabel(CS, inline=True, fontsize=10, colors='black')
    ax.set_xlabel('$λ_A$', fontsize=20)
    ax.set_ylabel('$λ_B$', fontsize=20)
    ax.set_title('Our method', fontsize=24)
    plt.xticks(size=16)
    plt.yticks(size=16)

    # Baseline
    ax = plt.subplot(1, 2, 2)  #figsize=(8, 8), dpi=100
    CS = ax.contourf(X, Y, Z_baseline, [.5, .55, .6, .65, .7, .75, .8, .85, .9, .95, 1.])
    cbar = plt.colorbar(CS)
    cbar.set_label('Accuracy', fontdict={'size': 20})
    cbar.ax.tick_params(labelsize=16)
    CS = ax.contour(X, Y, Z_baseline, [.5, .55, .6, .65, .7, .75, .8, .85, .9, .95, 1.])
    #plt.clabel(CS, inline=True, fontsize=10, colors='black')
    ax.set_xlabel('$λ_A$', fontsize=20)
    ax.set_ylabel('$λ_B$', fontsize=20)
    ax.set_title('Baseline', fontsize=24)
    plt.xticks(size=16)
    plt.yticks(size=16)

    plt.suptitle('Comparasion of our method and baseline', fontsize=30)
    plt.savefig('tmp.png')


def paint_line_chart(option, n_reviewer, which_base):
    # Paint the precise correct rate of our method and baseline fixing lambda (of the first paper)

    print('paint_line_chart')

    wr = [8, 8]
    hr = [5, 5]
    plt.figure(figsize=(sum(wr) * 1.2, sum(hr) * 1.2))
    gs = gridspec.GridSpec(nrows=len(hr), ncols=len(wr), width_ratios=wr, height_ratios=hr)

    for fig_id in range(3):
        ax = [plt.subplot(gs[0, 1]), plt.subplot(gs[1, 0]), plt.subplot(gs[1, 1])][fig_id]
        lam_A = [0.0, 0.3, 0.6][fig_id]
        ax.set_xlim(0, 1)
        ax.set_xticks([0., .2, .4, .6, .8, 1.])
        ax.tick_params(axis='x', labelsize=small_font)
        ax.set_ylim(.5, .8)
        ax.set_yticks([.5, .6, .7, .8])
        if n_reviewer == 5:
            ax.set_ylim(.5, .9)
            ax.set_yticks([.5, .6, .7, .8, .9])
        ax.tick_params(axis='y', labelsize=small_font)
        #ax.set_xticks(size=small_font)
        #ax.set_yticks(size=small_font)
        ax.set_xlabel('$\lambda_B$', fontsize=medium_font)
        ax.set_ylabel('Accuracy', fontsize=medium_font)
        ax.set_title('$\lambda_A=%.1f$' % (lam_A), fontsize=big_font)

        for id in range(3):
            tp = ['Beta(.5,.5)', 'Beta(1,1)', 'Beta(3,3)'][id]
            para_alpha = [.5, 1., 3.][id]
            para_beta = [.5, 1., 3.][id]
            # tp = 'Beta(.1,.1)'
            # para_alpha = 1. / 10
            # para_beta = 1. / 10
            linestyle = ['dotted', 'solid', 'dashed'][id]
            calc.set_U(tp)

            glo = 200

            Z_ours = [0. for i in range(glo)]
            Z_baseline = [0. for i in range(glo)]
            if option == 'calc':
                for i in range(glo):
                    print(i)
                    lam_B = (i + .5) / glo
                    sli = 100
                    step = 1. / sli
                    data = [[0. for i in range(sli + 1)] for i in range(sli + 1)]
                    for tmp_A in range(sli + 1):
                        for tmp_B in range(sli + 1):
                            p_A = float(tmp_A) / sli
                            p_B = float(tmp_B) / sli
                            data[tmp_A][tmp_B] = calc.calc_correct_rate(p_A, p_B, lam_A, lam_B, n_reviewer, which_base)

                    for tmp_A in range(sli):
                        for tmp_B in range(sli):
                            p_A = float(tmp_A) / sli
                            p_B = float(tmp_B) / sli
                            prob = beta.cdf(p_A + step, para_alpha, para_beta) - beta.cdf(p_A, para_alpha, para_beta)
                            prob *= beta.cdf(p_B + step, para_alpha, para_beta) - beta.cdf(p_B, para_alpha, para_beta)

                            tmp_ours = (data[tmp_A + 1][tmp_B][0] + data[tmp_A][tmp_B + 1][0]) / 2
                            tmp_baseline = (data[tmp_A + 1][tmp_B][1] + data[tmp_A][tmp_B + 1][1]) / 2
                            Z_ours[i] += tmp_ours * prob
                            Z_baseline[i] += tmp_baseline * prob
                np.save('./save/%s/%s_%.1f_Z_ours_%d.npy' % (tp, calc.bias_type, lam_A, n_reviewer), Z_ours)
                np.save('./save/%s/%s_%.1f_Z_%s_%d.npy' % (tp, calc.bias_type, lam_A, which_base, n_reviewer), Z_baseline)
            else:
                Z_ours = np.load('./save/%s/%s_%.1f_Z_ours_%d.npy' % (tp, calc.bias_type, lam_A, n_reviewer))
                Z_baseline = np.load('./save/%s/%s_%.1f_Z_%s_%d.npy' % (tp, calc.bias_type, lam_A, which_base, n_reviewer))

            x = [(i + .5) / glo for i in range(glo)]
            ax.plot(x, Z_ours, label='Our method (%s)' % (tp), color='red', linestyle=linestyle, linewidth=3)
            ax.plot(x, Z_baseline, label='%s (%s)' % (which_base, tp), color='green', linestyle=linestyle, linewidth=3)

    legend_lines = [
        Line2D([0], [0], color='red', lw=3, linestyle='dotted', label='Our Method (Beta(.5,.5))'),
        Line2D([0], [0], color='green', lw=3, linestyle='dotted', label='%s (Beta(.5,.5))' % (which_base)),
        Line2D([0], [0], color='red', lw=3, linestyle='solid', label='Our Method (Beta(1,1))'),
        Line2D([0], [0], color='green', lw=3, linestyle='solid', label='%s (Beta(1,1))' % (which_base)),
        Line2D([0], [0], color='red', lw=3, linestyle='dashed', label='Our Method (Beta(3,3))'),
        Line2D([0], [0], color='green', lw=3, linestyle='dashed', label='%s (Beta(3,3))' % (which_base))
    ]
    ax = plt.subplot(gs[0, 0])
    legend = ax.legend(handles=legend_lines, loc='center', fontsize=small_font + 4)
    ax.add_artist(legend)
    ax.axis('off')

    plt.tight_layout()
    plt.savefig('./fig/line_chart/%s_n%d_%s.pdf' % (calc.bias_type, n_reviewer, which_base))


def test(lam_A, lam_B, sv):
    assert (0)
    for id in range(1):
        tp = ['Beta(.5,.5)', 'Beta(1,1)', 'Beta(3,3)'][id]
        para = [.5, 1., 3.][id]
        linestyle = ['dotted', 'solid', 'dashed'][id]
        calc.set_U(tp)

        Z_ours5 = 0.
        Z_baseline5 = 0.

        sli = 50
        step = 1. / sli
        data5 = [[0. for i in range(sli + 1)] for i in range(sli + 1)]
        for tmp_A in range(sli + 1):
            for tmp_B in range(sli + 1):
                p_A = float(tmp_A) / sli
                p_B = float(tmp_B) / sli
                data5[tmp_A][tmp_B] = calc.calc_correct_rate(p_A, p_B, lam_A, lam_B, 5)

        for tmp_A in range(sli):
            for tmp_B in range(sli):
                p_A = float(tmp_A) / sli
                p_B = float(tmp_B) / sli
                prob = beta.cdf(p_A + step, para, para) - beta.cdf(p_A, para, para)
                prob *= beta.cdf(p_B + step, para, para) - beta.cdf(p_B, para, para)

                tmp_ours5 = (data5[tmp_A + 1][tmp_B][0] + data5[tmp_A][tmp_B + 1][0]) / 2
                tmp_baseline5 = (data5[tmp_A + 1][tmp_B][1] + data5[tmp_A][tmp_B + 1][1]) / 2
                Z_ours5 += tmp_ours5 * prob
                Z_baseline5 += tmp_baseline5 * prob

                sv[tmp_A][tmp_B] = tmp_ours5

        print(Z_ours5)


def paint_score_sketch():  # Paint the sketch map of our score when n = 3 (figure 1)
    plt.cla()
    print('paint_score_sketch')
    plt.figure(figsize=(25, 10.4))

    x = np.linspace(0.05, 0.95, 500)
    y = np.linspace(0.05, 0.95, 500)
    X, Y = np.meshgrid(x, y)
    Z_ours0 = copy.deepcopy(X)
    Z_ours1 = copy.deepcopy(X)

    for i in range(len(X)):
        if i % 10 == 0: print(i)
        for j in range(len(X[i])):
            if X[i][j] == 0.: X[i][j] += calc.eps
            if X[i][j] == 1.: X[i][j] -= calc.eps
            if Y[i][j] == 0.: Y[i][j] += calc.eps
            if Y[i][j] == 1.: Y[i][j] -= calc.eps

            data0 = [(0, float(X[i][j])), (0, float(X[i][j])), (1, float(Y[i][j]))]
            data1 = [(0, float(X[i][j])), (1, float(Y[i][j])), (1, float(Y[i][j]))]

            Z_ours0[i][j] = calc.calc_score_from_feedback(data0)
            Z_ours1[i][j] = calc.calc_score_from_feedback(data1)

            if Z_ours0[i][j] > 32.: Z_ours0[i][j] = 32.
            if Z_ours0[i][j] < -32.: Z_ours0[i][j] = -32.
            if Z_ours1[i][j] > 32.: Z_ours1[i][j] = 32.
            if Z_ours1[i][j] < -32.: Z_ours1[i][j] = -32.

    col = ('#ff0000', '#ff3300', '#ff6600', '#ff9900', '#ffbb00', '#ffee00', '#eeff00', '#bbff00', '#99ff00', '#66ff00', '#33ff00', '#00ff00')
    tick = [-32., -16., -8., -4., -2., -1., 0., 1., 2., 4., 8., 16., 32.]

    # Ours 1acc, 2rej
    ax = plt.subplot(1, 2, 1)  #figsize=(8, 8), dpi=100
    CS = ax.contourf(X, Y, Z_ours0, tick, colors=col)
    cbar = plt.colorbar(CS, ticks=tick)
    cbar.set_label('Score', fontdict={'size': small_font})
    cbar.ax.tick_params(labelsize=small_font)
    CS = ax.contour(X, Y, Z_ours0, tick, colors=col)
    #plt.clabel(CS, inline=True, fontsize=10, colors='black')
    ax.set_xlabel('Negative reviewer\'s prediction $\hat{P}_{0,1}$', fontsize=small_font)
    ax.set_ylabel('Positive reviewer\'s prediction $\hat{P}_{1,1}$', fontsize=small_font)
    ax.set_title('1 accept and 2 reject', fontsize=big_font)
    plt.xticks(size=small_font)
    plt.yticks(size=small_font)

    # Ours 2acc, 1rej
    ax = plt.subplot(1, 2, 2)  #figsize=(8, 8), dpi=100
    CS = ax.contourf(X, Y, Z_ours1, tick, colors=col)
    cbar = plt.colorbar(CS, ticks=tick)
    cbar.set_label('Score', fontdict={'size': small_font})
    cbar.ax.tick_params(labelsize=small_font)
    CS = ax.contour(X, Y, Z_ours1, tick, colors=col)
    #plt.clabel(CS, inline=True, fontsize=10, colors='black')
    ax.set_xlabel('Negative reviewer\'s prediction $\hat{P}_{0,1}$', fontsize=small_font)
    ax.set_ylabel('Positive reviewer\'s prediction $\hat{P}_{1,1}$', fontsize=small_font)
    ax.set_title('2 accept and 1 reject', fontsize=big_font)
    plt.xticks(size=small_font)
    plt.yticks(size=small_font)

    #plt.suptitle('Comparasion of our method and baseline under $p_A=%.2f$ and $p_B=%.2f$' % (p_A, p_B), fontsize=30)
    plt.tight_layout()
    plt.savefig('./fig/score_sketch/output.pdf')


import sys
if __name__ == '__main__':

    args = sys.argv

    arg1 = -1 if len(args) == 1 else int(args[1])

    if arg1 == 0 or arg1 == -1:
        calc.bias_type = 'opposite'
        paint_line_chart('load', 3, which_base='SP')
        paint_line_chart('load', 5, which_base='SP')
    if arg1 == 1 or arg1 == -1:
        calc.bias_type = 'accept'
        paint_line_chart('load', 3, which_base='SP')
        paint_line_chart('load', 5, which_base='SP')
    if arg1 == 2 or arg1 == -1:
        calc.bias_type = 'opposite'
        paint_line_chart('load', 3, which_base='Baseline')
        paint_line_chart('load', 5, which_base='Baseline')
    if arg1 == 3 or arg1 == -1:
        calc.bias_type = 'accept'
        paint_line_chart('load', 3, which_base='Baseline')
        paint_line_chart('load', 5, which_base='Baseline')
