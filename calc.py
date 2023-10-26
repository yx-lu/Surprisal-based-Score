from cmath import inf
import numpy as np
import copy

debug = False  # Set debug=True to allow debug output
eps = 1e-12  # Error tolerance for determining equality of real numbers
U = np.array([[1. / 3, 1. / 6], [1. / 6, 1. / 3]])  # Setting of  prior distribution Q: uniform
#U = np.array([[3. / 8, 1. / 8], [1. / 8, 3. / 8]])  # Setting of  prior distribution Q: Beta(.5,.5)
#U = np.array([[2. / 7, 3. / 14], [3. / 14, 2. / 7]])  # Setting of  prior distribution Q: Beta(3,3)
#U = np.array([[101. / 402, 50. / 201], [50. / 201, 101. / 402]])  # Setting of prior distribution Q: Beta(100,100)
bias_type = 'opposite'  # Setting of the bias type, options: 'opposite', 'accept', 'reject'


def set_U(_U):  # Reset the prior distribution Q
    global U
    if type(_U) == str:
        if _U == 'Beta(.5,.5)':
            U = np.array([[3. / 8, 1. / 8], [1. / 8, 3. / 8]])  # Beta(.5,.5)
        elif _U == 'uniform' or _U == 'Beta(1,1)':
            U = np.array([[1. / 3, 1. / 6], [1. / 6, 1. / 3]])
        elif _U == 'Beta(3,3)':
            U = np.array([[2. / 7, 3. / 14], [3. / 14, 2. / 7]])
        elif _U == 'Beta(1,3)':
            U = np.array([[3. / 5, 3. / 20], [3. / 20, 1. / 10]])
        elif _U == 'Beta(2,5)':
            U = np.array([[15. / 28, 5. / 28], [5. / 28, 3. / 28]])
        elif _U == 'Beta(.33,.33)':
            U = np.array([[2. / 5, 1. / 10], [1. / 10, 2. / 5]])
        elif _U == 'Beta(.1,.1)':
            U = np.array([[11. / 24, 1. / 24], [1. / 24, 11. / 24]])
        else:
            assert (0)
    else:
        print(type(_U))
        assert (0)
        U = _U


def sample(p):  # Sampling a Bernoulli random variable with expected value p
    return 1 if (np.random.uniform() < p) else 0


def calc_score_from_feedback(_data):  # Calculating empirical surprisal based score
    data = copy.deepcopy(_data)
    data.sort()
    n = len(data)
    sum = 0  # number of 1
    for x in data:
        sum += x[0]
    if sum == n: return float('inf')
    if sum == 0: return -float('inf')
    P = [[0., 0.], [0., 0.]]
    for i in range(n - sum):
        assert (data[i][0] == 0)
        P[0][0] += 1. - data[i][1]
        P[0][1] += data[i][1]
    P[0][0] /= n - sum
    P[0][1] /= n - sum
    for i in range(n - sum, n):
        assert (data[i][0] == 1)
        P[1][0] += 1. - data[i][1]
        P[1][1] += data[i][1]
    P[1][0] /= sum
    P[1][1] /= sum

    def gendiv(p, q):
        if p == 0. and q == 0.: return 1.
        if q == 0.: return float('inf') if p > 0 else -float('inf')
        return p / q

    q = [0., 0.]
    q[0] = 1. / (1. + gendiv(P[0][1], P[1][0]))
    q[1] = 1. / (gendiv(P[1][0], P[0][1]) + 1.)

    w = [float(n - sum) / n, float(sum) / n]

    score = (w[1] - q[1]) / np.sqrt(q[0] * q[1] * (P[1][1] - P[0][1]))

    return score


def numerical_correct_rate(lam_A, lam_B, n):  # the correct frequency (10000 times) of our method and baseline when fixing lambda and n
    if bias_type == 'opposite':
        M_A = np.array([[1. - lam_A, lam_A], [0., 1.]])
        M_B = np.array([[1., 0.], [lam_B, 1. - lam_B]])
    elif bias_type == 'accept':
        M_A = np.array([[1. - lam_A, lam_A], [0., 1.]])
        M_B = np.array([[1. - lam_A, lam_A], [0., 1.]])
    elif bias_type == 'reject':
        M_A = np.array([[1., 0.], [lam_B, 1. - lam_B]])
        M_B = np.array([[1., 0.], [lam_B, 1. - lam_B]])
    hatU_A = np.dot(np.dot(np.transpose(M_A), U), M_A)
    hatU_B = np.dot(np.dot(np.transpose(M_B), U), M_B)
    cnt_ours = 0.
    cnt_baseline = 0.
    tms = 10000
    for t in range(tms):
        # paper A has a long proof, thus has M+ as the bias
        p_A = np.random.uniform(0., 1.)
        data_A = []
        base_A = 0
        for i in range(n):
            truesig_Ai = sample(p_A)
            sig_Ai = 1 if truesig_Ai == 1 else sample(lam_A)  # not adaptive! [opposite]
            pre_Ai = hatU_A[sig_Ai][1] / (hatU_A[sig_Ai][0] + hatU_A[sig_Ai][1])
            data_A.append((sig_Ai, pre_Ai))
            base_A += sig_Ai
        score_A = calc_score_from_feedback(data_A)

        # paper B has a short proof, thus has M- as the bias
        p_B = np.random.uniform(0., 1.)
        data_B = []
        base_B = 0
        for i in range(n):
            truesig_Bi = sample(p_B)
            sig_Bi = 0 if truesig_Bi == 0 else sample(1 - lam_B)
            pre_Bi = hatU_B[sig_Bi][1] / (hatU_B[sig_Bi][0] + hatU_B[sig_Bi][1])
            data_B.append((sig_Bi, pre_Bi))
            base_B += sig_Bi
        score_B = calc_score_from_feedback(data_B)

        truth = int(p_A < p_B)
        ours = sample(.5) if (score_A == score_B) or (abs(score_A - score_B) < eps) else int(score_A < score_B)
        baseline = sample(.5) if base_A == base_B or (abs(base_A - base_B) < eps) else int(base_A < base_B)
        # (inf==inf) == True
        cnt_ours += (ours == truth)
        cnt_baseline += (baseline == truth)

        if debug:
            print(data_A)
            print(data_B)
            print(p_A, p_B)
            print(score_A, score_B)
            print(ours == truth)
            print("")
    cnt_ours /= tms
    cnt_baseline /= tms
    print("our performance =", cnt_ours)
    print("baseline performance =", cnt_baseline)
    return (cnt_ours, cnt_baseline)


def calc_prob_greater(p_A, p_B, lam_A, lam_B, n, which_base='Baseline'):  # n = 0 means n = inf
    # calculating the probability of score_A > score_B of our method and baseline when fixing p, lambda and n

    if bias_type == 'opposite':
        M_A = np.array([[1. - lam_A, lam_A], [0., 1.]])
        M_B = np.array([[1., 0.], [lam_B, 1. - lam_B]])
        bp_A = p_A + lam_A * (1 - p_A)
        bp_B = (1 - lam_B) * p_B
    elif bias_type == 'accept':
        M_A = np.array([[1. - lam_A, lam_A], [0., 1.]])
        M_B = np.array([[1. - lam_B, lam_B], [0., 1.]])
        bp_A = p_A + lam_A * (1 - p_A)
        bp_B = p_B + lam_B * (1 - p_B)
    elif bias_type == 'reject':
        M_A = np.array([[1., 0.], [lam_A, 1. - lam_A]])
        M_B = np.array([[1., 0.], [lam_B, 1. - lam_B]])
        bp_A = (1 - lam_A) * p_A
        bp_B = (1 - lam_B) * p_B
    elif bias_type == 'twoside':
        M_A = np.array([[1. - lam_A / 2, lam_A / 2], [lam_A / 2, 1. - lam_A / 2]])
        M_B = np.array([[1. - lam_B / 2, lam_B / 2], [lam_B / 2, 1. - lam_B / 2]])
        bp_A = (1 - lam_A / 2) * p_A + (lam_A / 2) * (1 - p_A)
        bp_B = (1 - lam_B / 2) * p_B + (lam_B / 2) * (1 - p_B)
    hatU_A = np.dot(np.dot(np.transpose(M_A), U), M_A)
    hatU_B = np.dot(np.dot(np.transpose(M_B), U), M_B)

    ans_ours = 0.
    ans_baseline = 0.
    ans_sp = 0.

    def binom(n, m):
        ret = 1.
        for i in range(n - m + 1, n + 1):
            ret *= i
        for i in range(1, m + 1):
            ret /= i
        return ret

    bin = [0. for i in range(n + 1)]
    for i in range(n + 1):
        bin[i] = binom(n, i)

    for sum_A in range(n + 1):
        for sum_B in range(n + 1):
            # sig_A = [(1 if i < sum_A else 0) for i in range(n)]
            # sig_B = [(1 if i < sum_B else 0) for i in range(n)]

            def cmp(sc_A, sc_B):
                if abs(sc_A - sc_B) < eps: return .5
                elif sc_A > sc_B: return 1.
                else: return 0.

            if n > 0:
                w_A = [float(n - sum_A) / n, float(sum_A) / n]
                w_B = [float(n - sum_B) / n, float(sum_B) / n]
                prob = bin[sum_A] * bin[sum_B]
                for i in range(n):
                    prob *= (bp_A if i < sum_A else (1 - bp_A))
                    prob *= (bp_B if i < sum_B else (1 - bp_B))

            else:
                w_A = [1 - bp_A, bp_A]
                w_B = [1 - bp_B, bp_B]
                prob = 1.

            if prob < eps: continue

            ans_baseline += prob * cmp(w_A[1], w_B[1])

            if (w_A[0] < eps) or (w_A[1] < eps) or (w_B[0] < eps) or (w_B[1] < eps):
                ans_ours += prob * cmp(w_A[1], w_B[1])
                ans_sp += prob * cmp(w_A[1], w_B[1])
            else:
                q_A = [hatU_A[0][0] + hatU_A[0][1], hatU_A[1][0] + hatU_A[1][1]]

                score_A = (w_A[1] - q_A[1]) / np.sqrt(q_A[0] * q_A[1] * (hatU_A[1][1] / q_A[1] - hatU_A[0][1] / q_A[0]))
                spscore_A = w_A[1] / q_A[1] - w_A[0] / q_A[0]

                q_B = [hatU_B[0][0] + hatU_B[0][1], hatU_B[1][0] + hatU_B[1][1]]

                score_B = (w_B[1] - q_B[1]) / np.sqrt(q_B[0] * q_B[1] * (hatU_B[1][1] / q_B[1] - hatU_B[0][1] / q_B[0]))
                spscore_B = w_B[1] / q_B[1] - w_B[0] / q_B[0]

                if abs(score_A - score_B) < eps:
                    ans_ours += prob * .5
                elif score_A > score_B:
                    ans_ours += prob

                if abs(spscore_A - spscore_B) < eps:
                    ans_sp += prob * .5
                elif spscore_A > spscore_B:
                    ans_sp += prob

                # if (abs(score_A - score_B) > eps) and (abs(spscore_A - spscore_B) > eps) and ((score_A > score_B) != (spscore_A > spscore_B)):
                #     if (score_A > score_B) and (spscore_A < spscore_B):
                #         print(p_A, p_B, lam_A, lam_B, n, "   |   ", sum_A, sum_B, (score_A > score_B), (spscore_A > spscore_B))

                if debug:
                    if score_A > score_B:
                        ans_ours += prob
                        print(prob, score_A, score_B)
                        print(w_A[1], q_A[1], w_B[1], q_B[1])
    if which_base == 'SP':
        ans_baseline = ans_sp
    return (ans_ours, ans_baseline)


def calc_correct_rate(p_A,
                      p_B,
                      lam_A,
                      lam_B,
                      n,
                      which_base='Baseline'):  # calculating the correct rate of our method and baseline when fixing p, lambda and n
    if p_A == p_B: return (.5, .5)
    elif p_A > p_B: return calc_prob_greater(p_A, p_B, lam_A, lam_B, n, which_base)
    else:
        res = calc_prob_greater(p_A, p_B, lam_A, lam_B, n, which_base)
        return (1. - res[0], 1. - res[1])