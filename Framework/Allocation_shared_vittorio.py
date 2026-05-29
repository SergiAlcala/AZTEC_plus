import numpy as np
import scipy.optimize
from statsmodels.distributions.empirical_distribution import ECDF
import scipy.stats
import pybobyqa
import sys


def upload_static_shared(phi, alpha, delay, num_services, B):
    cap_static = np.load('')  # load static forecasted by block I
    cap_shared = np.load('')  # load shared forecasted by block II

    cd_static = np.zeros((cap_static.shape[0] * delay, num_services, B))
    cs_shared = np.zeros((cap_shared.shape[0] * delay, B))

    # You can replace the for loops with np.repeat()
    for idx, element in enumerate(cap_static):
        cd_static[idx*delay:(idx+1)*delay] = element

    for idx, element in enumerate(cap_shared):
        cs_shared[idx*delay:(idx+1)*delay] = cap_shared[idx, 0]

    dist_cd = scipy.stats.norm(loc=cd_static.mean(axis=-1),
                               scale=cd_static.std(axis=-1))
    dist_cs = scipy.stats.norm(loc=cs_shared.mean(axis=-1),
                               scale=cs_shared.std(axis=-1))

    upper_cd_static = dist_cd.ppf(0.999)
    upper_cd_static[np.where(np.isnan(upper_cd_static))] = (
        cd_static[np.where(np.isnan(upper_cd_static))][:, 0])
    upper_cs_shared = dist_cs.ppf(0.999)
    upper_cs_shared[np.where(np.isnan(upper_cs_shared))] = (
        cs_shared[np.where(np.isnan(upper_cs_shared))][:, 0])

    return upper_cd_static, upper_cs_shared


def cost_func_evaluation(c_plus, forecasting, static, phi, alpha):
    ''' Evaluate the cost of the allocation given cplus selected.'''
    # cost_shared = phi
    cost_sla = alpha
    total_cost = 0
    for i in range(forecasting.shape[0]):  # shape[0] = num services
        # total_cost += cost_shared * c_plus[i]
        ecdf = ECDF(forecasting[i])
        total_cost += (1-ecdf(static[i] + c_plus[i])) * cost_sla
    return total_cost


def apps_need_shared(forecasting, static):
    ''' Return the slices that need shared capacity '''
    index_app = []
    for app in range(forecasting.shape[0]):
        dist = ECDF(forecasting[app])
        if (1 - dist(static[app])) != 0:
            index_app.append(app)
    return index_app


def fun(pn, dist, static, max_shared, phi, alpha, lower_bound, upper_bound):
    """ Function with only one variable to be minimized through
        bounded golden search
    """
    num_app = dist.shape[0]
    p_0 = np.ones(num_app) * 0.5
    opt = pybobyqa.solve(cost_func_evaluation_p_fix_p4, p_0,
                         bounds=(lower_bound[:num_app], upper_bound[:num_app]),
                         args=(dist, static, max_shared, pn, phi, alpha),
                         objfun_has_noise=True)
    return opt.f


def cost_func_evaluation_p(p, forecasting, static, shared_available, phi,
                           alpha):
    ''' Evaluate the cost of the allocation given cplus selected.'''
    # cost_shared = phi
    cost_sla = alpha
    total_cost = 0
    cplus = return_cplus(p, shared_available)
    for i in range(forecasting.shape[0]):  # shape[0] = num services
        # total_cost += cost_shared * cplus[i]
        ecdf = ECDF(forecasting[i])
        total_cost += (1-ecdf(static[i] + cplus[i])) * cost_sla
    return total_cost


def cost_func_evaluation_p_fix_p4(p, forecasting, static, shared_available, p4,
                                  phi, alpha):
    ''' Evaluate the cost of the allocation given cplus selected.'''
    # cost_shared = phi
    cost_sla = alpha
    total_cost = 0
    cplus, cplus_4 = return_cplus_fix_p4(p, shared_available, p4)
    for i in range(forecasting.shape[0]):  # shape[0] = num services
        # total_cost += cplus[i] * phi
        ecdf = ECDF(forecasting[i])
        total_cost += (1-ecdf(static[i] + cplus[i])) * cost_sla
    return total_cost


def return_cplus(p_vector, shared_available):
    """ Return c_plus given p and max amount of shared for
    that phi,alpha and time. """
    cplus = np.zeros(p_vector.shape)
    products = np.zeros(p_vector.shape[0]-1)
    for i in range(len(cplus)-1):
        products[i] = np.true_divide(np.prod(p_vector[i:-1]),
                                     np.prod((1-p_vector)[i:-1]))
    cplus[-1] = np.true_divide(shared_available * p_vector[-1],
                               np.sum(products) + 1)
    for i in range(len(cplus)-1):
        cplus[i] = products[i] * cplus[-1]
    return cplus


def return_cplus_fix_p4(p_vector, shared_available, p4):
    """ Return c_plus given p and max amount of shared for
    that phi,alpha and time. """
    cplus = np.zeros(p_vector.shape)
    products = np.zeros(p_vector.shape[0])
    for i in range(len(cplus)):
        products[i] = np.true_divide(np.prod(p_vector[i:]),
                                     np.prod((1-p_vector)[i:]))
    cplus4 = np.true_divide(shared_available * p4,
                            np.sum(products) + 1)
    for i in range(len(cplus)):
        cplus[i] = products[i] * cplus4
    return cplus, cplus4


def return_p(cplus, max_shared):
    p = np.zeros(cplus.shape)
    for i in range(cplus.shape[0] - 1):
        p[i] = cplus[i] / (cplus[i]+cplus[i+1])
    p[-1] = np.sum(cplus) / max_shared
    return p


def main(phi, alpha):
    lookback = 6
    num_services = 5
    delay = 12
    B = 100
    youtube = np.load('youtube/time_load_matrix.npy')
    snapchat = np.load('snapchat/time_load_matrix.npy')
    facebook = np.load('facebook/time_load_matrix.npy')
    instagram = np.load('instagram/time_load_matrix.npy')
    itunes = np.load('itunes/time_load_matrix.npy')
    max_youtube = np.max(np.sum(np.sum(youtube, axis=-1), axis=-1))
    max_snapchat = np.max(np.sum(np.sum(snapchat, axis=-1), axis=-1))
    max_facebook = np.max(np.sum(np.sum(facebook, axis=-1), axis=-1))
    max_instagram = np.max(np.sum(np.sum(instagram, axis=-1), axis=-1))
    max_itunes = np.max(np.sum(np.sum(itunes, axis=-1), axis=-1))
    agg_output_youtube = np.true_divide(np.sum(
        np.sum(youtube, axis=-1), axis=-1), max_youtube)
    agg_output_snapchat = np.true_divide(np.sum(
        np.sum(snapchat, axis=-1), axis=-1), max_snapchat)
    agg_output_facebook = np.true_divide(np.sum(
        np.sum(facebook, axis=-1), axis=-1), max_facebook)
    agg_output_itunes = np.true_divide(np.sum(
        np.sum(itunes, axis=-1), axis=-1), max_itunes)
    agg_output_instagram = np.true_divide(np.sum(
        np.sum(instagram, axis=-1), axis=-1), max_instagram)
    test_index = range(2016*10+6, 2016*11)
    agg_output = np.stack((agg_output_youtube[:-1], agg_output_snapchat,
                           agg_output_facebook[:-1], agg_output_instagram,
                           agg_output_itunes[:-1]), axis=-1)
    max_concatenated = np.stack((max_youtube, max_snapchat,
                                 max_facebook, max_instagram, max_itunes))
    mae_forecasting = np.load('')  # Load the real load forecasted by helper

    static, shared = upload_static_shared(phi, alpha, delay, num_services, B)

    max_shared = np.load('')  # load the value used to normalize the shared
    shared_norm = shared * max_shared / 10e9
    static_norm = static * max_concatenated / 10e9
    mae_norm = np.zeros(mae_forecasting.shape)
    for sample in range(B):
        mae_norm[:, :, sample] = (mae_forecasting[:, :, sample]
                                  * max_concatenated / 10e9)
    output_norm = agg_output[test_index] * max_concatenated / 10e9
    slas = np.zeros((num_services, 2010))
    cplus = np.zeros((num_services, 2010))
    p = np.zeros((num_services, 2010))
    lower_bound = np.ones(num_services) * 1e-10
    upper_bound = np.ones(num_services) * 0.99999
    idx_time = 1
    for time in range(0, 2010):
        forecasting = mae_norm[time]
        idx_app = apps_need_shared(forecasting, static_norm[time])
        if len(idx_app) > 1:
            test = scipy.optimize.minimize_scalar(fun,
                                                  bounds=(1e-10, 0.99999),
                                                  method='bounded',
                                                  args=(forecasting
                                                        [idx_app[:-1]],
                                                        static_norm[time,
                                                                    idx_app[
                                                                        :-1]],
                                                        shared_norm[time], phi,
                                                        alpha, lower_bound,
                                                        upper_bound))
            p4 = test.x
            p_0 = np.ones(len(idx_app)-1) * 0.5
            prova = pybobyqa.solve(cost_func_evaluation_p_fix_p4, p_0,
                                   bounds=(lower_bound[:len(idx_app)-1],
                                           upper_bound[:len(idx_app)-1]),
                                   args=(forecasting[idx_app[:-1]],
                                         static_norm[time, idx_app[:-1]],
                                         shared_norm[time], p4, phi, alpha),
                                   objfun_has_noise=True)
            p_0 = np.zeros(len(idx_app))
            p_0[:len(idx_app)-1] = prova.x
            p_0[-1] = p4
            prova_2 = pybobyqa.solve(cost_func_evaluation_p, p_0,
                                     bounds=(lower_bound[:len(idx_app)],
                                             upper_bound[:len(idx_app)]),
                                     args=(forecasting[idx_app],
                                           static_norm[time, idx_app],
                                           shared_norm[time], phi, alpha),
                                     objfun_has_noise=True)
            cplus[idx_app, time] = return_cplus(prova_2.x, shared_norm[time])
            p[idx_app, time] = prova_2.x
        elif len(idx_app) == 1:
            cplus[idx_app, time] = shared_norm[time]
        shared_needed = output_norm[time] - static_norm[time]
        for app in range(num_services):
            if shared_needed[app] > cplus[app, time]:
                slas[app, time] += 1
        if time == idx_time * 200:
            print(time)
            print('Youtube :', slas[0, :time].sum())
            print('Snapchat :', slas[1, :time].sum())
            print('Facebook :', slas[2, :time].sum())
            print('iTunes :', slas[3, :time].sum())
            print('Instagram :', slas[4, :time].sum())
            np.save('', cplus)  # save the shared allocated
            np.save('', slas)  # save sla violations
            idx_time += 1
    np.save('', cplus)  # save the final shared allocated
    np.save('', slas)  # save the final sla violations
    return 0


if __name__ == "__main__":
    main(float(sys.argv[1]), int(sys.argv[2]))
