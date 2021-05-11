import numpy as np

from .cohort import Cohort

SPEND_TRACKER_PMPM_DOMAIN = np.linspace(0, 5, 100 + 1)


def run_sim(cohort, uncertainty=2):
    data = list(cohort.generate_sim_data(uncertainty=uncertainty))
    trials = []
    for h_terminal, m_terminal in data:
        trials.append(cohort.estimate_ltv(h_terminal=h_terminal, m_terminal=m_terminal))
    return trials


def calculate_spend_tracker_breakeven(test_group=None, segment=None, comp_cohort=None):
    trials = []
    for cost in SPEND_TRACKER_PMPM_DOMAIN:
        cohort = Cohort(test_group, segment, cost)
        trials.append(cohort.estimate_ltv())

    # Identify breakeven price
    diff_vector = np.array(trials) - comp_cohort.estimate_ltv()
    i = np.where(diff_vector > 0, diff_vector, np.inf).argmin()
    return SPEND_TRACKER_PMPM_DOMAIN[i]
