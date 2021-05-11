import numpy as np
from lifelines.fitters.exponential_fitter import ExponentialFitter
from lifelines.fitters.kaplan_meier_fitter import KaplanMeierFitter
from lifelines.fitters.piecewise_exponential_fitter import \
    PiecewiseExponentialFitter
from lifelines.fitters.weibull_fitter import WeibullFitter
from lifelines.utils import survival_events_from_table

from helpers import GROSS_MARGIN, SPEND_TRACKER_PMPM
from helpers.load import df_acq_agg, df_perf


class Cohort:
    def __init__(
        self, test_group=None, segment=None, spend_tracker_pmpm=SPEND_TRACKER_PMPM
    ):
        self._test_group = test_group
        self._segment = segment
        self._cac = df_acq_agg.at[test_group, "CAC"]
        self._spend_tracker_pmpm = spend_tracker_pmpm

        # Properly cohort based on user inputs
        if segment and test_group:
            data = df_perf.query("`Test Group` == @test_group and Segment == @segment")
        elif test_group:
            data = df_perf.query("`Test Group` == @test_group")
        else:
            raise KeyError("Assert a cohort definition.")

        # Index on Months Since Conversion
        self._df = (
            data.sort_values("Months Since Conversion")
            .groupby(["Months Since Conversion"])
            .sum()
        ).sort_index()

        # Add analysis columns for each cohort
        self.__add_ltv_cols(spend_tracker_pmpm)

    def __repr__(self):
        return f"Cohort({self._test_group}, {self._segment})"

    @staticmethod
    def cohort_roi(control, test, cohort_str="Overall"):
        roi_cols = [
            "Gross Profit per Cohort User",
            "Average Spend Tracker Cost per Cohort User",
        ]
        term_nominal_cashflows = (test.df[roi_cols] - control.df[roi_cols]).sum()
        lower_cac_benefit = np.array([control.cac - test.cac, 0])
        roi = term_nominal_cashflows + lower_cac_benefit
        print(f"{cohort_str} 18-month ROI: {roi[0]/roi[1]-1:.2%}")

    @property
    def survival_events(self):
        survival_table = self.df.reset_index()[
            ["Months Since Conversion", "MoM AU Churn Count", "censor"]
        ].set_index("Months Since Conversion")
        T, E, W = survival_events_from_table(
            survival_table, "MoM AU Churn Count", "censor"
        )
        return T, E, W

    def __add_ltv_cols(self, spend_tracker_pmpm=SPEND_TRACKER_PMPM):
        """
        Adds analysis columns used in the determination of LTV
        """

        # Add analysis columns
        self.df["Original Cohort User Count"] = self.df["Active User Count"].max()

        self.df["Gross Profit per Cohort User"] = (
            self.df["Total Purchase Dollars"]
            * GROSS_MARGIN
            / self.df["Original Cohort User Count"]
        )

        self.df["Gross Profit per Active User"] = (
            self.df["Total Purchase Dollars"]
            * GROSS_MARGIN
            / self.df["Active User Count"]
        )

        self.df["Prior Month Active User Count"] = (
            self.df["Active User Count"]
            .shift(1)
            .combine_first(self.df["Original Cohort User Count"])
        )

        self.df["MoM AU Retention Rate"] = (
            self.df["Active User Count"] / self.df["Prior Month Active User Count"]
        )

        self.df["Overall AU Retention Rate"] = (
            self.df["Active User Count"] / self.df["Original Cohort User Count"]
        )

        self.df["MoM AU Churn Count"] = (
            self.df["Prior Month Active User Count"] - self.df["Active User Count"]
        )

        self.df["Spend Tracker Usage Rate"] = (
            self.df["Spend Tracker Active User Count"]
            / self.df["Original Cohort User Count"]
        )

        self.df["Average Spend Tracker Cost per Cohort User"] = (
            self.df["Spend Tracker Usage Rate"] * spend_tracker_pmpm
        )

        # Calculate empirical Contribution Margin
        self.df["Contribution Margin per Cohort User"] = (
            self.df["Gross Profit per Cohort User"]
            - self.df["Average Spend Tracker Cost per Cohort User"]
        )

        # Censored (for lifelines)
        self.df["censor"] = 0
        self.df.loc[self.df.index.max(), "censor"] = self.df.at[
            self.df.index.max(), "Active User Count"
        ]

        return self.df

    @property
    def test_group(self):
        return self._test_group

    @property
    def segment(self):
        return self._segment

    @property
    def df(self):
        return self._df

    @property
    def cac(self):
        return self._cac

    @property
    def overall_au_retention(self):
        return self._df["Overall AU Retention Rate"]

    @property
    def original_cohort_size(self):
        return self._df["Original Cohort User Count"].max()

    def fit_survival_model(
        self, model="PiecewiseExponential", breakpoints=None, label=None
    ):
        """
        Fits a survival model to the cohort's performance data
        """
        T, E, W = self.survival_events
        if model == "PiecewiseExponential":
            if not breakpoints:
                breakpoints = [12]
            return PiecewiseExponentialFitter(breakpoints=breakpoints).fit(
                T, E, weights=W, timeline=np.arange(0, 12 * 1000)
            )
        elif model == "KaplanMeier":
            return KaplanMeierFitter().fit(T, E, weights=W, label=label)
        elif model == "Exponential":
            return ExponentialFitter().fit(T, E, weights=W, label=label)
        elif model == "Weibull":
            return WeibullFitter().fit(T, E, weights=W, label=label)
        else:
            raise ValueError("Selected model not available.")

    @property
    def h_terminal(self):
        survival_model = self.fit_survival_model()
        h_terminal = survival_model.hazard_.iat[-1, 0]
        h_terminal_ci = survival_model.confidence_interval_hazard_.iloc[
            -1, :
        ].to_numpy()
        return h_terminal, h_terminal_ci

    @property
    def m_terminal(self):
        m_terminal = self.df.at[
            self.df.index.max(), "Contribution Margin per Cohort User"
        ]
        return m_terminal

    def generate_sim_data(self, uncertainty=1, n=10_000):

        # Define upper and lower bounds for h_terminal uniform distribution
        h_terminal_lower_95_ci, h_terminal_upper_95_ci = self.h_terminal[1]

        # Generate data
        h_terminals = np.random.uniform(
            low=h_terminal_lower_95_ci, high=h_terminal_upper_95_ci, size=n
        )
        m_terminals = np.random.normal(loc=self.m_terminal, scale=uncertainty, size=n)

        return np.stack([h_terminals, m_terminals]).T

    def estimate_ltv(
        self, m_terminal=None, h_terminal=None, d=0.01, use_empirical_estimator=True
    ):

        # Set up
        discount_factors = np.array([(1 + d) ** (-i) for i in self.df.index])
        if use_empirical_estimator:
            S_hat = self.df["Overall AU Retention Rate"]
        else:
            S_hat = self.fit_survival_model().predict(self.df.index)

        # Modeled LTV
        modeled_ltv = (self.df["Contribution Margin per Cohort User"] * S_hat).dot(
            discount_factors
        )

        # Terminal LTV
        if not h_terminal:
            h_terminal = self.h_terminal[0]
        if not m_terminal:
            m_terminal = self.m_terminal

        terminal_ltv = (
            discount_factors[-1]  # Discount residual value to PV
            * (1 - h_terminal)  # Retention rate
            * m_terminal  # Terminal contribution margin
        ) / (1 + d - (1 - h_terminal))

        return modeled_ltv + terminal_ltv


# Instantiate cohort objects
control = Cohort("Control")
test = Cohort("Test")

control_A = Cohort("Control", "Segment A")
control_B = Cohort("Control", "Segment B")
test_A = Cohort("Test", "Segment A")
test_B = Cohort("Test", "Segment B")
