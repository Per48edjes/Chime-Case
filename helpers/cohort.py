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
    def __init__(self, variant=None, segment=None):
        self._variant = variant
        self._segment = segment
        self._cac = df_acq_agg.at[variant, "CAC"]

        # Properly cohort based on user inputs
        if segment and variant:
            data = df_perf.query("`Test Group` == @variant and Segment == @segment")
        elif variant:
            data = df_perf.query("`Test Group` == @variant")
        else:
            raise KeyError("Assert a cohort definition.")

        # Index on Months Since Conversion
        self._df = (
            data.sort_values("Months Since Conversion")
            .groupby(["Months Since Conversion"])
            .sum()
        ).sort_index()

        # Add analysis columns for each cohort
        self.__add_ltv_cols()

    def __repr__(self):
        return f"Cohort({self._variant}, {self._segment})"

    @property
    def survival_events(self):
        survival_table = self.df.reset_index()[
            ["Months Since Conversion", "MoM AU Churn Count", "censor"]
        ].set_index("Months Since Conversion")
        T, E, W = survival_events_from_table(
            survival_table, "MoM AU Churn Count", "censor"
        )
        return T, E, W

    def __add_ltv_cols(self):
        """
        Adds analysis columns used in the determination of LTV
        """

        # Add analysis columns
        self.df["Original Cohort User Count"] = self.df["Active User Count"].max()

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
            self.df["Spend Tracker Active User Count"] / self.df["Active User Count"]
        )

        self.df["Average Spend Tracker Cost per Active User"] = (
            self.df["Spend Tracker Usage Rate"] * SPEND_TRACKER_PMPM
        )

        # Calculate empirical Contribution Margin
        self.df["Contribution Margin per Active User"] = (
            self.df["Gross Profit per Active User"]
            - self.df["Average Spend Tracker Cost per Active User"]
        )

        # Censored (for lifelines)
        self.df["censor"] = 0
        self.df.loc[self.df.index.max(), "censor"] = self.df.at[
            self.df.index.max(), "Active User Count"
        ]

        return self.df

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
                breakpoints = [14]
            return PiecewiseExponentialFitter(breakpoints=breakpoints).fit(
                T, E, weights=W
            )
        elif model == "KaplanMeier":
            return KaplanMeierFitter().fit(T, E, weights=W, label=label)
        elif model == "Exponential":
            return ExponentialFitter().fit(T, E, weights=W, label=label)
        elif model == "Weibull":
            return WeibullFitter().fit(T, E, weights=W, label=label)
        else:
            raise ValueError("Selected model not available.")

    def estimate_ltv(
        self,
        d=0.01,
        m_terminal_multiplier=1,
        h_terminal_multiplier=1,
        use_empirical_estimator=True,
    ):
        """
        Calculates the expected LTV for the cohort, returning estimate LTV and
        the portions from what's been modeled vs. what is in the terminal value,
        respectively
        """
        # Set up
        survival_model = self.fit_survival_model()
        discount_factors = np.array([(1 + d) ** (-i) for i in self.df.index])
        if use_empirical_estimator:
            S_hat = self.df["Overall AU Retention Rate"]
        else:
            S_hat = survival_model.predict(self.df.index)

        h_terminal_default = survival_model.hazard_.iat[-1, 0]
        h_terminal = h_terminal_default * h_terminal_multiplier
        m_terminal = self.df.at[
            self.df.index.max(), "Contribution Margin per Active User"
        ]

        # Modeled LTV
        modeled_ltv = (self.df["Contribution Margin per Active User"] * S_hat).dot(
            discount_factors
        )

        # Terminal LTV
        terminal_ltv = (
            discount_factors[-1] * h_terminal * m_terminal_multiplier * m_terminal
        ) / (1 + d - h_terminal)
        print(f"m_terminal: {m_terminal}")
        return modeled_ltv + terminal_ltv, modeled_ltv, terminal_ltv


# Instantiate cohort objects
control = Cohort("Control")
test = Cohort("Test")

control_A = Cohort("Control", "Segment A")
control_B = Cohort("Control", "Segment B")
test_A = Cohort("Test", "Segment A")
test_B = Cohort("Test", "Segment B")
