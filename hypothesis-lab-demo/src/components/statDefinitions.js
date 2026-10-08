export const STAT_TOOLTIPS = {
  statistic: {
    title: 'STATISTIC (Test Statistic)',
    body: 'A numerical value computed from your sample data that quantifies how far the observed results deviate from what the null hypothesis predicts, relative to expected sampling variability.',
  },
  p_value: {
    title: 'p-VALUE (Probability Value)',
    body: 'The probability of obtaining results at least as extreme as those observed, assuming the null hypothesis is true. It indicates statistical significance but does not measure the size or importance of an effect.',
  },
  effect_size: {
    title: 'EFFECT SIZE (Eta Squared / η² / Cohen’s d)',
    body: 'A quantitative measure of the magnitude of an observed effect or relationship, independent of sample size. For eta squared (η²), it represents the proportion of total variance in the dependent variable explained by the independent variable (.01 = small, .06 = medium, .14+ = large).',
  },
  effect_size_d: {
    title: "EFFECT SIZE (Cohen's d)",
    body: 'A standardized measure of the magnitude of difference between two groups in standard deviation units, independent of sample size. Standard benchmarks: 0.20 = small, 0.50 = medium, 0.80+ = large.',
  },
  ci_95: {
    title: 'CONFIDENCE INTERVAL (95% CI)',
    body: 'Provides a range of plausible values for the true population parameter, centered on the sample estimate. Narrower intervals indicate greater precision of the estimate.',
  },
  ci: {
    title: 'CONFIDENCE INTERVAL (CI)',
    body: 'A range of plausible values for the true population parameter centered on the sample estimate. A 95% CI means that across repeated sampling, 95% of computed intervals would capture the true parameter.',
  },
  sample_size: {
    title: 'SAMPLE SIZE (N)',
    body: 'The total number of observations, participants, or data points in your sample. Larger sample sizes increase estimate precision, narrow confidence intervals, and boost statistical power to detect true effects.',
  },
  significance_level: {
    title: 'SIGNIFICANCE LEVEL (α / Alpha)',
    body: 'The pre-set probability threshold of committing a Type I error (falsely rejecting a true null hypothesis). Setting α = 0.05 allows a 5% maximum acceptable risk of false-positive conclusions.',
  },
  power: {
    title: 'STATISTICAL POWER (1−β)',
    body: 'The probability that a statistical test will correctly reject a false null hypothesis (detecting a true effect when one exists). Expressed as 1−β; the accepted standard benchmark for adequate power is ≥ 80% (0.80).',
  },
};

export const FOLLOWUP_EXPLANATIONS = {
  followup_covariate: {
    title: 'Add a Covariate and Re-analyze',
    description: 'Include an additional continuous variable (covariate) in the model to statistically control for its influence, reduce error variance, and increase analysis sensitivity.',
    body: 'Include an additional continuous variable (covariate) in the model to statistically control for its influence, reduce error variance, and increase the sensitivity of the analysis. This extends ANOVA to ANCOVA.',
  },
  followup_nonparametric: {
    title: 'Compare with a Non-Parametric Alternative',
    description: 'Run a distribution-free test (e.g., Kruskal-Wallis or Mann-Whitney U) that does not assume normality.',
    body: 'Run a distribution-free test (e.g., Kruskal-Wallis or Mann-Whitney U) that does not assume normality. Use when your data violates parametric assumptions such as normal distribution or homogeneity of variance.',
  },
  followup_subgroup: {
    title: 'Test a Subgroup or Interaction',
    description: 'Examine whether the treatment effect differs across specific subsets of your data, or whether one variable depends on another.',
    body: 'Examine whether the treatment effect differs across specific subsets of your data (subgroup analysis), or whether the effect of one variable depends on the level of another (interaction effect).',
  },
  followup_new_hypothesis: {
    title: 'Create a Related Hypothesis with a Different Outcome',
    description: 'Generate a new hypothesis by changing the outcome variable while keeping the same predictor(s) as an exploratory analysis.',
    body: 'Generate a new, related hypothesis by changing the outcome variable while keeping the same predictor(s). This is an exploratory analysis and should be clearly distinguished from your original confirmatory test.',
  },
};

export const FOLLOWUP_MAP = {
  'Add a covariate': 'followup_covariate',
  'Add a covariate and re-analyze': 'followup_covariate',
  'Compare with non-parametric alternative': 'followup_nonparametric',
  'Compare with a non-parametric alternative': 'followup_nonparametric',
  'Test a subgroup': 'followup_subgroup',
  'Test a subgroup or interaction': 'followup_subgroup',
  'Create a related hypothesis': 'followup_new_hypothesis',
  'Create a related hypothesis with a different outcome': 'followup_new_hypothesis',
};
